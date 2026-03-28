"""
Training script for CryTransformer models with DDP (DistributedDataParallel) support

Supports:
- Single GPU training
- Multi-GPU training with DDP
- Focal loss for class imbalance
- SpecAugment
- SWA (Stochastic Weight Averaging)
- Mixed precision training
- Early stopping and checkpointing

Single GPU Usage:
    python train.py --config configs/model_medium.yaml --train_list audio_list/train.json

Multi-GPU Usage:
    # Method 1: Using torchrun (recommended)
    torchrun --nproc_per_node=4 train.py --config configs/model_medium.yaml --train_list audio_list/train.json

    # Method 2: Using python -m torch.distributed.launch
    python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/model_medium.yaml --train_list audio_list/train.json
"""

import argparse
import os
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))

from dataset.dataset import CryDataset
from dataset.dataloader import collate_fn, worker_init_fn, load_data_dict
from dataset.feature import FeatureExtractor
from dataset.sampler import CrySampler, DistributedCrySampler, SequentialCrySampler
from model import create_model, get_model_summary
from model.ema import ExponentialMovingAverage
from model.loss import create_loss
from model.scheduler import WarmupCosineScheduler
from model.distributed import setup_distributed, cleanup_distributed
from utils import setup_logger, setup_file_logger, get_logger
from utils.config import Config, load_config, save_config


def format_duration(seconds: float) -> str:
    """Format seconds to HH:MM:SS string."""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


class ETATracker:
    """Track epoch times and estimate time remaining."""
    def __init__(self, window_size: int = 5):
        self.epoch_times: list[float] = []
        self.window_size = window_size

    def update(self, elapsed_seconds: float) -> None:
        """Record elapsed time for an epoch."""
        self.epoch_times.append(elapsed_seconds)

    def estimate(self, remaining_epochs: int) -> tuple[str, str]:
        """Estimate remaining time based on average epoch time.

        Returns:
            Tuple of (eta_string, elapsed_string) in HH:MM:SS format
        """
        recent_times = self.epoch_times[-self.window_size:]
        avg_epoch_time = sum(recent_times) / len(recent_times)
        eta_seconds = avg_epoch_time * remaining_epochs
        elapsed_seconds = sum(self.epoch_times)
        return format_duration(eta_seconds), format_duration(elapsed_seconds)


class Trainer:
    """Trainer class for CryTransformer with DDP support"""

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        checkpoint_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        feature_extractor: Optional[torch.nn.Module] = None
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.feature_extractor = feature_extractor

        # Initialize logger early for torch.compile messages
        self.logger = get_logger(__name__)

        # Compile feature extractor with torch.compile (if available and not already compiled)
        if (feature_extractor is not None and
            device.type == 'cuda' and
            hasattr(torch, 'compile') and
            not hasattr(feature_extractor, '_compiled')):
            try:
                self.feature_extractor = torch.compile(
                    feature_extractor,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True
                )
                self.feature_extractor._compiled = True
                if rank == 0:
                    self.logger.info("FeatureExtractor compiled with torch.compile")
            except Exception as e:
                if rank == 0:
                    self.logger.warning(f"torch.compile failed for FeatureExtractor: {e}, using eager mode")

        model = model.to(device)
        self.raw_model = model
        self.model = DDP(model, device_ids=[rank], output_device=rank) if self.is_distributed else model

        # Training config
        self.train_cfg = config.training
        self.model_cfg = config.model

        self.criterion = self._create_loss().to(device)

        # Optimizer (operates on unwrapped model parameters)
        self.optimizer = self._create_optimizer()

        # Training state
        self.current_epoch = 0
        self.start_epoch = 0  # For resuming training
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.global_step = 0
        self.steps_per_epoch = len(train_loader) if train_loader else 0

        # Scheduler (created after steps_per_epoch is known)
        self.scheduler = self._create_scheduler()

        # TensorBoard writer (only on rank 0 when log_dir is provided)
        self.writer = None
        if rank == 0 and log_dir is not None:
            self.writer = SummaryWriter(log_dir)
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # SWA
        self.swa_model = None
        self.swa_n = 0
        if self.train_cfg.use_swa and rank == 0:
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.raw_model)

        # EMA (Exponential Moving Average)
        self.ema = None
        if self.train_cfg.use_ema:
            self.ema = ExponentialMovingAverage(self.raw_model, decay=self.train_cfg.ema_decay)

        # Scaler for mixed precision
        self.scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

        # Use root logger (configured in main) - inherits handlers from root
        # Logger was already set up earlier in __init__

        # Load checkpoint if provided (handles model weights and training state)
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint from file and restore all training state."""
        if self.rank == 0:
            self.logger.info(f"Loading checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._restore_from_checkpoint(checkpoint)

    def _restore_from_checkpoint(self, checkpoint: Dict):
        """Restore model weights and training state from checkpoint dict."""
        def _restore(key: str, restore_fn, name: str) -> None:
            """Helper to restore state with logging."""
            if key in checkpoint:
                restore_fn(checkpoint[key])
                if self.rank == 0:
                    self.logger.info(f"Restored {name}")

        # Restore model weights
        _restore('model_state_dict', self.raw_model.load_state_dict, "model weights")

        # Restore epoch (for resuming training loop)
        if 'epoch' in checkpoint:
            self.current_epoch = checkpoint['epoch']
            self.start_epoch = self.current_epoch
            if self.rank == 0:
                self.logger.info(f"Restored epoch: {self.current_epoch}")

        # Restore optimizer, EMA, scheduler states
        _restore('optimizer_state_dict', self.optimizer.load_state_dict, "optimizer state")

        if self.ema is not None:
            _restore('ema_state_dict', self.ema.load_state_dict, "EMA state")

        if self.scheduler is not None:
            _restore('scheduler_state_dict', self.scheduler.load_state_dict, "scheduler state")

        # Restore global step and best metric
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
            if self.rank == 0:
                self.logger.info(f"Restored global_step: {self.global_step}")

        if 'best_val_f1' in checkpoint:
            self.best_val_f1 = checkpoint['best_val_f1']

    def _create_loss(self) -> nn.Module:
        """Create loss function from configuration."""
        loss_cfg = self.train_cfg.loss
        return create_loss(
            loss_type=loss_cfg.loss_type,
            alpha=loss_cfg.focal_alpha,
            gamma=loss_cfg.focal_gamma,
            label_smoothing=loss_cfg.label_smoothing,
            focal_weight=loss_cfg.focal_weight,
            ohem_hard_ratio=loss_cfg.ohem_hard_ratio,
            ohem_min_hard_num=loss_cfg.ohem_min_hard_num
        )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from configuration."""
        cfg = self.train_cfg.optimizer
        params = self.raw_model.parameters()

        optimizers = {
            'adamw': lambda: AdamW(
                params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                betas=cfg.betas, eps=cfg.eps
            ),
            'adam': lambda: torch.optim.Adam(
                params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                betas=cfg.betas, eps=cfg.eps
            ),
            'sgd': lambda: torch.optim.SGD(
                params, lr=cfg.lr, momentum=cfg.momentum,
                weight_decay=cfg.weight_decay, nesterov=cfg.nesterov
            ),
        }

        if cfg.type not in optimizers:
            raise ValueError(f"Unknown optimizer: {cfg.type}")

        return optimizers[cfg.type]()

    def _create_scheduler(self):
        """Create learning rate scheduler from configuration."""
        cfg = self.train_cfg.scheduler
        opt = self.optimizer

        if cfg.type == 'cosine_warmup':
            return WarmupCosineScheduler(
                optimizer=opt,
                warmup_epochs=cfg.warmup_epochs,
                total_epochs=self.train_cfg.num_epochs,
                steps_per_epoch=self.steps_per_epoch,
                base_lr=self.train_cfg.optimizer.lr,
                min_lr=cfg.min_lr,
                warmup_steps=cfg.warmup_steps
            )

        schedulers = {
            'cosine': lambda: CosineAnnealingWarmRestarts(
                opt, T_0=10, T_mult=2, eta_min=cfg.min_lr
            ),
            'plateau': lambda: ReduceLROnPlateau(
                opt, mode=cfg.plateau_mode, factor=cfg.plateau_factor,
                patience=cfg.plateau_patience, verbose=True
            ),
            'step': lambda: torch.optim.lr_scheduler.StepLR(
                opt, step_size=cfg.step_size, gamma=cfg.gamma
            ),
        }

        return schedulers.get(cfg.type, lambda: None)()

    def _clip_gradients(self):
        """Clip gradients if configured."""
        if self.train_cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.train_cfg.grad_clip)

    def _forward_backward_step(self, features: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute forward and backward pass with optional mixed precision."""
        self.optimizer.zero_grad()

        if self.scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                outputs = self.model(features)
                loss = self.criterion(outputs, targets) / self.world_size
            self.scaler.scale(loss).backward()

            if self.is_distributed or self.train_cfg.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
            self._clip_gradients()

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(features)
            loss = self.criterion(outputs, targets) / self.world_size
            loss.backward()
            self._clip_gradients()
            self.optimizer.step()

        # Update EMA after optimizer step
        if self.ema is not None:
            self.ema.update(self.raw_model)

        return outputs, loss

    def _aggregate_metrics(self, total_loss: float, correct: int, total: int) -> tuple[float, int, int]:
        """Aggregate metrics across distributed processes."""
        if not self.is_distributed:
            return total_loss, correct, total

        metrics = torch.tensor([total_loss, correct, total], device=self.device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        return metrics[0].item(), int(metrics[1].item()), int(metrics[2].item())

    @torch.no_grad()
    def _extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract features from waveforms if feature extractor is configured."""
        if self.feature_extractor is None:
            return waveforms.to(self.device, non_blocking=True)
        # Move waveforms to device before feature extraction
        waveforms = waveforms.to(self.device, non_blocking=True)
        return self.feature_extractor(waveforms)

    def _train_epoch(self, log_interval: int = 10) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()

        # 设置特征提取器模式
        if self.feature_extractor is not None:
            mask_config = self.config.feature.mask
            current = self.current_epoch
            start = mask_config.start_epoch
            end = mask_config.end_epoch
            mask_enabled = mask_config.enable and current >= start and (end < 0 or current < end)
            self.feature_extractor.train(mask_enabled)

        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.current_epoch)

        total_loss = correct = total = 0
        # 使用预分配的张量列表避免频繁CPU同步
        local_preds, local_targets = [], []
        num_batches = len(self.train_loader)

        # 使用CUDA事件进行异步计时，避免同步开销
        start_event = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        end_event = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        if start_event:
            start_event.record()

        for batch_idx, (waveforms, targets) in enumerate(self.train_loader):
            # 提取特征并训练
            features = self._extract_features(waveforms)
            targets = targets.to(self.device, non_blocking=True)

            if isinstance(self.scheduler, WarmupCosineScheduler) and self.train_cfg.scheduler.warmup_steps > 0:
                self.scheduler.step(step=self.global_step)

            outputs, loss = self._forward_backward_step(features, targets)

            # 异步获取结果，减少同步点
            with torch.no_grad():
                unscaled_loss = loss.item() * self.world_size  # 必须同步获取loss
                total_loss += unscaled_loss
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                # 延迟CPU转换，累积在GPU
                local_preds.append(predicted)
                local_targets.append(targets)

            # 日志记录（减少同步频率）
            if self.rank == 0 and (batch_idx + 1) % log_interval == 0:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = 100. * correct / total

                throughput_info = ""
                if start_event and end_event:
                    end_event.record()
                    # 非阻塞计算时间
                    elapsed_ms = start_event.elapsed_time(end_event)
                    elapsed_s = elapsed_ms / 1000
                    samples_per_sec = (batch_idx + 1) * self.train_cfg.batch_size / elapsed_s
                    throughput_info = f", {samples_per_sec:.1f} samples/s"
                    # 重置开始事件
                    start_event.record()

                lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f"Epoch [{self.current_epoch}] Batch [{batch_idx + 1}/{num_batches}] "
                    f"Loss: {unscaled_loss:.4f} (avg: {current_loss:.4f}), Acc: {current_acc:.2f}%, "
                    f"LR: {lr:.6f}{throughput_info}"
                )

            if self.writer and self.global_step % self.train_cfg.log_interval == 0:
                self.writer.add_scalar('train/loss_step', unscaled_loss, self.global_step)
                self.writer.add_scalar('train/acc_step', correct/total, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        # 在epoch结束时一次性聚合metrics（减少同步次数）
        total_loss, correct, total = self._aggregate_metrics(total_loss, correct, total)

        epoch_loss = total_loss / (len(self.train_loader) * self.world_size)
        epoch_acc = correct / total

        # 只在需要F1计算时才gather预测结果
        if self.is_distributed:
            # 使用更高效的all_reduce计算F1，而不是gather所有预测
            precision, recall, f1 = self._compute_f1_distributed(local_preds, local_targets)
        else:
            f1 = precision = recall = 0.0
            # 单卡模式：直接计算
            all_preds = torch.cat(local_preds).cpu().numpy()
            all_targets = torch.cat(local_targets).cpu().numpy()
            if len(all_targets) > 0:
                f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
                precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
                recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

        return {'loss': epoch_loss, 'accuracy': epoch_acc, 'f1': f1, 'precision': precision, 'recall': recall}

    def _compute_f1_distributed(self, local_preds: list, local_targets: list) -> tuple[float, float, float]:
        """使用all_reduce计算F1，避免gather所有数据，返回(precision, recall, f1)"""
        if not local_preds:
            return 0.0, 0.0, 0.0

        # 在GPU上拼接
        preds_tensor = torch.cat(local_preds)
        targets_tensor = torch.cat(local_targets)

        # 计算混淆矩阵元素（GPU上）
        tp = ((preds_tensor == 1) & (targets_tensor == 1)).sum().float()
        fp = ((preds_tensor == 1) & (targets_tensor == 0)).sum().float()
        fn = ((preds_tensor == 0) & (targets_tensor == 1)).sum().float()

        # 聚合到所有rank
        metrics = torch.stack([tp, fp, fn])
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        tp, fp, fn = metrics[0].item(), metrics[1].item(), metrics[2].item()

        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0

        return precision_val, recall_val, f1_val

    def _compute_auc_distributed(self, local_probs: list, local_targets: list) -> float:
        """使用all_gather计算AUC，收集所有rank的预测概率和标签"""
        if not local_probs:
            return 0.5

        # 在GPU上拼接
        probs_tensor = torch.cat(local_probs)
        targets_tensor = torch.cat(local_targets)

        # 收集所有rank的数据
        if self.is_distributed:
            # 获取每个rank的数据量
            local_size = torch.tensor([probs_tensor.size(0)], dtype=torch.long, device=self.device)
            all_sizes = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(self.world_size)]
            dist.all_gather(all_sizes, local_size)

            # 创建gather列表
            all_probs = [torch.zeros(size.item(), dtype=probs_tensor.dtype, device=self.device) for size in all_sizes]
            all_targets = [torch.zeros(size.item(), dtype=targets_tensor.dtype, device=self.device) for size in all_sizes]

            # 收集数据
            dist.all_gather(all_probs, probs_tensor)
            dist.all_gather(all_targets, targets_tensor)

            # 在rank 0上计算AUC
            if self.rank == 0:
                all_probs_cat = torch.cat(all_probs).cpu().numpy()
                all_targets_cat = torch.cat(all_targets).cpu().numpy()
                try:
                    auc = roc_auc_score(all_targets_cat, all_probs_cat)
                except Exception:
                    auc = 0.5
            else:
                auc = 0.5

            # 广播AUC到所有rank
            auc_tensor = torch.tensor([auc], dtype=torch.float32, device=self.device)
            dist.broadcast(auc_tensor, src=0)
            return auc_tensor.item()
        else:
            # 单卡模式
            all_probs = probs_tensor.cpu().numpy()
            all_targets = targets_tensor.cpu().numpy()
            try:
                return roc_auc_score(all_targets, all_probs)
            except Exception:
                return 0.5

    def _gather_lists(self, local_list: list) -> list:
        """Gather lists from all ranks using all_gather_object."""
        if not self.is_distributed:
            return local_list

        gathered_lists = [None] * self.world_size
        try:
            dist.all_gather_object(gathered_lists, local_list)
            return [item for sublist in gathered_lists for item in sublist]
        except Exception as e:
            self.logger.warning(f"Failed to gather lists: {e}")
            return local_list

    @torch.no_grad()
    def _validate(self, log_interval: int = 10, use_ema: bool = True) -> Dict[str, float]:
        """Validate on validation set."""
        if self.val_loader is None:
            return {}

        ema_active = use_ema and self.ema is not None
        if ema_active:
            self.ema.apply_shadow(self.raw_model)

        try:
            self.model.eval()
            if self.feature_extractor is not None:
                self.feature_extractor.eval()

            total_loss = 0.0
            correct = 0
            total = 0
            # 累积GPU张量，减少CPU同步
            local_preds, local_targets, local_probs = [], [], []

            for batch_idx, (waveforms, targets) in enumerate(self.val_loader):
                features = self._extract_features(waveforms)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(features)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                probs = F.softmax(outputs, dim=1)[:, 1]

                # 延迟CPU转换
                local_preds.append(predicted)
                local_targets.append(targets)
                local_probs.append(probs)

                if self.rank == 0 and (batch_idx + 1) % log_interval == 0:
                    current_loss = total_loss / (batch_idx + 1)
                    current_acc = 100. * correct / total
                    self.logger.info(
                        f"Validation Batch [{batch_idx + 1}/{len(self.val_loader)}] "
                        f"Loss: {loss.item():.4f} (avg: {current_loss:.4f}), Acc: {current_acc:.2f}%"
                    )

            # 聚合metrics
            if self.is_distributed:
                local_batches = len(self.val_loader)
                batches_tensor = torch.tensor([local_batches], dtype=torch.long, device=self.device)
                dist.all_reduce(batches_tensor, op=dist.ReduceOp.SUM)
                total_batches = int(batches_tensor.item())

                metrics = torch.tensor([total_loss, correct, total])
                if self.device.type == 'cuda':
                    metrics = metrics.cuda(self.device)
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                total_loss, correct, total = metrics[0].item(), int(metrics[1].item()), int(metrics[2].item())

                # 使用分布式F1计算
                precision, recall, f1 = self._compute_f1_distributed(local_preds, local_targets)
                # 计算分布式AUC
                auc = self._compute_auc_distributed(local_probs, local_targets)
            else:
                total_batches = len(self.val_loader)
                # 单卡：拼接后转CPU计算
                all_preds = torch.cat(local_preds).cpu().numpy()
                all_targets = torch.cat(local_targets).cpu().numpy()
                all_probs = torch.cat(local_probs).cpu().numpy()

                f1 = precision_score(all_targets, all_preds, average='macro', zero_division=0)
                recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
                precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
                try:
                    auc = roc_auc_score(all_targets, all_probs)
                except Exception:
                    auc = 0.5

            val_loss = total_loss / max(total_batches, 1)
            val_acc = correct / max(total, 1)

        finally:
            if ema_active:
                self.ema.restore(self.raw_model)

        return {
            'loss': val_loss,
            'accuracy': val_acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

    def _save_checkpoint(self, filename: str, is_best: bool = False, use_ema: bool = True):
        """Save model checkpoint (only on rank 0 when checkpoint_dir is set)."""
        if self.rank != 0 or self.checkpoint_dir is None:
            return

        # Apply EMA shadow for saving if enabled
        ema_active = use_ema and self.ema is not None
        if ema_active:
            self.ema.apply_shadow(self.raw_model)

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'config': self.config,
        }

        # Restore original parameters after saving state dict
        if ema_active:
            self.ema.restore(self.raw_model)

        # Save EMA state separately
        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        if self.swa_model is not None:
            checkpoint['swa_state_dict'] = self.swa_model.state_dict()

        # Save scheduler state if available
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        checkpoint_dir = Path(self.checkpoint_dir)
        path = checkpoint_dir / filename
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")

        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")

    def _check_early_stopping(self, val_f1: float, epoch: int) -> bool:
        """Check if early stopping should be triggered. Returns True if should stop."""
        cfg = self.train_cfg

        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.patience_counter = 0
            if cfg.save_best_only:
                self._save_checkpoint('best_model.pt')
        else:
            self.patience_counter += 1
            if cfg.early_stopping_patience and self.patience_counter >= cfg.early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                return True

        if not cfg.save_best_only:
            self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

        return False

    def _sync_early_stopping(self, should_stop: bool) -> bool:
        """Synchronize early stopping decision across all ranks."""
        if not self.is_distributed:
            return should_stop

        should_stop_tensor = torch.tensor([should_stop], dtype=torch.int32, device=self.device)
        dist.broadcast(should_stop_tensor, src=0)
        return should_stop_tensor.item() != 0

    def train(self):
        """Main training loop"""
        start_epoch = self.start_epoch
        total_epochs = self.train_cfg.num_epochs

        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {total_epochs}, start epoch: {start_epoch}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Distributed: {self.is_distributed} (world_size={self.world_size})")
        if hasattr(self.feature_extractor, '_compiled'):
            self.logger.info("Feature extractor: torch.compile ENABLED")

        eta_tracker = ETATracker(window_size=5)

        for epoch in range(start_epoch, total_epochs):
            self.current_epoch = epoch
            should_stop = False

            # Step epoch-based scheduler at the beginning of each epoch
            if isinstance(self.scheduler, WarmupCosineScheduler) and self.train_cfg.scheduler.warmup_steps == 0:
                self.scheduler.step(epoch=epoch)

            epoch_start = time.time()
            train_metrics = self._train_epoch()
            epoch_elapsed = time.time() - epoch_start
            eta_tracker.update(epoch_elapsed)

            if self.rank == 0:
                lr = self.optimizer.param_groups[0]['lr']
                eta_str, elapsed_str = eta_tracker.estimate(total_epochs - (epoch + 1))

                self.logger.info(
                    f"Epoch {epoch}/{total_epochs - 1}: Train Loss={train_metrics['loss']:.4f}, "
                    f"Acc={train_metrics['accuracy']:.4f}, F1={train_metrics['f1']:.4f}, LR={lr:.6f} "
                    f"| epoch={epoch_elapsed:.1f}s, elapsed={elapsed_str}, ETA={eta_str}"
                )

                if self.writer:
                    for key, value in train_metrics.items():
                        self.writer.add_scalar(f'train/{key}', value, epoch)

            if self.val_loader is not None and epoch % self.train_cfg.val_interval == 0:
                val_metrics = self._validate()

                if self.rank == 0:
                    self.logger.info(
                        f"Epoch {epoch}: Val Loss={val_metrics['loss']:.4f}, "
                        f"Acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}, "
                        f"AUC={val_metrics.get('auc', 0):.4f}"
                    )

                    if self.writer:
                        for key, value in val_metrics.items():
                            self.writer.add_scalar(f'val/{key}', value, epoch)

                    if self.scheduler is not None:
                        if isinstance(self.scheduler, ReduceLROnPlateau):
                            self.scheduler.step(val_metrics['f1'])
                        elif not isinstance(self.scheduler, WarmupCosineScheduler):
                            self.scheduler.step()

                    should_stop = self._check_early_stopping(val_metrics['f1'], epoch)

            if self.is_distributed and self.val_loader is not None:
                should_stop = self._sync_early_stopping(should_stop)
                if should_stop:
                    break

            # Update SWA
            if self.train_cfg.use_swa and self.swa_model is not None and epoch >= self.train_cfg.swa_start:
                self.swa_model.update_parameters(self.raw_model)
                self.swa_n += 1

        if self.train_cfg.use_swa and self.swa_model is not None and self.rank == 0:
            torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device=self.device)
            self._save_checkpoint('swa_model.pt')

        if self.writer:
            self.writer.close()
        self.logger.info("Training completed!")


def set_seed(seed: int = 42, rank: int = 0) -> None:
    """
    Set random seeds for reproducibility.

    Different seeds for each rank in distributed training to avoid
    identical data augmentation across ranks.

    Args:
        seed: Base random seed (default: 42)
        rank: Process rank for distributed training (default: 0)
    """
    # Offset seed by rank to avoid identical augmentation in DDP
    rank_seed = seed + rank

    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)  # For multi-GPU

    # Deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Note: torch.use_deterministic_algorithms(True) may impact performance
    # and is not always compatible with all operations. Enable with caution.
    # torch.use_deterministic_algorithms(True)


def main():
    parser = argparse.ArgumentParser(
        description='Train CryTransformer model with DDP support'
    )
    parser.add_argument('--config', type=str, default='configs/model_medium.yaml',
                        help='Path to config file')
    parser.add_argument('--train_list', type=str, required=True,
                        help='Path to training data list JSON')
    parser.add_argument('--val_list', type=str, default=None,
                        help='Path to validation data list JSON')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size per GPU (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training (set automatically by torchrun)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to log file (logs to console only if not specified)')

    # Model architecture overrides
    parser.add_argument('--d_model', type=int, default=None,
                        help='Override model hidden dimension')
    parser.add_argument('--n_layers', type=int, default=None,
                        help='Override number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=None,
                        help='Override number of attention heads')
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, device = setup_distributed()

    # Early logger setup (console only, will re-configure with file after run_dir is created)
    logger = setup_logger(rank, name=None)

    # Set random seed for reproducibility
    set_seed(args.seed, rank)
    logger.info(f"Random seed set to {args.seed} (rank offset: {rank})")

    try:
        # Load config
        if Path(args.config).exists():
            config = load_config(args.config)
        else:
            logger.warning(f"Config file not found: {args.config}, using defaults")
            config = Config()

        # Override training config
        if args.epochs:
            config.training.num_epochs = args.epochs
        if args.batch_size:
            # Note: batch_size is per GPU in distributed training
            config.training.batch_size = args.batch_size
        if args.lr:
            config.training.optimizer.lr = args.lr

        # Override model architecture
        if args.d_model:
            config.model.d_model = args.d_model
        if args.n_layers:
            config.model.n_layers = args.n_layers
        if args.n_heads:
            config.model.n_heads = args.n_heads

        config.model.__post_init__()

        # Create unified output directory based on timestamp (only rank 0 creates, others use None)
        checkpoint_dir = log_dir = None
        if rank == 0:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_run_dir = Path(config.training.run_dir)
            run_dir = base_run_dir / timestamp
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints").mkdir(exist_ok=True)
            (run_dir / "tensorboard").mkdir(exist_ok=True)
            (run_dir / "logs").mkdir(exist_ok=True)

            checkpoint_dir = str(run_dir / "checkpoints")
            log_dir = str(run_dir / "tensorboard")

            # Auto-generate log_file if not specified
            if args.log_file is None:
                args.log_file = str(run_dir / "logs" / "train.log")

            # Re-configure logger with file output
            logger = setup_file_logger(args.log_file, rank=rank, name=None)
            logger.info(f"Training outputs saved to: {run_dir}")
            logger.info(f"  Checkpoints: {checkpoint_dir}")
            logger.info(f"  TensorBoard: {log_dir}")
            logger.info(f"  Logs: {args.log_file}")

            # Save training configuration
            config_save_path = run_dir / "config.yaml"
            save_config(config, str(config_save_path))
            logger.info(f"  Config: {config_save_path}")

        # Load data
        if rank == 0:
            logger.info(f"Loading training data from: {args.train_list}")
        train_dict = load_data_dict(args.train_list)

        # Create datasets
        train_dataset = CryDataset(
            train_dict, config.dataset,
            aug_config=config.augmentation if config.training.use_augmentation else None
        )

        val_dataset = val_loader = None
        if args.val_list:
            if rank == 0:
                logger.info(f"Loading validation data from: {args.val_list}")
            val_dict = load_data_dict(args.val_list)
            val_dataset = CryDataset(val_dict, config.dataset)

        # Create samplers and loaders
        if world_size > 1:
            # Use DistributedCrySampler for distributed training
            train_sampler = DistributedCrySampler(
                train_dataset,
                cry_rate=config.dataset.cry_rate,
                shuffle=True,
                seed=0
            )
        else:
            train_sampler = CrySampler(train_dataset, cry_rate=config.dataset.cry_rate, shuffle=True)

        # Create worker init function with seed for reproducibility
        worker_init = partial(worker_init_fn, base_seed=args.seed) if config.training.num_workers > 0 else None

        # Determine persistent_workers (disable for DDP to avoid deadlock)
        persistent_workers = config.training.persistent_workers and world_size == 1

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=train_sampler,
            num_workers=config.training.num_workers,
            prefetch_factor=config.training.prefetch_factor if config.training.num_workers > 0 else None,
            pin_memory=config.training.pin_memory and device.type == 'cuda',
            collate_fn=collate_fn,
            worker_init_fn=worker_init,
            persistent_workers=persistent_workers if config.training.num_workers > 0 else False,
            drop_last=True  # Important for DDP: avoid uneven batch sizes
        )

        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                sampler=SequentialCrySampler(val_dataset, partition_rank=True),
                num_workers=config.training.num_workers,
                prefetch_factor=config.training.prefetch_factor if config.training.num_workers > 0 else None,
                pin_memory=config.training.pin_memory and device.type == 'cuda',
                collate_fn=collate_fn,
                worker_init_fn=worker_init,
                persistent_workers=persistent_workers if config.training.num_workers > 0 else False
            )

        # Create model
        if rank == 0:
            logger.info(f"Creating model: d_model={config.model.d_model}, n_layers={config.model.n_layers}")
        model = create_model(
            config=config.model,
            in_channels=config.feature.feature_dim,
            num_classes=config.model.num_classes,
            use_spec_augment=config.training.use_spec_augment
        )

        if rank == 0:
            logger.info("Model architecture summary:\n" + get_model_summary(model))

        # Create feature extractor with sample_rate and move to device
        feature_extractor = FeatureExtractor(config.feature, sr=config.dataset.sample_rate).to(device)

        # Synchronize before training to ensure all ranks are ready
        if world_size > 1:
            dist.barrier()

        # Create trainer and train (checkpoint loading handled inside Trainer)
        trainer = Trainer(
            model, config, train_loader, val_loader, device,
            rank, world_size,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            checkpoint_path=args.resume,
            feature_extractor=feature_extractor
        )
        trainer.train()

    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
