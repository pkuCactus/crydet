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
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import Config, load_config
from dataset.dataset import CryDataset
from dataset.dataloader import collate_fn, worker_init_fn, load_data_dict
from dataset.sampler import CrySampler, DistributedCrySampler, SequentialCrySampler
from model import create_model, print_model_summary
from model.ema import ExponentialMovingAverage
from model.loss import create_loss
from model.scheduler import WarmupCosineScheduler
from model.distributed import setup_distributed, cleanup_distributed
from utils import setup_logger


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
        checkpoint: Optional[Dict] = None
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1

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
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.global_step = 0
        self.steps_per_epoch = len(train_loader) if train_loader else 0

        # Scheduler (created after steps_per_epoch is known)
        self.scheduler = self._create_scheduler()

        # TensorBoard writer (only on rank 0)
        self.writer = None
        if rank == 0:
            self.writer = SummaryWriter(self.train_cfg.log_dir)
            os.makedirs(self.train_cfg.checkpoint_dir, exist_ok=True)

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

        self.logger = setup_logger(rank)

        # Restore from checkpoint if provided
        if checkpoint is not None:
            self._restore_from_checkpoint(checkpoint)

    def _restore_from_checkpoint(self, checkpoint: Dict):
        """Restore training state from checkpoint."""
        # Restore optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info("Restored optimizer state")

        # Restore EMA state
        if self.ema is not None and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
            self.logger.info("Restored EMA state")

        # Restore scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            if isinstance(self.scheduler, WarmupCosineScheduler):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("Restored scheduler state")

        # Restore best validation metric
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
        """Create optimizer."""
        cfg = self.train_cfg
        lr, wd = cfg.learning_rate, cfg.weight_decay

        if cfg.optimizer == 'adamw':
            return AdamW(self.raw_model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
        elif cfg.optimizer == 'adam':
            return torch.optim.Adam(self.raw_model.parameters(), lr=lr, weight_decay=wd)
        elif cfg.optimizer == 'sgd':
            return torch.optim.SGD(self.raw_model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        cfg = self.train_cfg
        opt = self.optimizer

        if cfg.scheduler == 'cosine_warmup':
            # Warmup + Cosine Decay
            return WarmupCosineScheduler(
                optimizer=opt,
                warmup_epochs=cfg.warmup_epochs,
                total_epochs=cfg.num_epochs,
                steps_per_epoch=self.steps_per_epoch,
                base_lr=cfg.learning_rate,
                min_lr=cfg.min_lr,
                warmup_steps=cfg.warmup_steps
            )
        elif cfg.scheduler == 'cosine':
            return CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=cfg.min_lr)
        elif cfg.scheduler == 'plateau':
            return ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, verbose=True)
        elif cfg.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1)
        return None

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

    def _train_epoch(self, log_interval: int = 10) -> Dict[str, float]:
        """Train one epoch with periodic logging instead of tqdm."""
        self.model.train()

        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.current_epoch)

        total_loss = correct = total = 0
        all_preds, all_targets = [], []
        num_batches = len(self.train_loader)

        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        if start_time:
            start_time.record()

        for batch_idx, (features, targets) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)

            outputs, loss = self._forward_backward_step(features, targets)

            with torch.no_grad():
                unscaled_loss = loss.item() * self.world_size
                total_loss += unscaled_loss
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            if self.rank == 0 and (batch_idx + 1) % log_interval == 0:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = 100. * correct / total

                throughput_info = ""
                if start_time:
                    end_time = torch.cuda.Event(enable_timing=True)
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_s = start_time.elapsed_time(end_time) / 1000
                    samples_per_sec = (batch_idx + 1) * self.train_cfg.batch_size / elapsed_s
                    throughput_info = f", {samples_per_sec:.1f} samples/s"

                self.logger.info(
                    f"Epoch [{self.current_epoch}] Batch [{batch_idx + 1}/{num_batches}] "
                    f"Loss: {unscaled_loss:.4f} (avg: {current_loss:.4f}), Acc: {current_acc:.2f}%{throughput_info}"
                )

            # TensorBoard logging
            if self.writer and self.global_step % self.train_cfg.log_interval == 0:
                self.writer.add_scalar('train/loss_step', unscaled_loss, self.global_step)
                self.writer.add_scalar('train/acc_step', correct/total, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            # Step warmup cosine scheduler per step if using step-based warmup
            if isinstance(self.scheduler, WarmupCosineScheduler) and self.train_cfg.warmup_steps > 0:
                self.scheduler.step(step=self.global_step)

            self.global_step += 1

        total_loss, correct, total = self._aggregate_metrics(total_loss, correct, total)

        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = correct / total

        if self.is_distributed:
            all_preds = self._gather_lists(all_preds)
            all_targets = self._gather_lists(all_targets)

        if all_targets and all_preds:
            f1 = f1_score(all_targets, all_preds, average='macro')
            precision = precision_score(all_targets, all_preds, average='macro')
            recall = recall_score(all_targets, all_preds, average='macro')
        else:
            f1 = precision = recall = 0.0

        return {'loss': epoch_loss, 'accuracy': epoch_acc, 'f1': f1, 'precision': precision, 'recall': recall}

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
        """Validate on validation set with periodic logging.

        Args:
            log_interval: Logging interval in batches
            use_ema: Whether to use EMA model for validation (if available)
        """
        if self.val_loader is None:
            return {}

        # Apply EMA shadow parameters if enabled
        ema_active = use_ema and self.ema is not None
        if ema_active:
            self.ema.apply_shadow(self.raw_model)

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []

        num_batches = len(self.val_loader)

        for batch_idx, (features, targets) in enumerate(self.val_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(features)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            probs = F.softmax(outputs, dim=1)[:, 1]

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if self.rank == 0 and (batch_idx + 1) % log_interval == 0:
                current_loss = total_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                self.logger.info(
                    f"Validation Batch [{batch_idx + 1}/{num_batches}] "
                    f"Loss: {loss.item():.4f} (avg: {current_loss:.4f}), Acc: {current_acc:.2f}%"
                )

        if self.is_distributed:
            # Gather num_batches from all ranks for correct loss averaging
            local_batches = len(self.val_loader)
            batches_tensor = torch.tensor([local_batches], dtype=torch.long, device=self.device)
            dist.all_reduce(batches_tensor, op=dist.ReduceOp.SUM)
            total_batches = int(batches_tensor.item())

            # Create tensor on CPU then move to device to avoid NCCL issue
            metrics = torch.tensor([total_loss, correct, total])
            if self.device.type == 'cuda':
                metrics = metrics.cuda(self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, correct, total = metrics[0].item(), int(metrics[1].item()), int(metrics[2].item())

            all_preds = self._gather_lists(all_preds)
            all_targets = self._gather_lists(all_targets)
            all_probs = self._gather_lists(all_probs)
        else:
            total_batches = len(self.val_loader)

        val_loss = total_loss / max(total_batches, 1)
        val_acc = correct / max(total, 1)

        f1 = precision = recall = auc = 0.0
        if all_targets and all_preds:
            from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
            try:
                f1 = f1_score(all_targets, all_preds, average='macro')
                precision = precision_score(all_targets, all_preds, average='macro')
                recall = recall_score(all_targets, all_preds, average='macro')
                auc = roc_auc_score(all_targets, all_probs)
            except Exception as e:
                self.logger.warning(f"Failed to compute metrics: {e}")
                auc = 0.5

        # Restore original parameters after validation
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
        """Save model checkpoint (only on rank 0).

        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model
            use_ema: Whether to save EMA version as primary model
        """
        if self.rank != 0:
            return

        # Apply EMA shadow for saving if enabled
        ema_active = use_ema and self.ema is not None
        if ema_active:
            self.ema.apply_shadow(self.raw_model)

        checkpoint = {
            'epoch': self.current_epoch,
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
            if isinstance(self.scheduler, WarmupCosineScheduler):
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            else:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        path = os.path.join(self.train_cfg.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")

        if is_best:
            best_path = os.path.join(self.train_cfg.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")

    def _check_early_stopping(self, val_f1: float, epoch: int) -> bool:
        """Check if early stopping should be triggered. Returns True if should stop."""
        cfg = self.train_cfg

        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.patience_counter = 0
            if cfg.save_best_only:
                self._save_checkpoint('best_model.pt', is_best=True)
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
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.train_cfg.num_epochs}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Distributed: {self.is_distributed} (world_size={self.world_size})")

        for epoch in range(self.train_cfg.num_epochs):
            self.current_epoch = epoch
            should_stop = False

            train_metrics = self._train_epoch()

            if self.rank == 0:
                self.logger.info(
                    f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                    f"Acc={train_metrics['accuracy']:.4f}, F1={train_metrics['f1']:.4f}"
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
                        elif isinstance(self.scheduler, WarmupCosineScheduler):
                            # WarmupCosineScheduler is stepped per-step or per-epoch
                            if self.train_cfg.warmup_steps == 0:
                                self.scheduler.step(epoch=epoch)
                        else:
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
    logger = setup_logger(rank)

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
            config.training.learning_rate = args.lr

        # Override model architecture
        if args.d_model:
            config.model.d_model = args.d_model
        if args.n_layers:
            config.model.n_layers = args.n_layers
        if args.n_heads:
            config.model.n_heads = args.n_heads

        config.model.__post_init__()

        # Load data
        if rank == 0:
            logger.info(f"Loading training data from: {args.train_list}")
        train_dict = load_data_dict(args.train_list)

        # Create datasets
        train_dataset = CryDataset(
            train_dict, config.dataset,
            aug_config=config.augmentation if config.training.use_augmentation else None,
            feat_config=config.feature
        )

        val_dataset = val_loader = None
        if args.val_list:
            if rank == 0:
                logger.info(f"Loading validation data from: {args.val_list}")
            val_dict = load_data_dict(args.val_list)
            val_dataset = CryDataset(val_dict, config.dataset, feat_config=config.feature)

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

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=train_sampler,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory and device.type == 'cuda',
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn if config.training.num_workers > 0 else None,
            persistent_workers=False,  # Disable to avoid deadlock in DDP
            drop_last=True  # Important for DDP: avoid uneven batch sizes
        )

        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                sampler=SequentialCrySampler(val_dataset, partition_rank=True),
                num_workers=config.training.num_workers,
                pin_memory=config.training.pin_memory and device.type == 'cuda',
                collate_fn=collate_fn,
                worker_init_fn=worker_init_fn if config.training.num_workers > 0 else None,
                persistent_workers=False
            )

        # Create model
        if rank == 0:
            logger.info(f"Creating model: d_model={config.model.d_model}, n_layers={config.model.n_layers}")
        in_channels = config.feature.n_mels * config.feature.num_channels
        model = create_model(
            config=config.model,
            in_channels=in_channels,
            num_classes=config.model.num_classes,
            use_spec_augment=config.training.use_spec_augment
        )

        if rank == 0:
            print_model_summary(model)

        # Resume from checkpoint if specified
        checkpoint = None
        start_epoch = 0
        if args.resume:
            if rank == 0:
                logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)

        # Synchronize before training to ensure all ranks are ready
        if world_size > 1:
            dist.barrier()

        # Create trainer and train
        trainer = Trainer(model, config, train_loader, val_loader, device, rank, world_size, checkpoint)
        trainer.current_epoch = start_epoch
        trainer.train()

    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
