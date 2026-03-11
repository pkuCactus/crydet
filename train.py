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
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, load_config, ModelConfig, TrainingConfig
from dataset.dataset import CryDataset
from dataset.sampler import CrySampler, DistributedCrySampler, SequentialCrySampler
from model import create_model, get_model_info, print_model_summary
from model.loss import FocalLoss, LabelSmoothingCrossEntropy, CombinedLoss


# Setup logging
def setup_logger(rank: int = 0):
    """Setup logger with rank info for distributed training"""
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Input: List of (features, label) tuples where features is [T, F] numpy array
    Output: (features_tensor, label_indices) where features_tensor is [B, T, F]
    """
    features_list, labels = zip(*batch)

    # Stack features directly (they should all have same shape [T, F])
    features = torch.from_numpy(np.stack(features_list)).float()  # [B, T, F]

    label_to_idx = {'cry': 1, 'other': 0}
    label_indices = torch.tensor([label_to_idx.get(l, 0) for l in labels], dtype=torch.long)

    return features, label_indices


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
        world_size: int = 1
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1

        # Wrap model with DDP if distributed
        if self.is_distributed:
            self.model = DDP(model.to(device), device_ids=[rank], output_device=rank)
            self.raw_model = model  # Keep reference to unwrapped model for saving
        else:
            self.model = model.to(device)
            self.raw_model = model

        # Training config
        self.train_cfg = config.training
        self.model_cfg = config.model

        # Loss function
        if self.train_cfg.use_focal_loss:
            self.criterion = CombinedLoss(
                alpha=self.train_cfg.focal_alpha,
                gamma=self.train_cfg.focal_gamma,
                label_smoothing=self.model_cfg.label_smoothing,
                focal_weight=0.5
            )
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.model_cfg.label_smoothing)

        self.criterion = self.criterion.to(device)

        # Optimizer (operates on unwrapped model parameters)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Training state
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.global_step = 0

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

        # Scaler for mixed precision
        self.scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

        self.logger = setup_logger(rank)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        params = self.raw_model.parameters()

        if self.train_cfg.optimizer == 'adamw':
            return AdamW(
                params,
                lr=self.train_cfg.learning_rate,
                weight_decay=self.train_cfg.weight_decay,
                betas=(0.9, 0.98)
            )
        elif self.train_cfg.optimizer == 'adam':
            return torch.optim.Adam(
                params,
                lr=self.train_cfg.learning_rate,
                weight_decay=self.train_cfg.weight_decay
            )
        elif self.train_cfg.optimizer == 'sgd':
            return torch.optim.SGD(
                params,
                lr=self.train_cfg.learning_rate,
                momentum=0.9,
                weight_decay=self.train_cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.train_cfg.optimizer}")

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.train_cfg.scheduler == 'cosine':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=self.train_cfg.min_lr
            )
        elif self.train_cfg.scheduler == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif self.train_cfg.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        return None

    def _train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()

        # Set epoch for sampler to regenerate data schedules
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.current_epoch)

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            disable=self.rank != 0
        )

        for _, (features, targets) in enumerate(pbar):
            # Features are already in [B, T, F] format from dataset
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model(features)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.world_size  # Scale loss for gradient averaging
            else:
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                loss = loss / self.world_size

            # Backward pass
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()

                # Gradient synchronization for DDP
                if self.is_distributed:
                    self.scaler.unscale_(self.optimizer)

                if self.train_cfg.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(),
                        self.train_cfg.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                if self.train_cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.raw_model.parameters(),
                        self.train_cfg.grad_clip
                    )
                self.optimizer.step()

            # Statistics (use unscaled loss for logging)
            with torch.no_grad():
                unscaled_loss = loss.item() * self.world_size
                total_loss += unscaled_loss
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

            if self.rank == 0:
                pbar.set_postfix({
                    'loss': f'{unscaled_loss:.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

                if self.global_step % self.train_cfg.log_interval == 0 and self.writer:
                    self.writer.add_scalar('train/loss_step', unscaled_loss, self.global_step)
                    self.writer.add_scalar('train/acc_step', correct/total, self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        # Aggregate metrics across all processes
        if self.is_distributed:
            metrics_tensor = torch.tensor([total_loss, correct, total], device=self.device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            total_loss, correct, total = metrics_tensor.tolist()

        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = correct / total

        # Calculate F1 score (only on rank 0 to avoid redundant computation)
        if self.rank == 0:
            from sklearn.metrics import f1_score, precision_score, recall_score
            f1 = f1_score(all_targets, all_preds, average='macro')
            precision = precision_score(all_targets, all_preds, average='macro')
            recall = recall_score(all_targets, all_preds, average='macro')
        else:
            f1 = precision = recall = 0.0

        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []

        for features, targets in tqdm(
            self.val_loader,
            desc="Validation",
            disable=self.rank != 0
        ):
            # Features are already in [B, T, F] format from dataset
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

        # Aggregate metrics across all processes
        if self.is_distributed:
            metrics_tensor = torch.tensor([total_loss, correct, total], device=self.device)
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
            total_loss, correct, total = metrics_tensor.tolist()

        val_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        val_acc = correct / total if total > 0 else 0

        if self.rank == 0:
            from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
            f1 = f1_score(all_targets, all_preds, average='macro')
            precision = precision_score(all_targets, all_preds, average='macro')
            recall = recall_score(all_targets, all_preds, average='macro')
            try:
                auc = roc_auc_score(all_targets, all_probs)
            except:
                auc = 0.5
        else:
            f1 = precision = recall = auc = 0.0

        return {
            'loss': val_loss,
            'accuracy': val_acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }

    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint (only on rank 0)"""
        if self.rank != 0:
            return

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.raw_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'config': self.config,
        }

        if self.swa_model is not None:
            checkpoint['swa_state_dict'] = self.swa_model.state_dict()

        path = os.path.join(self.train_cfg.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")

        if is_best:
            best_path = os.path.join(self.train_cfg.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.train_cfg.num_epochs}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Distributed: {self.is_distributed} (world_size={self.world_size})")

        for epoch in range(self.train_cfg.num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self._train_epoch()

            if self.rank == 0:
                self.logger.info(
                    f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                    f"Acc={train_metrics['accuracy']:.4f}, F1={train_metrics['f1']:.4f}"
                )

                if self.writer:
                    for key, value in train_metrics.items():
                        self.writer.add_scalar(f'train/{key}', value, epoch)

            # Validation (only on rank 0 to avoid redundant computation)
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

                    # Learning rate scheduling
                    if self.scheduler is not None:
                        if isinstance(self.scheduler, ReduceLROnPlateau):
                            self.scheduler.step(val_metrics['f1'])
                        else:
                            self.scheduler.step()

                    # Early stopping check
                    val_f1 = val_metrics['f1']
                    if val_f1 > self.best_val_f1:
                        self.best_val_f1 = val_f1
                        self.patience_counter = 0
                        if self.train_cfg.save_best_only:
                            self._save_checkpoint('best_model.pt', is_best=True)
                    else:
                        self.patience_counter += 1
                        if self.train_cfg.early_stopping_patience and \
                           self.patience_counter >= self.train_cfg.early_stopping_patience:
                            self.logger.info(f"Early stopping triggered at epoch {epoch}")
                            break

                    # Regular checkpoint
                    if not self.train_cfg.save_best_only:
                        self._save_checkpoint(f'checkpoint_epoch_{epoch}.pt')

            # Update SWA
            if self.train_cfg.use_swa and self.swa_model is not None and epoch >= self.train_cfg.swa_start:
                self.swa_model.update_parameters(self.raw_model)
                self.swa_n += 1

        # Final SWA model
        if self.train_cfg.use_swa and self.swa_model is not None and self.rank == 0:
            torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device=self.device)
            self._save_checkpoint('swa_model.pt')

        if self.writer:
            self.writer.close()

        self.logger.info("Training completed!")


def setup_distributed() -> Tuple[int, int, torch.device]:
    """
    Setup distributed training environment.

    Returns:
        Tuple of (rank, world_size, device)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Launched with torchrun or torch.distributed.launch
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # Single GPU training
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return rank, world_size, device


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_data_dict(json_path: str) -> dict:
    """Load data dictionary from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


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
            train_dict,
            config.dataset,
            aug_config=config.augmentation if config.training.use_augmentation else None,
            feat_config=config.feature
        )

        val_dataset = None
        val_loader = None
        if args.val_list:
            if rank == 0:
                logger.info(f"Loading validation data from: {args.val_list}")
            val_dict = load_data_dict(args.val_list)
            val_dataset = CryDataset(
                val_dict,
                config.dataset,
                aug_config=None,
                feat_config=config.feature
            )

        # Create samplers and loaders
        if world_size > 1:
            # Use DistributedCrySampler for distributed training
            train_sampler = DistributedCrySampler(
                train_dataset,
                cry_rate=config.dataset.cry_rate,
                num_replicas=world_size,
                rank=rank
            )
        else:
            train_sampler = CrySampler(train_dataset, cry_rate=config.dataset.cry_rate)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=train_sampler,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            collate_fn=collate_fn
        )

        if val_dataset:
            # Validation should use all samples without balanced sampling
            val_sampler = SequentialCrySampler(val_dataset)
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                sampler=val_sampler,
                num_workers=config.training.num_workers,
                pin_memory=config.training.pin_memory,
                collate_fn=collate_fn
            )

        # Create model
        if rank == 0:
            logger.info(f"Creating model: d_model={config.model.d_model}, n_layers={config.model.n_layers}")
        # Calculate input feature dimension (with deltas if enabled)
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
        start_epoch = 0
        if args.resume:
            if rank == 0:
                logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)

        # Create trainer and train
        trainer = Trainer(model, config, train_loader, val_loader, device, rank, world_size)
        trainer.current_epoch = start_epoch
        trainer.train()

    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
