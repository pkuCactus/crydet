"""
Custom learning rate schedulers for transformer training.

Includes warmup + cosine decay scheduler optimized for stable training
of deep models.
"""

from typing import Dict, Optional
import numpy as np
import torch


class WarmupCosineScheduler:
    """
    Warmup + Cosine Decay Learning Rate Scheduler.

    Learning rate schedule:
    1. Warmup phase: linear increase from 0 to base_lr
    2. Cosine decay phase: cosine annealing from base_lr to min_lr

    This scheduler is particularly effective for training transformers,
    as the warmup phase stabilizes early training and cosine decay
    provides smooth reduction of learning rate.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup
        total_epochs: Total number of training epochs
        steps_per_epoch: Number of steps per epoch (for step-based warmup)
        base_lr: Initial learning rate after warmup
        min_lr: Minimum learning rate at end of training
        warmup_steps: If > 0, use step-based warmup instead of epoch-based

    Example:
        >>> scheduler = WarmupCosineScheduler(
        ...     optimizer,
        ...     warmup_epochs=5,
        ...     total_epochs=100,
        ...     steps_per_epoch=len(train_loader),
        ...     base_lr=1e-3,
        ...     min_lr=1e-6
        ... )
        >>> for epoch in range(num_epochs):
        ...     for step, batch in enumerate(train_loader):
        ...         # For step-based warmup:
        ...         scheduler.step(step=epoch * len(train_loader) + step)
        ...         # Or for epoch-based:
        ...         # scheduler.step(epoch=epoch)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int = 0,
        base_lr: float = 1e-3,
        min_lr: float = 1e-6,
        warmup_steps: int = 0
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps

        # Calculate total steps for warmup and cosine decay
        if warmup_steps > 0:
            self.total_warmup_steps = warmup_steps
        else:
            self.total_warmup_steps = warmup_epochs * steps_per_epoch if steps_per_epoch > 0 else warmup_epochs

        self.total_steps = total_epochs * steps_per_epoch if steps_per_epoch > 0 else total_epochs
        self.current_step = 0

    def step(self, epoch: Optional[int] = None, step: Optional[int] = None) -> float:
        """
        Update learning rate.

        Args:
            epoch: Current epoch number (for epoch-based scheduling)
            step: Current global step (for step-based scheduling)

        Returns:
            Current learning rate
        """
        if step is not None:
            self.current_step = step
        elif epoch is not None and self.steps_per_epoch > 0:
            self.current_step = epoch * self.steps_per_epoch
        elif epoch is not None:
            self.current_step = epoch

        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def _get_lr(self) -> float:
        """
        Calculate current learning rate.

        Returns:
            Learning rate for current step
        """
        if self.current_step < self.total_warmup_steps:
            # Warmup phase: linear increase from min_lr to base_lr
            progress = self.current_step / max(1, self.total_warmup_steps)
            return self.min_lr + (self.base_lr - self.min_lr) * progress
        else:
            # Cosine decay phase
            progress = (self.current_step - self.total_warmup_steps) / max(
                1, self.total_steps - self.total_warmup_steps
            )
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay

    def get_last_lr(self) -> float:
        """
        Get the last computed learning rate.

        Returns:
            Current learning rate
        """
        return self._get_lr()

    def state_dict(self) -> Dict:
        """
        Return state dict for checkpointing.

        Returns:
            Dictionary containing scheduler state
        """
        return {
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'steps_per_epoch': self.steps_per_epoch,
            'base_lr': self.base_lr,
            'min_lr': self.min_lr,
            'warmup_steps': self.warmup_steps,
            'current_step': self.current_step
        }

    def load_state_dict(self, state_dict: Dict):
        """
        Load state dict from checkpoint.

        Args:
            state_dict: Dictionary containing scheduler state
        """
        self.warmup_epochs = state_dict['warmup_epochs']
        self.total_epochs = state_dict['total_epochs']
        self.steps_per_epoch = state_dict['steps_per_epoch']
        self.base_lr = state_dict['base_lr']
        self.min_lr = state_dict['min_lr']
        self.warmup_steps = state_dict['warmup_steps']
        self.current_step = state_dict['current_step']


class LinearWarmupPolyDecayScheduler:
    """
    Linear warmup followed by polynomial decay.

    Alternative to cosine decay with polynomial falloff.
    Sometimes works better for certain model architectures.

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        base_lr: Peak learning rate after warmup
        min_lr: Minimum learning rate
        power: Polynomial power (default: 1.0 for linear decay)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float = 1e-3,
        min_lr: float = 1e-6,
        power: float = 1.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.power = power
        self.current_step = 0

    def step(self, step: Optional[int] = None) -> float:
        """Update learning rate."""
        if step is not None:
            self.current_step = step

        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def _get_lr(self) -> float:
        """Calculate current learning rate."""
        if self.current_step < self.warmup_steps:
            # Linear warmup from min_lr to base_lr
            progress = self.current_step / max(1, self.warmup_steps)
            return self.min_lr + (self.base_lr - self.min_lr) * progress
        else:
            # Polynomial decay
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(1.0, max(0.0, progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * ((1 - progress) ** self.power)
            return lr

    def state_dict(self) -> Dict:
        """Return state dict for checkpointing."""
        return {
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'base_lr': self.base_lr,
            'min_lr': self.min_lr,
            'power': self.power,
            'current_step': self.current_step
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state dict from checkpoint."""
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.base_lr = state_dict['base_lr']
        self.min_lr = state_dict['min_lr']
        self.power = state_dict['power']
        self.current_step = state_dict['current_step']
