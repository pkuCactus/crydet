"""
Exponential Moving Average (EMA) for model parameters.

EMA maintains a shadow copy of model parameters that is updated with:
    shadow_param = decay * shadow_param + (1 - decay) * model_param

This is commonly used to create a more stable model for evaluation
while allowing the training model to continue updating aggressively.
"""

from typing import Dict
import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) for model parameters.

    Maintains a shadow copy of model parameters that is updated with:
        shadow_param = decay * shadow_param + (1 - decay) * model_param

    The shadow parameters are typically used for validation and saving,
    as they tend to generalize better than the raw training parameters.

    Args:
        model: PyTorch model to track
        decay: EMA decay rate (default: 0.9999)
               Closer to 1.0 means slower updates (more smoothing)
               Typical values: 0.999, 0.9999

    Example:
        >>> ema = ExponentialMovingAverage(model, decay=0.9999)
        >>>
        >>> # During training
        >>> for batch in dataloader:
        >>>     loss = compute_loss(model(batch))
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>     ema.update(model)  # Update EMA after optimizer step
        >>>
        >>> # During validation - use EMA parameters
        >>> ema.apply_shadow(model)
        >>> validate(model)
        >>> ema.restore(model)  # Restore training parameters
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.backup_params: Dict[str, torch.Tensor] = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()

    def update(self, model: nn.Module):
        """
        Update shadow parameters from model parameters.

        Should be called after each optimizer step during training.

        Args:
            model: PyTorch model with updated parameters
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """
        Apply shadow parameters to model (for validation/saving).

        This temporarily replaces model parameters with EMA parameters.
        Call restore() after validation to revert to original parameters.

        Args:
            model: PyTorch model to apply EMA parameters to
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.backup_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])

    def restore(self, model: nn.Module):
        """
        Restore original parameters after validation.

        Must be called after apply_shadow() to restore training parameters.

        Args:
            model: PyTorch model to restore original parameters to
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup_params:
                param.data.copy_(self.backup_params[name])
        self.backup_params.clear()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Return state dict for checkpointing.

        Returns:
            Dictionary containing decay rate and shadow parameters
        """
        return {
            'decay': torch.tensor(self.decay),
            'shadow_params': self.shadow_params
        }

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load state dict from checkpoint.

        Args:
            state_dict: Dictionary containing decay rate and shadow parameters
        """
        self.decay = state_dict['decay'].item()
        self.shadow_params = state_dict['shadow_params']
