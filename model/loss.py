"""
Loss functions for CryTransformer models

Includes:
- FocalLoss: For handling class imbalance
- LabelSmoothingCrossEntropy: Label smoothing for better generalization
- CombinedLoss: Combined Focal + Label Smoothing
- OHEMLoss: Online Hard Example Mining wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance

    Down-weights easy examples and focuses on hard negatives.
    Based on "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    Args:
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits tensor of shape [B, num_classes]
            targets: Ground truth class indices of shape [B]

        Returns:
            Focal loss scalar
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss

    Prevents over-confidence by smoothing target distribution.
    Based on "Rethinking the Inception Architecture for Computer Vision" (Szegedy et al., 2015)

    Args:
        smoothing: Smoothing factor (default: 0.1)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits tensor of shape [B, num_classes]
            targets: Ground truth class indices of shape [B]

        Returns:
            Label smoothing cross entropy loss scalar
        """
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes

        loss = -(targets_smooth * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """Combined Focal Loss + Label Smoothing

    Combines the benefits of focal loss (handling class imbalance)
    and label smoothing (better generalization).

    Args:
        alpha: Focal loss alpha parameter (default: 0.25)
        gamma: Focal loss gamma parameter (default: 2.0)
        label_smoothing: Label smoothing factor (default: 0.1)
        focal_weight: Weight for focal loss vs label smoothing (default: 0.5)
        reduction: 'mean', 'sum', or 'none' (default: 'mean')
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        focal_weight: float = 0.5,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha, gamma, reduction='none')
        self.ce_loss = LabelSmoothingCrossEntropy(label_smoothing, reduction='none')
        self.focal_weight = focal_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits tensor of shape [B, num_classes]
            targets: Ground truth class indices of shape [B]

        Returns:
            Combined loss scalar or per-sample losses
        """
        focal = self.focal_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        loss = self.focal_weight * focal + (1 - self.focal_weight) * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


_LOSS_CREATORS = {
    'focal': lambda **kw: FocalLoss(alpha=kw['alpha'], gamma=kw['gamma']),
    'label_smoothing': lambda **kw: LabelSmoothingCrossEntropy(smoothing=kw['label_smoothing']),
    'combined': lambda **kw: CombinedLoss(
        alpha=kw['alpha'],
        gamma=kw['gamma'],
        label_smoothing=kw['label_smoothing'],
        focal_weight=kw['focal_weight'],
        reduction='mean'
    ),
    'cross_entropy': lambda **kw: nn.CrossEntropyLoss(label_smoothing=kw['label_smoothing']),
}


class OHEMLoss(nn.Module):
    """Online Hard Example Mining (OHEM) Loss Wrapper

    Selects hard examples based on loss values and only backpropagates through them.
    Based on "Training Region-based Object Detectors with Online Hard Example Mining"
    (Shrivastava et al., 2016)

    Args:
        base_loss: Base loss function (e.g., FocalLoss, CrossEntropyLoss)
        hard_ratio: Ratio of hard examples to keep (default: 0.25)
        min_hard_num: Minimum number of hard examples to keep (default: 4)
        reduction: 'mean' or 'sum' for final aggregation

    Example:
        >>> base_loss = FocalLoss(alpha=0.25, gamma=2.0)
        >>> ohem_loss = OHEMLoss(base_loss, hard_ratio=0.25)
        >>> loss = ohem_loss(inputs, targets)
    """

    def __init__(
        self,
        base_loss: nn.Module,
        hard_ratio: float = 0.25,
        min_hard_num: int = 4,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.base_loss = base_loss
        self.hard_ratio = hard_ratio
        self.min_hard_num = min_hard_num
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits tensor of shape [B, num_classes]
            targets: Ground truth class indices of shape [B]

        Returns:
            OHEM loss scalar (only hard examples contribute)
        """
        batch_size = inputs.size(0)

        # Compute per-sample loss (no reduction)
        with torch.no_grad():
            # Temporarily disable reduction for base loss
            original_reduction = getattr(self.base_loss, 'reduction', None)
            if original_reduction is not None:
                self.base_loss.reduction = 'none'

            per_sample_loss = self.base_loss(inputs, targets)

            # Restore original reduction
            if original_reduction is not None:
                self.base_loss.reduction = original_reduction

        # Handle multi-dimensional loss (e.g., from label smoothing)
        if per_sample_loss.dim() > 1:
            per_sample_loss = per_sample_loss.mean(dim=-1)

        # Calculate number of hard examples to keep (ensure valid range)
        num_hard = max(int(batch_size * self.hard_ratio), min(self.min_hard_num, batch_size))
        num_hard = min(num_hard, batch_size)

        # Select hard examples (top-k by loss value)
        _, hard_indices = torch.topk(per_sample_loss, num_hard, largest=True, sorted=False)

        # Get hard examples
        hard_inputs = inputs[hard_indices]
        hard_targets = targets[hard_indices]

        # Compute loss only for hard examples
        hard_loss = self.base_loss(hard_inputs, hard_targets)

        # Aggregate
        if self.reduction == 'mean':
            return hard_loss.mean() if hard_loss.dim() > 0 else hard_loss
        elif self.reduction == 'sum':
            return hard_loss.sum() if hard_loss.dim() > 0 else hard_loss
        return hard_loss

    def get_hard_mask(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Get boolean mask indicating which samples are selected as hard examples.

        Useful for debugging or visualization.

        Returns:
            Boolean mask of shape [B], True for hard examples
        """
        batch_size = inputs.size(0)

        with torch.no_grad():
            original_reduction = getattr(self.base_loss, 'reduction', None)
            if original_reduction is not None:
                self.base_loss.reduction = 'none'

            per_sample_loss = self.base_loss(inputs, targets)

            if original_reduction is not None:
                self.base_loss.reduction = original_reduction

        if per_sample_loss.dim() > 1:
            per_sample_loss = per_sample_loss.mean(dim=-1)

        num_hard = max(int(batch_size * self.hard_ratio), min(self.min_hard_num, batch_size))
        num_hard = min(num_hard, batch_size)

        _, hard_indices = torch.topk(per_sample_loss, num_hard, largest=True, sorted=False)

        mask = torch.zeros(batch_size, dtype=torch.bool, device=inputs.device)
        mask[hard_indices] = True
        return mask


def create_loss(
    loss_type: str = 'combined',
    alpha: float = 0.25,
    gamma: float = 2.0,
    label_smoothing: float = 0.1,
    focal_weight: float = 0.5,
    ohem_hard_ratio: float = 0.25,
    ohem_min_hard_num: int = 4
) -> nn.Module:
    """
    Factory function to create loss function

    Args:
        loss_type: 'focal', 'label_smoothing', 'combined', 'cross_entropy',
                   'ohem_focal', 'ohem_ce', or 'ohem_combined'
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor
        focal_weight: Weight for focal loss in combined mode
        ohem_hard_ratio: Ratio of hard examples for OHEM
        ohem_min_hard_num: Minimum number of hard examples for OHEM

    Returns:
        Loss module

    Raises:
        ValueError: If loss_type is not recognized
    """
    # Handle OHEM variants
    if loss_type.startswith('ohem_'):
        base_type = loss_type[5:]  # Remove 'ohem_' prefix
        # Map aliases to full names
        type_aliases = {'ce': 'cross_entropy'}
        base_type = type_aliases.get(base_type, base_type)
        if base_type not in _LOSS_CREATORS:
            raise ValueError(f"Unknown OHEM base loss type: {base_type}")

        base_loss = _LOSS_CREATORS[base_type](
            alpha=alpha, gamma=gamma, label_smoothing=label_smoothing, focal_weight=focal_weight
        )
        return OHEMLoss(base_loss, hard_ratio=ohem_hard_ratio, min_hard_num=ohem_min_hard_num)

    # Standard loss types
    if loss_type not in _LOSS_CREATORS:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return _LOSS_CREATORS[loss_type](
        alpha=alpha, gamma=gamma, label_smoothing=label_smoothing, focal_weight=focal_weight
    )
