"""Loss functions for CryTransformer models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """Apply reduction to per-sample loss tensor."""
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (Lin et al., 2017).

    Down-weights easy examples and focuses on hard negatives.

    Args:
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * ce_loss
        return _apply_reduction(loss, self.reduction)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss (Szegedy et al., 2015).

    Prevents over-confidence by smoothing target distribution.

    Args:
        smoothing: Smoothing factor
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing cross entropy loss."""
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_smooth = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes

        loss = -(targets_smooth * log_probs).sum(dim=-1)
        return _apply_reduction(loss, self.reduction)


class CombinedLoss(nn.Module):
    """Combined Focal Loss + Label Smoothing.

    Combines focal loss (class imbalance handling) with label smoothing.

    Args:
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor
        focal_weight: Weight for focal loss vs label smoothing
        reduction: 'mean', 'sum', or 'none'
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
        """Compute combined loss."""
        focal = self.focal_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        loss = self.focal_weight * focal + (1 - self.focal_weight) * ce
        return _apply_reduction(loss, self.reduction)


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
    """Online Hard Example Mining (OHEM) Loss Wrapper.

    Selects hard examples based on loss values and only backpropagates through them.
    Based on Shrivastava et al., 2016.

    Args:
        base_loss: Base loss function (e.g., FocalLoss, CrossEntropyLoss)
        hard_ratio: Ratio of hard examples to keep
        min_hard_num: Minimum number of hard examples to keep
        reduction: 'mean' or 'sum' for final aggregation
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

    def _compute_per_sample_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute per-sample loss without gradient tracking."""
        with torch.no_grad():
            original_reduction = getattr(self.base_loss, 'reduction', None)
            if original_reduction is not None:
                self.base_loss.reduction = 'none'
            per_sample_loss = self.base_loss(inputs, targets)
            if original_reduction is not None:
                self.base_loss.reduction = original_reduction

        if per_sample_loss.dim() > 1:
            per_sample_loss = per_sample_loss.mean(dim=-1)
        return per_sample_loss

    def _select_hard_indices(self, per_sample_loss: torch.Tensor) -> torch.Tensor:
        """Select indices of hard examples based on loss values."""
        batch_size = per_sample_loss.size(0)
        num_hard = max(int(batch_size * self.hard_ratio), min(self.min_hard_num, batch_size))
        num_hard = min(num_hard, batch_size)
        _, hard_indices = torch.topk(per_sample_loss, num_hard, largest=True, sorted=False)
        return hard_indices

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute OHEM loss on hard examples only."""
        per_sample_loss = self._compute_per_sample_loss(inputs, targets)
        hard_indices = self._select_hard_indices(per_sample_loss)

        hard_inputs = inputs[hard_indices]
        hard_targets = targets[hard_indices]
        hard_loss = self.base_loss(hard_inputs, hard_targets)

        return _apply_reduction(hard_loss, self.reduction)

    def get_hard_mask(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Get boolean mask indicating hard example selection."""
        batch_size = inputs.size(0)
        per_sample_loss = self._compute_per_sample_loss(inputs, targets)
        hard_indices = self._select_hard_indices(per_sample_loss)

        mask = torch.zeros(batch_size, dtype=torch.bool, device=inputs.device)
        mask[hard_indices] = True
        return mask


# Loss type aliases for convenience
_LOSS_ALIASES = {'ce': 'cross_entropy'}


def create_loss(
    loss_type: str = 'combined',
    alpha: float = 0.25,
    gamma: float = 2.0,
    label_smoothing: float = 0.1,
    focal_weight: float = 0.5,
    ohem_hard_ratio: float = 0.25,
    ohem_min_hard_num: int = 4
) -> nn.Module:
    """Factory function to create loss function.

    Args:
        loss_type: One of 'focal', 'label_smoothing', 'combined', 'cross_entropy',
                   'ohem_focal', 'ohem_ce', 'ohem_combined'
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor
        focal_weight: Weight for focal loss in combined mode
        ohem_hard_ratio: Ratio of hard examples for OHEM
        ohem_min_hard_num: Minimum number of hard examples for OHEM

    Returns:
        Configured loss module

    Raises:
        ValueError: If loss_type is not recognized
    """
    # Handle OHEM variants
    if loss_type.startswith('ohem_'):
        base_type = loss_type[5:]  # Remove 'ohem_' prefix
        base_type = _LOSS_ALIASES.get(base_type, base_type)

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
