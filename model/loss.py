"""
Loss functions for CryTransformer models

Includes:
- FocalLoss: For handling class imbalance
- LabelSmoothingCrossEntropy: Label smoothing for better generalization
- CombinedLoss: Combined Focal + Label Smoothing
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
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        focal_weight: float = 0.5
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.ce_loss = LabelSmoothingCrossEntropy(label_smoothing)
        self.focal_weight = focal_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits tensor of shape [B, num_classes]
            targets: Ground truth class indices of shape [B]

        Returns:
            Combined loss scalar
        """
        focal = self.focal_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)
        return self.focal_weight * focal + (1 - self.focal_weight) * ce


# Factory function to create loss from config
def create_loss(
    loss_type: str = 'combined',
    alpha: float = 0.25,
    gamma: float = 2.0,
    label_smoothing: float = 0.1,
    focal_weight: float = 0.5
) -> nn.Module:
    """
    Factory function to create loss function

    Args:
        loss_type: 'focal', 'label_smoothing', 'combined', or 'cross_entropy'
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor
        focal_weight: Weight for focal loss in combined mode

    Returns:
        Loss module
    """
    if loss_type == 'focal':
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == 'label_smoothing':
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    elif loss_type == 'combined':
        return CombinedLoss(
            alpha=alpha,
            gamma=gamma,
            label_smoothing=label_smoothing,
            focal_weight=focal_weight
        )
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
