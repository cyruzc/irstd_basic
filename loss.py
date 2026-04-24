"""Loss functions for IR small target segmentation.

All losses operate on logits (raw model output, no sigmoid).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELogits(nn.Module):
    """Binary cross-entropy with logits (numerically stable)."""

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets)


class SoftIoU(nn.Module):
    """Soft IoU loss — 1 - IoU, computed from sigmoid probabilities."""

    def forward(self, logits, targets):
        preds = logits.sigmoid()
        inter = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - inter
        return 1 - (inter / (union + 1)).mean()


class FocalLogits(nn.Module):
    """Logits-based focal loss (numerically stable).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Built on F.binary_cross_entropy_with_logits for numerical stability.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = logits.sigmoid()
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class CombinedLoss(nn.Module):
    """Weighted combination of multiple losses."""

    def __init__(self, losses, weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights or [1.0] * len(losses)

    def forward(self, logits, targets):
        total = 0.0
        for loss_fn, w in zip(self.losses, self.weights):
            total = total + w * loss_fn(logits, targets)
        return total
