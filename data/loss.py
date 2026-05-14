"""
Loss functions for anomaly segmentation training.
Adapted from the original LLaVA-OV anomaly detection pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in anomaly segmentation.
    
    Focuses training on hard-to-classify pixels by down-weighting
    well-classified examples.
    
    Args:
        alpha: Weighting factor for the rare class. Default: 0.25.
        gamma: Focusing parameter. Higher values focus more on hard examples. Default: 2.0.
        reduction: Specifies the reduction to apply to the output. Default: 'mean'.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted anomaly map of shape [B, 2, H, W] (softmax probabilities).
            targets: Ground truth mask of shape [B, 1, H, W] or [B, H, W] with values in {0, 1}.
        """
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Ensure targets are long for cross_entropy
        targets_long = targets.squeeze(1).long()

        # Compute cross-entropy loss per pixel
        ce_loss = F.cross_entropy(inputs, targets_long, reduction='none')

        # Get probability of the correct class
        pt = torch.exp(-ce_loss)

        # Apply focal weighting
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class BinaryDiceLoss(nn.Module):
    """
    Binary Dice Loss for anomaly segmentation.
    
    Measures overlap between predicted anomaly map and ground truth,
    particularly effective for imbalanced segmentation tasks.
    
    Args:
        smooth: Smoothing factor to avoid division by zero. Default: 1.0.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted defect probability map of shape [B, H, W] (values in [0, 1]).
            targets: Ground truth mask of shape [B, H, W] with values in {0, 1}.
        """
        # Flatten
        inputs_flat = inputs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1).float()

        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )

        return 1.0 - dice
