"""
Advanced Loss Functions for Moonboard Grade Classification

Implements specialized loss functions to handle class imbalance and
ordinal nature of climbing grades.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal loss down-weights easy examples and focuses training on hard negatives.
    This is particularly useful for imbalanced datasets where some classes
    (like 8A+, 8B) have very few samples.
    
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Class weights. If None, all classes weighted equally.
               Can be a float or tensor of shape (num_classes,)
        gamma: Focusing parameter. Higher values increase focus on hard examples.
               Typical values: 1.5-3.0. Default: 2.0
        reduction: 'mean', 'sum', or 'none'
        
    Examples:
        >>> # Basic usage
        >>> criterion = FocalLoss(gamma=2.0)
        >>> loss = criterion(logits, targets)
        
        >>> # With class weights
        >>> weights = torch.tensor([1.0, 2.0, 3.0, ...])  # 19 classes
        >>> criterion = FocalLoss(alpha=weights, gamma=2.0)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predicted logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Loss value (scalar if reduction='mean'/'sum', tensor if 'none')
        """
        # Get true class probabilities via softmax
        probs = F.softmax(inputs, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute unweighted cross-entropy (needed for base loss)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute focal modulation: (1 - p_t)^gamma
        # This correctly down-weights easy examples (high p_t) and focuses on hard ones
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal modulation to CE loss
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights (alpha) if provided
        if self.alpha is not None:
            # Ensure alpha is on the same device as inputs
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class OrdinalCrossEntropyLoss(nn.Module):
    """
    Ordinal Cross-Entropy Loss for climbing grades.
    
    Standard cross-entropy treats all misclassifications equally:
    - Predicting 6A as 6B: same penalty as 6A as 8C
    
    Ordinal loss penalizes predictions proportionally to distance:
    - Predicting 6A as 6B: small penalty
    - Predicting 6A as 8C: large penalty
    
    This better captures the ordinal nature of climbing grades.
    
    Args:
        num_classes: Number of grade classes (default: 19)
        alpha: Distance penalty multiplier (default: 2.0)
               Higher values = stronger penalty for distant predictions
        reduction: 'mean', 'sum', or 'none'
        
    Examples:
        >>> criterion = OrdinalCrossEntropyLoss(num_classes=19, alpha=2.0)
        >>> loss = criterion(logits, targets)
    """
    
    def __init__(
        self,
        num_classes: int = 19,
        alpha: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.reduction = reduction
        
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ordinal cross-entropy loss.
        
        Args:
            inputs: Predicted logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Loss value (scalar if reduction='mean'/'sum', tensor if 'none')
        """
        # Get probabilities
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create one-hot encoded targets
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        
        # Calculate distance matrix
        # Distance from each class to all other classes
        class_indices = torch.arange(self.num_classes, device=inputs.device)
        distances = torch.abs(class_indices[None, :] - targets[:, None])
        
        # Calculate distance-based weights
        # weight = 1 + alpha * distance
        # Correct class (distance=0): weight=1
        # Adjacent class (distance=1): weight=1+alpha
        # Far class (distance=10): weight=1+10*alpha
        weights = 1.0 + self.alpha * distances.float()
        
        # Weighted negative log-likelihood
        loss = -torch.sum(targets_one_hot * log_probs * weights, dim=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalOrdinalLoss(nn.Module):
    """
    Combined Focal + Ordinal Loss.
    
    Combines the benefits of both:
    - Focal component: Handles class imbalance
    - Ordinal component: Respects grade ordering
    
    Args:
        num_classes: Number of grade classes (default: 19)
        alpha: Class weights for focal loss
        gamma: Focusing parameter for focal loss (default: 2.0)
        ordinal_weight: Weight for ordinal component (default: 1.0)
        ordinal_alpha: Distance penalty for ordinal loss (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
        
    Examples:
        >>> weights = compute_class_weights(train_labels)
        >>> criterion = FocalOrdinalLoss(
        ...     alpha=weights, 
        ...     gamma=2.0, 
        ...     ordinal_weight=0.5
        ... )
    """
    
    def __init__(
        self,
        num_classes: int = 19,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ordinal_weight: float = 1.0,
        ordinal_alpha: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.ordinal_loss = OrdinalCrossEntropyLoss(
            num_classes=num_classes,
            alpha=ordinal_alpha,
            reduction=reduction
        )
        self.ordinal_weight = ordinal_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined focal + ordinal loss.
        
        Args:
            inputs: Predicted logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Combined loss value
        """
        focal = self.focal_loss(inputs, targets)
        ordinal = self.ordinal_loss(inputs, targets)
        
        return focal + self.ordinal_weight * ordinal


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-Entropy Loss with Label Smoothing.
    
    Label smoothing prevents overconfident predictions by replacing
    hard targets (0/1) with smoothed targets (e.g., 0.1/0.9).
    
    Args:
        smoothing: Smoothing parameter (0.0 = no smoothing, 0.1 = typical)
        reduction: 'mean', 'sum', or 'none'
        
    Examples:
        >>> criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy.
        
        Args:
            inputs: Predicted logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            Loss value
        """
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # Apply label smoothing
        # True class: 1 - smoothing + smoothing / num_classes
        # Other classes: smoothing / num_classes
        smoothed_targets = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        
        # Compute loss
        loss = -torch.sum(smoothed_targets * log_probs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_loss_function(
    loss_type: str = 'focal',
    num_classes: int = 19,
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss function.
    
    Args:
        loss_type: Type of loss function
            - 'ce': Standard cross-entropy
            - 'focal': Focal loss (for class imbalance)
            - 'ordinal': Ordinal cross-entropy (for grade ordering)
            - 'focal_ordinal': Combined focal + ordinal
            - 'label_smoothing': Cross-entropy with label smoothing
        num_classes: Number of grade classes
        class_weights: Class weights tensor
        **kwargs: Additional arguments passed to loss constructor
        
    Returns:
        Loss function instance
        
    Examples:
        >>> # Focal loss with class weights
        >>> loss_fn = create_loss_function(
        ...     'focal', 
        ...     num_classes=19, 
        ...     class_weights=weights,
        ...     gamma=2.0
        ... )
        
        >>> # Ordinal loss
        >>> loss_fn = create_loss_function('ordinal', alpha=2.0)
        
        >>> # Combined focal + ordinal
        >>> loss_fn = create_loss_function(
        ...     'focal_ordinal',
        ...     class_weights=weights,
        ...     gamma=2.0,
        ...     ordinal_weight=0.5
        ... )
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_type == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)
    
    elif loss_type == 'ordinal':
        alpha = kwargs.get('alpha', 2.0)
        return OrdinalCrossEntropyLoss(num_classes=num_classes, alpha=alpha)
    
    elif loss_type == 'focal_ordinal':
        gamma = kwargs.get('gamma', 2.0)
        ordinal_weight = kwargs.get('ordinal_weight', 0.5)
        ordinal_alpha = kwargs.get('ordinal_alpha', 2.0)
        return FocalOrdinalLoss(
            num_classes=num_classes,
            alpha=class_weights,
            gamma=gamma,
            ordinal_weight=ordinal_weight,
            ordinal_alpha=ordinal_alpha
        )
    
    elif loss_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    
    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. "
            f"Must be one of: 'ce', 'focal', 'ordinal', 'focal_ordinal', 'label_smoothing'"
        )

