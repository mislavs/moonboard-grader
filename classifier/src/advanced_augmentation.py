"""
Advanced Data Augmentation for Moonboard Problems

Implements sophisticated augmentation techniques beyond simple horizontal flipping
to increase dataset diversity and reduce overfitting.
"""

import torch
import random
import numpy as np
from typing import Optional, List, Tuple


class AdvancedMoonboardAugmentation:
    """
    Advanced augmentation pipeline for Moonboard climbing problems.
    
    Augmentations:
        1. Horizontal flip: Mirrors the board left-to-right
        2. Gaussian noise: Simulates uncertainty in hold marking
        3. Hold dropout: Randomly removes middle holds to create variations
        4. Intensity jitter: Small variations in hold confidence
        
    These augmentations preserve the fundamental difficulty while
    increasing training data diversity.
    
    Args:
        flip_prob: Probability of horizontal flip (default: 0.5)
        noise_prob: Probability of adding Gaussian noise (default: 0.3)
        noise_level: Standard deviation of Gaussian noise (default: 0.05)
        dropout_prob: Probability of hold dropout (default: 0.2)
        dropout_rate: Fraction of middle holds to drop (default: 0.1)
        jitter_prob: Probability of intensity jitter (default: 0.3)
        jitter_range: Range of intensity variation (default: 0.1)
        
    Examples:
        >>> aug = AdvancedMoonboardAugmentation(
        ...     flip_prob=0.5,
        ...     noise_prob=0.3,
        ...     dropout_prob=0.2
        ... )
        >>> augmented_grid = aug(original_grid)
    """
    
    def __init__(
        self,
        flip_prob: float = 0.5,
        noise_prob: float = 0.3,
        noise_level: float = 0.05,
        dropout_prob: float = 0.2,
        dropout_rate: float = 0.1,
        jitter_prob: float = 0.3,
        jitter_range: float = 0.1
    ):
        # Validate probabilities
        for name, value in [
            ('flip_prob', flip_prob),
            ('noise_prob', noise_prob),
            ('dropout_prob', dropout_prob),
            ('jitter_prob', jitter_prob)
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        
        if noise_level < 0:
            raise ValueError(f"noise_level must be non-negative, got {noise_level}")
        
        if jitter_range < 0:
            raise ValueError(f"jitter_range must be non-negative, got {jitter_range}")
        
        self.flip_prob = flip_prob
        self.noise_prob = noise_prob
        self.noise_level = noise_level
        self.dropout_prob = dropout_prob
        self.dropout_rate = dropout_rate
        self.jitter_prob = jitter_prob
        self.jitter_range = jitter_range
    
    def __call__(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation pipeline to a grid tensor.
        
        Args:
            grid: Input tensor of shape (3, 18, 11)
                  - Channel 0: Start holds
                  - Channel 1: Middle holds
                  - Channel 2: End holds
                  
        Returns:
            Augmented grid tensor of same shape (3, 18, 11)
        """
        # Clone to avoid modifying original
        grid = grid.clone()
        
        # 1. Horizontal flip
        if random.random() < self.flip_prob:
            grid = self._horizontal_flip(grid)
        
        # 2. Gaussian noise
        if random.random() < self.noise_prob:
            grid = self._add_noise(grid)
        
        # 3. Hold dropout (only on middle holds)
        if random.random() < self.dropout_prob:
            grid = self._dropout_holds(grid)
        
        # 4. Intensity jitter
        if random.random() < self.jitter_prob:
            grid = self._intensity_jitter(grid)
        
        # Ensure values stay in valid range [0, 1]
        grid = torch.clamp(grid, 0.0, 1.0)
        
        return grid
    
    def _horizontal_flip(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Flip the grid horizontally (mirror left-right).
        
        Simulates the same problem but mirrored, which is still
        a valid climbing problem with similar difficulty.
        """
        return torch.flip(grid, dims=[2])
    
    def _add_noise(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to simulate uncertainty in hold marking.
        
        Real climbers might interpret holds slightly differently,
        so small noise helps model generalize.
        """
        noise = torch.randn_like(grid) * self.noise_level
        return grid + noise
    
    def _dropout_holds(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Randomly remove middle holds to create problem variations.
        
        Only drops middle holds (channel 1), never start/end holds,
        since those are critical for grade determination.
        """
        if grid[1].sum() > 0:  # Only if there are middle holds
            # Create random mask for middle holds
            mask = (torch.rand_like(grid[1]) > self.dropout_rate).float()
            grid[1] = grid[1] * mask
        
        return grid
    
    def _intensity_jitter(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Apply small random variations to hold intensities.
        
        Simulates variation in how holds are marked (confidence levels).
        """
        jitter = (torch.rand_like(grid) - 0.5) * 2 * self.jitter_range
        return grid + jitter
    
    def __repr__(self) -> str:
        """String representation of the augmentation."""
        return (
            f"AdvancedMoonboardAugmentation("
            f"flip_prob={self.flip_prob}, "
            f"noise_prob={self.noise_prob}, "
            f"dropout_prob={self.dropout_prob})"
        )


class MixUpAugmentation:
    """
    MixUp augmentation for climbing problems.
    
    Creates synthetic training samples by linearly interpolating between
    two climbing problems. This is a powerful regularization technique.
    
    Reference: Zhang et al. "mixup: Beyond Empirical Risk Minimization" (2017)
    
    Note: This augmentation requires special handling in training loop
    since it modifies both data and labels.
    
    Args:
        alpha: Beta distribution parameter (default: 0.2)
               Lower values = less mixing
               Higher values = more mixing
        
    Examples:
        >>> mixup = MixUpAugmentation(alpha=0.2)
        >>> mixed_grids, y_a, y_b, lam = mixup(grids, labels)
        >>> # In training loop:
        >>> outputs = model(mixed_grids)
        >>> loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
    """
    
    def __init__(self, alpha: float = 0.2):
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.alpha = alpha
    
    def __call__(
        self,
        grids: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp augmentation to a batch.
        
        Args:
            grids: Batch of grid tensors, shape (batch_size, 3, 18, 11)
            labels: Batch of labels, shape (batch_size,)
            
        Returns:
            Tuple of (mixed_grids, labels_a, labels_b, lambda) where:
            - mixed_grids: Interpolated grids
            - labels_a: Original labels
            - labels_b: Shuffled labels for mixing
            - lambda: Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = grids.size(0)
        
        # Generate random permutation
        index = torch.randperm(batch_size, device=grids.device)
        
        # Mix grids
        mixed_grids = lam * grids + (1 - lam) * grids[index]
        
        # Return mixed data and both label sets
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_grids, labels_a, labels_b, lam
    
    def __repr__(self) -> str:
        return f"MixUpAugmentation(alpha={self.alpha})"


class CutMixAugmentation:
    """
    CutMix augmentation for climbing problems.
    
    Replaces a rectangular region of one problem with a patch from another.
    More localized than MixUp - can create hybrid problems.
    
    Reference: Yun et al. "CutMix: Regularization Strategy to Train Strong Classifiers" (2019)
    
    Args:
        alpha: Beta distribution parameter (default: 1.0)
        
    Examples:
        >>> cutmix = CutMixAugmentation(alpha=1.0)
        >>> mixed_grids, y_a, y_b, lam = cutmix(grids, labels)
    """
    
    def __init__(self, alpha: float = 1.0):
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.alpha = alpha
    
    def __call__(
        self,
        grids: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation to a batch.
        
        Args:
            grids: Batch of grid tensors, shape (batch_size, 3, 18, 11)
            labels: Batch of labels, shape (batch_size,)
            
        Returns:
            Tuple of (mixed_grids, labels_a, labels_b, lambda)
        """
        batch_size, _, height, width = grids.size()
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Generate random permutation
        index = torch.randperm(batch_size, device=grids.device)
        
        # Calculate cut area
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)
        
        # Random center point
        cx = np.random.randint(width)
        cy = np.random.randint(height)
        
        # Calculate bounding box
        x1 = np.clip(cx - cut_w // 2, 0, width)
        y1 = np.clip(cy - cut_h // 2, 0, height)
        x2 = np.clip(cx + cut_w // 2, 0, width)
        y2 = np.clip(cy + cut_h // 2, 0, height)
        
        # Apply CutMix
        mixed_grids = grids.clone()
        mixed_grids[:, :, y1:y2, x1:x2] = grids[index, :, y1:y2, x1:x2]
        
        # Adjust lambda to match actual area ratio
        actual_lam = 1 - ((x2 - x1) * (y2 - y1) / (width * height))
        
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_grids, labels_a, labels_b, actual_lam
    
    def __repr__(self) -> str:
        return f"CutMixAugmentation(alpha={self.alpha})"


def create_augmentation_pipeline(
    aug_type: str = 'advanced',
    **kwargs
) -> callable:
    """
    Factory function to create augmentation pipeline.
    
    Args:
        aug_type: Type of augmentation
            - 'none': No augmentation (identity)
            - 'basic': Horizontal flip only
            - 'advanced': Full augmentation suite
            - 'mixup': MixUp augmentation
            - 'cutmix': CutMix augmentation
        **kwargs: Additional arguments for augmentation
        
    Returns:
        Augmentation callable
        
    Examples:
        >>> # For training
        >>> train_aug = create_augmentation_pipeline('advanced', flip_prob=0.5)
        
        >>> # For validation/test
        >>> val_aug = create_augmentation_pipeline('none')
        
        >>> # MixUp for training loop
        >>> mixup = create_augmentation_pipeline('mixup', alpha=0.2)
    """
    from .augmentation import no_augmentation, MoonboardAugmentation
    
    if aug_type == 'none':
        return no_augmentation
    
    elif aug_type == 'basic':
        flip_prob = kwargs.get('flip_prob', 0.5)
        return MoonboardAugmentation(flip_prob=flip_prob)
    
    elif aug_type == 'advanced':
        return AdvancedMoonboardAugmentation(**kwargs)
    
    elif aug_type == 'mixup':
        alpha = kwargs.get('alpha', 0.2)
        return MixUpAugmentation(alpha=alpha)
    
    elif aug_type == 'cutmix':
        alpha = kwargs.get('alpha', 1.0)
        return CutMixAugmentation(alpha=alpha)
    
    else:
        raise ValueError(
            f"Unknown aug_type '{aug_type}'. "
            f"Must be one of: 'none', 'basic', 'advanced', 'mixup', 'cutmix'"
        )

