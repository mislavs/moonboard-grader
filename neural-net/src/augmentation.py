"""
Data Augmentation Module

Implements augmentation techniques for Moonboard climbing problems
to increase dataset diversity and reduce overfitting.
"""

import torch
import random
from typing import Optional


class MoonboardAugmentation:
    """
    Data augmentation for Moonboard grid tensors.
    
    Applies random transformations to climbing problems to increase
    training data diversity without changing the fundamental difficulty.
    
    Augmentations:
        - Horizontal flip: Mirrors the board left-to-right
        - This simulates climbers with different dominant hands
        
    Attributes:
        flip_prob: Probability of applying horizontal flip (0.0 to 1.0)
    """
    
    def __init__(self, flip_prob: float = 0.5):
        """
        Initialize the augmentation pipeline.
        
        Args:
            flip_prob: Probability of horizontal flip (default: 0.5)
                      Set to 0.0 to disable flipping
                      Set to 1.0 to always flip
                      
        Raises:
            ValueError: If flip_prob is not in range [0, 1]
            
        Examples:
            >>> aug = MoonboardAugmentation(flip_prob=0.5)
            >>> augmented_grid = aug(original_grid)
        """
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError(f"flip_prob must be in [0, 1], got {flip_prob}")
        
        self.flip_prob = flip_prob
    
    def __call__(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a grid tensor.
        
        Args:
            grid: Input tensor of shape (3, 18, 11)
                  - Channel 0: Start holds
                  - Channel 1: Middle holds
                  - Channel 2: End holds
                  
        Returns:
            Augmented grid tensor of same shape (3, 18, 11)
            
        Examples:
            >>> aug = MoonboardAugmentation(flip_prob=0.5)
            >>> grid = torch.randn(3, 18, 11)
            >>> aug_grid = aug(grid)
            >>> assert aug_grid.shape == grid.shape
        """
        # Apply horizontal flip with given probability
        if random.random() < self.flip_prob:
            grid = self._horizontal_flip(grid)
        
        return grid
    
    def _horizontal_flip(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Flip the grid horizontally (mirror left-right).
        
        This simulates the same problem but mirrored, which is still
        a valid climbing problem with similar difficulty.
        
        Args:
            grid: Input tensor of shape (3, 18, 11)
            
        Returns:
            Horizontally flipped tensor of shape (3, 18, 11)
        """
        # Flip along the width dimension (dim=2)
        # Keep channels (dim=0) and height (dim=1) unchanged
        return torch.flip(grid, dims=[2])
    
    def __repr__(self) -> str:
        """String representation of the augmentation."""
        return f"MoonboardAugmentation(flip_prob={self.flip_prob})"


def no_augmentation(grid: torch.Tensor) -> torch.Tensor:
    """
    Identity function for no augmentation.
    
    Useful for validation/test sets where augmentation should be disabled.
    
    Args:
        grid: Input tensor
        
    Returns:
        Same tensor unchanged
        
    Examples:
        >>> grid = torch.randn(3, 18, 11)
        >>> result = no_augmentation(grid)
        >>> assert torch.equal(result, grid)
    """
    return grid


def create_augmentation(
    enabled: bool = True,
    flip_prob: float = 0.5
) -> callable:
    """
    Factory function to create an augmentation callable.
    
    Args:
        enabled: Whether to enable augmentation (default: True)
                If False, returns identity function
        flip_prob: Probability of horizontal flip (default: 0.5)
        
    Returns:
        Callable that takes a grid tensor and returns augmented grid
        
    Examples:
        >>> # For training set
        >>> train_aug = create_augmentation(enabled=True, flip_prob=0.5)
        
        >>> # For validation/test sets
        >>> val_aug = create_augmentation(enabled=False)
    """
    if enabled:
        return MoonboardAugmentation(flip_prob=flip_prob)
    else:
        return no_augmentation

