"""
PyTorch Dataset class for Moonboard climbing problems.

This module provides a PyTorch Dataset wrapper for processed moonboard data
and utilities for creating stratified train/validation/test splits.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple, List, Optional

from .split_validation import (
    raise_friendly_stratify_error,
    validate_two_stage_stratified_feasibility,
)


class MoonboardDataset(Dataset):
    """
    PyTorch Dataset for moonboard climbing problems.
    
    Args:
        data: List of numpy arrays or single stacked numpy array of shape (N, 3, 18, 11)
        labels: List of integer labels or numpy array of shape (N,)
        
    The dataset returns (tensor, label) pairs where:
        - tensor: torch.FloatTensor of shape (3, 18, 11) representing the grid
        - label: integer representing the grade label
    """
    
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray
    ):
        """
        Initialize the dataset.
        
        Args:
            data: numpy array of shape (N, 3, 18, 11) or list of (3, 18, 11) arrays
            labels: numpy array of shape (N,) or list of integers
            
        Raises:
            ValueError: If data and labels have different lengths
            TypeError: If inputs are not numpy arrays or convertible to arrays
        """
        # Convert to numpy arrays if not already
        if isinstance(data, list):
            data = np.array(data)
        if isinstance(labels, list):
            labels = np.array(labels)
            
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy array or list of arrays")
        if not isinstance(labels, np.ndarray):
            raise TypeError("labels must be a numpy array or list of integers")
            
        # Validate data dimensionality first
        if data.ndim == 4:
            if data.shape[1:] != (3, 18, 11):
                raise ValueError(
                    f"Expected data shape (N, 3, 18, 11), got {data.shape}"
                )
        elif data.ndim == 3:
            # Single sample
            if data.shape != (3, 18, 11):
                raise ValueError(
                    f"Expected data shape (3, 18, 11), got {data.shape}"
                )
            data = data[np.newaxis, ...]  # Add batch dimension
        else:
            raise ValueError(
                f"Expected data to be 3D or 4D array, got {data.ndim}D"
            )
            
        # Now validate lengths match
        if len(data) != len(labels):
            raise ValueError(
                f"data and labels must have the same length. "
                f"Got {len(data)} samples and {len(labels)} labels"
            )
            
        # Ensure labels are integers
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("labels must be integers")
            
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (tensor, label) where:
                - tensor: torch.FloatTensor of shape (3, 18, 11)
                - label: integer grade label
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
            
        # Convert to PyTorch tensors
        sample = torch.from_numpy(self.data[idx]).float()
        label = int(self.labels[idx])
        
        return sample, label
    
    def get_label_distribution(self) -> dict:
        """
        Get the distribution of labels in the dataset.
        
        Returns:
            Dictionary mapping label -> count
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        return {int(label): int(count) for label, count in zip(unique, counts)}


def create_data_splits(
    data: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[MoonboardDataset, MoonboardDataset, MoonboardDataset]:
    """
    Create stratified train/validation/test splits of the data.
    
    Uses scikit-learn's StratifiedShuffleSplit to ensure grade distribution
    is preserved across all splits.
    
    Args:
        data: numpy array of shape (N, 3, 18, 11)
        labels: numpy array of shape (N,) with integer labels
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
        
    Raises:
        ValueError: If ratios don't sum to 1.0 or are invalid
        ValueError: If dataset is too small for splitting
    """
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0. Got {train_ratio + val_ratio + test_ratio}"
        )
    
    if train_ratio >= 1.0 or val_ratio >= 1.0 or test_ratio >= 1.0:
        raise ValueError("All ratios must be less than 1.0")
    
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("All ratios must be positive")
    
    # Convert to numpy arrays if needed
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(labels, list):
        labels = np.array(labels)
    
    # Validate minimum dataset size
    n_samples = len(data)
    if n_samples < 3:
        raise ValueError(
            f"Dataset too small for splitting. Need at least 3 samples, got {n_samples}"
        )
    
    context = validate_two_stage_stratified_feasibility(
        labels, train_ratio, val_ratio, test_ratio
    )

    # First split: separate test set
    test_size = test_ratio
    splitter_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )
    try:
        train_val_idx, test_idx = next(splitter_test.split(data, labels))
    except ValueError as e:
        raise_friendly_stratify_error("test split", e, context)
    
    # Second split: separate train and validation from remaining data
    # val_ratio_adjusted is the proportion of validation in the train_val set
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    splitter_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio_adjusted,
        random_state=random_state
    )
    
    try:
        train_idx, val_idx = next(
            splitter_val.split(data[train_val_idx], labels[train_val_idx])
        )
    except ValueError as e:
        raise_friendly_stratify_error("train/validation split", e, context)
    
    # Map back to original indices
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]
    
    # Create datasets
    train_dataset = MoonboardDataset(data[train_idx], labels[train_idx])
    val_dataset = MoonboardDataset(data[val_idx], labels[val_idx])
    test_dataset = MoonboardDataset(data[test_idx], labels[test_idx])
    
    return train_dataset, val_dataset, test_dataset


def get_split_info(
    train_dataset: MoonboardDataset,
    val_dataset: MoonboardDataset,
    test_dataset: MoonboardDataset
) -> dict:
    """
    Get information about the data splits.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        
    Returns:
        Dictionary with split information including sizes and distributions
    """
    total = len(train_dataset) + len(val_dataset) + len(test_dataset)
    
    return {
        'total_samples': total,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'train_ratio': len(train_dataset) / total,
        'val_ratio': len(val_dataset) / total,
        'test_ratio': len(test_dataset) / total,
        'train_distribution': train_dataset.get_label_distribution(),
        'val_distribution': val_dataset.get_label_distribution(),
        'test_distribution': test_dataset.get_label_distribution(),
    }

