"""
Data Splitting Module

Provides utilities for creating stratified train/val/test splits,
including an optional group-aware mode that prevents hold-layout
leakage across splits.
"""

import hashlib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from typing import Tuple, Optional, Dict, Any
from torch.utils.data import DataLoader

from .dataset import MoonboardDataset
from .split_validation import (
    raise_friendly_stratify_error,
    validate_two_stage_stratified_feasibility,
)


def compute_layout_hashes(tensors: np.ndarray) -> np.ndarray:
    """
    Compute a canonical hash for each tensor's hold layout.

    Two tensors with identical hold positions (regardless of grade label)
    receive the same hash. This allows grouping duplicate layouts.

    Args:
        tensors: Array of shape (N, 3, 18, 11)

    Returns:
        Array of N integer hash values (one per sample)
    """
    hashes = np.empty(len(tensors), dtype=np.int64)
    for i in range(len(tensors)):
        digest = hashlib.md5(tensors[i].tobytes()).digest()
        hashes[i] = int.from_bytes(digest[:8], 'little', signed=True)
    return hashes


def create_stratified_splits(
    tensors: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified train/validation/test splits.
    
    Args:
        tensors: Array of shape (N, 3, 18, 11) containing grid tensors
        labels: Array of shape (N,) containing grade labels
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Tuple of (train_idx, val_idx, test_idx, train_data, val_data, test_data)
    
    Examples:
        >>> tensors = np.random.randn(100, 3, 18, 11)
        >>> labels = np.random.randint(0, 19, 100)
        >>> train_idx, val_idx, test_idx, _, _, _ = create_stratified_splits(tensors, labels)
    """
    context = validate_two_stage_stratified_feasibility(
        labels, train_ratio, val_ratio, test_ratio
    )

    # First split: separate test set
    test_size = test_ratio
    splitter_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_seed
    )
    try:
        train_val_idx, test_idx = next(splitter_test.split(tensors, labels))
    except ValueError as e:
        raise_friendly_stratify_error("test split", e, context)
    
    # Second split: separate train and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    splitter_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio_adjusted, random_state=random_seed
    )
    try:
        train_idx, val_idx = next(
            splitter_val.split(tensors[train_val_idx], labels[train_val_idx])
        )
    except ValueError as e:
        raise_friendly_stratify_error("train/validation split", e, context)
    
    # Map back to original indices
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]
    
    # Extract splits
    train_data = (tensors[train_idx], labels[train_idx])
    val_data = (tensors[val_idx], labels[val_idx])
    test_data = (tensors[test_idx], labels[test_idx])
    
    return train_idx, val_idx, test_idx, train_data, val_data, test_data


def create_grouped_splits(
    tensors: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, tuple, tuple, tuple]:
    """
    Create train/val/test splits grouped by hold-layout hash.

    Problems with the same hold layout are always assigned to the same split,
    preventing data leakage from duplicate or near-identical layouts.

    Args:
        tensors: Array of shape (N, 3, 18, 11)
        labels: Array of shape (N,)
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_idx, val_idx, test_idx, train_data, val_data, test_data)
    """
    groups = compute_layout_hashes(tensors)

    # First split: separate test groups
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=random_seed
    )
    train_val_idx, test_idx = next(gss_test.split(tensors, labels, groups))

    # Second split: separate train and validation groups from remaining
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_ratio_adjusted, random_state=random_seed
    )
    remaining_groups = groups[train_val_idx]
    rel_train_idx, rel_val_idx = next(
        gss_val.split(tensors[train_val_idx], labels[train_val_idx], remaining_groups)
    )

    # Map back to original indices
    train_idx_final = train_val_idx[rel_train_idx]
    val_idx_final = train_val_idx[rel_val_idx]

    train_data = (tensors[train_idx_final], labels[train_idx_final])
    val_data = (tensors[val_idx_final], labels[val_idx_final])
    test_data = (tensors[test_idx], labels[test_idx])

    return train_idx_final, val_idx_final, test_idx, train_data, val_data, test_data


def create_datasets(
    tensors: np.ndarray,
    labels: np.ndarray,
    config: Dict[str, Any],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[MoonboardDataset, MoonboardDataset, MoonboardDataset]:
    """
    Create train/val/test datasets.

    Supports optional group-aware splitting via ``config['data']['group_by_layout']``
    to prevent hold-layout leakage across splits.

    Args:
        tensors: Array of shape (N, 3, 18, 11) containing grid tensors
        labels: Array of shape (N,) containing grade labels
        config: Configuration dictionary with data settings
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Examples:
        >>> config = {'data': {}}
        >>> train_ds, val_ds, test_ds = create_datasets(tensors, labels, config)
    """
    group_by_layout = config.get('data', {}).get('group_by_layout', False)

    if group_by_layout:
        train_idx, val_idx, test_idx, _, _, _ = create_grouped_splits(
            tensors, labels, train_ratio, val_ratio, test_ratio, random_seed
        )
    else:
        train_idx, val_idx, test_idx, _, _, _ = create_stratified_splits(
            tensors, labels, train_ratio, val_ratio, test_ratio, random_seed
        )

    train_dataset = MoonboardDataset(tensors[train_idx], labels[train_idx])
    val_dataset = MoonboardDataset(tensors[val_idx], labels[val_idx])
    test_dataset = MoonboardDataset(tensors[test_idx], labels[test_idx])

    return train_dataset, val_dataset, test_dataset


def create_data_loaders(
    train_dataset: MoonboardDataset,
    val_dataset: MoonboardDataset,
    test_dataset: MoonboardDataset,
    batch_size: int = 64,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for data loading (default: 64)
        shuffle_train: Whether to shuffle training data (default: True)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    
    Examples:
        >>> train_loader, val_loader, test_loader = create_data_loaders(
        ...     train_ds, val_ds, test_ds, batch_size=32
        ... )
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

