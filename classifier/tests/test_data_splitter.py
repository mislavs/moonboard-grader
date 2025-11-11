"""
Tests for Data Splitter Module

Tests the stratified splitting and dataset creation functionality.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_splitter import (
    create_stratified_splits,
    create_datasets,
    create_data_loaders
)
from src.dataset import MoonboardDataset


class TestStratifiedSplits:
    """Test stratified data splitting."""
    
    def test_create_stratified_splits_basic(self):
        """Test basic stratified splitting."""
        # Create dummy data with controlled distribution
        # Ensure each class has at least 5 samples for proper stratification
        tensors = np.random.randn(190, 3, 18, 11).astype(np.float32)
        # Create labels with 10 samples per class (19 classes * 10 = 190)
        labels = np.repeat(np.arange(19), 10)
        np.random.shuffle(labels)
        
        # Create splits
        train_idx, val_idx, test_idx, train_data, val_data, test_data = create_stratified_splits(
            tensors, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
        )
        
        # Check indices don't overlap
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0
        
        # Check all indices are covered
        all_indices = set(train_idx) | set(val_idx) | set(test_idx)
        assert len(all_indices) == 190
        
        # Check split sizes are approximately correct
        assert len(train_idx) == pytest.approx(133, abs=10)
        assert len(val_idx) == pytest.approx(28, abs=10)
        assert len(test_idx) == pytest.approx(28, abs=10)
        
        # Check data shapes
        assert train_data[0].shape[1:] == (3, 18, 11)
        assert val_data[0].shape[1:] == (3, 18, 11)
        assert test_data[0].shape[1:] == (3, 18, 11)
    
    def test_create_stratified_splits_reproducibility(self):
        """Test that splits are reproducible with same random seed."""
        tensors = np.random.randn(190, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(19), 10)
        np.random.shuffle(labels)
        
        # Create splits twice with same seed
        train_idx1, val_idx1, test_idx1, _, _, _ = create_stratified_splits(
            tensors, labels, random_seed=42
        )
        train_idx2, val_idx2, test_idx2, _, _, _ = create_stratified_splits(
            tensors, labels, random_seed=42
        )
        
        # Check they're identical
        assert np.array_equal(train_idx1, train_idx2)
        assert np.array_equal(val_idx1, val_idx2)
        assert np.array_equal(test_idx1, test_idx2)
    
    def test_create_stratified_splits_different_seeds(self):
        """Test that different seeds produce different splits."""
        tensors = np.random.randn(190, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(19), 10)
        np.random.shuffle(labels)
        
        # Create splits with different seeds
        train_idx1, _, _, _, _, _ = create_stratified_splits(
            tensors, labels, random_seed=42
        )
        train_idx2, _, _, _, _, _ = create_stratified_splits(
            tensors, labels, random_seed=123
        )
        
        # Check they're different
        assert not np.array_equal(train_idx1, train_idx2)


class TestDataLoaders:
    """Test DataLoader creation."""
    
    def test_create_data_loaders_basic(self):
        """Test basic DataLoader creation."""
        tensors = np.random.randn(100, 3, 18, 11).astype(np.float32)
        labels = np.random.randint(0, 19, 100)
        
        train_ds = MoonboardDataset(tensors[:70], labels[:70])
        val_ds = MoonboardDataset(tensors[70:85], labels[70:85])
        test_ds = MoonboardDataset(tensors[85:], labels[85:])
        
        train_loader, val_loader, test_loader = create_data_loaders(
            train_ds, val_ds, test_ds, batch_size=32
        )
        
        # Check all are DataLoader instances
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        
        # Check batch sizes
        assert train_loader.batch_size == 32
        assert val_loader.batch_size == 32
        assert test_loader.batch_size == 32
        
        # Check we can iterate
        for batch_data, batch_labels in train_loader:
            assert batch_data.shape[1:] == (3, 18, 11)
            assert isinstance(batch_labels, torch.Tensor)
            break  # Just check first batch
    
    def test_create_data_loaders_custom_batch_size(self):
        """Test DataLoader creation with custom batch size."""
        tensors = np.random.randn(50, 3, 18, 11).astype(np.float32)
        labels = np.random.randint(0, 19, 50)
        
        train_ds = MoonboardDataset(tensors[:35], labels[:35])
        val_ds = MoonboardDataset(tensors[35:42], labels[35:42])
        test_ds = MoonboardDataset(tensors[42:], labels[42:])
        
        train_loader, val_loader, test_loader = create_data_loaders(
            train_ds, val_ds, test_ds, batch_size=16
        )
        
        assert train_loader.batch_size == 16
        assert val_loader.batch_size == 16
        assert test_loader.batch_size == 16


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

