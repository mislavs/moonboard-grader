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


class TestGroupedSplits:
    """Test group-aware splitting to prevent layout leakage."""

    def test_grouped_split_no_layout_overlap(self):
        """Same layout hash must not appear in more than one split."""
        from src.data_splitter import create_grouped_splits, compute_layout_hashes

        # Create data with some duplicate layouts
        base = np.random.randn(50, 3, 18, 11).astype(np.float32)
        # Duplicate first 5 samples (same layout, different positions in array)
        tensors = np.concatenate([base, base[:5]], axis=0)
        labels = np.concatenate([
            np.random.randint(0, 3, 50),
            np.random.randint(0, 3, 5),
        ])

        train_idx, val_idx, test_idx, _, _, _ = create_grouped_splits(
            tensors, labels, 0.7, 0.15, 0.15, 42
        )

        hashes = compute_layout_hashes(tensors)
        train_hashes = set(hashes[train_idx])
        val_hashes = set(hashes[val_idx])
        test_hashes = set(hashes[test_idx])

        assert len(train_hashes & val_hashes) == 0, "Layout overlap between train and val"
        assert len(train_hashes & test_hashes) == 0, "Layout overlap between train and test"
        assert len(val_hashes & test_hashes) == 0, "Layout overlap between val and test"

    def test_grouped_split_covers_all_indices(self):
        """All samples must be assigned to exactly one split."""
        from src.data_splitter import create_grouped_splits

        tensors = np.random.randn(100, 3, 18, 11).astype(np.float32)
        labels = np.random.randint(0, 5, 100)

        train_idx, val_idx, test_idx, _, _, _ = create_grouped_splits(
            tensors, labels, 0.7, 0.15, 0.15, 42
        )

        all_idx = set(train_idx) | set(val_idx) | set(test_idx)
        assert len(all_idx) == 100

    def test_create_datasets_with_group_by_layout(self):
        """create_datasets should use grouped mode when config enables it."""
        from src.data_splitter import create_datasets, compute_layout_hashes

        base = np.random.randn(40, 3, 18, 11).astype(np.float32)
        tensors = np.concatenate([base, base[:5]], axis=0)
        labels = np.concatenate([np.random.randint(0, 3, 40), np.random.randint(0, 3, 5)])

        config = {'data': {'group_by_layout': True}}
        train_ds, val_ds, test_ds = create_datasets(
            tensors, labels, config, 0.7, 0.15, 0.15, 42
        )

        # Verify no layout overlap
        hashes = compute_layout_hashes(tensors)
        all_data = np.concatenate([train_ds.data, val_ds.data, test_ds.data])
        all_hashes = compute_layout_hashes(all_data)
        train_h = set(compute_layout_hashes(train_ds.data))
        val_h = set(compute_layout_hashes(val_ds.data))
        test_h = set(compute_layout_hashes(test_ds.data))

        assert len(train_h & val_h) == 0
        assert len(train_h & test_h) == 0
        assert len(val_h & test_h) == 0


class TestStratifiedSplitEdgeCases:
    """Test split pre-validation for small/imbalanced datasets."""

    def test_test_split_too_small_for_classes(self):
        """When test split yields fewer samples than classes, raise domain error."""
        # 10 classes with 3 samples each -> test split (0.15) = ~5 samples < 10 classes
        tensors = np.random.randn(30, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(10), 3)

        with pytest.raises(ValueError, match="Test split too small"):
            create_stratified_splits(tensors, labels, 0.7, 0.15, 0.15, 42)

    def test_val_split_too_small_for_classes(self):
        """When val split yields fewer samples than classes, raise domain error."""
        # 5 classes, 15 samples -> test=3 ok, but train_val=12, val=12*0.176~3 < 5
        tensors = np.random.randn(15, 3, 18, 11).astype(np.float32)
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

        with pytest.raises(ValueError, match="split too small for stratification"):
            create_stratified_splits(tensors, labels, 0.7, 0.15, 0.15, 42)

    def test_sufficient_data_passes_validation(self):
        """Large enough dataset with few classes should pass validation."""
        tensors = np.random.randn(100, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(5), 20)

        train_idx, val_idx, test_idx, _, _, _ = create_stratified_splits(
            tensors, labels, 0.7, 0.15, 0.15, 42
        )
        all_idx = set(train_idx) | set(val_idx) | set(test_idx)
        assert len(all_idx) == 100

    def test_minority_class_two_stage_failure_returns_domain_error(self):
        """Class count=2 should fail early with actionable, non-sklearn message."""
        tensors = np.random.randn(8, 3, 18, 11).astype(np.float32)
        labels = np.array([0, 0, 1, 1, 1, 1, 1, 1])

        with pytest.raises(ValueError) as exc:
            create_stratified_splits(tensors, labels, 0.7, 0.15, 0.15, 42)

        msg = str(exc.value).lower()
        assert "at least 3 samples per class" in msg
        assert "least populated class" not in msg

    def test_sklearn_error_is_wrapped_with_domain_message(self, monkeypatch):
        """Unexpected sklearn ValueError should be wrapped with project guidance."""
        import src.data_splitter as data_splitter

        class FailingStratifiedShuffleSplit:
            def __init__(self, *args, **kwargs):
                pass

            def split(self, tensors, labels):
                raise ValueError(
                    "The least populated class in y has only 1 member, which is too few."
                )

        monkeypatch.setattr(
            data_splitter, "StratifiedShuffleSplit", FailingStratifiedShuffleSplit
        )

        tensors = np.random.randn(100, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(5), 20)

        with pytest.raises(ValueError) as exc:
            data_splitter.create_stratified_splits(tensors, labels, 0.7, 0.15, 0.15, 42)

        msg = str(exc.value).lower()
        assert "stratified split failed during test split" in msg
        assert "least populated class" not in msg


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

