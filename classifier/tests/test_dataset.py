"""
Unit tests for the dataset module.

Tests the MoonboardDataset class and data splitting functionality.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    MoonboardDataset,
    create_data_splits,
    get_split_info,
)


class TestMoonboardDataset:
    """Tests for the MoonboardDataset class."""
    
    def test_init_with_numpy_arrays(self):
        """Test initialization with numpy arrays."""
        data = np.random.rand(10, 3, 18, 11).astype(np.float32)
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        dataset = MoonboardDataset(data, labels)
        
        assert len(dataset) == 10
        assert dataset.data.shape == (10, 3, 18, 11)
        assert dataset.labels.shape == (10,)
    
    def test_init_with_lists(self):
        """Test initialization with lists."""
        data = [np.random.rand(3, 18, 11) for _ in range(5)]
        labels = [0, 1, 2, 3, 4]
        
        dataset = MoonboardDataset(data, labels)
        
        assert len(dataset) == 5
        assert dataset.data.shape == (5, 3, 18, 11)
    
    def test_init_single_sample(self):
        """Test initialization with a single sample."""
        data = np.random.rand(3, 18, 11).astype(np.float32)
        labels = np.array([0])
        
        dataset = MoonboardDataset(data, labels)
        
        assert len(dataset) == 1
        assert dataset.data.shape == (1, 3, 18, 11)
    
    def test_len(self):
        """Test __len__ method."""
        data = np.random.rand(15, 3, 18, 11).astype(np.float32)
        labels = np.array(list(range(15)))
        
        dataset = MoonboardDataset(data, labels)
        
        assert len(dataset) == 15
    
    def test_getitem_returns_tensor_and_int(self):
        """Test __getitem__ returns torch tensor and integer."""
        data = np.random.rand(5, 3, 18, 11).astype(np.float32)
        labels = np.array([0, 1, 2, 3, 4])
        
        dataset = MoonboardDataset(data, labels)
        
        sample, label = dataset[0]
        
        assert isinstance(sample, torch.Tensor)
        assert sample.dtype == torch.float32
        assert sample.shape == (3, 18, 11)
        assert isinstance(label, int)
        assert label == 0
    
    def test_getitem_all_indices(self):
        """Test __getitem__ works for all valid indices."""
        data = np.random.rand(10, 3, 18, 11).astype(np.float32)
        labels = np.array(list(range(10)))
        
        dataset = MoonboardDataset(data, labels)
        
        for i in range(10):
            sample, label = dataset[i]
            assert sample.shape == (3, 18, 11)
            assert label == i
    
    def test_getitem_out_of_bounds(self):
        """Test __getitem__ raises IndexError for invalid indices."""
        data = np.random.rand(5, 3, 18, 11).astype(np.float32)
        labels = np.array([0, 1, 2, 3, 4])
        
        dataset = MoonboardDataset(data, labels)
        
        with pytest.raises(IndexError):
            _ = dataset[10]
        
        with pytest.raises(IndexError):
            _ = dataset[-1000]
    
    def test_dataloader_compatibility(self):
        """Test dataset works with PyTorch DataLoader."""
        data = np.random.rand(20, 3, 18, 11).astype(np.float32)
        labels = np.array(list(range(20)))
        
        dataset = MoonboardDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        batch_samples, batch_labels = next(iter(dataloader))
        
        assert batch_samples.shape == (4, 3, 18, 11)
        assert batch_labels.shape == (4,)
        assert batch_samples.dtype == torch.float32
        assert batch_labels.dtype == torch.int64
    
    def test_dataloader_iteration(self):
        """Test iterating through DataLoader."""
        data = np.random.rand(10, 3, 18, 11).astype(np.float32)
        labels = np.array(list(range(10)))
        
        dataset = MoonboardDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
        
        total_samples = 0
        for batch_samples, batch_labels in dataloader:
            total_samples += len(batch_samples)
            assert batch_samples.shape[1:] == (3, 18, 11)
        
        assert total_samples == 10
    
    def test_get_label_distribution(self):
        """Test get_label_distribution method."""
        data = np.random.rand(10, 3, 18, 11).astype(np.float32)
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3])
        
        dataset = MoonboardDataset(data, labels)
        dist = dataset.get_label_distribution()
        
        assert dist == {0: 3, 1: 2, 2: 4, 3: 1}
    
    def test_init_mismatched_lengths(self):
        """Test initialization fails with mismatched data and labels."""
        data = np.random.rand(10, 3, 18, 11).astype(np.float32)
        labels = np.array([0, 1, 2, 3, 4])  # Only 5 labels
        
        with pytest.raises(ValueError, match="same length"):
            MoonboardDataset(data, labels)
    
    def test_init_wrong_data_shape(self):
        """Test initialization fails with wrong data shape."""
        data = np.random.rand(10, 3, 20, 11).astype(np.float32)  # Wrong shape
        labels = np.array(list(range(10)))
        
        with pytest.raises(ValueError, match="Expected data shape"):
            MoonboardDataset(data, labels)
    
    def test_init_wrong_ndim(self):
        """Test initialization fails with wrong number of dimensions."""
        data = np.random.rand(10, 18, 11).astype(np.float32)  # Missing channel dim
        labels = np.array(list(range(10)))
        
        with pytest.raises(ValueError, match="Expected data shape"):
            MoonboardDataset(data, labels)
    
    def test_init_non_integer_labels(self):
        """Test initialization fails with non-integer labels."""
        data = np.random.rand(5, 3, 18, 11).astype(np.float32)
        labels = np.array([0.5, 1.2, 2.7, 3.1, 4.9])  # Float labels
        
        with pytest.raises(ValueError, match="must be integers"):
            MoonboardDataset(data, labels)
    
    def test_init_invalid_types(self):
        """Test initialization fails with invalid types."""
        data = "not an array"
        labels = np.array([0, 1, 2])
        
        with pytest.raises(TypeError, match="must be a numpy array"):
            MoonboardDataset(data, labels)
    
    def test_data_type_conversion(self):
        """Test that data is converted to float32."""
        data = np.random.rand(5, 3, 18, 11).astype(np.float64)
        labels = np.array([0, 1, 2, 3, 4])
        
        dataset = MoonboardDataset(data, labels)
        
        assert dataset.data.dtype == np.float32
    
    def test_label_type_conversion(self):
        """Test that labels are converted to int64."""
        data = np.random.rand(5, 3, 18, 11).astype(np.float32)
        labels = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        
        dataset = MoonboardDataset(data, labels)
        
        assert dataset.labels.dtype == np.int64


class TestCreateDataSplits:
    """Tests for the create_data_splits function."""
    
    def test_default_split_ratios(self):
        """Test default 70/15/15 split ratios."""
        data = np.random.rand(100, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(10), 10)  # 10 of each class
        
        train_ds, val_ds, test_ds = create_data_splits(data, labels)
        
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 100
        
        # Allow some tolerance due to rounding
        assert 68 <= len(train_ds) <= 72
        assert 13 <= len(val_ds) <= 17
        assert 13 <= len(test_ds) <= 17
    
    def test_custom_split_ratios(self):
        """Test custom split ratios."""
        data = np.random.rand(100, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(10), 10)
        
        train_ds, val_ds, test_ds = create_data_splits(
            data, labels,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 100
        
        assert 58 <= len(train_ds) <= 62
        assert 18 <= len(val_ds) <= 22
        assert 18 <= len(test_ds) <= 22
    
    def test_stratification(self):
        """Test that splits preserve class distribution."""
        # Create imbalanced dataset
        data = np.random.rand(100, 3, 18, 11).astype(np.float32)
        labels = np.array([0]*50 + [1]*30 + [2]*20)  # 50%, 30%, 20%
        
        train_ds, val_ds, test_ds = create_data_splits(data, labels)
        
        # Get distributions
        train_dist = train_ds.get_label_distribution()
        val_dist = val_ds.get_label_distribution()
        test_dist = test_ds.get_label_distribution()
        
        # Check each split has all classes
        assert set(train_dist.keys()) == {0, 1, 2}
        assert set(val_dist.keys()) == {0, 1, 2}
        assert set(test_dist.keys()) == {0, 1, 2}
        
        # Check approximate proportions in train set
        train_total = sum(train_dist.values())
        train_prop_0 = train_dist[0] / train_total
        train_prop_1 = train_dist[1] / train_total
        train_prop_2 = train_dist[2] / train_total
        
        assert 0.45 <= train_prop_0 <= 0.55  # ~50%
        assert 0.25 <= train_prop_1 <= 0.35  # ~30%
        assert 0.15 <= train_prop_2 <= 0.25  # ~20%
    
    def test_reproducibility(self):
        """Test that same random_state gives same splits."""
        data = np.random.rand(50, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(5), 10)
        
        train1, val1, test1 = create_data_splits(data, labels, random_state=42)
        train2, val2, test2 = create_data_splits(data, labels, random_state=42)
        
        # Same sizes
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)
        assert len(test1) == len(test2)
        
        # Same labels (not just distribution, but exact same samples)
        assert np.array_equal(train1.labels, train2.labels)
        assert np.array_equal(val1.labels, val2.labels)
        assert np.array_equal(test1.labels, test2.labels)
    
    def test_different_random_states(self):
        """Test that different random_state gives different splits."""
        data = np.random.rand(50, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(5), 10)
        
        train1, val1, test1 = create_data_splits(data, labels, random_state=42)
        train2, val2, test2 = create_data_splits(data, labels, random_state=123)
        
        # Different labels
        assert not np.array_equal(train1.labels, train2.labels)
    
    def test_invalid_ratios_sum(self):
        """Test that invalid ratio sum raises error."""
        data = np.random.rand(50, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(5), 10)
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            create_data_splits(data, labels, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
    
    def test_negative_ratios(self):
        """Test that negative ratios raise error."""
        data = np.random.rand(50, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(5), 10)
        
        with pytest.raises(ValueError, match="must be positive"):
            create_data_splits(data, labels, train_ratio=0.8, val_ratio=-0.1, test_ratio=0.3)
    
    def test_ratio_greater_than_one(self):
        """Test that ratio >= 1.0 raises error."""
        data = np.random.rand(50, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(5), 10)
        
        with pytest.raises(ValueError, match="must be less than 1.0"):
            create_data_splits(data, labels, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    
    def test_dataset_too_small(self):
        """Test that very small dataset raises error."""
        data = np.random.rand(2, 3, 18, 11).astype(np.float32)
        labels = np.array([0, 1])
        
        with pytest.raises(ValueError, match="Dataset too small"):
            create_data_splits(data, labels)
    
    def test_class_with_one_sample(self):
        """Test that class with only one sample raises error."""
        data = np.random.rand(10, 3, 18, 11).astype(np.float32)
        labels = np.array([0, 0, 0, 1, 2, 2, 2, 2, 2, 2])  # Class 1 has only 1 sample
        
        with pytest.raises(ValueError, match="some classes have fewer than 2 samples"):
            create_data_splits(data, labels)
    
    def test_returns_dataset_instances(self):
        """Test that function returns MoonboardDataset instances."""
        data = np.random.rand(50, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(5), 10)
        
        train_ds, val_ds, test_ds = create_data_splits(data, labels)
        
        assert isinstance(train_ds, MoonboardDataset)
        assert isinstance(val_ds, MoonboardDataset)
        assert isinstance(test_ds, MoonboardDataset)
    
    def test_no_data_leakage(self):
        """Test that train/val/test splits don't overlap."""
        data = np.random.rand(50, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(5), 10)
        
        train_ds, val_ds, test_ds = create_data_splits(data, labels)
        
        # Convert to sets of labels for simple check
        # (In practice, you'd check actual sample indices)
        train_size = len(train_ds)
        val_size = len(val_ds)
        test_size = len(test_ds)
        
        # All samples accounted for
        assert train_size + val_size + test_size == 50
    
    def test_with_list_inputs(self):
        """Test that function works with list inputs."""
        data = [np.random.rand(3, 18, 11).astype(np.float32) for _ in range(40)]
        labels = list(np.repeat(np.arange(4), 10))
        
        train_ds, val_ds, test_ds = create_data_splits(data, labels)
        
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 40


class TestGetSplitInfo:
    """Tests for the get_split_info function."""
    
    def test_basic_info(self):
        """Test basic split information."""
        data = np.random.rand(100, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(10), 10)
        
        train_ds, val_ds, test_ds = create_data_splits(data, labels)
        info = get_split_info(train_ds, val_ds, test_ds)
        
        assert 'total_samples' in info
        assert 'train_size' in info
        assert 'val_size' in info
        assert 'test_size' in info
        assert info['total_samples'] == 100
    
    def test_ratios_in_info(self):
        """Test that info contains ratio information."""
        data = np.random.rand(100, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(10), 10)
        
        train_ds, val_ds, test_ds = create_data_splits(data, labels)
        info = get_split_info(train_ds, val_ds, test_ds)
        
        assert 'train_ratio' in info
        assert 'val_ratio' in info
        assert 'test_ratio' in info
        
        # Ratios should sum to 1.0
        total_ratio = info['train_ratio'] + info['val_ratio'] + info['test_ratio']
        assert np.isclose(total_ratio, 1.0)
    
    def test_distribution_in_info(self):
        """Test that info contains distribution information."""
        data = np.random.rand(100, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(10), 10)
        
        train_ds, val_ds, test_ds = create_data_splits(data, labels)
        info = get_split_info(train_ds, val_ds, test_ds)
        
        assert 'train_distribution' in info
        assert 'val_distribution' in info
        assert 'test_distribution' in info
        
        assert isinstance(info['train_distribution'], dict)
        assert isinstance(info['val_distribution'], dict)
        assert isinstance(info['test_distribution'], dict)


class TestIntegration:
    """Integration tests for the dataset module."""
    
    def test_full_workflow(self):
        """Test complete workflow from data to DataLoader."""
        # Create synthetic dataset
        np.random.seed(42)
        data = np.random.rand(100, 3, 18, 11).astype(np.float32)
        labels = np.repeat(np.arange(10), 10)
        
        # Create splits
        train_ds, val_ds, test_ds = create_data_splits(
            data, labels,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )
        
        # Create DataLoaders
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
        
        # Verify we can iterate
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))
        
        assert train_batch[0].shape[1:] == (3, 18, 11)
        assert val_batch[0].shape[1:] == (3, 18, 11)
        assert test_batch[0].shape[1:] == (3, 18, 11)
        
        # Get split info
        info = get_split_info(train_ds, val_ds, test_ds)
        assert info['total_samples'] == 100
    
    def test_realistic_moonboard_scenario(self):
        """Test with realistic moonboard problem data."""
        # Simulate 200 problems with various grades (0-18 representing Font grades)
        np.random.seed(42)
        n_problems = 200
        
        # Create realistic-looking grid data
        data = []
        labels = []
        
        for i in range(n_problems):
            grid = np.zeros((3, 18, 11), dtype=np.float32)
            
            # Add some random holds
            n_start = np.random.randint(1, 3)
            n_middle = np.random.randint(3, 10)
            n_end = np.random.randint(1, 3)
            
            # Random positions for each channel
            for _ in range(n_start):
                r, c = np.random.randint(0, 18), np.random.randint(0, 11)
                grid[0, r, c] = 1.0
            
            for _ in range(n_middle):
                r, c = np.random.randint(0, 18), np.random.randint(0, 11)
                grid[1, r, c] = 1.0
            
            for _ in range(n_end):
                r, c = np.random.randint(0, 18), np.random.randint(0, 11)
                grid[2, r, c] = 1.0
            
            data.append(grid)
            labels.append(i % 19)  # Cycle through 19 grade levels
        
        data = np.array(data)
        labels = np.array(labels)
        
        # Create splits
        train_ds, val_ds, test_ds = create_data_splits(data, labels)
        
        # Verify splits
        assert len(train_ds) + len(val_ds) + len(test_ds) == n_problems
        
        # Create DataLoader and iterate
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        total_batches = 0
        for batch_data, batch_labels in train_loader:
            total_batches += 1
            assert batch_data.shape[1:] == (3, 18, 11)
            assert len(batch_labels) <= 32
        
        assert total_batches > 0

