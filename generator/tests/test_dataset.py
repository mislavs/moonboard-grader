"""
Tests for the MoonBoard Dataset.
"""

import pytest
import torch
from pathlib import Path

from src.dataset import MoonBoardDataset, create_data_loaders


class TestMoonBoardDataset:
    """Test suite for MoonBoardDataset."""
    
    @pytest.fixture
    def data_path(self):
        """Get path to test data."""
        return "../data/problems.json"
    
    @pytest.fixture
    def dataset(self, data_path):
        """Create a dataset for testing."""
        return MoonBoardDataset(data_path)
    
    def test_dataset_initialization(self, dataset):
        """Test that dataset initializes correctly."""
        assert len(dataset) > 0
        assert dataset.grade_to_label is not None
        assert dataset.label_to_grade is not None
        assert len(dataset.grade_to_label) > 0
    
    def test_dataset_getitem(self, dataset):
        """Test getting a single item from the dataset."""
        grid, grade_label = dataset[0]
        
        # Check types
        assert isinstance(grid, torch.Tensor)
        assert isinstance(grade_label, int) or isinstance(grade_label, torch.Tensor)
        
        # Check shapes
        assert grid.shape == (3, 18, 11)
        
        # Check grade label is valid
        if isinstance(grade_label, torch.Tensor):
            grade_label = grade_label.item()
        assert 0 <= grade_label < len(dataset.grade_to_label)
    
    def test_dataset_length(self, dataset):
        """Test that dataset length is correct."""
        length = len(dataset)
        assert length > 0
        
        # Should be able to iterate through all items
        count = 0
        for _ in dataset:
            count += 1
            if count >= 10:  # Just test first 10 to save time
                break
        assert count == 10
    
    def test_grid_tensor_values(self, dataset):
        """Test that grid tensors contain valid values."""
        grid, _ = dataset[0]
        
        # Grid should be binary (0 or 1)
        unique_values = torch.unique(grid)
        assert all(v in [0.0, 1.0] for v in unique_values.tolist())
    
    def test_grade_mappings(self, dataset):
        """Test grade to label and label to grade mappings."""
        # Test forward and backward mapping
        for grade, label in dataset.grade_to_label.items():
            assert dataset.label_to_grade[label] == grade
            
            # Test get methods
            assert dataset.get_label_from_grade(grade) == label
            assert dataset.get_grade_from_label(label) == grade
    
    def test_get_num_grades(self, dataset):
        """Test getting number of unique grades."""
        num_grades = dataset.get_num_grades()
        assert num_grades == len(dataset.grade_to_label)
        assert num_grades > 0
    
    def test_custom_grade_filter(self, data_path):
        """Test dataset with grade filtering (6A to 6B+)."""
        # Filter to grades 1-4 (6A, 6A+, 6B, 6B+)
        dataset = MoonBoardDataset(data_path, min_grade_index=1, max_grade_index=4)
        
        assert dataset.get_num_grades() == 4
        # Check grade names are correct
        assert '6A' in dataset.grade_names
        assert '6B+' in dataset.grade_names
    
    def test_multiple_items(self, dataset):
        """Test getting multiple items from the dataset."""
        items = [dataset[i] for i in range(min(5, len(dataset)))]
        
        for grid, grade_label in items:
            assert grid.shape == (3, 18, 11)
            if isinstance(grade_label, torch.Tensor):
                grade_label = grade_label.item()
            assert 0 <= grade_label < len(dataset.grade_to_label)


class TestDataLoaders:
    """Test suite for data loader creation."""
    
    @pytest.fixture
    def data_path(self):
        """Get path to test data."""
        return "../data/problems.json"
    
    def test_create_data_loaders(self, data_path):
        """Test creating train and validation data loaders."""
        train_loader, val_loader, dataset = create_data_loaders(
            data_path=data_path,
            batch_size=32,
            train_split=0.8,
            shuffle=True,
            num_workers=0
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert dataset is not None
        
        # Check that dataset is split correctly
        total_size = len(dataset)
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        
        assert train_size + val_size == total_size
        assert abs(train_size / total_size - 0.8) < 0.01  # Within 1% of 80%
    
    def test_data_loader_batching(self, data_path):
        """Test that data loaders produce correct batch sizes."""
        batch_size = 16
        train_loader, val_loader, _ = create_data_loaders(
            data_path=data_path,
            batch_size=batch_size,
            train_split=0.8,
            shuffle=True,
            num_workers=0
        )
        
        # Get first batch
        grids, grades = next(iter(train_loader))
        
        # Check batch dimensions
        assert grids.shape[0] <= batch_size  # Can be less for last batch
        assert grids.shape[1:] == (3, 18, 11)
        assert grades.shape[0] == grids.shape[0]
    
    def test_data_loader_iteration(self, data_path):
        """Test iterating through data loaders."""
        train_loader, _, _ = create_data_loaders(
            data_path=data_path,
            batch_size=32,
            train_split=0.8,
            shuffle=False,
            num_workers=0
        )
        
        # Iterate through a few batches
        batch_count = 0
        for grids, grades in train_loader:
            assert grids.shape[1:] == (3, 18, 11)
            assert grades.shape[0] == grids.shape[0]
            
            batch_count += 1
            if batch_count >= 3:  # Just test first 3 batches
                break
        
        assert batch_count == 3
    
    def test_different_batch_sizes(self, data_path):
        """Test data loaders with different batch sizes."""
        for batch_size in [8, 16, 32, 64]:
            train_loader, val_loader, _ = create_data_loaders(
                data_path=data_path,
                batch_size=batch_size,
                train_split=0.8,
                num_workers=0
            )
            
            grids, grades = next(iter(train_loader))
            assert grids.shape[0] <= batch_size
    
    def test_different_train_splits(self, data_path):
        """Test data loaders with different train/val splits."""
        for split in [0.7, 0.8, 0.9]:
            train_loader, val_loader, dataset = create_data_loaders(
                data_path=data_path,
                batch_size=32,
                train_split=split,
                num_workers=0
            )
            
            total_size = len(dataset)
            train_size = len(train_loader.dataset)
            val_size = len(val_loader.dataset)
            
            assert train_size + val_size == total_size
            assert abs(train_size / total_size - split) < 0.01

