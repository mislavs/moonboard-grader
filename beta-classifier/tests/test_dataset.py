"""Tests for dataset module."""

import numpy as np
import pytest
import torch

from src.dataset import (
    FeatureNormalizer,
    MoveSequenceDataset,
    collate_fn,
    extract_features,
    create_data_splits
)


class TestFeatureNormalizer:
    """Tests for FeatureNormalizer class."""
    
    def test_fit_computes_mean_and_std(self):
        """Normalizer should compute mean and std from training data."""
        # Create sample sequences
        seq1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        seq2 = np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
        sequences = [seq1, seq2]
        
        normalizer = FeatureNormalizer()
        normalizer.fit(sequences)
        
        # Concatenated: [[1,2], [3,4], [5,6], [7,8], [9,10]]
        # Mean: [5, 6], Std: [2.83, 2.83] approx
        assert normalizer.mean is not None
        assert normalizer.std is not None
        assert normalizer.mean.shape == (2,)
        assert normalizer.std.shape == (2,)
        assert normalizer._fitted is True
    
    def test_transform_normalizes_correctly(self):
        """Transform should apply z-score normalization."""
        seq = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
        
        normalizer = FeatureNormalizer()
        normalizer.mean = np.array([1.0, 1.0])
        normalizer.std = np.array([1.0, 1.0])
        normalizer._fitted = True
        
        result = normalizer.transform([seq])
        
        expected = np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result[0], expected)
    
    def test_transform_without_fit_raises(self):
        """Transform should raise error if not fitted."""
        normalizer = FeatureNormalizer()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            normalizer.transform([np.array([[1.0, 2.0]])])
    
    def test_save_and_load(self, tmp_path):
        """Normalizer should save and load correctly."""
        normalizer = FeatureNormalizer()
        normalizer.mean = np.array([1.0, 2.0, 3.0])
        normalizer.std = np.array([0.5, 1.0, 1.5])
        normalizer._fitted = True
        
        path = tmp_path / "normalizer.npz"
        normalizer.save(str(path))
        
        loaded = FeatureNormalizer.load(str(path))
        
        np.testing.assert_array_equal(loaded.mean, normalizer.mean)
        np.testing.assert_array_equal(loaded.std, normalizer.std)
        assert loaded._fitted is True
    
    def test_zero_std_handling(self):
        """Normalizer should handle zero std (constant features)."""
        # All same values for second feature
        seq = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]], dtype=np.float32)
        
        normalizer = FeatureNormalizer()
        normalizer.fit([seq])
        
        # Std for constant feature should be set to 1.0
        assert normalizer.std[1] == 1.0


class TestMoveSequenceDataset:
    """Tests for MoveSequenceDataset class."""
    
    def test_getitem_returns_correct_types(self):
        """Dataset should return tensor and label."""
        sequences = [np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]
        labels = np.array([5])
        
        dataset = MoveSequenceDataset(sequences, labels)
        seq, label = dataset[0]
        
        assert isinstance(seq, torch.Tensor)
        assert seq.dtype == torch.float32
        assert label == 5
    
    def test_len_returns_correct_count(self):
        """Dataset should return correct length."""
        sequences = [np.zeros((3, 2)), np.zeros((4, 2)), np.zeros((5, 2))]
        labels = np.array([0, 1, 2])
        
        dataset = MoveSequenceDataset(sequences, labels)
        
        assert len(dataset) == 3


class TestCollateFn:
    """Tests for collate_fn function."""
    
    def test_pads_to_max_length(self):
        """Collate should pad sequences to max length in batch."""
        seq1 = torch.randn(2, 15)  # length 2, 15 features
        seq2 = torch.randn(3, 15)  # length 3, 15 features
        
        batch = [(seq1, 0), (seq2, 1)]
        padded, mask, labels = collate_fn(batch)
        
        assert padded.shape == (2, 3, 15)  # batch=2, max_len=3, features=15
        assert mask.shape == (2, 3)
        assert labels.shape == (2,)
    
    def test_mask_is_correct(self):
        """Mask should be 1 for real tokens, 0 for padding."""
        seq1 = torch.randn(1, 15)  # length 1
        seq2 = torch.randn(3, 15)  # length 3
        
        batch = [(seq1, 0), (seq2, 1)]
        _, mask, _ = collate_fn(batch)
        
        # First sequence: 1 real, 2 padding
        assert mask[0].tolist() == [1.0, 0.0, 0.0]
        # Second sequence: 3 real
        assert mask[1].tolist() == [1.0, 1.0, 1.0]
    
    def test_labels_are_long_tensor(self):
        """Labels should be LongTensor."""
        seq = torch.randn(1, 15)
        batch = [(seq, 5)]
        
        _, _, labels = collate_fn(batch)
        
        assert labels.dtype == torch.long


class TestExtractFeatures:
    """Tests for extract_features function."""
    
    def test_extracts_15_features(self):
        """Should extract exactly 15 features per move."""
        moves = [{
            'targetX': 1, 'targetY': 2,
            'stationaryX': 3, 'stationaryY': 4,
            'originX': 5, 'originY': 6,
            'targetDifficulty': 7.0, 'stationaryDifficulty': 8.0, 'originDifficulty': 9.0,
            'bodyStretchDx': 10, 'bodyStretchDy': 11,
            'travelDx': 12, 'travelDy': 13,
            'hand': 0, 'successScore': 0.5
        }]
        
        features = extract_features(moves)
        
        assert features.shape == (1, 15)
        assert features.dtype == np.float32
    
    def test_correct_feature_order(self):
        """Features should be in expected order."""
        moves = [{
            'targetX': 1, 'targetY': 2,
            'stationaryX': 3, 'stationaryY': 4,
            'originX': 5, 'originY': 6,
            'targetDifficulty': 7.0, 'stationaryDifficulty': 8.0, 'originDifficulty': 9.0,
            'bodyStretchDx': -1, 'bodyStretchDy': -2,
            'travelDx': 10, 'travelDy': 11,
            'hand': 1, 'successScore': 0.75
        }]
        
        features = extract_features(moves)
        
        expected = [1, 2, 3, 4, 5, 6, 7.0, 8.0, 9.0, -1, -2, 10, 11, 1, 0.75]
        np.testing.assert_array_almost_equal(features[0], expected)


class TestCreateDataSplits:
    """Tests for create_data_splits function."""
    
    def test_split_ratios(self):
        """Splits should respect specified ratios approximately."""
        n_samples = 100
        sequences = [np.zeros((3, 15)) for _ in range(n_samples)]
        # Create balanced labels for stratification
        labels = np.array([i % 5 for i in range(n_samples)])
        
        splits = create_data_splits(
            sequences, labels,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            random_seed=42
        )
        
        assert len(splits['train'][0]) == 70
        assert len(splits['val'][0]) == 15
        assert len(splits['test'][0]) == 15
    
    def test_no_overlap_between_splits(self):
        """Samples should not appear in multiple splits."""
        # Need enough samples per class for stratified split (min 2 per class)
        sequences = [np.array([[i]]) for i in range(40)]
        labels = np.array([i % 4 for i in range(40)])  # 10 samples per class
        
        splits = create_data_splits(sequences, labels, random_seed=42)
        
        # Get unique first elements from each split
        train_vals = {s[0, 0] for s in splits['train'][0]}
        val_vals = {s[0, 0] for s in splits['val'][0]}
        test_vals = {s[0, 0] for s in splits['test'][0]}
        
        assert len(train_vals & val_vals) == 0
        assert len(train_vals & test_vals) == 0
        assert len(val_vals & test_vals) == 0
    
    def test_reproducible_with_seed(self):
        """Same seed should produce same splits."""
        sequences = [np.zeros((3, 15)) for _ in range(50)]
        labels = np.array([i % 5 for i in range(50)])
        
        splits1 = create_data_splits(sequences, labels, random_seed=42)
        splits2 = create_data_splits(sequences, labels, random_seed=42)
        
        np.testing.assert_array_equal(splits1['train'][1], splits2['train'][1])

