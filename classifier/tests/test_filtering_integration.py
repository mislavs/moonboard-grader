"""
Integration tests for filtered grade classifier.

Tests that Trainer and Predictor correctly handle grade filtering metadata.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src import (
    Trainer,
    Predictor,
    MoonboardDataset,
    create_model,
    encode_grade,
    decode_grade,
    remap_label,
    unmap_label,
)


class TestTrainerFilteringMetadata:
    """Tests for Trainer saving filtering metadata in checkpoints."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for checkpoints."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_data(self):
        """Create mock training data."""
        # 50 samples, 10 classes (0-9, representing remapped 6A+ through 7C)
        tensors = np.random.rand(50, 3, 18, 11).astype(np.float32)
        labels = np.random.randint(0, 10, 50).astype(np.int64)
        return tensors, labels
    
    def test_trainer_saves_default_metadata(self, temp_dir, mock_data):
        """Test that trainer saves default (non-filtered) metadata."""
        tensors, labels = mock_data
        dataset = MoonboardDataset(tensors, labels)
        train_loader = DataLoader(dataset, batch_size=8)
        
        model = create_model('fc', num_classes=10)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint_dir=temp_dir
        )
        
        trainer.save_checkpoint('test.pth')
        
        checkpoint = torch.load(Path(temp_dir) / 'test.pth')
        assert 'grade_offset' in checkpoint
        assert checkpoint['grade_offset'] == 0
        assert checkpoint['min_grade_index'] == 0
        assert checkpoint['max_grade_index'] == 18
    
    def test_trainer_saves_filtered_metadata(self, temp_dir, mock_data):
        """Test that trainer saves filtered model metadata."""
        tensors, labels = mock_data
        dataset = MoonboardDataset(tensors, labels)
        train_loader = DataLoader(dataset, batch_size=8)
        
        model = create_model('fc', num_classes=10)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Create trainer with filtering metadata
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint_dir=temp_dir,
            grade_offset=2,
            min_grade_index=2,
            max_grade_index=11
        )
        
        trainer.save_checkpoint('filtered.pth')
        
        checkpoint = torch.load(Path(temp_dir) / 'filtered.pth')
        assert checkpoint['grade_offset'] == 2
        assert checkpoint['min_grade_index'] == 2
        assert checkpoint['max_grade_index'] == 11


class TestPredictorFilteringMetadata:
    """Tests for Predictor loading and using filtering metadata."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for checkpoints."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def create_checkpoint(self, checkpoint_path, num_classes, grade_offset=0, 
                         min_grade_index=0, max_grade_index=18):
        """Helper to create a checkpoint with metadata."""
        model = create_model('fc', num_classes=num_classes)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'grade_offset': grade_offset,
            'min_grade_index': min_grade_index,
            'max_grade_index': max_grade_index
        }
        torch.save(checkpoint, checkpoint_path)
    
    def test_predictor_loads_default_metadata(self, temp_dir):
        """Test that predictor loads default metadata correctly."""
        checkpoint_path = Path(temp_dir) / 'default.pth'
        self.create_checkpoint(checkpoint_path, 19)
        
        predictor = Predictor(str(checkpoint_path))
        
        assert predictor.grade_offset == 0
        assert predictor.min_grade_index == 0
        assert predictor.max_grade_index == 18
    
    def test_predictor_loads_filtered_metadata(self, temp_dir):
        """Test that predictor loads filtered metadata correctly."""
        checkpoint_path = Path(temp_dir) / 'filtered.pth'
        self.create_checkpoint(
            checkpoint_path, 
            num_classes=10,
            grade_offset=2,
            min_grade_index=2,
            max_grade_index=11
        )
        
        predictor = Predictor(str(checkpoint_path))
        
        assert predictor.grade_offset == 2
        assert predictor.min_grade_index == 2
        assert predictor.max_grade_index == 11
    
    def test_predictor_model_info_shows_filtering(self, temp_dir):
        """Test that model info indicates filtered model."""
        checkpoint_path = Path(temp_dir) / 'filtered.pth'
        self.create_checkpoint(
            checkpoint_path,
            num_classes=10,
            grade_offset=2,
            min_grade_index=2,
            max_grade_index=11
        )
        
        predictor = Predictor(str(checkpoint_path))
        info = predictor.get_model_info()
        
        assert info['filtered'] is True
        assert info['grade_offset'] == 2
        assert info['min_grade'] == '6A+'
        assert info['max_grade'] == '7C'
    
    def test_predictor_model_info_non_filtered(self, temp_dir):
        """Test that model info shows non-filtered correctly."""
        checkpoint_path = Path(temp_dir) / 'default.pth'
        self.create_checkpoint(checkpoint_path, 19)
        
        predictor = Predictor(str(checkpoint_path))
        info = predictor.get_model_info()
        
        assert info['filtered'] is False
        assert 'min_grade' not in info
        assert 'max_grade' not in info
    
    def test_predictor_unmaps_predictions(self, temp_dir):
        """Test that predictor unmaps predictions correctly."""
        checkpoint_path = Path(temp_dir) / 'filtered.pth'
        self.create_checkpoint(
            checkpoint_path,
            num_classes=10,
            grade_offset=2,
            min_grade_index=2,
            max_grade_index=11
        )
        
        predictor = Predictor(str(checkpoint_path))
        
        # Create a simple problem
        problem = {
            'moves': [
                {'description': 'A1', 'isStart': True, 'isEnd': False},
                {'description': 'K18', 'isStart': False, 'isEnd': True}
            ]
        }
        
        result = predictor.predict(problem)
        
        # Check that predicted_label is in original space (2-11)
        predicted_label = result['predicted_label']
        assert 2 <= predicted_label <= 11
        
        # Check that predicted grade is valid
        predicted_grade = result['predicted_grade']
        assert predicted_grade in ['6A+', '6B', '6B+', '6C', '6C+', 
                                   '7A', '7A+', '7B', '7B+', '7C']
    
    def test_predictor_probabilities_use_original_grades(self, temp_dir):
        """Test that all_probabilities dict uses original grade names."""
        checkpoint_path = Path(temp_dir) / 'filtered.pth'
        self.create_checkpoint(
            checkpoint_path,
            num_classes=10,
            grade_offset=2,
            min_grade_index=2,
            max_grade_index=11
        )
        
        predictor = Predictor(str(checkpoint_path))
        
        problem = {
            'moves': [
                {'description': 'A1', 'isStart': True, 'isEnd': False},
                {'description': 'K18', 'isStart': False, 'isEnd': True}
            ]
        }
        
        result = predictor.predict(problem)
        probs = result['all_probabilities']
        
        # Should have exactly 10 grades
        assert len(probs) == 10
        
        # Should only have grades in the filtered range
        expected_grades = ['6A+', '6B', '6B+', '6C', '6C+', 
                          '7A', '7A+', '7B', '7B+', '7C']
        assert set(probs.keys()) == set(expected_grades)
        
        # Probabilities should sum to ~1.0
        assert abs(sum(probs.values()) - 1.0) < 0.01


class TestEndToEndFilteredWorkflow:
    """End-to-end tests for complete filtered workflow."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_train_and_predict_filtered_model(self, temp_dir):
        """Test training with filtered data and making predictions."""
        # Create filtered dataset (grades 2-11 in original space, 0-9 in model space)
        np.random.seed(42)
        tensors = np.random.rand(100, 3, 18, 11).astype(np.float32)
        labels = np.random.randint(0, 10, 100).astype(np.int64)  # Already remapped to 0-9
        
        dataset = MoonboardDataset(tensors, labels)
        train_loader = DataLoader(dataset, batch_size=16)
        
        # Create and train model
        model = create_model('fc', num_classes=10)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint_dir=temp_dir,
            grade_offset=2,
            min_grade_index=2,
            max_grade_index=11
        )
        
        # Train for 1 epoch
        trainer.fit(num_epochs=1, verbose=False)
        
        # Load with predictor
        checkpoint_path = Path(temp_dir) / 'best_model.pth'
        predictor = Predictor(str(checkpoint_path))
        
        # Verify metadata
        assert predictor.grade_offset == 2
        
        # Make prediction
        problem = {
            'moves': [
                {'description': 'A1', 'isStart': True, 'isEnd': False},
                {'description': 'F10', 'isStart': False, 'isEnd': False},
                {'description': 'K18', 'isStart': False, 'isEnd': True}
            ]
        }
        
        result = predictor.predict(problem)
        
        # Verify prediction is in correct range
        assert result['predicted_label'] in range(2, 12)
        assert result['predicted_grade'] in ['6A+', '6B', '6B+', '6C', '6C+',
                                             '7A', '7A+', '7B', '7B+', '7C']
    
    def test_label_consistency_through_workflow(self, temp_dir):
        """Test that labels are consistent through complete workflow."""
        # Start with original grade indices
        original_indices = [2, 3, 5, 7, 11]  # 6A+, 6B, 6C, 7A, 7C
        offset = 2
        
        # Remap for training
        model_indices = [remap_label(idx, offset) for idx in original_indices]
        assert model_indices == [0, 1, 3, 5, 9]
        
        # Create dataset with remapped labels
        tensors = np.random.rand(5, 3, 18, 11).astype(np.float32)
        labels = np.array(model_indices, dtype=np.int64)
        
        dataset = MoonboardDataset(tensors, labels)
        train_loader = DataLoader(dataset, batch_size=5)
        
        # Train model
        model = create_model('fc', num_classes=10)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint_dir=temp_dir,
            grade_offset=offset,
            min_grade_index=2,
            max_grade_index=11
        )
        
        trainer.fit(num_epochs=1, verbose=False)
        
        # Load and verify predictions are unmapped
        checkpoint_path = Path(temp_dir) / 'best_model.pth'
        predictor = Predictor(str(checkpoint_path))
        
        # Test on one of the training samples
        problem = {
            'moves': [
                {'description': 'A1', 'isStart': True, 'isEnd': False},
                {'description': 'K18', 'isStart': False, 'isEnd': True}
            ]
        }
        
        result = predictor.predict(problem)
        
        # Prediction should be in original space (2-11)
        predicted_original = result['predicted_label']
        assert 2 <= predicted_original <= 11
        
        # We can unmap it and it should still be valid
        # (This is a consistency check - predictor already unmaps)
        predicted_model = remap_label(predicted_original, offset)
        assert 0 <= predicted_model <= 9


