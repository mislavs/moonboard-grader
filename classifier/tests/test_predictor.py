"""
Tests for Predictor Module

Tests the inference interface for loading trained models and making predictions.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import os

from src.predictor import Predictor
from src.models import FullyConnectedModel, ConvolutionalModel
from src.grade_encoder import encode_grade, decode_grade, get_num_grades
from src.grid_builder import create_grid_tensor


# Test data
EXAMPLE_PROBLEM = {
    'moves': [
        {'description': 'A1', 'isStart': True, 'isEnd': False},
        {'description': 'B5', 'isStart': False, 'isEnd': False},
        {'description': 'F10', 'isStart': False, 'isEnd': True}
    ]
}

EXAMPLE_PROBLEM_2 = {
    'moves': [
        {'description': 'K1', 'isStart': True, 'isEnd': False},
        {'description': 'E8', 'isStart': False, 'isEnd': False},
        {'description': 'A18', 'isStart': False, 'isEnd': True}
    ]
}


@pytest.fixture
def temp_checkpoint_fc():
    """Create a temporary checkpoint file with FC model."""
    model = FullyConnectedModel(num_classes=get_num_grades())
    
    checkpoint = {
        'epoch': 10,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'best_val_loss': 1.5,
        'history': {}
    }
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
        torch.save(checkpoint, f.name)
        checkpoint_path = f.name
    
    yield checkpoint_path
    
    # Cleanup
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


@pytest.fixture
def temp_checkpoint_cnn():
    """Create a temporary checkpoint file with CNN model."""
    model = ConvolutionalModel(num_classes=get_num_grades())
    
    checkpoint = {
        'epoch': 20,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'best_val_loss': 1.2,
        'history': {}
    }
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
        torch.save(checkpoint, f.name)
        checkpoint_path = f.name
    
    yield checkpoint_path
    
    # Cleanup
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


class TestPredictorInitialization:
    """Test Predictor initialization."""
    
    def test_init_with_fc_model(self, temp_checkpoint_fc):
        """Test initialization with FC model checkpoint."""
        predictor = Predictor(temp_checkpoint_fc)
        
        assert predictor.device == 'cpu'
        assert predictor.checkpoint_path == Path(temp_checkpoint_fc)
        assert isinstance(predictor.model, FullyConnectedModel)
        assert predictor.model.training is False  # Should be in eval mode
    
    def test_init_with_cnn_model(self, temp_checkpoint_cnn):
        """Test initialization with CNN model checkpoint."""
        predictor = Predictor(temp_checkpoint_cnn)
        
        assert predictor.device == 'cpu'
        assert isinstance(predictor.model, ConvolutionalModel)
        assert predictor.model.training is False
    
    def test_init_with_path_object(self, temp_checkpoint_fc):
        """Test initialization with Path object."""
        predictor = Predictor(Path(temp_checkpoint_fc))
        assert isinstance(predictor.model, FullyConnectedModel)
    
    def test_init_nonexistent_checkpoint(self):
        """Test initialization with nonexistent checkpoint."""
        with pytest.raises(FileNotFoundError):
            Predictor('nonexistent_checkpoint.pth')
    
    def test_init_invalid_device(self, temp_checkpoint_fc):
        """Test initialization with invalid device."""
        with pytest.raises(ValueError, match="Invalid device"):
            Predictor(temp_checkpoint_fc, device='tpu')
    
    def test_init_cuda_unavailable(self, temp_checkpoint_fc):
        """Test initialization with CUDA when unavailable."""
        if not torch.cuda.is_available():
            with pytest.raises(ValueError, match="CUDA is not available"):
                Predictor(temp_checkpoint_fc, device='cuda')
        else:
            pytest.skip("CUDA is available")
    
    def test_init_malformed_checkpoint(self):
        """Test initialization with malformed checkpoint."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            # Save invalid checkpoint
            torch.save({'invalid': 'checkpoint'}, f.name)
            checkpoint_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid checkpoint"):
                Predictor(checkpoint_path)
        finally:
            os.remove(checkpoint_path)


class TestPredictSingle:
    """Test single problem prediction."""
    
    def test_predict_basic(self, temp_checkpoint_fc):
        """Test basic prediction on a single problem."""
        predictor = Predictor(temp_checkpoint_fc)
        result = predictor.predict(EXAMPLE_PROBLEM)
        
        # Check result structure
        assert 'predicted_grade' in result
        assert 'predicted_label' in result
        assert 'confidence' in result
        assert 'all_probabilities' in result
        assert 'top_k_predictions' in result
        
        # Check types
        assert isinstance(result['predicted_grade'], str)
        assert isinstance(result['predicted_label'], int)
        assert isinstance(result['confidence'], float)
        assert isinstance(result['all_probabilities'], dict)
        assert isinstance(result['top_k_predictions'], list)
        
        # Check value ranges
        assert 0 <= result['predicted_label'] < get_num_grades()
        assert 0 <= result['confidence'] <= 1
        
        # Check all_probabilities
        assert len(result['all_probabilities']) == get_num_grades()
        assert abs(sum(result['all_probabilities'].values()) - 1.0) < 1e-5
        
        # Check top_k_predictions
        assert len(result['top_k_predictions']) == 3
        for grade, prob in result['top_k_predictions']:
            assert isinstance(grade, str)
            assert isinstance(prob, float)
            assert 0 <= prob <= 1
    
    def test_predict_different_top_k(self, temp_checkpoint_fc):
        """Test prediction with different top_k values."""
        predictor = Predictor(temp_checkpoint_fc)
        
        result_1 = predictor.predict(EXAMPLE_PROBLEM, return_top_k=1)
        assert len(result_1['top_k_predictions']) == 1
        
        result_5 = predictor.predict(EXAMPLE_PROBLEM, return_top_k=5)
        assert len(result_5['top_k_predictions']) == 5
        
        # Top predictions should be in descending order
        probs = [prob for _, prob in result_5['top_k_predictions']]
        assert probs == sorted(probs, reverse=True)
    
    def test_predict_consistency(self, temp_checkpoint_fc):
        """Test that predictions are deterministic in eval mode."""
        predictor = Predictor(temp_checkpoint_fc)
        
        result1 = predictor.predict(EXAMPLE_PROBLEM)
        result2 = predictor.predict(EXAMPLE_PROBLEM)
        
        assert result1['predicted_grade'] == result2['predicted_grade']
        assert result1['confidence'] == result2['confidence']
    
    def test_predict_different_problems(self, temp_checkpoint_fc):
        """Test prediction on different problems."""
        predictor = Predictor(temp_checkpoint_fc)
        
        result1 = predictor.predict(EXAMPLE_PROBLEM)
        result2 = predictor.predict(EXAMPLE_PROBLEM_2)
        
        # Both should have valid predictions
        assert isinstance(result1['predicted_grade'], str)
        assert isinstance(result2['predicted_grade'], str)
    
    def test_predict_invalid_input(self, temp_checkpoint_fc):
        """Test prediction with invalid input."""
        predictor = Predictor(temp_checkpoint_fc)
        
        # Not a dict
        with pytest.raises(ValueError, match="Problem must be a dictionary"):
            predictor.predict("not a dict")
        
        # Missing moves key
        with pytest.raises(ValueError, match="must contain 'moves' key"):
            predictor.predict({'grade': '6A'})
        
        # Invalid moves
        with pytest.raises(ValueError, match="Failed to process problem"):
            predictor.predict({'moves': [{'description': 'Z99'}]})
    
    def test_predict_empty_moves(self, temp_checkpoint_fc):
        """Test prediction with empty moves list."""
        predictor = Predictor(temp_checkpoint_fc)
        
        # Empty moves should still work (all zeros tensor)
        result = predictor.predict({'moves': []})
        assert isinstance(result['predicted_grade'], str)


class TestPredictBatch:
    """Test batch prediction."""
    
    def test_predict_batch_basic(self, temp_checkpoint_fc):
        """Test basic batch prediction."""
        predictor = Predictor(temp_checkpoint_fc)
        
        problems = [EXAMPLE_PROBLEM, EXAMPLE_PROBLEM_2]
        results = predictor.predict_batch(problems)
        
        assert len(results) == 2
        for result in results:
            assert 'predicted_grade' in result
            assert 'confidence' in result
    
    def test_predict_batch_single_problem(self, temp_checkpoint_fc):
        """Test batch prediction with single problem."""
        predictor = Predictor(temp_checkpoint_fc)
        
        results = predictor.predict_batch([EXAMPLE_PROBLEM])
        assert len(results) == 1
    
    def test_predict_batch_many_problems(self, temp_checkpoint_fc):
        """Test batch prediction with many problems."""
        predictor = Predictor(temp_checkpoint_fc)
        
        # Create multiple variations
        problems = [EXAMPLE_PROBLEM] * 10
        results = predictor.predict_batch(problems)
        
        assert len(results) == 10
        # All should be identical (same problem)
        for result in results[1:]:
            assert result['predicted_grade'] == results[0]['predicted_grade']
    
    def test_predict_batch_custom_top_k(self, temp_checkpoint_fc):
        """Test batch prediction with custom top_k."""
        predictor = Predictor(temp_checkpoint_fc)
        
        problems = [EXAMPLE_PROBLEM, EXAMPLE_PROBLEM_2]
        results = predictor.predict_batch(problems, return_top_k=5)
        
        for result in results:
            assert len(result['top_k_predictions']) == 5
    
    def test_predict_batch_invalid_input(self, temp_checkpoint_fc):
        """Test batch prediction with invalid input."""
        predictor = Predictor(temp_checkpoint_fc)
        
        # Not a list
        with pytest.raises(ValueError, match="Problems must be a list"):
            predictor.predict_batch("not a list")
        
        # Empty list
        with pytest.raises(ValueError, match="cannot be empty"):
            predictor.predict_batch([])
    
    def test_predict_batch_with_invalid_problem(self, temp_checkpoint_fc):
        """Test batch prediction with one invalid problem."""
        predictor = Predictor(temp_checkpoint_fc)
        
        problems = [
            EXAMPLE_PROBLEM,
            {'invalid': 'problem'}  # Missing moves
        ]
        
        with pytest.raises(ValueError, match="Failed to process problem"):
            predictor.predict_batch(problems)


class TestPredictFromTensor:
    """Test prediction from pre-processed tensors."""
    
    def test_predict_from_numpy_tensor(self, temp_checkpoint_fc):
        """Test prediction from numpy array."""
        predictor = Predictor(temp_checkpoint_fc)
        
        tensor = create_grid_tensor(EXAMPLE_PROBLEM['moves'])
        result = predictor.predict_from_tensor(tensor)
        
        assert isinstance(result, dict)
        assert 'predicted_grade' in result
    
    def test_predict_from_torch_tensor(self, temp_checkpoint_fc):
        """Test prediction from torch tensor."""
        predictor = Predictor(temp_checkpoint_fc)
        
        tensor = create_grid_tensor(EXAMPLE_PROBLEM['moves'])
        torch_tensor = torch.FloatTensor(tensor)
        result = predictor.predict_from_tensor(torch_tensor)
        
        assert isinstance(result, dict)
    
    def test_predict_from_batch_tensor(self, temp_checkpoint_fc):
        """Test prediction from batch tensor."""
        predictor = Predictor(temp_checkpoint_fc)
        
        # Create batch of tensors
        tensor1 = create_grid_tensor(EXAMPLE_PROBLEM['moves'])
        tensor2 = create_grid_tensor(EXAMPLE_PROBLEM_2['moves'])
        batch_tensor = np.stack([tensor1, tensor2])
        
        results = predictor.predict_from_tensor(batch_tensor)
        
        assert isinstance(results, list)
        assert len(results) == 2
    
    def test_predict_from_tensor_consistency(self, temp_checkpoint_fc):
        """Test consistency between predict and predict_from_tensor."""
        predictor = Predictor(temp_checkpoint_fc)
        
        # Predict using problem dict
        result1 = predictor.predict(EXAMPLE_PROBLEM)
        
        # Predict using tensor
        tensor = create_grid_tensor(EXAMPLE_PROBLEM['moves'])
        result2 = predictor.predict_from_tensor(tensor)
        
        # Should be identical
        assert result1['predicted_grade'] == result2['predicted_grade']
        assert abs(result1['confidence'] - result2['confidence']) < 1e-6
    
    def test_predict_from_tensor_invalid_shape(self, temp_checkpoint_fc):
        """Test prediction with invalid tensor shape."""
        predictor = Predictor(temp_checkpoint_fc)
        
        # Wrong shape
        with pytest.raises(ValueError, match="Invalid tensor shape"):
            predictor.predict_from_tensor(np.zeros((5, 5)))
        
        # Wrong channels
        with pytest.raises(ValueError, match="Invalid tensor shape"):
            predictor.predict_from_tensor(np.zeros((1, 18, 11)))


class TestModelInfo:
    """Test model information retrieval."""
    
    def test_get_model_info_fc(self, temp_checkpoint_fc):
        """Test getting model info for FC model."""
        predictor = Predictor(temp_checkpoint_fc)
        info = predictor.get_model_info()
        
        assert info['model_type'] == 'FullyConnected'
        assert info['num_parameters'] > 0
        assert info['num_classes'] == get_num_grades()
        assert info['device'] == 'cpu'
        assert 'checkpoint_path' in info
    
    def test_get_model_info_cnn(self, temp_checkpoint_cnn):
        """Test getting model info for CNN model."""
        predictor = Predictor(temp_checkpoint_cnn)
        info = predictor.get_model_info()
        
        assert info['model_type'] == 'Convolutional'
        assert info['num_parameters'] > 0
        assert info['num_classes'] == get_num_grades()
    
    def test_model_info_parameter_count(self, temp_checkpoint_fc):
        """Test that parameter count is reasonable."""
        predictor = Predictor(temp_checkpoint_fc)
        info = predictor.get_model_info()
        
        # FC model with 19 classes should have ~187k parameters
        assert 150000 < info['num_parameters'] < 250000


class TestCNNPredictor:
    """Test predictor with CNN model."""
    
    def test_cnn_predict_basic(self, temp_checkpoint_cnn):
        """Test basic prediction with CNN model."""
        predictor = Predictor(temp_checkpoint_cnn)
        result = predictor.predict(EXAMPLE_PROBLEM)
        
        assert isinstance(result['predicted_grade'], str)
        assert 0 <= result['confidence'] <= 1
    
    def test_cnn_predict_batch(self, temp_checkpoint_cnn):
        """Test batch prediction with CNN model."""
        predictor = Predictor(temp_checkpoint_cnn)
        
        problems = [EXAMPLE_PROBLEM, EXAMPLE_PROBLEM_2]
        results = predictor.predict_batch(problems)
        
        assert len(results) == 2


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_predict_all_zeros_tensor(self, temp_checkpoint_fc):
        """Test prediction on empty problem (all zeros)."""
        predictor = Predictor(temp_checkpoint_fc)
        
        tensor = np.zeros((3, 18, 11), dtype=np.float32)
        result = predictor.predict_from_tensor(tensor)
        
        # Should still make a prediction
        assert isinstance(result['predicted_grade'], str)
        assert 0 <= result['confidence'] <= 1
    
    def test_predict_single_hold(self, temp_checkpoint_fc):
        """Test prediction with single hold."""
        predictor = Predictor(temp_checkpoint_fc)
        
        problem = {
            'moves': [
                {'description': 'E9', 'isStart': True, 'isEnd': True}
            ]
        }
        result = predictor.predict(problem)
        
        assert isinstance(result['predicted_grade'], str)
    
    def test_predict_many_holds(self, temp_checkpoint_fc):
        """Test prediction with many holds."""
        predictor = Predictor(temp_checkpoint_fc)
        
        # Create problem with many holds
        moves = []
        for i in range(1, 18):
            moves.append({
                'description': f'A{i}',
                'isStart': i == 1,
                'isEnd': i == 17
            })
        
        problem = {'moves': moves}
        result = predictor.predict(problem)
        
        assert isinstance(result['predicted_grade'], str)
    
    def test_probability_sum(self, temp_checkpoint_fc):
        """Test that all probabilities sum to 1."""
        predictor = Predictor(temp_checkpoint_fc)
        result = predictor.predict(EXAMPLE_PROBLEM)
        
        total_prob = sum(result['all_probabilities'].values())
        assert abs(total_prob - 1.0) < 1e-5
    
    def test_top_k_probabilities_descending(self, temp_checkpoint_fc):
        """Test that top-k predictions are in descending order."""
        predictor = Predictor(temp_checkpoint_fc)
        result = predictor.predict(EXAMPLE_PROBLEM, return_top_k=get_num_grades())
        
        probs = [prob for _, prob in result['top_k_predictions']]
        assert probs == sorted(probs, reverse=True)
    
    def test_confidence_matches_top_prediction(self, temp_checkpoint_fc):
        """Test that confidence matches the top prediction probability."""
        predictor = Predictor(temp_checkpoint_fc)
        result = predictor.predict(EXAMPLE_PROBLEM)
        
        top_grade, top_prob = result['top_k_predictions'][0]
        assert result['predicted_grade'] == top_grade
        assert abs(result['confidence'] - top_prob) < 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAPredictor:
    """Test predictor with CUDA device."""
    
    def test_init_with_cuda(self, temp_checkpoint_fc):
        """Test initialization with CUDA device."""
        predictor = Predictor(temp_checkpoint_fc, device='cuda')
        assert predictor.device == 'cuda'
        assert next(predictor.model.parameters()).is_cuda
    
    def test_predict_on_cuda(self, temp_checkpoint_fc):
        """Test prediction on CUDA device."""
        predictor = Predictor(temp_checkpoint_fc, device='cuda')
        result = predictor.predict(EXAMPLE_PROBLEM)
        
        assert isinstance(result['predicted_grade'], str)
        assert 0 <= result['confidence'] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

