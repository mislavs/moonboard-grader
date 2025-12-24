"""Tests for predictor module."""

import json
import numpy as np
import pytest
import torch

from src.dataset import FeatureNormalizer
from src.model import TransformerSequenceClassifier
from src.predictor import Predictor


@pytest.fixture
def sample_problem():
    """Create a sample problem with moves."""
    return {
        "name": "Test Problem",
        "grade": "7A",
        "moves": [
            {
                "targetX": 6, "targetY": 5,
                "stationaryX": 9, "stationaryY": 3,
                "originX": 9, "originY": 3,
                "targetDifficulty": 5.5,
                "stationaryDifficulty": 4.5,
                "originDifficulty": 4.5,
                "bodyStretchDx": -3, "bodyStretchDy": 2,
                "travelDx": -3, "travelDy": 2,
                "hand": 0, "successScore": 0.67
            },
            {
                "targetX": 5, "targetY": 6,
                "stationaryX": 6, "stationaryY": 5,
                "originX": 9, "originY": 3,
                "targetDifficulty": 5.0,
                "stationaryDifficulty": 5.5,
                "originDifficulty": 4.5,
                "bodyStretchDx": -1, "bodyStretchDy": 1,
                "travelDx": -4, "travelDy": 3,
                "hand": 1, "successScore": 0.75
            }
        ]
    }


@pytest.fixture
def trained_artifacts(tmp_path):
    """Create mock trained model and normalizer artifacts."""
    # Create and save normalizer
    normalizer = FeatureNormalizer()
    normalizer.mean = np.zeros(15, dtype=np.float32)
    normalizer.std = np.ones(15, dtype=np.float32)
    normalizer._fitted = True
    
    normalizer_path = tmp_path / "normalizer.npz"
    normalizer.save(str(normalizer_path))
    
    # Create and save model
    model = TransformerSequenceClassifier(
        input_dim=15,
        d_model=32,
        n_heads=2,
        n_layers=1,
        num_classes=19
    )
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config()
    }
    
    checkpoint_path = tmp_path / "model.pth"
    torch.save(checkpoint, checkpoint_path)
    
    return str(checkpoint_path), str(normalizer_path)


class TestPredictor:
    """Tests for Predictor class."""
    
    def test_initialization(self, trained_artifacts):
        """Predictor should load model and normalizer."""
        checkpoint_path, normalizer_path = trained_artifacts
        
        predictor = Predictor(
            checkpoint_path=checkpoint_path,
            normalizer_path=normalizer_path,
            device='cpu'
        )
        
        assert predictor.model is not None
        assert predictor.normalizer is not None
        assert predictor.device == 'cpu'
    
    def test_predict_returns_expected_keys(self, trained_artifacts, sample_problem):
        """Predict should return dict with expected keys."""
        checkpoint_path, normalizer_path = trained_artifacts
        predictor = Predictor(checkpoint_path, normalizer_path, 'cpu')
        
        result = predictor.predict(sample_problem)
        
        assert 'predicted_grade' in result
        assert 'predicted_index' in result
        assert 'confidence' in result
        assert 'all_probabilities' in result
    
    def test_predict_grade_is_valid(self, trained_artifacts, sample_problem):
        """Predicted grade should be a valid grade string."""
        checkpoint_path, normalizer_path = trained_artifacts
        predictor = Predictor(checkpoint_path, normalizer_path, 'cpu')
        
        result = predictor.predict(sample_problem)
        
        from moonboard_core import get_all_grades
        assert result['predicted_grade'] in get_all_grades()
    
    def test_predict_confidence_is_probability(self, trained_artifacts, sample_problem):
        """Confidence should be between 0 and 1."""
        checkpoint_path, normalizer_path = trained_artifacts
        predictor = Predictor(checkpoint_path, normalizer_path, 'cpu')
        
        result = predictor.predict(sample_problem)
        
        assert 0.0 <= result['confidence'] <= 1.0
    
    def test_all_probabilities_sum_to_one(self, trained_artifacts, sample_problem):
        """All probabilities should sum to approximately 1."""
        checkpoint_path, normalizer_path = trained_artifacts
        predictor = Predictor(checkpoint_path, normalizer_path, 'cpu')
        
        result = predictor.predict(sample_problem)
        
        total_prob = sum(result['all_probabilities'].values())
        assert abs(total_prob - 1.0) < 1e-5
    
    def test_predict_empty_moves_raises(self, trained_artifacts):
        """Predict should raise error for empty moves."""
        checkpoint_path, normalizer_path = trained_artifacts
        predictor = Predictor(checkpoint_path, normalizer_path, 'cpu')
        
        with pytest.raises(ValueError, match="non-empty 'moves'"):
            predictor.predict({"moves": []})
    
    def test_predict_missing_moves_raises(self, trained_artifacts):
        """Predict should raise error for missing moves key."""
        checkpoint_path, normalizer_path = trained_artifacts
        predictor = Predictor(checkpoint_path, normalizer_path, 'cpu')
        
        with pytest.raises(ValueError, match="non-empty 'moves'"):
            predictor.predict({"name": "No moves"})
    
    def test_predict_batch(self, trained_artifacts, sample_problem):
        """predict_batch should handle multiple problems."""
        checkpoint_path, normalizer_path = trained_artifacts
        predictor = Predictor(checkpoint_path, normalizer_path, 'cpu')
        
        problems = [sample_problem, sample_problem]
        results = predictor.predict_batch(problems)
        
        assert len(results) == 2
        assert all('predicted_grade' in r for r in results)
    
    def test_predict_with_alternatives(self, trained_artifacts, sample_problem):
        """predict_with_alternatives should return top-k grades."""
        checkpoint_path, normalizer_path = trained_artifacts
        predictor = Predictor(checkpoint_path, normalizer_path, 'cpu')
        
        result = predictor.predict_with_alternatives(sample_problem, top_k=3)
        
        assert 'alternatives' in result
        assert len(result['alternatives']) == 3
        
        # Alternatives should be sorted by probability (descending)
        probs = [alt['probability'] for alt in result['alternatives']]
        assert probs == sorted(probs, reverse=True)
    
    def test_compare_with_actual(self, trained_artifacts, sample_problem):
        """compare_with_actual should add comparison fields."""
        checkpoint_path, normalizer_path = trained_artifacts
        predictor = Predictor(checkpoint_path, normalizer_path, 'cpu')
        
        result = predictor.compare_with_actual(sample_problem)
        
        assert 'actual_grade' in result
        assert 'actual_index' in result
        assert 'error' in result
        assert 'correct' in result
        assert 'within_1' in result
        assert 'within_2' in result
    
    def test_compare_with_override_grade(self, trained_artifacts, sample_problem):
        """compare_with_actual should use override grade when provided."""
        checkpoint_path, normalizer_path = trained_artifacts
        predictor = Predictor(checkpoint_path, normalizer_path, 'cpu')
        
        result = predictor.compare_with_actual(sample_problem, actual_grade="6B+")
        
        assert result['actual_grade'] == "6B+"
    
    def test_deterministic_predictions(self, trained_artifacts, sample_problem):
        """Same input should produce same output."""
        checkpoint_path, normalizer_path = trained_artifacts
        predictor = Predictor(checkpoint_path, normalizer_path, 'cpu')
        
        result1 = predictor.predict(sample_problem)
        result2 = predictor.predict(sample_problem)
        
        assert result1['predicted_grade'] == result2['predicted_grade']
        assert result1['confidence'] == result2['confidence']


class TestPredictorWithFile:
    """Tests for predictor with file I/O."""
    
    def test_predict_from_json_file(self, trained_artifacts, sample_problem, tmp_path):
        """Predictor should work with problems loaded from JSON."""
        checkpoint_path, normalizer_path = trained_artifacts
        
        # Save problem to file
        problem_path = tmp_path / "problem.json"
        with open(problem_path, 'w') as f:
            json.dump(sample_problem, f)
        
        # Load and predict
        with open(problem_path, 'r') as f:
            loaded_problem = json.load(f)
        
        predictor = Predictor(checkpoint_path, normalizer_path, 'cpu')
        result = predictor.predict(loaded_problem)
        
        assert 'predicted_grade' in result

