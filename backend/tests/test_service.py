"""
Tests for the PredictorService.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.services.predictor_service import PredictorService


class TestPredictorService:
    """Test suite for PredictorService."""
    
    def test_init(self, tmp_path: Path):
        """Test service initialization."""
        model_path = tmp_path / "model.pth"
        service = PredictorService(model_path=model_path, device="cpu")
        
        assert service.model_path == model_path
        assert service.device == "cpu"
        assert not service.is_loaded
        assert service.predictor is None
    
    def test_init_with_defaults(self):
        """Test service initialization with default settings."""
        service = PredictorService()
        
        assert service.model_path is not None
        assert service.device is not None
        assert not service.is_loaded
    
    def test_load_model_file_not_found(self, tmp_path: Path):
        """Test loading model when file doesn't exist."""
        model_path = tmp_path / "nonexistent.pth"
        service = PredictorService(model_path=model_path, device="cpu")
        
        with pytest.raises(FileNotFoundError) as exc_info:
            service.load_model()
        
        assert "Model file not found" in str(exc_info.value)
        assert not service.is_loaded
    
    @patch('app.services.predictor_service.Predictor')
    def test_load_model_success(self, mock_predictor_class, tmp_path: Path):
        """Test successful model loading."""
        # Create a fake model file
        model_path = tmp_path / "model.pth"
        model_path.touch()
        
        # Setup mock
        mock_predictor_instance = MagicMock()
        mock_predictor_class.return_value = mock_predictor_instance
        
        service = PredictorService(model_path=model_path, device="cpu")
        service.load_model()
        
        assert service.is_loaded
        assert service.predictor is not None
        mock_predictor_class.assert_called_once_with(
            checkpoint_path=str(model_path),
            device="cpu"
        )
    
    def test_predict_without_loading(self, tmp_path: Path):
        """Test prediction without loading model."""
        model_path = tmp_path / "model.pth"
        service = PredictorService(model_path=model_path, device="cpu")
        
        problem = {"moves": [{"description": "A1", "isStart": True}]}
        
        with pytest.raises(RuntimeError) as exc_info:
            service.predict(problem=problem, top_k=3)
        
        assert "Model not loaded" in str(exc_info.value)
    
    def test_predict_success(self, predictor_service, mock_predictor):
        """Test successful prediction."""
        problem = {"moves": [{"description": "A1", "isStart": True}]}
        
        result = predictor_service.predict(problem=problem, top_k=3)
        
        assert result['predicted_grade'] == '6B+'
        assert result['confidence'] == 0.87
        assert len(result['top_k_predictions']) == 3
        mock_predictor.predict.assert_called_once_with(
            problem=problem,
            return_top_k=3
        )
    
    def test_predict_invalid_data(self, predictor_service, mock_predictor):
        """Test prediction with invalid data."""
        mock_predictor.predict.side_effect = ValueError("Invalid problem data")
        
        problem = {"invalid": "data"}
        
        with pytest.raises(ValueError) as exc_info:
            predictor_service.predict(problem=problem, top_k=3)
        
        assert "Invalid problem data" in str(exc_info.value)
    
    def test_get_model_info(self, tmp_path: Path):
        """Test getting model information."""
        model_path = tmp_path / "model.pth"
        model_path.touch()
        
        service = PredictorService(model_path=model_path, device="cuda")
        info = service.get_model_info()
        
        assert info['model_path'] == str(model_path)
        assert info['device'] == "cuda"
        assert info['model_exists'] is True
        assert info['is_loaded'] is False
    
    def test_is_loaded_property(self, predictor_service):
        """Test is_loaded property."""
        assert predictor_service.is_loaded is True
    
    def test_predictor_property(self, predictor_service, mock_predictor):
        """Test predictor property."""
        assert predictor_service.predictor == mock_predictor

