"""
Pytest configuration and shared fixtures.
"""

import pytest
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, MagicMock

from fastapi.testclient import TestClient

from app.main import create_application
from app.services.predictor_service import PredictorService
from app.api.dependencies import set_predictor_service


@pytest.fixture
def mock_predictor():
    """Mock Predictor instance for testing."""
    mock = MagicMock()
    mock.predict.return_value = {
        'predicted_grade': '6B+',
        'confidence': 0.87,
        'top_k_predictions': [
            ('6B+', 0.87),
            ('6C', 0.09),
            ('6B', 0.03)
        ]
    }
    return mock


@pytest.fixture
def predictor_service(mock_predictor, tmp_path: Path) -> PredictorService:
    """Create a predictor service with mocked predictor."""
    # Create a fake model file
    model_path = tmp_path / "test_model.pth"
    model_path.touch()
    
    service = PredictorService(model_path=model_path, device="cpu")
    service._predictor = mock_predictor
    service._is_loaded = True
    
    return service


@pytest.fixture
def unloaded_predictor_service(tmp_path: Path) -> PredictorService:
    """Create a predictor service without loading the model."""
    model_path = tmp_path / "test_model.pth"
    service = PredictorService(model_path=model_path, device="cpu")
    return service


@pytest.fixture
def app_with_loaded_model(predictor_service: PredictorService) -> Generator:
    """Create test app with loaded model."""
    app = create_application()
    set_predictor_service(predictor_service)
    yield app
    # Cleanup
    set_predictor_service(None)


@pytest.fixture
def app_with_unloaded_model(unloaded_predictor_service: PredictorService) -> Generator:
    """Create test app with unloaded model."""
    app = create_application()
    set_predictor_service(unloaded_predictor_service)
    yield app
    # Cleanup
    set_predictor_service(None)


@pytest.fixture
def client_with_loaded_model(app_with_loaded_model) -> TestClient:
    """Test client with loaded model."""
    return TestClient(app_with_loaded_model)


@pytest.fixture
def client_with_unloaded_model(app_with_unloaded_model) -> TestClient:
    """Test client with unloaded model."""
    return TestClient(app_with_unloaded_model)


@pytest.fixture
def sample_problem_request():
    """Sample problem request data."""
    return {
        "moves": [
            {"description": "A1", "isStart": True, "isEnd": False},
            {"description": "B5", "isStart": False, "isEnd": False},
            {"description": "K10", "isStart": False, "isEnd": True}
        ],
        "top_k": 3
    }

