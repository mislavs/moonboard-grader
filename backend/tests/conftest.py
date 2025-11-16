"""
Pytest configuration and shared fixtures.
"""

import json
import pytest
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, MagicMock

from fastapi.testclient import TestClient

from app.main import create_application
from app.services.predictor_service import PredictorService
from app.services.problem_service import ProblemService
from app.services.generator_service import GeneratorService
from app.api.dependencies import set_predictor_service, set_problem_service, set_generator_service


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


# Test data constants
SAMPLE_PROBLEM_ID_1 = 305445
SAMPLE_PROBLEM_ID_2 = 123456


@pytest.fixture
def sample_problems_data():
    """Sample problems data for testing."""
    return {
        "total": 2,
        "data": [
            {
                "name": "Fat Guy In A Little Suit",
                "grade": "6B+",
                "apiId": SAMPLE_PROBLEM_ID_1,
                "isBenchmark": True,
                "holdsetup": {
                    "apiId": 15
                },
                "moves": [
                    {
                        "problemId": SAMPLE_PROBLEM_ID_1,
                        "description": "J4",
                        "isStart": True,
                        "isEnd": False
                    },
                    {
                        "problemId": SAMPLE_PROBLEM_ID_1,
                        "description": "F18",
                        "isStart": False,
                        "isEnd": True
                    }
                ]
            },
            {
                "name": "Test Problem",
                "grade": "7A",
                "apiId": SAMPLE_PROBLEM_ID_2,
                "isBenchmark": False,
                "holdsetup": {
                    "apiId": 15
                },
                "moves": [
                    {
                        "problemId": SAMPLE_PROBLEM_ID_2,
                        "description": "A1",
                        "isStart": True,
                        "isEnd": False
                    },
                    {
                        "problemId": SAMPLE_PROBLEM_ID_2,
                        "description": "K18",
                        "isStart": False,
                        "isEnd": True
                    }
                ]
            }
        ]
    }


@pytest.fixture
def problem_service(tmp_path: Path, sample_problems_data):
    """Create a problem service with test data."""
    # Create a test problems.json file
    problems_file = tmp_path / "problems.json"
    problems_file.write_text(json.dumps(sample_problems_data, indent=2))
    
    service = ProblemService(problems_path=problems_file)
    return service


@pytest.fixture
def app_with_problem_service(predictor_service: PredictorService, problem_service: ProblemService) -> Generator:
    """Create test app with both predictor and problem services."""
    app = create_application()
    set_predictor_service(predictor_service)
    set_problem_service(problem_service)
    yield app
    # Cleanup
    set_predictor_service(None)
    set_problem_service(None)


@pytest.fixture
def client_with_problem_service(app_with_problem_service) -> TestClient:
    """Test client with both predictor and problem services."""
    return TestClient(app_with_problem_service)


@pytest.fixture
def mock_generator():
    """Mock Generator instance for testing."""
    mock = MagicMock()
    # Mock the generate_with_retry method (which is what GeneratorService calls)
    mock.generate_with_retry.return_value = [
        {
            'moves': [
                {'description': 'A1', 'isStart': True, 'isEnd': False},
                {'description': 'B5', 'isStart': False, 'isEnd': False},
                {'description': 'K10', 'isStart': False, 'isEnd': True}
            ],
            'grade_label': 2,  # 6A+
            'validation': {
                'valid': True,
                'errors': [],
                'warnings': []
            }
        }
    ]
    return mock


@pytest.fixture
def generator_service(mock_generator, tmp_path: Path) -> GeneratorService:
    """Create a generator service with mocked generator."""
    # Create a fake model file
    model_path = tmp_path / "test_generator_model.pth"
    model_path.touch()
    
    service = GeneratorService(model_path=model_path, device="cpu")
    service._generator = mock_generator
    service._is_loaded = True
    service._min_grade_index = 2  # Matches the filtered training (6A+)
    
    return service


@pytest.fixture
def unloaded_generator_service(tmp_path: Path) -> GeneratorService:
    """Create a generator service without loading the model."""
    model_path = tmp_path / "test_generator_model.pth"
    service = GeneratorService(model_path=model_path, device="cpu")
    return service


@pytest.fixture
def app_with_loaded_generator(generator_service: GeneratorService) -> Generator:
    """Create test app with loaded generator."""
    app = create_application()
    set_generator_service(generator_service)
    yield app
    # Cleanup
    set_generator_service(None)


@pytest.fixture
def app_with_unloaded_generator(unloaded_generator_service: GeneratorService) -> Generator:
    """Create test app with unloaded generator."""
    app = create_application()
    set_generator_service(unloaded_generator_service)
    yield app
    # Cleanup
    set_generator_service(None)


@pytest.fixture
def client_with_loaded_generator(app_with_loaded_generator) -> TestClient:
    """Test client with loaded generator."""
    return TestClient(app_with_loaded_generator)


@pytest.fixture
def client_with_unloaded_generator(app_with_unloaded_generator) -> TestClient:
    """Test client with unloaded generator."""
    return TestClient(app_with_unloaded_generator)

