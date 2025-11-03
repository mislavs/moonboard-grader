"""
Tests for Pydantic models/schemas.
"""

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    Move,
    ProblemRequest,
    TopKPrediction,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)


class TestMoveModel:
    """Test suite for Move model."""
    
    def test_valid_move(self):
        """Test creating a valid move."""
        move = Move(description="A1", isStart=True, isEnd=False)
        
        assert move.description == "A1"
        assert move.isStart is True
        assert move.isEnd is False
    
    def test_move_defaults(self):
        """Test move with default values."""
        move = Move(description="B5")
        
        assert move.description == "B5"
        assert move.isStart is False
        assert move.isEnd is False
    
    def test_move_missing_description(self):
        """Test that description is required."""
        with pytest.raises(ValidationError):
            Move(isStart=True)


class TestProblemRequestModel:
    """Test suite for ProblemRequest model."""
    
    def test_valid_problem_request(self):
        """Test creating a valid problem request."""
        request = ProblemRequest(
            moves=[
                Move(description="A1", isStart=True),
                Move(description="B5"),
            ],
            top_k=3
        )
        
        assert len(request.moves) == 2
        assert request.top_k == 3
    
    def test_problem_request_default_top_k(self):
        """Test default top_k value."""
        request = ProblemRequest(
            moves=[Move(description="A1")]
        )
        
        assert request.top_k == 3
    
    def test_problem_request_top_k_validation_min(self):
        """Test top_k minimum validation."""
        with pytest.raises(ValidationError):
            ProblemRequest(
                moves=[Move(description="A1")],
                top_k=0
            )
    
    def test_problem_request_top_k_validation_max(self):
        """Test top_k maximum validation."""
        with pytest.raises(ValidationError):
            ProblemRequest(
                moves=[Move(description="A1")],
                top_k=11
            )
    
    def test_problem_request_missing_moves(self):
        """Test that moves are required."""
        with pytest.raises(ValidationError):
            ProblemRequest(top_k=3)


class TestTopKPredictionModel:
    """Test suite for TopKPrediction model."""
    
    def test_valid_prediction(self):
        """Test creating a valid top-k prediction."""
        pred = TopKPrediction(grade="6B+", probability=0.87)
        
        assert pred.grade == "6B+"
        assert pred.probability == 0.87
    
    def test_probability_validation_min(self):
        """Test probability minimum validation."""
        with pytest.raises(ValidationError):
            TopKPrediction(grade="6B+", probability=-0.1)
    
    def test_probability_validation_max(self):
        """Test probability maximum validation."""
        with pytest.raises(ValidationError):
            TopKPrediction(grade="6B+", probability=1.1)


class TestPredictionResponseModel:
    """Test suite for PredictionResponse model."""
    
    def test_valid_response(self):
        """Test creating a valid prediction response."""
        response = PredictionResponse(
            predicted_grade="6B+",
            confidence=0.87,
            top_k_predictions=[
                TopKPrediction(grade="6B+", probability=0.87),
                TopKPrediction(grade="6C", probability=0.09),
            ]
        )
        
        assert response.predicted_grade == "6B+"
        assert response.confidence == 0.87
        assert len(response.top_k_predictions) == 2
    
    def test_confidence_validation(self):
        """Test confidence validation."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                predicted_grade="6B+",
                confidence=1.5,
                top_k_predictions=[]
            )


class TestHealthResponseModel:
    """Test suite for HealthResponse model."""
    
    def test_valid_health_response(self):
        """Test creating a valid health response."""
        response = HealthResponse(status="healthy", model_loaded=True)
        
        assert response.status == "healthy"
        assert response.model_loaded is True


class TestModelInfoResponseModel:
    """Test suite for ModelInfoResponse model."""
    
    def test_valid_model_info_response(self):
        """Test creating a valid model info response."""
        response = ModelInfoResponse(
            model_path="/path/to/model.pth",
            device="cpu",
            model_exists=True
        )
        
        assert response.model_path == "/path/to/model.pth"
        assert response.device == "cpu"
        assert response.model_exists is True

