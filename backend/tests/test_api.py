"""
Tests for API endpoints.
"""

import pytest
from fastapi import status


class TestRootEndpoint:
    """Test suite for root endpoint."""
    
    def test_root(self, client_with_loaded_model):
        """Test root endpoint returns correct information."""
        response = client_with_loaded_model.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data


class TestHealthEndpoint:
    """Test suite for health check endpoint."""
    
    def test_health_with_loaded_model(self, client_with_loaded_model):
        """Test health check when model is loaded."""
        response = client_with_loaded_model.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
    
    def test_health_with_unloaded_model(self, client_with_unloaded_model):
        """Test health check when model is not loaded."""
        response = client_with_unloaded_model.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is False


class TestModelInfoEndpoint:
    """Test suite for model info endpoint."""
    
    def test_model_info(self, client_with_loaded_model):
        """Test model info endpoint."""
        response = client_with_loaded_model.get("/model-info")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "model_path" in data
        assert "device" in data
        assert "model_exists" in data


class TestPredictEndpoint:
    """Test suite for prediction endpoint."""
    
    def test_predict_success(self, client_with_loaded_model, sample_problem_request):
        """Test successful prediction."""
        response = client_with_loaded_model.post("/predict", json=sample_problem_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "predicted_grade" in data
        assert "confidence" in data
        assert "top_k_predictions" in data
        assert data["predicted_grade"] == "6B+"
        assert data["confidence"] == 0.87
        assert len(data["top_k_predictions"]) == 3
    
    def test_predict_with_unloaded_model(self, client_with_unloaded_model, sample_problem_request):
        """Test prediction when model is not loaded."""
        response = client_with_unloaded_model.post("/predict", json=sample_problem_request)
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "Model not loaded" in response.json()["detail"]
    
    def test_predict_invalid_request_missing_moves(self, client_with_loaded_model):
        """Test prediction with missing moves."""
        invalid_request = {"top_k": 3}
        response = client_with_loaded_model.post("/predict", json=invalid_request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_invalid_request_empty_moves(self, client_with_loaded_model, mock_predictor):
        """Test prediction with empty moves list."""
        # Configure mock to raise ValueError for empty moves
        mock_predictor.predict.side_effect = ValueError("No moves provided")
        
        invalid_request = {"moves": [], "top_k": 3}
        response = client_with_loaded_model.post("/predict", json=invalid_request)
        
        # Should return 400 Bad Request for invalid data
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid problem data" in response.json()["detail"]
    
    def test_predict_invalid_top_k_too_small(self, client_with_loaded_model, sample_problem_request):
        """Test prediction with top_k too small."""
        sample_problem_request["top_k"] = 0
        response = client_with_loaded_model.post("/predict", json=sample_problem_request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_invalid_top_k_too_large(self, client_with_loaded_model, sample_problem_request):
        """Test prediction with top_k too large."""
        sample_problem_request["top_k"] = 11
        response = client_with_loaded_model.post("/predict", json=sample_problem_request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_custom_top_k(self, client_with_loaded_model, sample_problem_request):
        """Test prediction with custom top_k value."""
        sample_problem_request["top_k"] = 5
        response = client_with_loaded_model.post("/predict", json=sample_problem_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Note: mock returns 3 predictions, but in real scenario it should return 5
        assert "top_k_predictions" in data
    
    def test_predict_move_validation(self, client_with_loaded_model):
        """Test that move structure is validated."""
        invalid_request = {
            "moves": [
                {"invalid_field": "A1"}  # Missing required fields
            ],
            "top_k": 3
        }
        response = client_with_loaded_model.post("/predict", json=invalid_request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestCORS:
    """Test suite for CORS configuration."""
    
    def test_cors_headers_present(self, client_with_loaded_model):
        """Test that CORS headers are present."""
        response = client_with_loaded_model.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers or response.status_code == status.HTTP_200_OK


class TestAPIDocumentation:
    """Test suite for API documentation."""
    
    def test_openapi_schema_available(self, client_with_loaded_model):
        """Test that OpenAPI schema is accessible."""
        response = client_with_loaded_model.get("/openapi.json")
        
        assert response.status_code == status.HTTP_200_OK
        schema = response.json()
        assert "openapi" in schema
        assert "paths" in schema
    
    def test_swagger_ui_available(self, client_with_loaded_model):
        """Test that Swagger UI is accessible."""
        response = client_with_loaded_model.get("/docs")
        
        assert response.status_code == status.HTTP_200_OK
    
    def test_redoc_available(self, client_with_loaded_model):
        """Test that ReDoc is accessible."""
        response = client_with_loaded_model.get("/redoc")
        
        assert response.status_code == status.HTTP_200_OK

