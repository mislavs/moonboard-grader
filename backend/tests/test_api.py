"""
Tests for API endpoints.
"""

import pytest
from fastapi import status

from .conftest import SAMPLE_PROBLEM_ID_1, SAMPLE_PROBLEM_ID_2


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
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    
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
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    
    def test_predict_invalid_top_k_too_large(self, client_with_loaded_model, sample_problem_request):
        """Test prediction with top_k too large."""
        sample_problem_request["top_k"] = 11
        response = client_with_loaded_model.post("/predict", json=sample_problem_request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    
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
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


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


class TestProblemsEndpoint:
    """Test suite for problems list endpoint."""
    
    def test_get_problems_list_default_pagination(self, client_with_problem_service):
        """Test GET /problems returns paginated response with defaults."""
        response = client_with_problem_service.get("/problems")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Should return paginated response structure
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data
        
        # Check pagination metadata
        assert data["total"] == 2
        assert data["page"] == 1
        assert data["page_size"] == 20
        assert data["total_pages"] == 1
        
        # Check items
        items = data["items"]
        assert isinstance(items, list)
        assert len(items) == 2
        
        # Check first problem
        assert items[0]["id"] == SAMPLE_PROBLEM_ID_1
        assert items[0]["name"] == "Fat Guy In A Little Suit"
        assert items[0]["grade"] == "6B+"
        assert "moves" not in items[0]  # List endpoint should not include moves
        
        # Check second problem
        assert items[1]["id"] == SAMPLE_PROBLEM_ID_2
        assert items[1]["name"] == "Test Problem"
        assert items[1]["grade"] == "7A"
    
    def test_get_problems_list_custom_page_size(self, client_with_problem_service):
        """Test GET /problems with custom page size."""
        response = client_with_problem_service.get("/problems?page=1&page_size=1")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["total"] == 2
        assert data["page"] == 1
        assert data["page_size"] == 1
        assert data["total_pages"] == 2
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == SAMPLE_PROBLEM_ID_1
    
    def test_get_problems_list_second_page(self, client_with_problem_service):
        """Test GET /problems for second page."""
        response = client_with_problem_service.get("/problems?page=2&page_size=1")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["total"] == 2
        assert data["page"] == 2
        assert data["page_size"] == 1
        assert data["total_pages"] == 2
        assert len(data["items"]) == 1
        assert data["items"][0]["id"] == SAMPLE_PROBLEM_ID_2
    
    def test_get_problems_list_page_beyond_range(self, client_with_problem_service):
        """Test GET /problems for page beyond available data."""
        response = client_with_problem_service.get("/problems?page=10&page_size=20")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["total"] == 2
        assert data["page"] == 10
        assert data["page_size"] == 20
        assert data["total_pages"] == 1
        assert len(data["items"]) == 0  # Empty page
    
    def test_get_problems_list_invalid_page(self, client_with_problem_service):
        """Test GET /problems with invalid page number."""
        response = client_with_problem_service.get("/problems?page=0")
        
        # Should return 422 for validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    
    def test_get_problems_list_invalid_page_size(self, client_with_problem_service):
        """Test GET /problems with invalid page size."""
        response = client_with_problem_service.get("/problems?page_size=0")
        
        # Should return 422 for validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    
    def test_get_problems_list_page_size_too_large(self, client_with_problem_service):
        """Test GET /problems with page size exceeding maximum."""
        response = client_with_problem_service.get("/problems?page_size=101")
        
        # Should return 422 for validation error
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    
    def test_get_problems_list_schema(self, client_with_problem_service):
        """Test that problems list matches expected schema."""
        response = client_with_problem_service.get("/problems")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify paginated response structure
        assert "items" in data
        assert "total" in data and isinstance(data["total"], int)
        assert "page" in data and isinstance(data["page"], int)
        assert "page_size" in data and isinstance(data["page_size"], int)
        assert "total_pages" in data and isinstance(data["total_pages"], int)
        
        # Verify each item has required fields with correct types
        for problem in data["items"]:
            assert "id" in problem and isinstance(problem["id"], int)
            assert "name" in problem and isinstance(problem["name"], str)
            assert "grade" in problem and isinstance(problem["grade"], str)


class TestProblemDetailEndpoint:
    """Test suite for problem detail endpoint."""
    
    def test_get_problem_by_id(self, client_with_problem_service):
        """Test GET /problems/{id} returns full problem details."""
        response = client_with_problem_service.get(f"/problems/{SAMPLE_PROBLEM_ID_1}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check basic fields
        assert data["id"] == SAMPLE_PROBLEM_ID_1
        assert data["name"] == "Fat Guy In A Little Suit"
        assert data["grade"] == "6B+"
        
        # Check moves array is included
        assert "moves" in data
        assert isinstance(data["moves"], list)
        assert len(data["moves"]) == 2
        
        # Check first move structure (problemId should be removed)
        move = data["moves"][0]
        assert "problemId" not in move
        assert move["description"] == "J4"
        assert move["isStart"] is True
        assert move["isEnd"] is False
    
    def test_get_problem_by_id_not_found(self, client_with_problem_service):
        """Test GET /problems/{id} with invalid ID returns 404."""
        response = client_with_problem_service.get("/problems/999999")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_problem_detail_schema(self, client_with_problem_service):
        """Test that problem detail matches expected schema."""
        response = client_with_problem_service.get(f"/problems/{SAMPLE_PROBLEM_ID_2}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify required fields with correct types
        assert "id" in data and isinstance(data["id"], int)
        assert "name" in data and isinstance(data["name"], str)
        assert "grade" in data and isinstance(data["grade"], str)
        assert "moves" in data and isinstance(data["moves"], list)
        
        # Verify move structure (problemId should be removed)
        for move in data["moves"]:
            assert "problemId" not in move
            assert all(key in move for key in ["description", "isStart", "isEnd"])
            assert isinstance(move["description"], str)
            assert isinstance(move["isStart"], bool)
            assert isinstance(move["isEnd"], bool)


class TestCheckDuplicateEndpoint:
    """Test suite for duplicate check endpoint."""
    
    def test_check_duplicate_exists(self, client_with_problem_service):
        """Test checking for duplicate when problem exists."""
        # Use moves from the first test problem (ID 305445)
        response = client_with_problem_service.post("/problems/check-duplicate", json={
            "moves": [
                {"description": "J4", "isStart": True, "isEnd": False},
                {"description": "F18", "isStart": False, "isEnd": True}
            ]
        })
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["exists"] is True
        assert data["problem_id"] == SAMPLE_PROBLEM_ID_1
    
    def test_check_duplicate_not_exists(self, client_with_problem_service):
        """Test checking for duplicate when problem doesn't exist."""
        response = client_with_problem_service.post("/problems/check-duplicate", json={
            "moves": [
                {"description": "Z99", "isStart": True, "isEnd": False},
                {"description": "Y88", "isStart": False, "isEnd": True}
            ]
        })
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["exists"] is False
        assert data["problem_id"] is None
    
    def test_check_duplicate_order_independent(self, client_with_problem_service):
        """Test that move order doesn't affect duplicate check."""
        # Same moves as problem 305445 but in reversed order
        response = client_with_problem_service.post("/problems/check-duplicate", json={
            "moves": [
                {"description": "F18", "isStart": False, "isEnd": True},
                {"description": "J4", "isStart": True, "isEnd": False}
            ]
        })
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["exists"] is True
        assert data["problem_id"] == SAMPLE_PROBLEM_ID_1
    
    def test_check_duplicate_empty_moves(self, client_with_problem_service):
        """Test checking duplicate with empty moves list."""
        response = client_with_problem_service.post("/problems/check-duplicate", json={
            "moves": []
        })
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Empty moves should not match anything (or match empty problems if any)
        assert "exists" in data
        assert "problem_id" in data
    
    def test_check_duplicate_invalid_request_missing_moves(self, client_with_problem_service):
        """Test checking duplicate without moves field."""
        response = client_with_problem_service.post("/problems/check-duplicate", json={})
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    
    def test_check_duplicate_invalid_move_structure(self, client_with_problem_service):
        """Test checking duplicate with invalid move structure."""
        response = client_with_problem_service.post("/problems/check-duplicate", json={
            "moves": [
                {"invalid_field": "A1"}
            ]
        })
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    
    def test_check_duplicate_response_schema(self, client_with_problem_service):
        """Test that duplicate check response matches expected schema."""
        response = client_with_problem_service.post("/problems/check-duplicate", json={
            "moves": [
                {"description": "J4", "isStart": True, "isEnd": False},
                {"description": "F18", "isStart": False, "isEnd": True}
            ]
        })
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Verify response structure
        assert "exists" in data and isinstance(data["exists"], bool)
        assert "problem_id" in data
        if data["problem_id"] is not None:
            assert isinstance(data["problem_id"], int)