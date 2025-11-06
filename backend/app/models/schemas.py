"""
Pydantic models for request/response validation.

Defines all data schemas used by the API endpoints.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class Move(BaseModel):
    """Represents a single climbing hold/move for prediction requests."""
    
    description: str = Field(..., description="Hold position (e.g., 'A1', 'B5')")
    isStart: bool = Field(default=False, description="Whether this is a starting hold")
    isEnd: bool = Field(default=False, description="Whether this is a finishing hold")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "A1",
                "isStart": True,
                "isEnd": False
            }
        }
    )


class ProblemRequest(BaseModel):
    """Request body for grade prediction."""
    
    moves: List[Move] = Field(..., description="List of holds/moves in the problem")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top predictions to return")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "moves": [
                    {"description": "A1", "isStart": True, "isEnd": False},
                    {"description": "B3", "isStart": False, "isEnd": False},
                    {"description": "C5", "isStart": False, "isEnd": True}
                ],
                "top_k": 3
            }
        }
    )


class TopKPrediction(BaseModel):
    """A single top-k prediction."""
    
    grade: str = Field(..., description="Predicted grade")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")


class PredictionResponse(BaseModel):
    """Response body for grade prediction."""
    
    predicted_grade: str = Field(..., description="Most likely grade")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in prediction")
    top_k_predictions: List[TopKPrediction] = Field(
        ..., 
        description="Top K most likely grades"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_grade": "6B+",
                "confidence": 0.87,
                "top_k_predictions": [
                    {"grade": "6B+", "probability": 0.87},
                    {"grade": "6C", "probability": 0.09},
                    {"grade": "6B", "probability": 0.03}
                ]
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    
    model_config = ConfigDict(protected_namespaces=())


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_path: str = Field(..., description="Path to the loaded model")
    device: str = Field(..., description="Device used for inference (cpu/cuda)")
    model_exists: bool = Field(..., description="Whether the model file exists")
    
    model_config = ConfigDict(protected_namespaces=())


# ============================================================================
# Problem Data Schemas
# ============================================================================


class ProblemMove(BaseModel):
    """Move data from the problems JSON file (read-only from database)."""
    
    description: str = Field(..., description="Hold position (e.g., 'A1', 'B5')")
    isStart: bool = Field(..., description="Whether this is a starting hold")
    isEnd: bool = Field(..., description="Whether this is a finishing hold")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "J4",
                "isStart": True,
                "isEnd": False
            }
        }
    )


class ProblemListItem(BaseModel):
    """Basic problem information for list endpoint (lightweight response)."""
    
    id: int = Field(..., description="Unique identifier for the problem")
    name: str = Field(..., description="Problem name")
    grade: str = Field(..., description="Problem grade")
    isBenchmark: bool = Field(..., description="Whether this is a benchmark problem")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 305445,
                "name": "Fat Guy In A Little Suit",
                "grade": "6B+",
                "isBenchmark": False
            }
        }
    )


class ProblemDetail(BaseModel):
    """Complete problem data for detail endpoint."""
    
    id: int = Field(..., description="Unique identifier for the problem")
    name: str = Field(..., description="Problem name")
    grade: str = Field(..., description="Problem grade")
    setby: str = Field(..., description="Name of who set the problem")
    repeats: int = Field(..., description="Number of times the problem has been repeated")
    isBenchmark: bool = Field(..., description="Whether this is a benchmark problem")
    moves: List[ProblemMove] = Field(..., description="List of moves in the problem")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 305445,
                "name": "Fat Guy In A Little Suit",
                "grade": "6B+",
                "setby": "Kyle Knapp",
                "repeats": 187,
                "isBenchmark": False,
                "moves": [
                    {
                        "description": "J4",
                        "isStart": True,
                        "isEnd": False
                    },
                    {
                        "description": "F18",
                        "isStart": False,
                        "isEnd": True
                    }
                ]
            }
        }
    )


class PaginatedProblemsResponse(BaseModel):
    """Paginated response for problems list endpoint."""
    
    items: List[ProblemListItem] = Field(..., description="List of problems for current page")
    total: int = Field(..., description="Total number of problems across all pages")
    page: int = Field(..., ge=1, description="Current page number (1-indexed)")
    page_size: int = Field(..., ge=1, description="Number of items per page")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "id": 305445,
                        "name": "Fat Guy In A Little Suit",
                        "grade": "6B+"
                    },
                    {
                        "id": 305446,
                        "name": "Another Problem",
                        "grade": "7A"
                    }
                ],
                "total": 100,
                "page": 1,
                "page_size": 20,
                "total_pages": 5
            }
        }
    )


class DuplicateCheckRequest(BaseModel):
    """Request body for checking duplicate problems."""
    
    moves: List[ProblemMove] = Field(..., description="List of moves to check for duplicates")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "moves": [
                    {"description": "J4", "isStart": True, "isEnd": False},
                    {"description": "F18", "isStart": False, "isEnd": True}
                ]
            }
        }
    )


class DuplicateCheckResponse(BaseModel):
    """Response for duplicate check endpoint."""
    
    exists: bool = Field(..., description="Whether a problem with these moves exists")
    problem_id: Optional[int] = Field(None, description="ID of the matching problem if found")
