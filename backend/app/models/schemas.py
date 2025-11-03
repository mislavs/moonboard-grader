"""
Pydantic models for request/response validation.

Defines all data schemas used by the API endpoints.
"""

from pydantic import BaseModel, Field
from typing import List


class Move(BaseModel):
    """Represents a single climbing hold/move."""
    
    description: str = Field(..., description="Hold position (e.g., 'A1', 'B5')")
    isStart: bool = Field(default=False, description="Whether this is a starting hold")
    isEnd: bool = Field(default=False, description="Whether this is a finishing hold")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "description": "A1",
                "isStart": True,
                "isEnd": False
            }
        }
    }


class ProblemRequest(BaseModel):
    """Request body for grade prediction."""
    
    moves: List[Move] = Field(..., description="List of holds/moves in the problem")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top predictions to return")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "moves": [
                    {"description": "A1", "isStart": True, "isEnd": False},
                    {"description": "B3", "isStart": False, "isEnd": False},
                    {"description": "C5", "isStart": False, "isEnd": True}
                ],
                "top_k": 3
            }
        }
    }


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
    
    model_config = {
        "json_schema_extra": {
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
    }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    
    model_config = {
        "protected_namespaces": ()
    }


class ModelInfoResponse(BaseModel):
    """Model information response."""
    
    model_path: str = Field(..., description="Path to the loaded model")
    device: str = Field(..., description="Device used for inference (cpu/cuda)")
    model_exists: bool = Field(..., description="Whether the model file exists")
    
    model_config = {
        "protected_namespaces": ()
    }

