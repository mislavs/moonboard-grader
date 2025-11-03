"""
FastAPI Backend for Moonboard Grade Prediction

Provides REST API endpoints for predicting climbing grades using the
moonboard-classifier package.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging

# Import from the classifier package
from src.predictor import Predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Moonboard Grade Predictor API",
    description="REST API for predicting climbing problem grades using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[Predictor] = None

# Model path (hardcoded as per spec)
MODEL_PATH = Path("models/model_for_inference.pth")


# Pydantic models for request/response validation
class Move(BaseModel):
    """Represents a single climbing hold/move"""
    description: str = Field(..., description="Hold position (e.g., 'A1', 'B5')")
    isStart: bool = Field(False, description="Whether this is a starting hold")
    isEnd: bool = Field(False, description="Whether this is a finishing hold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "description": "A1",
                "isStart": True,
                "isEnd": False
            }
        }


class ProblemRequest(BaseModel):
    """Request body for grade prediction"""
    moves: List[Move] = Field(..., description="List of holds/moves in the problem")
    top_k: int = Field(3, ge=1, le=10, description="Number of top predictions to return")
    
    class Config:
        json_schema_extra = {
            "example": {
                "moves": [
                    {"description": "A1", "isStart": True, "isEnd": False},
                    {"description": "B3", "isStart": False, "isEnd": False},
                    {"description": "C5", "isStart": False, "isEnd": True}
                ],
                "top_k": 3
            }
        }


class TopKPrediction(BaseModel):
    """A single top-k prediction"""
    grade: str = Field(..., description="Predicted grade")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")


class PredictionResponse(BaseModel):
    """Response body for grade prediction"""
    predicted_grade: str = Field(..., description="Most likely grade")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in prediction")
    top_k_predictions: List[TopKPrediction] = Field(..., description="Top K most likely grades")
    all_probabilities: Optional[Dict[str, float]] = Field(None, description="All grade probabilities")
    
    class Config:
        json_schema_extra = {
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


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_path: str = Field(..., description="Path to the loaded model")
    device: str = Field(..., description="Device used for inference (cpu/cuda)")
    model_exists: bool = Field(..., description="Whether the model file exists")


# Startup event - load the model
@app.on_event("startup")
async def startup_event():
    """Load the model on application startup"""
    global predictor
    
    logger.info(f"Loading model from {MODEL_PATH}...")
    
    try:
        if not MODEL_PATH.exists():
            logger.warning(
                f"Model file not found at {MODEL_PATH}. "
                "API will be available but predictions will fail. "
                "Please add model_for_inference.pth to the models/ directory."
            )
            predictor = None
        else:
            # Load predictor with CPU device (can be made configurable later)
            predictor = Predictor(
                checkpoint_path=str(MODEL_PATH),
                device='cpu'
            )
            logger.info("Model loaded successfully!")
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None
        # Don't fail startup - let the API start but predictions will return errors


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "Moonboard Grade Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get information about the loaded model"""
    return ModelInfoResponse(
        model_path=str(MODEL_PATH),
        device=predictor.device if predictor else "N/A",
        model_exists=MODEL_PATH.exists()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_grade(request: ProblemRequest):
    """
    Predict the grade of a climbing problem.
    
    Args:
        request: Problem data with moves and optional top_k parameter
        
    Returns:
        Prediction response with predicted grade, confidence, and top-k predictions
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not loaded. Please ensure model_for_inference.pth "
                "exists in the models/ directory and restart the server."
            )
        )
    
    # Convert Pydantic models to dict format expected by Predictor
    problem_dict = {
        "moves": [move.model_dump() for move in request.moves]
    }
    
    try:
        # Make prediction
        result = predictor.predict(
            problem=problem_dict,
            return_top_k=request.top_k
        )
        
        # Convert top_k_predictions format from tuple to dict
        top_k_formatted = [
            TopKPrediction(grade=grade, probability=prob)
            for grade, prob in result['top_k_predictions']
        ]
        
        # Build response
        response = PredictionResponse(
            predicted_grade=result['predicted_grade'],
            confidence=result['confidence'],
            top_k_predictions=top_k_formatted,
            all_probabilities=result.get('all_probabilities')
        )
        
        return response
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid problem data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

