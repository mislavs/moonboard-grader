"""
Prediction route handlers.

Provides endpoints for grade prediction using ML models.
"""

from fastapi import APIRouter, Depends, HTTPException, status
import logging

from ...models.schemas import (
    ProblemRequest,
    PredictionResponse,
    TopKPrediction,
)
from ...services.predictor_service import PredictorService
from ..dependencies import get_loaded_predictor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_grade(
    request: ProblemRequest,
    predictor_service: PredictorService = Depends(get_loaded_predictor)
):
    """
    Predict the grade of a climbing problem.
    
    Args:
        request: Problem data with moves and optional top_k parameter
        predictor_service: Injected predictor service (dependency)
        
    Returns:
        Prediction response with predicted grade, confidence, and top-k predictions
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    # Convert Pydantic models to dict format expected by Predictor
    problem_dict = {
        "moves": [move.model_dump() for move in request.moves]
    }
    
    try:
        # Make prediction
        result = predictor_service.predict(
            problem=problem_dict,
            top_k=request.top_k
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
            top_k_predictions=top_k_formatted
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

