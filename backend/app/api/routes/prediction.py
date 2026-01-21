"""
Prediction route handlers.

Provides endpoints for grade prediction using ML models.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from typing import Optional
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


@router.post(
    "/predict", response_model=PredictionResponse, tags=["Prediction"]
)
async def predict_grade(
    request: ProblemRequest,
    hold_setup: Optional[str] = Query(
        None,
        description="Hold setup ID (e.g., 'masters-2017'). Uses default if not specified."
    ),
    angle: Optional[int] = Query(
        None,
        description="Wall angle in degrees (e.g., 40). Uses default if not specified."
    ),
    predictor_service: PredictorService = Depends(get_loaded_predictor)
):
    """
    Predict the grade of a climbing problem.

    Args:
        request: Problem data with moves and optional top_k parameter
        hold_setup: Optional hold setup ID to use for prediction
        angle: Optional wall angle to use for prediction
        predictor_service: Injected predictor service (dependency)

    Returns:
        Prediction response with predicted grade, confidence,
        and top-k predictions

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    # Log the setup being used (for future multi-model support)
    if hold_setup or angle:
        logger.debug(f"Prediction requested for setup={hold_setup}, angle={angle}")
    # Convert Pydantic models to dict format expected by Predictor
    problem_dict = {
        "moves": [move.model_dump() for move in request.moves]
    }

    try:
        # Make prediction with attention map
        result = predictor_service.predict_with_attention(
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
            top_k_predictions=top_k_formatted,
            attention_map=result.get('attention_map')
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
