"""
API route handlers.

Defines all HTTP endpoints for the application.
"""

from fastapi import APIRouter, Depends, HTTPException, status
import logging

from ..models.schemas import (
    ProblemRequest,
    PredictionResponse,
    TopKPrediction,
    HealthResponse,
    ModelInfoResponse,
    ProblemListItem,
    ProblemDetail,
)
from ..services.predictor_service import PredictorService
from ..services.problem_service import ProblemService
from .dependencies import get_predictor_service, get_loaded_predictor, get_problem_service
from ..core.config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/health"
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(
    predictor_service: PredictorService = Depends(get_predictor_service)
):
    """
    Health check endpoint.
    
    Returns the service status and whether the model is loaded.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=predictor_service.is_loaded
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info(
    predictor_service: PredictorService = Depends(get_predictor_service)
):
    """
    Get information about the loaded model.
    
    Returns model path, device, and existence status.
    """
    info = predictor_service.get_model_info()
    return ModelInfoResponse(
        model_path=info["model_path"],
        device=info["device"],
        model_exists=info["model_exists"]
    )


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


def _handle_problem_service_error(e: Exception, context: str = "loading problems") -> HTTPException:
    """
    Convert problem service errors to appropriate HTTP exceptions.
    
    Args:
        e: The exception to handle
        context: Description of what operation failed
        
    Returns:
        HTTPException with appropriate status code and message
    """
    if isinstance(e, HTTPException):
        return e
    
    if isinstance(e, FileNotFoundError):
        logger.error(f"Problems file not found: {e}")
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Problems data file not found"
        )
    
    logger.error(f"Error {context}: {e}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to load problems: {str(e)}"
    )


@router.get("/problems", response_model=list[ProblemListItem], tags=["Problems"])
async def get_problems_list(
    problem_service: ProblemService = Depends(get_problem_service)
):
    """
    Get list of all problems with basic information.
    
    Returns a lightweight list of problems containing only apiId, name, and grade.
    Useful for dropdowns and problem selection interfaces.
    
    Args:
        problem_service: Injected problem service (dependency)
        
    Returns:
        List of ProblemListItem with basic problem information
        
    Raises:
        HTTPException: If problems data cannot be loaded
    """
    try:
        return problem_service.get_all_problems()
    except Exception as e:
        raise _handle_problem_service_error(e)


@router.get("/problems/{api_id}", response_model=ProblemDetail, tags=["Problems"])
async def get_problem_detail(
    api_id: int,
    problem_service: ProblemService = Depends(get_problem_service)
):
    """
    Get detailed information for a specific problem.
    
    Returns complete problem data including all moves for the specified problem.
    
    Args:
        api_id: The unique identifier (apiId) of the problem
        problem_service: Injected problem service (dependency)
        
    Returns:
        ProblemDetail with complete problem information including moves
        
    Raises:
        HTTPException: 404 if problem not found, 500 for other errors
    """
    try:
        problem = problem_service.get_problem_by_id(api_id)
        if problem is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Problem with apiId {api_id} not found"
            )
        return problem
    except Exception as e:
        raise _handle_problem_service_error(e, f"loading problem {api_id}")

