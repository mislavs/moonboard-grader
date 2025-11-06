"""
Health and system information route handlers.

Provides endpoints for health checks and system status.
"""

from fastapi import APIRouter, Depends

from ...models.schemas import HealthResponse, ModelInfoResponse
from ...services.predictor_service import PredictorService
from ..dependencies import get_predictor_service
from ...core.config import settings

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

