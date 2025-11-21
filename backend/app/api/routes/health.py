"""
Health and system information route handlers.

Provides endpoints for health checks and system status.
"""

from fastapi import APIRouter, Depends

from ...models.schemas import HealthResponse, ModelInfoResponse
from ...services.predictor_service import PredictorService
from ...services.generator_service import GeneratorService
from ..dependencies import get_predictor_service, get_generator_service
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
    predictor_service: PredictorService = Depends(get_predictor_service),
    generator_service: GeneratorService = Depends(get_generator_service)
):
    """
    Health check endpoint.

    Returns the service status and whether the models are loaded.
    Status will be "healthy" if both models are loaded, "degraded" otherwise.

    Note: This endpoint requires both services to be initialized.
    It will return 503 if services are not set up.
    """
    predictor_loaded = predictor_service.is_loaded
    generator_loaded = generator_service.is_loaded

    # Determine overall status
    if predictor_loaded and generator_loaded:
        status_value = "healthy"
    else:
        status_value = "degraded"

    return HealthResponse(
        status=status_value,
        model_loaded=predictor_loaded,
        generator_model_loaded=generator_loaded
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
