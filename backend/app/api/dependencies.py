"""
FastAPI dependencies for dependency injection.

Provides reusable dependencies for routes.
"""

from typing import Optional
from fastapi import HTTPException, status

from ..services.predictor_service import PredictorService

# Global predictor service instance
_predictor_service: Optional[PredictorService] = None


def get_predictor_service() -> PredictorService:
    """
    Dependency that returns the predictor service.
    
    Returns:
        PredictorService instance
        
    Raises:
        HTTPException: If service is not initialized
    """
    if _predictor_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor service not initialized"
        )
    return _predictor_service


def set_predictor_service(service: PredictorService) -> None:
    """
    Set the global predictor service instance.
    
    Args:
        service: PredictorService instance to set
    """
    global _predictor_service
    _predictor_service = service


def get_loaded_predictor() -> PredictorService:
    """
    Dependency that returns a loaded predictor service.
    
    Returns:
        Loaded PredictorService instance
        
    Raises:
        HTTPException: If model is not loaded
    """
    service = get_predictor_service()
    
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not loaded. Please ensure model_for_inference.pth "
                "exists in the models/ directory and restart the server."
            )
        )
    
    return service

