"""
FastAPI dependencies for dependency injection.

Provides reusable dependencies for routes.
"""

from typing import Optional
from fastapi import HTTPException, status

from ..services.predictor_service import PredictorService
from ..services.problem_service import ProblemService
from ..services.generator_service import GeneratorService
from ..services.service_registry import ServiceRegistry

# Global service registry instance
_service_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """
    Dependency that returns the service registry.

    Returns:
        ServiceRegistry instance

    Raises:
        HTTPException: If registry is not initialized
    """
    if _service_registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service registry not initialized",
        )
    return _service_registry


def set_service_registry(registry: Optional[ServiceRegistry]) -> None:
    """
    Set the global service registry instance.

    Args:
        registry: ServiceRegistry instance to set
    """
    global _service_registry
    _service_registry = registry


def get_predictor_service() -> PredictorService:
    """
    Dependency that returns the predictor service.

    Returns:
        PredictorService instance

    Raises:
        HTTPException: If service is not initialized
    """
    return get_service_registry().get_predictor()


def get_loaded_predictor() -> PredictorService:
    """
    Dependency that returns a loaded predictor service.

    Returns:
        Loaded PredictorService instance

    Raises:
        HTTPException: If model is not loaded
    """
    return get_service_registry().get_loaded_predictor()


def get_problem_service() -> ProblemService:
    """
    Dependency that returns the problem service.

    Returns:
        ProblemService instance

    Raises:
        HTTPException: If service is not initialized
    """
    return get_service_registry().get_problem_service()


def get_generator_service() -> GeneratorService:
    """
    Dependency that returns the generator service.

    Returns:
        GeneratorService instance

    Raises:
        HTTPException: If service is not initialized
    """
    return get_service_registry().get_generator()


def get_loaded_generator() -> GeneratorService:
    """
    Dependency that returns a loaded generator service.

    Returns:
        Loaded GeneratorService instance

    Raises:
        HTTPException: If model is not loaded
    """
    return get_service_registry().get_loaded_generator()
