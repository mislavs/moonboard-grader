"""
API route modules.

Aggregates all route handlers from domain-specific modules.
"""

from fastapi import APIRouter
from .health import router as health_router
from .prediction import router as prediction_router
from .problems import router as problems_router
from .generation import router as generation_router
from .analytics import router as analytics_router
from .board_setups import router as board_setups_router

router = APIRouter()
router.include_router(health_router)
router.include_router(prediction_router)
router.include_router(problems_router)
router.include_router(generation_router)
router.include_router(analytics_router)
router.include_router(board_setups_router)

__all__ = ["router"]
