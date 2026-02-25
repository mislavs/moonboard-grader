"""
Board analytics route handlers.

Provides endpoints for board-level analytics and hold statistics.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...models.schemas import BoardAnalyticsResponse
from ...services.service_registry import ServiceRegistry
from ..dependencies import get_service_registry

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/analytics/board",
    response_model=BoardAnalyticsResponse,
    tags=["Analytics"]
)
async def get_board_analytics(
    hold_setup: Optional[str] = Query(
        None,
        description="Hold setup ID (e.g., 'masters-2017'). Uses default if not specified."
    ),
    angle: Optional[int] = Query(
        None,
        description="Wall angle in degrees (e.g., 40). Uses default if not specified."
    ),
    registry: ServiceRegistry = Depends(get_service_registry),
):
    """
    Get board-level analytics including hold difficulty statistics.

    Returns pre-computed statistics for each hold position on the board,
    including difficulty ratings, usage frequency, and heatmap data.

    The data is pre-computed from problems with 5+ repeats to ensure
    statistical reliability.

    Args:
        hold_setup: Optional hold setup ID for analytics data
        angle: Optional wall angle for analytics data

    Returns:
        BoardAnalyticsResponse with holds stats, heatmaps, and metadata

    Raises:
        HTTPException: 500 if analytics data file is not found or invalid
    """
    if hold_setup or angle:
        logger.debug("Analytics requested for setup=%s, angle=%s", hold_setup, angle)

    analytics_path = registry.get_analytics_path(hold_setup, angle)
    if analytics_path is None or not analytics_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analytics data unavailable for the requested configuration",
        )

    try:
        with open(analytics_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return BoardAnalyticsResponse(**data)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in analytics file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analytics data file is corrupted"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading analytics data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load analytics data: {str(e)}"
        )

