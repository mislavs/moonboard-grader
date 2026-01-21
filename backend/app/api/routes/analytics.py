"""
Board analytics route handlers.

Provides endpoints for board-level analytics and hold statistics.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from ...models.schemas import BoardAnalyticsResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Path to the pre-computed analytics data
# This file should be copied from analysis/hold_stats.json
ANALYTICS_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "hold_stats.json"


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
    )
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
    # Log the setup being used (for future multi-data-source support)
    if hold_setup or angle:
        logger.debug(f"Analytics requested for setup={hold_setup}, angle={angle}")

    if not ANALYTICS_DATA_PATH.exists():
        logger.error(f"Analytics data file not found: {ANALYTICS_DATA_PATH}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analytics data not available. Please ensure hold_stats.json is present in the data folder."
        )

    try:
        with open(ANALYTICS_DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return BoardAnalyticsResponse(**data)

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in analytics file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analytics data file is corrupted"
        )
    except Exception as e:
        logger.error(f"Error loading analytics data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load analytics data: {str(e)}"
        )

