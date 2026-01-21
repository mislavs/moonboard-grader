"""
Board setups route handlers.

Provides endpoints for retrieving available board configurations.
"""

import logging
import sys
from pathlib import Path

from fastapi import APIRouter

from ...core.config import settings
from ...models.schemas import (
    AngleConfigResponse,
    BoardSetupsResponse,
    HoldSetupResponse,
)

# Add moonboard_core to path for imports
_MOONBOARD_CORE_PATH = str(Path(__file__).parent.parent.parent.parent.parent)
if _MOONBOARD_CORE_PATH not in sys.path:
    sys.path.insert(0, _MOONBOARD_CORE_PATH)

from moonboard_core.board_config import BoardConfigRegistry  # noqa: E402

logger = logging.getLogger(__name__)

router = APIRouter()


def _model_exists(model_file: str | None) -> bool:
    """Check if a model file exists."""
    if model_file is None:
        return False
    # Model paths in config are relative to project root
    model_path = Path(settings.board_config_path).parent.parent / model_file
    return model_path.exists()


@router.get(
    "/board-setups",
    response_model=BoardSetupsResponse,
    tags=["Configuration"]
)
async def get_board_setups():
    """
    Get all available board setups and their configurations.

    Returns a list of hold setups, each with their available wall angles
    and whether trained models exist for each configuration.
    """
    try:
        registry = BoardConfigRegistry(settings.board_config_path)
        hold_setups = registry.get_hold_setups()

        response_setups = []
        for setup in hold_setups:
            angles = []
            for angle_config in setup.angles:
                angles.append(AngleConfigResponse(
                    angle=angle_config.angle,
                    hasModel=_model_exists(angle_config.model_file),
                    isDefault=angle_config.is_default
                ))

            response_setups.append(HoldSetupResponse(
                id=setup.id,
                name=setup.name,
                angles=angles
            ))

        return BoardSetupsResponse(holdSetups=response_setups)

    except FileNotFoundError as e:
        logger.error(f"Board configuration file not found: {e}")
        # Return empty list if config doesn't exist
        return BoardSetupsResponse(holdSetups=[])
    except Exception as e:
        logger.error(f"Error loading board setups: {e}")
        raise

