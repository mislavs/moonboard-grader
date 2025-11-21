"""
Generation route handlers.

Provides endpoints for generating new climbing problems using VAE models.
"""

from fastapi import APIRouter, Depends, HTTPException, status
import logging

from ...models.schemas import (
    GenerateRequest,
    GenerateResponse,
    ProblemMove,
)
from ...services.generator_service import GeneratorService
from ..dependencies import get_loaded_generator

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/generate", response_model=GenerateResponse, tags=["Generation"]
)
async def generate_problem(
    request: GenerateRequest,
    generator_service: GeneratorService = Depends(get_loaded_generator)
):
    """
    Generate a new climbing problem at the specified grade.

    Uses a Conditional VAE model to generate novel problems.
    The endpoint will retry generation until a valid problem is produced
    (or max attempts reached).

    Args:
        request: Generation parameters including grade and temperature
        generator_service: Injected generator service (dependency)

    Returns:
        Generated problem with moves, grade, and statistics

    Raises:
        HTTPException: If model is not loaded or generation fails
    """
    try:
        logger.info(
            f"Generation request received for grade {request.grade}"
        )

        # Generate problem with retry logic
        result = generator_service.generate_problem(
            grade=request.grade,
            temperature=request.temperature,
            max_attempts=10
        )

        # Convert moves to ProblemMove schema
        moves = [
            ProblemMove(
                description=move['description'],
                isStart=move['isStart'],
                isEnd=move['isEnd']
            )
            for move in result['moves']
        ]

        # Build response
        response = GenerateResponse(
            moves=moves,
            grade=request.grade
        )

        logger.info(
            f"Successfully generated problem with {len(moves)} moves "
            f"at grade {request.grade}"
        )

        return response

    except ValueError as e:
        logger.error(f"Invalid generation parameters: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid generation parameters: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Problem generation failed: {str(e)}"
        )
