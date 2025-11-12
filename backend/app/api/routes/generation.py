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
    ProblemStats,
)
from ...services.generator_service import GeneratorService
from ..dependencies import get_loaded_generator

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate_problem(
    request: GenerateRequest,
    generator_service: GeneratorService = Depends(get_loaded_generator)
):
    """
    Generate a new climbing problem at the specified grade.
    
    Uses a Conditional VAE model to generate novel problems. The endpoint
    will retry generation until a valid problem is produced (or max attempts reached).
    
    NOTE: Currently ignores the input grade and always generates 6A+ problems.
    
    Args:
        request: Generation parameters including grade and temperature
        generator_service: Injected generator service (dependency)
        
    Returns:
        Generated problem with moves, grade, and statistics
        
    Raises:
        HTTPException: If model is not loaded or generation fails
    """
    try:
        # NOTE: Currently hardcoded to always generate 6A+ (as per plan)
        # The input grade is ignored for now
        hardcoded_grade = "6A+"
        
        logger.info(
            f"Generation request received for grade {request.grade} "
            f"(using hardcoded {hardcoded_grade})"
        )
        
        # Generate problem with retry logic
        result = generator_service.generate_problem(
            grade=hardcoded_grade,
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
        
        # Extract stats
        stats_dict = result.get('stats', {})
        stats = ProblemStats(
            num_moves=stats_dict.get('num_moves', len(moves)),
            num_start_holds=stats_dict.get('num_start_holds', 0),
            num_end_holds=stats_dict.get('num_end_holds', 0)
        )
        
        # Build response
        response = GenerateResponse(
            moves=moves,
            grade=hardcoded_grade,
            stats=stats
        )
        
        logger.info(
            f"Successfully generated problem with {len(moves)} moves at grade {hardcoded_grade}"
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

