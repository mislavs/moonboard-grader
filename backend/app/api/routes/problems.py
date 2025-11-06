"""
Problem data route handlers.

Provides endpoints for browsing and managing problem data.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
import logging
import math

from ...models.schemas import (
    ProblemDetail,
    PaginatedProblemsResponse,
    DuplicateCheckRequest,
    DuplicateCheckResponse,
)
from ...services.problem_service import ProblemService
from ..dependencies import get_problem_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _handle_problem_service_error(e: Exception, context: str = "loading problems") -> HTTPException:
    """
    Convert problem service errors to appropriate HTTP exceptions.
    
    Args:
        e: The exception to handle
        context: Description of what operation failed
        
    Returns:
        HTTPException with appropriate status code and message
    """
    if isinstance(e, HTTPException):
        return e
    
    if isinstance(e, FileNotFoundError):
        logger.error(f"Problems file not found: {e}")
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Problems data file not found"
        )
    
    logger.error(f"Error {context}: {e}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Failed to load problems: {str(e)}"
    )


@router.get("/problems", response_model=PaginatedProblemsResponse, tags=["Problems"])
async def get_problems_list(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page"),
    problem_service: ProblemService = Depends(get_problem_service)
):
    """
    Get paginated list of problems with basic information.
    
    Returns a paginated list of problems containing only id, name, and grade.
    Useful for dropdowns and problem selection interfaces.
    
    Args:
        page: Page number, starting from 1 (default: 1)
        page_size: Number of items per page, max 100 (default: 20)
        problem_service: Injected problem service (dependency)
        
    Returns:
        PaginatedProblemsResponse with items, pagination metadata, and total count
        
    Raises:
        HTTPException: If problems data cannot be loaded
    """
    try:
        items, total = problem_service.get_all_problems(page=page, page_size=page_size)
        total_pages = math.ceil(total / page_size) if total > 0 else 0
        
        return PaginatedProblemsResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
    except Exception as e:
        raise _handle_problem_service_error(e)


@router.get("/problems/{id}", response_model=ProblemDetail, tags=["Problems"])
async def get_problem_detail(
    id: int,
    problem_service: ProblemService = Depends(get_problem_service)
):
    """
    Get detailed information for a specific problem.
    
    Returns complete problem data including all moves for the specified problem.
    
    Args:
        id: The unique identifier of the problem
        problem_service: Injected problem service (dependency)
        
    Returns:
        ProblemDetail with complete problem information including moves
        
    Raises:
        HTTPException: 404 if problem not found, 500 for other errors
    """
    try:
        problem = problem_service.get_problem_by_id(id)
        if problem is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Problem with id {id} not found"
            )
        return problem
    except Exception as e:
        raise _handle_problem_service_error(e, f"loading problem {id}")


@router.post("/problems/check-duplicate", response_model=DuplicateCheckResponse, tags=["Problems"])
async def check_duplicate_problem(
    request: DuplicateCheckRequest,
    problem_service: ProblemService = Depends(get_problem_service)
):
    """
    Check if a problem with the exact same moves already exists.
    
    Compares moves in an order-independent manner (sorting by hold position).
    Both hold positions and start/end flags must match.
    
    Args:
        request: Request containing list of moves to check
        problem_service: Injected problem service
        
    Returns:
        Response indicating if duplicate exists and the problem ID if found
        
    Raises:
        HTTPException: If problems data cannot be loaded
    """
    try:
        duplicate_id = problem_service.find_duplicate_by_moves(request.moves)
        
        return DuplicateCheckResponse(
            exists=duplicate_id is not None,
            problem_id=duplicate_id
        )
            
    except Exception as e:
        raise _handle_problem_service_error(e, "checking for duplicate problem")

