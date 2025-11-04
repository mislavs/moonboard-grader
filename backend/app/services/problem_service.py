"""
Problem service for managing problem data.

Handles loading and serving problem data from JSON file.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..models.schemas import ProblemListItem, ProblemDetail, ProblemMove
from ..core.config import settings

logger = logging.getLogger(__name__)


class ProblemService:
    """
    Service class for managing problem data.
    
    Handles loading problem data from JSON and providing access methods.
    """
    
    def __init__(self, problems_path: Optional[Path] = None):
        """
        Initialize the problem service.
        
        Args:
            problems_path: Path to the problems JSON file. Defaults to settings.problems_data_path
        """
        self.problems_path = problems_path or settings.problems_data_path
        self._problems_cache: Optional[List[Dict[str, Any]]] = None
        logger.info(f"ProblemService initialized with path: {self.problems_path}")
    
    def _load_problems(self) -> None:
        """
        Load problems from JSON file into cache.
        
        Raises:
            FileNotFoundError: If problems file doesn't exist
            json.JSONDecodeError: If JSON is invalid
            KeyError: If JSON structure is unexpected
        """
        if not self.problems_path.exists():
            raise FileNotFoundError(
                f"Problems data file not found at {self.problems_path}"
            )
        
        logger.info(f"Loading problems from {self.problems_path}...")
        
        try:
            with open(self.problems_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if 'data' not in data:
                raise KeyError("Problems JSON must have a 'data' key")
            
            self._problems_cache = data['data']
            logger.info(f"Loaded {len(self._problems_cache)} problems successfully")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in problems file: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load problems: {e}")
            raise
    
    def _ensure_loaded(self) -> None:
        """Ensure problems are loaded into cache."""
        if self._problems_cache is None:
            self._load_problems()
    
    @staticmethod
    def _extract_api_id(problem: Dict[str, Any]) -> Optional[int]:
        """
        Extract API ID from problem data.
        
        Args:
            problem: Raw problem dictionary
            
        Returns:
            API ID if present, None otherwise
        """
        return problem.get('apiId')
    
    @staticmethod
    def _parse_problem_moves(moves_data: List[Dict[str, Any]]) -> List[ProblemMove]:
        """
        Parse raw move data into ProblemMove objects.
        
        Args:
            moves_data: List of raw move dictionaries
            
        Returns:
            List of ProblemMove objects
        """
        return [
            ProblemMove(
                problemId=move.get('problemId', 0),
                description=move.get('description', ''),
                isStart=move.get('isStart', False),
                isEnd=move.get('isEnd', False)
            )
            for move in moves_data
        ]
    
    def _create_problem_detail(self, problem: Dict[str, Any]) -> ProblemDetail:
        """
        Create ProblemDetail from raw problem data.
        
        Args:
            problem: Raw problem dictionary
            
        Returns:
            ProblemDetail object
        """
        api_id = self._extract_api_id(problem)
        moves = self._parse_problem_moves(problem.get('moves', []))
        
        return ProblemDetail(
            apiId=api_id,
            name=problem.get('name', 'Unnamed'),
            grade=problem.get('grade', 'Unknown'),
            moves=moves
        )
    
    def get_all_problems(self, page: int = 1, page_size: int = 20) -> tuple[List[ProblemListItem], int]:
        """
        Get list of all problems with basic info (ID, name, grade), with pagination support.
        
        Args:
            page: Page number (1-indexed). Defaults to 1.
            page_size: Number of items per page. Defaults to 20.
        
        Returns:
            Tuple of (paginated list of ProblemListItem objects, total count)
            
        Raises:
            FileNotFoundError: If problems file doesn't exist
            Exception: If loading or processing fails
        """
        self._ensure_loaded()
        
        result = []
        for idx, problem in enumerate(self._problems_cache):
            try:
                api_id = self._extract_api_id(problem)
                if api_id is None:
                    logger.warning(f"Problem at index {idx} missing apiId, skipping")
                    continue
                
                result.append(ProblemListItem(
                    apiId=api_id,
                    name=problem.get('name', 'Unnamed'),
                    grade=problem.get('grade', 'Unknown')
                ))
            except Exception as e:
                logger.warning(f"Error processing problem at index {idx}: {e}")
                continue
        
        # Calculate pagination
        total = len(result)
        offset = (page - 1) * page_size
        paginated_result = result[offset:offset + page_size]
        
        return paginated_result, total
    
    def get_problem_by_id(self, api_id: int) -> Optional[ProblemDetail]:
        """
        Get complete problem details by API ID.
        
        Args:
            api_id: The API ID (problem.apiId) to search for
            
        Returns:
            ProblemDetail object if found, None otherwise
            
        Raises:
            FileNotFoundError: If problems file doesn't exist
            Exception: If loading fails
        """
        self._ensure_loaded()
        
        # Find problem by problem.apiId
        for problem in self._problems_cache:
            if self._extract_api_id(problem) == api_id:
                try:
                    return self._create_problem_detail(problem)
                except Exception as e:
                    logger.error(f"Error processing problem with apiId {api_id}: {e}")
                    raise
        
        # Problem not found
        return None
    
    def reload(self) -> None:
        """
        Force reload of problems from disk.
        
        Useful for testing or if the file has been updated.
        """
        self._problems_cache = None
        self._load_problems()
    
    @property
    def is_loaded(self) -> bool:
        """Check if problems are loaded in cache."""
        return self._problems_cache is not None
    
    @property
    def problem_count(self) -> int:
        """Get count of loaded problems."""
        return len(self._problems_cache) if self._problems_cache else 0

