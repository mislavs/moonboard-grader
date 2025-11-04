"""
Tests for the ProblemService.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any

from app.services.problem_service import ProblemService
from app.models.schemas import ProblemListItem, ProblemDetail


@pytest.fixture
def sample_problems_data() -> Dict[str, Any]:
    """Sample problems data matching the JSON structure."""
    return {
        "total": 3,
        "data": [
            {
                "name": "Test Problem 1",
                "grade": "6B+",
                "holdsetup": {
                    "description": "MoonBoard Masters 2017",
                    "apiId": 1001
                },
                "moves": [
                    {
                        "problemId": 1001,
                        "description": "A1",
                        "isStart": True,
                        "isEnd": False
                    },
                    {
                        "problemId": 1001,
                        "description": "K18",
                        "isStart": False,
                        "isEnd": True
                    }
                ]
            },
            {
                "name": "Test Problem 2",
                "grade": "7A",
                "holdsetup": {
                    "description": "MoonBoard Masters 2017",
                    "apiId": 1002
                },
                "moves": [
                    {
                        "problemId": 1002,
                        "description": "B2",
                        "isStart": True,
                        "isEnd": False
                    },
                    {
                        "problemId": 1002,
                        "description": "J15",
                        "isStart": False,
                        "isEnd": True
                    }
                ]
            },
            {
                "name": "Test Problem 3",
                "grade": "6C",
                "holdsetup": {
                    "description": "MoonBoard Masters 2017",
                    "apiId": 1003
                },
                "moves": []
            }
        ]
    }


@pytest.fixture
def problems_json_file(tmp_path: Path, sample_problems_data: Dict[str, Any]) -> Path:
    """Create a temporary problems JSON file."""
    json_file = tmp_path / "problems.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(sample_problems_data, f)
    return json_file


@pytest.fixture
def problem_service(problems_json_file: Path) -> ProblemService:
    """Create a problem service with test data."""
    service = ProblemService(problems_path=problems_json_file)
    return service


class TestProblemService:
    """Test suite for ProblemService."""
    
    def test_multiple_instances_with_same_path(self, problems_json_file: Path):
        """Test that service can be instantiated multiple times for different paths."""
        service1 = ProblemService(problems_path=problems_json_file)
        service2 = ProblemService(problems_path=problems_json_file)
        
        # These are different instances but can work with the same data
        assert service1.problems_path == service2.problems_path
    
    def test_init(self, problems_json_file: Path):
        """Test service initialization."""
        service = ProblemService(problems_path=problems_json_file)
        
        assert service.problems_path == problems_json_file
        assert not service.is_loaded
        assert service.problem_count == 0
    
    def test_load_problems_file_not_found(self, tmp_path: Path):
        """Test loading when file doesn't exist."""
        nonexistent_path = tmp_path / "nonexistent.json"
        service = ProblemService(problems_path=nonexistent_path)
        
        with pytest.raises(FileNotFoundError) as exc_info:
            service._load_problems()
        
        assert "Problems data file not found" in str(exc_info.value)
        assert not service.is_loaded
    
    def test_load_problems_invalid_json(self, tmp_path: Path):
        """Test loading with invalid JSON."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{ invalid json }", encoding='utf-8')
        
        service = ProblemService(problems_path=json_file)
        
        with pytest.raises(json.JSONDecodeError):
            service._load_problems()
    
    def test_load_problems_missing_data_key(self, tmp_path: Path):
        """Test loading with missing 'data' key."""
        json_file = tmp_path / "missing_key.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({"total": 0}, f)
        
        service = ProblemService(problems_path=json_file)
        
        with pytest.raises(KeyError) as exc_info:
            service._load_problems()
        
        assert "must have a 'data' key" in str(exc_info.value)
    
    def test_load_problems_success(self, problem_service: ProblemService):
        """Test successful problem loading."""
        problem_service._load_problems()
        
        assert problem_service.is_loaded
        assert problem_service.problem_count == 3
    
    def test_get_all_problems(self, problem_service: ProblemService):
        """Test getting all problems with basic info."""
        problems = problem_service.get_all_problems()
        
        assert len(problems) == 3
        assert all(isinstance(p, ProblemListItem) for p in problems)
        
        # Check first problem
        assert problems[0].apiId == 1001
        assert problems[0].name == "Test Problem 1"
        assert problems[0].grade == "6B+"
        
        # Check second problem
        assert problems[1].apiId == 1002
        assert problems[1].name == "Test Problem 2"
        assert problems[1].grade == "7A"
    
    def test_get_all_problems_with_missing_apiid(self, tmp_path: Path):
        """Test getting problems when some have missing apiId."""
        data = {
            "total": 2,
            "data": [
                {
                    "name": "Good Problem",
                    "grade": "6B+",
                    "holdsetup": {"apiId": 1001},
                    "moves": []
                },
                {
                    "name": "Bad Problem",
                    "grade": "7A",
                    "holdsetup": {},  # Missing apiId
                    "moves": []
                }
            ]
        }
        
        json_file = tmp_path / "problems.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        
        service = ProblemService(problems_path=json_file)
        problems = service.get_all_problems()
        
        # Should only return the problem with valid apiId
        assert len(problems) == 1
        assert problems[0].apiId == 1001
    
    def test_get_problem_by_id_found(self, problem_service: ProblemService):
        """Test getting a specific problem by ID."""
        problem = problem_service.get_problem_by_id(1001)
        
        assert problem is not None
        assert isinstance(problem, ProblemDetail)
        assert problem.apiId == 1001
        assert problem.name == "Test Problem 1"
        assert problem.grade == "6B+"
        assert len(problem.moves) == 2
        
        # Check moves
        assert problem.moves[0].description == "A1"
        assert problem.moves[0].isStart is True
        assert problem.moves[0].isEnd is False
        assert problem.moves[1].description == "K18"
        assert problem.moves[1].isStart is False
        assert problem.moves[1].isEnd is True
    
    def test_get_problem_by_id_not_found(self, problem_service: ProblemService):
        """Test getting a non-existent problem."""
        problem = problem_service.get_problem_by_id(9999)
        
        assert problem is None
    
    def test_get_problem_by_id_with_empty_moves(self, problem_service: ProblemService):
        """Test getting a problem with no moves."""
        problem = problem_service.get_problem_by_id(1003)
        
        assert problem is not None
        assert problem.apiId == 1003
        assert problem.name == "Test Problem 3"
        assert problem.grade == "6C"
        assert len(problem.moves) == 0
    
    def test_reload(self, problem_service: ProblemService, problems_json_file: Path):
        """Test reloading problems from disk."""
        # First load
        problem_service._load_problems()
        assert problem_service.problem_count == 3
        
        # Modify the file
        new_data = {
            "total": 1,
            "data": [
                {
                    "name": "New Problem",
                    "grade": "8A",
                    "holdsetup": {"apiId": 2001},
                    "moves": []
                }
            ]
        }
        with open(problems_json_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f)
        
        # Reload
        problem_service.reload()
        
        assert problem_service.problem_count == 1
        problems = problem_service.get_all_problems()
        assert problems[0].apiId == 2001
        assert problems[0].name == "New Problem"
    
    def test_ensure_loaded_lazy_loading(self, problem_service: ProblemService):
        """Test that problems are loaded lazily on first access."""
        assert not problem_service.is_loaded
        
        # Access problems - should trigger loading
        problems = problem_service.get_all_problems()
        
        assert problem_service.is_loaded
        assert len(problems) > 0
    
    def test_is_loaded_property(self, problem_service: ProblemService):
        """Test is_loaded property."""
        assert not problem_service.is_loaded
        
        problem_service._load_problems()
        
        assert problem_service.is_loaded
    
    def test_problem_count_property(self, problem_service: ProblemService):
        """Test problem_count property."""
        assert problem_service.problem_count == 0
        
        problem_service._load_problems()
        
        assert problem_service.problem_count == 3

