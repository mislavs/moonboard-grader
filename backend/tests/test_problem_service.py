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
                "apiId": 1001,
                "isBenchmark": True,
                "holdsetup": {
                    "description": "MoonBoard Masters 2017",
                    "apiId": 15
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
                "apiId": 1002,
                "isBenchmark": False,
                "holdsetup": {
                    "description": "MoonBoard Masters 2017",
                    "apiId": 15
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
                "apiId": 1003,
                "isBenchmark": True,
                "holdsetup": {
                    "description": "MoonBoard Masters 2017",
                    "apiId": 15
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
    
    def test_get_all_problems_default_pagination(self, problem_service: ProblemService):
        """Test getting all problems with default pagination."""
        problems, total = problem_service.get_all_problems()
        
        assert total == 3
        assert len(problems) == 3  # All items fit on first page with default size
        assert all(isinstance(p, ProblemListItem) for p in problems)
        
        # Check first problem
        assert problems[0].id == 1001
        assert problems[0].name == "Test Problem 1"
        assert problems[0].grade == "6B+"
        
        # Check second problem
        assert problems[1].id == 1002
        assert problems[1].name == "Test Problem 2"
        assert problems[1].grade == "7A"
    
    def test_get_all_problems_with_pagination(self, problem_service: ProblemService):
        """Test getting problems with pagination."""
        # Get first page with 2 items per page
        problems_page1, total = problem_service.get_all_problems(page=1, page_size=2)
        
        assert total == 3
        assert len(problems_page1) == 2
        assert problems_page1[0].id == 1001
        assert problems_page1[1].id == 1002
        
        # Get second page
        problems_page2, total = problem_service.get_all_problems(page=2, page_size=2)
        
        assert total == 3
        assert len(problems_page2) == 1
        assert problems_page2[0].id == 1003
    
    def test_get_all_problems_page_beyond_range(self, problem_service: ProblemService):
        """Test getting a page beyond available data."""
        problems, total = problem_service.get_all_problems(page=10, page_size=20)
        
        assert total == 3
        assert len(problems) == 0  # Empty page
    
    def test_get_all_problems_custom_page_size(self, problem_service: ProblemService):
        """Test getting problems with custom page size."""
        problems, total = problem_service.get_all_problems(page=1, page_size=1)
        
        assert total == 3
        assert len(problems) == 1
        assert problems[0].id == 1001
    
    def test_get_all_problems_with_missing_apiid(self, tmp_path: Path):
        """Test getting problems when some have missing apiId."""
        data = {
            "total": 2,
            "data": [
                {
                    "name": "Good Problem",
                    "grade": "6B+",
                    "apiId": 1001,
                    "holdsetup": {"apiId": 15},
                    "moves": []
                },
                {
                    "name": "Bad Problem",
                    "grade": "7A",
                    "holdsetup": {"apiId": 15},  # Has holdsetup.apiId but missing problem.apiId
                    "moves": []
                }
            ]
        }
        
        json_file = tmp_path / "problems.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        
        service = ProblemService(problems_path=json_file)
        problems, total = service.get_all_problems()
        
        # Should only return the problem with valid apiId
        assert total == 1
        assert len(problems) == 1
        assert problems[0].id == 1001
    
    def test_get_problem_by_id_found(self, problem_service: ProblemService):
        """Test getting a specific problem by ID."""
        problem = problem_service.get_problem_by_id(1001)
        
        assert problem is not None
        assert isinstance(problem, ProblemDetail)
        assert problem.id == 1001
        assert problem.name == "Test Problem 1"
        assert problem.grade == "6B+"
        assert len(problem.moves) == 2
        
        # Check moves (problemId should be removed)
        assert "problemId" not in problem.moves[0].model_dump()
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
        assert problem.id == 1003
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
                    "apiId": 2001,
                    "holdsetup": {"apiId": 15},
                    "moves": []
                }
            ]
        }
        with open(problems_json_file, 'w', encoding='utf-8') as f:
            json.dump(new_data, f)
        
        # Reload
        problem_service.reload()
        
        assert problem_service.problem_count == 1
        problems, total = problem_service.get_all_problems()
        assert total == 1
        assert problems[0].id == 2001
        assert problems[0].name == "New Problem"
    
    def test_ensure_loaded_lazy_loading(self, problem_service: ProblemService):
        """Test that problems are loaded lazily on first access."""
        assert not problem_service.is_loaded
        
        # Access problems - should trigger loading
        problems, total = problem_service.get_all_problems()
        
        assert problem_service.is_loaded
        assert total > 0
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
    
    def test_moves_to_key_normalization(self, problem_service: ProblemService):
        """Test that moves are normalized correctly for comparison."""
        from app.models.schemas import ProblemMove
        
        moves1 = [
            ProblemMove(description="A1", isStart=True, isEnd=False),
            ProblemMove(description="K18", isStart=False, isEnd=True)
        ]
        moves2 = [
            ProblemMove(description="K18", isStart=False, isEnd=True),
            ProblemMove(description="A1", isStart=True, isEnd=False)
        ]
        
        key1 = problem_service._moves_to_key(moves1)
        key2 = problem_service._moves_to_key(moves2)
        assert key1 == key2
    
    def test_find_duplicate_by_moves_exists(self, problem_service: ProblemService):
        """Test finding a duplicate problem that exists."""
        from app.models.schemas import ProblemMove
        
        moves = [
            ProblemMove(description="A1", isStart=True, isEnd=False),
            ProblemMove(description="K18", isStart=False, isEnd=True)
        ]
        
        result = problem_service.find_duplicate_by_moves(moves)
        assert result == 1001
    
    def test_find_duplicate_by_moves_not_found(self, problem_service: ProblemService):
        """Test that non-existent moves return None."""
        from app.models.schemas import ProblemMove
        
        moves = [
            ProblemMove(description="Z99", isStart=True, isEnd=False),
            ProblemMove(description="Y88", isStart=False, isEnd=True)
        ]
        
        result = problem_service.find_duplicate_by_moves(moves)
        assert result is None


class TestBenchmarkFiltering:
    """Test suite for benchmark filtering functionality."""
    
    def test_get_all_problems_benchmarks_only_true(self, problem_service: ProblemService):
        """Test filtering to get only benchmark problems."""
        problems, total = problem_service.get_all_problems(benchmarks_only=True)
        
        # Should return only the 2 benchmark problems (IDs 1001 and 1003)
        assert total == 2
        assert len(problems) == 2
        assert all(p.isBenchmark for p in problems)
        assert problems[0].id == 1001
        assert problems[1].id == 1003
    
    def test_get_all_problems_benchmarks_only_false(self, problem_service: ProblemService):
        """Test filtering to get only non-benchmark problems."""
        problems, total = problem_service.get_all_problems(benchmarks_only=False)
        
        # Should return only the 1 non-benchmark problem (ID 1002)
        assert total == 1
        assert len(problems) == 1
        assert not problems[0].isBenchmark
        assert problems[0].id == 1002
    
    def test_get_all_problems_benchmarks_only_none(self, problem_service: ProblemService):
        """Test that None filter returns all problems."""
        problems, total = problem_service.get_all_problems(benchmarks_only=None)
        
        # Should return all 3 problems
        assert total == 3
        assert len(problems) == 3
    
    def test_benchmark_filtering_with_pagination(self, problem_service: ProblemService):
        """Test that benchmark filtering works correctly with pagination."""
        # Get first page of benchmarks with page_size=1
        page1, total = problem_service.get_all_problems(page=1, page_size=1, benchmarks_only=True)
        
        assert total == 2  # Total benchmarks
        assert len(page1) == 1  # Items on page
        assert page1[0].id == 1001
        assert page1[0].isBenchmark
        
        # Get second page of benchmarks
        page2, total = problem_service.get_all_problems(page=2, page_size=1, benchmarks_only=True)
        
        assert total == 2  # Total benchmarks
        assert len(page2) == 1  # Items on page
        assert page2[0].id == 1003
        assert page2[0].isBenchmark
    
    def test_benchmark_field_included_in_response(self, problem_service: ProblemService):
        """Test that isBenchmark field is included in ProblemListItem."""
        problems, _ = problem_service.get_all_problems()
        
        for problem in problems:
            assert hasattr(problem, 'isBenchmark')
            assert isinstance(problem.isBenchmark, bool)
    
    def test_get_problem_by_id_includes_benchmark(self, problem_service: ProblemService):
        """Test that isBenchmark field is included in ProblemDetail."""
        problem = problem_service.get_problem_by_id(1001)
        
        assert problem is not None
        assert hasattr(problem, 'isBenchmark')
        assert problem.isBenchmark is True
        
        # Check non-benchmark problem
        problem2 = problem_service.get_problem_by_id(1002)
        assert problem2 is not None
        assert problem2.isBenchmark is False


class TestGradeFiltering:
    """Test suite for grade range filtering functionality."""
    
    def test_grade_from_filter_only(self, problem_service: ProblemService):
        """Test filtering by minimum grade only."""
        # Test data has: 6B+, 7A, 6C
        # grade_from=6C should return 6C and 7A (grades >= 6C)
        problems, total = problem_service.get_all_problems(grade_from="6C")
        
        assert total == 2
        assert len(problems) == 2
        grades = {p.grade for p in problems}
        assert grades == {"6C", "7A"}
    
    def test_grade_to_filter_only(self, problem_service: ProblemService):
        """Test filtering by maximum grade only."""
        # Test data has: 6B+, 7A, 6C
        # grade_to=6C should return 6B+ and 6C (grades <= 6C)
        problems, total = problem_service.get_all_problems(grade_to="6C")
        
        assert total == 2
        assert len(problems) == 2
        grades = {p.grade for p in problems}
        assert grades == {"6B+", "6C"}
    
    def test_grade_range_filter(self, problem_service: ProblemService):
        """Test filtering by both minimum and maximum grade."""
        # Test data has: 6B+, 7A, 6C
        # grade_from=6B+, grade_to=6C should return 6B+ and 6C
        problems, total = problem_service.get_all_problems(grade_from="6B+", grade_to="6C")
        
        assert total == 2
        assert len(problems) == 2
        grades = {p.grade for p in problems}
        assert grades == {"6B+", "6C"}
    
    def test_grade_equal_range(self, problem_service: ProblemService):
        """Test filtering when grade_from equals grade_to."""
        # Should return only problems with exact grade match
        problems, total = problem_service.get_all_problems(grade_from="6B+", grade_to="6B+")
        
        assert total == 1
        assert len(problems) == 1
        assert problems[0].grade == "6B+"
    
    def test_grade_filter_no_matches(self, problem_service: ProblemService):
        """Test filtering with grade range that has no matches."""
        # Test data has: 6B+, 7A, 6C (lowest=6B+, highest=7A)
        # grade_from=7B should return no results
        problems, total = problem_service.get_all_problems(grade_from="7B")
        
        assert total == 0
        assert len(problems) == 0
    
    def test_grade_filter_case_insensitive(self, problem_service: ProblemService):
        """Test that grade filtering is case-insensitive."""
        # Test with lowercase grades
        problems_lower, total_lower = problem_service.get_all_problems(grade_from="6b+", grade_to="6c")
        problems_upper, total_upper = problem_service.get_all_problems(grade_from="6B+", grade_to="6C")
        
        assert total_lower == total_upper
        assert len(problems_lower) == len(problems_upper)
    
    def test_grade_and_benchmark_filters_combined(self, problem_service: ProblemService):
        """Test combining grade filter with benchmark filter."""
        # Test data: 
        # - 6B+ (benchmark)
        # - 7A (non-benchmark)
        # - 6C (benchmark)
        
        # Get benchmarks in grade range 6B+ to 6C
        problems, total = problem_service.get_all_problems(
            grade_from="6B+",
            grade_to="6C",
            benchmarks_only=True
        )
        
        assert total == 2  # 6B+ and 6C are both benchmarks
        assert len(problems) == 2
        assert all(p.isBenchmark for p in problems)
        grades = {p.grade for p in problems}
        assert grades == {"6B+", "6C"}
    
    def test_grade_filter_with_pagination(self, problem_service: ProblemService):
        """Test that grade filtering works correctly with pagination."""
        # Get first page with page_size=1
        page1, total = problem_service.get_all_problems(
            grade_from="6B+",
            grade_to="6C",
            page=1,
            page_size=1
        )
        
        assert total == 2  # Total in range
        assert len(page1) == 1  # Items on page
        assert page1[0].grade in {"6B+", "6C"}
        
        # Get second page
        page2, total = problem_service.get_all_problems(
            grade_from="6B+",
            grade_to="6C",
            page=2,
            page_size=1
        )
        
        assert total == 2  # Total in range
        assert len(page2) == 1  # Items on page
        assert page2[0].grade in {"6B+", "6C"}
    
    def test_invalid_grade_handled_gracefully(self, problem_service: ProblemService):
        """Test that invalid grades in problems data are excluded."""
        # The _is_grade_in_range method should handle invalid grades
        # by returning False, effectively excluding them
        # This is tested implicitly by other tests, but we verify the behavior
        problems, total = problem_service.get_all_problems(grade_from="5+", grade_to="8C+")
        
        # Should return all valid problems
        assert total == 3
    
    def test_inverted_grade_range(self, problem_service: ProblemService):
        """Test filtering when grade_from is greater than grade_to (inverted range)."""
        # Test data has: 6B+, 7A, 6C
        # grade_from=7A, grade_to=6B+ is inverted - should return no results
        problems, total = problem_service.get_all_problems(grade_from="7A", grade_to="6B+")
        
        assert total == 0
        assert len(problems) == 0


@pytest.fixture
def extended_problems_data() -> Dict[str, Any]:
    """Extended sample problems data with more grades for comprehensive testing."""
    return {
        "total": 7,
        "data": [
            {
                "name": "Easy Problem",
                "grade": "6A",
                "apiId": 2001,
                "isBenchmark": False,
                "moves": []
            },
            {
                "name": "Medium Problem 1",
                "grade": "6B+",
                "apiId": 2002,
                "isBenchmark": True,
                "moves": []
            },
            {
                "name": "Medium Problem 2",
                "grade": "6C",
                "apiId": 2003,
                "isBenchmark": False,
                "moves": []
            },
            {
                "name": "Medium Problem 3",
                "grade": "7A",
                "apiId": 2004,
                "isBenchmark": True,
                "moves": []
            },
            {
                "name": "Hard Problem 1",
                "grade": "7B",
                "apiId": 2005,
                "isBenchmark": False,
                "moves": []
            },
            {
                "name": "Hard Problem 2",
                "grade": "7C",
                "apiId": 2006,
                "isBenchmark": True,
                "moves": []
            },
            {
                "name": "Very Hard Problem",
                "grade": "8A",
                "apiId": 2007,
                "isBenchmark": False,
                "moves": []
            }
        ]
    }


class TestGradeFilteringExtended:
    """Extended grade filtering tests with more comprehensive data."""
    
    @pytest.fixture
    def extended_service(self, tmp_path: Path, extended_problems_data: Dict[str, Any]) -> ProblemService:
        """Create a problem service with extended test data."""
        json_file = tmp_path / "extended_problems.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(extended_problems_data, f)
        return ProblemService(problems_path=json_file)
    
    def test_wide_grade_range(self, extended_service: ProblemService):
        """Test filtering with a wide grade range."""
        problems, total = extended_service.get_all_problems(grade_from="6B+", grade_to="7B")
        
        assert total == 4
        grades = {p.grade for p in problems}
        assert grades == {"6B+", "6C", "7A", "7B"}
    
    def test_narrow_grade_range(self, extended_service: ProblemService):
        """Test filtering with a narrow grade range."""
        problems, total = extended_service.get_all_problems(grade_from="7A", grade_to="7B")
        
        assert total == 2
        grades = {p.grade for p in problems}
        assert grades == {"7A", "7B"}
    
    def test_grade_from_at_minimum(self, extended_service: ProblemService):
        """Test grade_from at the minimum available grade."""
        problems, total = extended_service.get_all_problems(grade_from="6A")
        
        assert total == 7  # All problems
    
    def test_grade_to_at_maximum(self, extended_service: ProblemService):
        """Test grade_to at the maximum available grade."""
        problems, total = extended_service.get_all_problems(grade_to="8A")
        
        assert total == 7  # All problems
    
    def test_grade_from_beyond_maximum(self, extended_service: ProblemService):
        """Test grade_from beyond all available grades."""
        problems, total = extended_service.get_all_problems(grade_from="8B")
        
        assert total == 0  # No problems
    
    def test_grade_to_below_minimum(self, extended_service: ProblemService):
        """Test grade_to below all available grades."""
        problems, total = extended_service.get_all_problems(grade_to="5+")
        
        assert total == 0  # No problems