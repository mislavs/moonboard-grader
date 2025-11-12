"""Tests for grid_to_moves module."""

import pytest
import numpy as np
from moonboard_core.grid_to_moves import grid_to_moves, validate_moves
from moonboard_core.grid_builder import create_grid_tensor


class TestGridToMoves:
    """Test grid_to_moves function."""
    
    def test_simple_problem(self):
        """Test converting a simple 3-hold problem."""
        # Create a simple problem: A1 (start), F7 (middle), K18 (end)
        tensor = np.zeros((3, 18, 11), dtype=np.float32)
        tensor[0, 0, 0] = 1.0  # A1 start (row 1, col A)
        tensor[1, 6, 5] = 1.0  # F7 middle (row 7, col F)
        tensor[2, 17, 10] = 1.0  # K18 end (row 18, col K)
        
        moves = grid_to_moves(tensor)
        
        assert len(moves) == 3
        
        # Check A1 start
        assert any(m['description'] == 'A1' and m['isStart'] and not m['isEnd'] 
                  for m in moves)
        
        # Check F7 middle
        assert any(m['description'] == 'F7' and not m['isStart'] and not m['isEnd'] 
                  for m in moves)
        
        # Check K18 end
        assert any(m['description'] == 'K18' and not m['isStart'] and m['isEnd'] 
                  for m in moves)
    
    def test_multiple_start_holds(self):
        """Test problem with multiple start holds."""
        tensor = np.zeros((3, 18, 11), dtype=np.float32)
        tensor[0, 0, 0] = 1.0  # A1 start
        tensor[0, 0, 1] = 1.0  # B1 start
        tensor[1, 5, 5] = 1.0  # F6 middle
        tensor[2, 10, 10] = 1.0  # K11 end
        
        moves = grid_to_moves(tensor)
        
        assert len(moves) == 4
        start_count = sum(1 for m in moves if m['isStart'])
        assert start_count == 2
    
    def test_threshold_filtering(self):
        """Test that threshold parameter filters values correctly."""
        tensor = np.zeros((3, 18, 11), dtype=np.float32)
        tensor[0, 0, 0] = 1.0  # A1 start (above threshold)
        tensor[0, 1, 0] = 0.3  # A2 start (below default threshold)
        tensor[2, 17, 10] = 0.8  # K18 end (above threshold)
        
        # With default threshold (0.5)
        moves = grid_to_moves(tensor, threshold=0.5)
        assert len(moves) == 2
        
        # With lower threshold
        moves = grid_to_moves(tensor, threshold=0.2)
        assert len(moves) == 3
    
    def test_hold_both_start_and_end(self):
        """Test a hold that is both start and end (single-move problem)."""
        tensor = np.zeros((3, 18, 11), dtype=np.float32)
        tensor[0, 0, 0] = 1.0  # A1 start
        tensor[2, 0, 0] = 1.0  # A1 also end
        
        moves = grid_to_moves(tensor)
        
        assert len(moves) == 1
        assert moves[0]['description'] == 'A1'
        assert moves[0]['isStart'] is True
        assert moves[0]['isEnd'] is True
    
    def test_sorted_output(self):
        """Test that moves are sorted by row then column."""
        tensor = np.zeros((3, 18, 11), dtype=np.float32)
        tensor[0, 5, 5] = 1.0  # F6
        tensor[1, 2, 0] = 1.0  # A3
        tensor[2, 5, 0] = 1.0  # A6
        
        moves = grid_to_moves(tensor)
        
        # Should be sorted: A3, A6, F6
        assert moves[0]['description'] == 'A3'
        assert moves[1]['description'] == 'A6'
        assert moves[2]['description'] == 'F6'
    
    def test_roundtrip_conversion(self):
        """Test that grid -> moves -> grid preserves data."""
        original_moves = [
            {"description": "A1", "isStart": True, "isEnd": False},
            {"description": "F7", "isStart": False, "isEnd": False},
            {"description": "G8", "isStart": False, "isEnd": False},
            {"description": "K18", "isStart": False, "isEnd": True}
        ]
        
        # Convert to grid
        grid = create_grid_tensor(original_moves)
        
        # Convert back to moves
        recovered_moves = grid_to_moves(grid)
        
        # Should have same number of holds
        assert len(recovered_moves) == len(original_moves)
        
        # All original positions should be present
        original_positions = set(m['description'] for m in original_moves)
        recovered_positions = set(m['description'] for m in recovered_moves)
        assert original_positions == recovered_positions
    
    def test_empty_grid(self):
        """Test converting an empty grid."""
        tensor = np.zeros((3, 18, 11), dtype=np.float32)
        moves = grid_to_moves(tensor)
        assert len(moves) == 0
    
    def test_invalid_tensor_shape(self):
        """Test that invalid tensor shape raises ValueError."""
        tensor = np.zeros((2, 18, 11))  # Wrong number of channels
        
        with pytest.raises(ValueError, match="must have shape"):
            grid_to_moves(tensor)
    
    def test_invalid_tensor_type(self):
        """Test that non-array input raises ValueError."""
        with pytest.raises(ValueError, match="must be a numpy array"):
            grid_to_moves([[1, 2, 3]])


class TestValidateMoves:
    """Test validate_moves function."""
    
    def test_valid_problem(self):
        """Test validation of a valid problem."""
        moves = [
            {"description": "A1", "isStart": True, "isEnd": False},
            {"description": "F7", "isStart": False, "isEnd": False},
            {"description": "K18", "isStart": False, "isEnd": True}
        ]
        
        result = validate_moves(moves)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert result['stats']['total_holds'] == 3
        assert result['stats']['start_holds'] == 1
        assert result['stats']['middle_holds'] == 1
        assert result['stats']['end_holds'] == 1
    
    def test_no_holds(self):
        """Test validation of problem with no holds."""
        result = validate_moves([])
        
        assert result['valid'] is False
        assert "no holds" in result['errors'][0].lower()
    
    def test_no_start_holds(self):
        """Test validation of problem with no start holds."""
        moves = [
            {"description": "F7", "isStart": False, "isEnd": False},
            {"description": "K18", "isStart": False, "isEnd": True}
        ]
        
        result = validate_moves(moves)
        
        assert result['valid'] is False
        assert any("start hold" in err.lower() for err in result['errors'])
    
    def test_no_end_holds(self):
        """Test validation of problem with no end holds."""
        moves = [
            {"description": "A1", "isStart": True, "isEnd": False},
            {"description": "F7", "isStart": False, "isEnd": False}
        ]
        
        result = validate_moves(moves)
        
        assert result['valid'] is False
        assert any("end hold" in err.lower() for err in result['errors'])
    
    def test_warnings_few_holds(self):
        """Test that problems with very few holds generate warnings."""
        moves = [
            {"description": "A1", "isStart": True, "isEnd": False},
            {"description": "K18", "isStart": False, "isEnd": True}
        ]
        
        result = validate_moves(moves)
        
        assert result['valid'] is True  # Still valid
        assert len(result['warnings']) > 0
        assert any("few holds" in warn.lower() for warn in result['warnings'])
    
    def test_warnings_many_holds(self):
        """Test that problems with many holds generate warnings."""
        # Create problem with 21 holds
        moves = [
            {"description": f"A{i}", "isStart": i == 1, "isEnd": i == 18}
            for i in range(1, 22)
        ]
        
        result = validate_moves(moves)
        
        assert result['valid'] is True
        assert len(result['warnings']) > 0
        assert any("many holds" in warn.lower() for warn in result['warnings'])
    
    def test_warnings_many_start_holds(self):
        """Test warning for many start holds."""
        moves = [
            {"description": f"A{i}", "isStart": i <= 5, "isEnd": i == 10}
            for i in range(1, 11)
        ]
        
        result = validate_moves(moves)
        
        assert len(result['warnings']) > 0
        assert any("start" in warn.lower() for warn in result['warnings'])
        assert result['stats']['start_holds'] == 5
    
    def test_multiple_errors(self):
        """Test that multiple validation errors are reported."""
        moves = []  # No holds at all
        
        result = validate_moves(moves)
        
        # Should have errors for: no holds, no start, no end
        assert len(result['errors']) == 3
        assert result['valid'] is False

