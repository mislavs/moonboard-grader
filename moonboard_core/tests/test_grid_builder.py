"""
Unit tests for grid_builder module
"""

import pytest
import numpy as np
from moonboard_core.grid_builder import (
    create_grid_tensor,
    get_channel_counts,
    tensor_to_moves,
)
from moonboard_core.position_parser import ROWS, COLS


class TestCreateGridTensor:
    """Test create_grid_tensor function"""
    
    def test_empty_moves_list(self):
        """Test with empty moves list"""
        tensor = create_grid_tensor([])
        assert tensor.shape == (3, ROWS, COLS)
        assert np.sum(tensor) == 0  # All zeros
    
    def test_single_start_hold(self):
        """Test with single start hold"""
        moves = [
            {"description": "F7", "isStart": True, "isEnd": False}
        ]
        tensor = create_grid_tensor(moves)
        
        assert tensor.shape == (3, ROWS, COLS)
        assert tensor[0, 6, 5] == 1.0  # Start channel, row 7, col F
        assert np.sum(tensor[0]) == 1.0
        assert np.sum(tensor[1]) == 0.0
        assert np.sum(tensor[2]) == 0.0
    
    def test_single_end_hold(self):
        """Test with single end hold"""
        moves = [
            {"description": "K18", "isStart": False, "isEnd": True}
        ]
        tensor = create_grid_tensor(moves)
        
        assert tensor.shape == (3, ROWS, COLS)
        assert tensor[2, 17, 10] == 1.0  # End channel, row 18, col K
        assert np.sum(tensor[0]) == 0.0
        assert np.sum(tensor[1]) == 0.0
        assert np.sum(tensor[2]) == 1.0
    
    def test_single_middle_hold(self):
        """Test with single middle hold"""
        moves = [
            {"description": "D10", "isStart": False, "isEnd": False}
        ]
        tensor = create_grid_tensor(moves)
        
        assert tensor.shape == (3, ROWS, COLS)
        assert tensor[1, 9, 3] == 1.0  # Middle channel, row 10, col D
        assert np.sum(tensor[0]) == 0.0
        assert np.sum(tensor[1]) == 1.0
        assert np.sum(tensor[2]) == 0.0
    
    def test_complete_problem(self):
        """Test with a complete problem (start, middle, end holds)"""
        moves = [
            {"description": "A1", "isStart": True, "isEnd": False},
            {"description": "F7", "isStart": False, "isEnd": False},
            {"description": "G10", "isStart": False, "isEnd": False},
            {"description": "K18", "isStart": False, "isEnd": True}
        ]
        tensor = create_grid_tensor(moves)
        
        assert tensor.shape == (3, ROWS, COLS)
        # Check start hold
        assert tensor[0, 0, 0] == 1.0  # A1
        assert np.sum(tensor[0]) == 1.0
        
        # Check middle holds
        assert tensor[1, 6, 5] == 1.0  # F7
        assert tensor[1, 9, 6] == 1.0  # G10
        assert np.sum(tensor[1]) == 2.0
        
        # Check end hold
        assert tensor[2, 17, 10] == 1.0  # K18
        assert np.sum(tensor[2]) == 1.0
    
    def test_both_start_and_end(self):
        """Test a single-move problem where a hold is both start and end"""
        moves = [
            {"description": "F7", "isStart": True, "isEnd": True}
        ]
        tensor = create_grid_tensor(moves)
        
        assert tensor.shape == (3, ROWS, COLS)
        # Should be in both start and end channels
        assert tensor[0, 6, 5] == 1.0
        assert tensor[2, 6, 5] == 1.0
        assert tensor[1, 6, 5] == 0.0
        assert np.sum(tensor) == 2.0
    
    def test_invalid_moves_list_type(self):
        """Test that non-list raises error"""
        with pytest.raises(ValueError, match="moves_list must be a list"):
            create_grid_tensor("not a list")
    
    def test_invalid_move_type(self):
        """Test that non-dict move raises error"""
        with pytest.raises(ValueError, match="Move at index 0 must be a dict"):
            create_grid_tensor(["not a dict"])
    
    def test_missing_description_field(self):
        """Test that missing description raises error"""
        with pytest.raises(ValueError, match="missing 'description' field"):
            create_grid_tensor([
                {"isStart": True, "isEnd": False}
            ])
    
    def test_invalid_position_format(self):
        """Test that invalid position raises error"""
        with pytest.raises(ValueError, match="Invalid position"):
            create_grid_tensor([
                {"description": "Z99", "isStart": True, "isEnd": False}
            ])


class TestGetChannelCounts:
    """Test get_channel_counts function"""
    
    def test_empty_tensor(self):
        """Test counts for empty tensor"""
        tensor = np.zeros((3, ROWS, COLS), dtype=np.float32)
        counts = get_channel_counts(tensor)
        
        assert counts['start'] == 0
        assert counts['middle'] == 0
        assert counts['end'] == 0
    
    def test_simple_counts(self):
        """Test counts for simple problem"""
        moves = [
            {"description": "A1", "isStart": True, "isEnd": False},
            {"description": "F7", "isStart": False, "isEnd": False},
            {"description": "G10", "isStart": False, "isEnd": False},
            {"description": "K18", "isStart": False, "isEnd": True}
        ]
        tensor = create_grid_tensor(moves)
        counts = get_channel_counts(tensor)
        
        assert counts['start'] == 1
        assert counts['middle'] == 2
        assert counts['end'] == 1
    
    def test_invalid_tensor_type(self):
        """Test that non-array raises error"""
        with pytest.raises(ValueError, match="tensor must be a numpy array"):
            get_channel_counts([[1, 2, 3]])
    
    def test_invalid_tensor_shape(self):
        """Test that wrong shape raises error"""
        tensor = np.zeros((2, ROWS, COLS))
        with pytest.raises(ValueError, match="tensor must have shape"):
            get_channel_counts(tensor)


class TestTensorToMoves:
    """Test tensor_to_moves function"""
    
    def test_empty_tensor(self):
        """Test conversion of empty tensor"""
        tensor = np.zeros((3, ROWS, COLS), dtype=np.float32)
        positions = tensor_to_moves(tensor)
        
        assert positions['start'] == []
        assert positions['middle'] == []
        assert positions['end'] == []
    
    def test_single_start_hold(self):
        """Test conversion with single start hold"""
        moves = [
            {"description": "F7", "isStart": True, "isEnd": False}
        ]
        tensor = create_grid_tensor(moves)
        positions = tensor_to_moves(tensor)
        
        assert positions['start'] == ['F7']
        assert positions['middle'] == []
        assert positions['end'] == []
    
    def test_complete_problem(self):
        """Test conversion of complete problem"""
        moves = [
            {"description": "A1", "isStart": True, "isEnd": False},
            {"description": "F7", "isStart": False, "isEnd": False},
            {"description": "G10", "isStart": False, "isEnd": False},
            {"description": "K18", "isStart": False, "isEnd": True}
        ]
        tensor = create_grid_tensor(moves)
        positions = tensor_to_moves(tensor)
        
        assert positions['start'] == ['A1']
        assert set(positions['middle']) == {'F7', 'G10'}
        assert positions['end'] == ['K18']
    
    def test_round_trip_conversion(self):
        """Test that tensor -> moves -> tensor preserves data"""
        original_moves = [
            {"description": "A1", "isStart": True, "isEnd": False},
            {"description": "F7", "isStart": False, "isEnd": False},
            {"description": "G10", "isStart": False, "isEnd": False},
            {"description": "K18", "isStart": False, "isEnd": True}
        ]
        tensor1 = create_grid_tensor(original_moves)
        positions = tensor_to_moves(tensor1)
        
        # Reconstruct moves from positions
        reconstructed_moves = []
        for pos in positions['start']:
            reconstructed_moves.append({"description": pos, "isStart": True, "isEnd": False})
        for pos in positions['middle']:
            reconstructed_moves.append({"description": pos, "isStart": False, "isEnd": False})
        for pos in positions['end']:
            reconstructed_moves.append({"description": pos, "isStart": False, "isEnd": True})
        
        tensor2 = create_grid_tensor(reconstructed_moves)
        
        # Tensors should be identical
        assert np.array_equal(tensor1, tensor2)

