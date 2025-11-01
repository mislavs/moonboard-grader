"""
Grid Tensor Builder for Moonboard Problems

This module converts a problem's moves array into a 3-channel tensor representation.
The tensor shape is (3, 18, 11) representing:
- Channel 0: Start holds
- Channel 1: Middle holds (neither start nor end)
- Channel 2: End holds

Each channel is an 18x11 grid (rows x columns) with binary values (1 if hold present, 0 otherwise).
"""

import numpy as np
from typing import List, Dict
from .position_parser import parse_position, ROWS, COLS


def create_grid_tensor(moves_list: List[Dict]) -> np.ndarray:
    """
    Convert a list of moves into a 3-channel grid tensor.
    
    Args:
        moves_list: List of move dictionaries, each with:
            - 'description': Position string (e.g., "F7")
            - 'isStart': Boolean indicating start hold
            - 'isEnd': Boolean indicating end hold
    
    Returns:
        numpy array of shape (3, 18, 11) where:
        - tensor[0] = start holds
        - tensor[1] = middle holds
        - tensor[2] = end holds
    
    Raises:
        ValueError: If moves_list is invalid or contains invalid positions
        
    Examples:
        >>> moves = [
        ...     {"description": "A1", "isStart": True, "isEnd": False},
        ...     {"description": "F7", "isStart": False, "isEnd": False},
        ...     {"description": "K18", "isStart": False, "isEnd": True}
        ... ]
        >>> tensor = create_grid_tensor(moves)
        >>> tensor.shape
        (3, 18, 11)
    """
    if not isinstance(moves_list, list):
        raise ValueError(f"moves_list must be a list, got {type(moves_list).__name__}")
    
    # Initialize empty 3-channel grid
    # Shape: (channels, rows, cols) = (3, 18, 11)
    grid = np.zeros((3, ROWS, COLS), dtype=np.float32)
    
    # Process each move
    for i, move in enumerate(moves_list):
        if not isinstance(move, dict):
            raise ValueError(f"Move at index {i} must be a dict, got {type(move).__name__}")
        
        # Extract required fields
        if 'description' not in move:
            raise ValueError(f"Move at index {i} missing 'description' field")
        if 'isStart' not in move:
            raise ValueError(f"Move at index {i} missing 'isStart' field")
        if 'isEnd' not in move:
            raise ValueError(f"Move at index {i} missing 'isEnd' field")
        
        position_str = move['description']
        is_start = move['isStart']
        is_end = move['isEnd']
        
        # Validate boolean fields
        if not isinstance(is_start, bool):
            raise ValueError(f"Move at index {i}: 'isStart' must be boolean, got {type(is_start).__name__}")
        if not isinstance(is_end, bool):
            raise ValueError(f"Move at index {i}: 'isEnd' must be boolean, got {type(is_end).__name__}")
        
        # Parse position
        try:
            row, col = parse_position(position_str)
        except ValueError as e:
            raise ValueError(f"Move at index {i}: Invalid position '{position_str}': {e}")
        
        # Determine which channel(s) to populate
        # A hold can theoretically be both start and end (single-move problem)
        if is_start:
            grid[0, row, col] = 1.0
        if is_end:
            grid[2, row, col] = 1.0
        if not is_start and not is_end:
            # Middle hold
            grid[1, row, col] = 1.0
    
    return grid


def get_channel_counts(tensor: np.ndarray) -> Dict[str, int]:
    """
    Count the number of holds in each channel of a grid tensor.
    
    Args:
        tensor: Grid tensor of shape (3, 18, 11)
    
    Returns:
        Dictionary with keys 'start', 'middle', 'end' and their counts
        
    Raises:
        ValueError: If tensor has invalid shape
        
    Examples:
        >>> tensor = create_grid_tensor(moves)
        >>> counts = get_channel_counts(tensor)
        >>> counts['start']
        1
    """
    if not isinstance(tensor, np.ndarray):
        raise ValueError(f"tensor must be a numpy array, got {type(tensor).__name__}")
    
    expected_shape = (3, ROWS, COLS)
    if tensor.shape != expected_shape:
        raise ValueError(f"tensor must have shape {expected_shape}, got {tensor.shape}")
    
    return {
        'start': int(np.sum(tensor[0])),
        'middle': int(np.sum(tensor[1])),
        'end': int(np.sum(tensor[2]))
    }


def tensor_to_moves(tensor: np.ndarray) -> Dict[str, List[str]]:
    """
    Convert a grid tensor back to lists of position strings (for debugging/visualization).
    
    Args:
        tensor: Grid tensor of shape (3, 18, 11)
    
    Returns:
        Dictionary with keys 'start', 'middle', 'end', each containing
        a list of position strings (e.g., ["A1", "F7"])
        
    Raises:
        ValueError: If tensor has invalid shape
        
    Examples:
        >>> tensor = create_grid_tensor(moves)
        >>> positions = tensor_to_moves(tensor)
        >>> 'F7' in positions['middle']
        True
    """
    if not isinstance(tensor, np.ndarray):
        raise ValueError(f"tensor must be a numpy array, got {type(tensor).__name__}")
    
    expected_shape = (3, ROWS, COLS)
    if tensor.shape != expected_shape:
        raise ValueError(f"tensor must have shape {expected_shape}, got {tensor.shape}")
    
    from .position_parser import COLUMNS
    
    result = {
        'start': [],
        'middle': [],
        'end': []
    }
    
    channel_names = ['start', 'middle', 'end']
    
    for channel_idx, channel_name in enumerate(channel_names):
        # Find all positions where the channel has a 1
        positions = np.argwhere(tensor[channel_idx] > 0.5)  # Use > 0.5 for float comparison
        
        for row, col in positions:
            # Convert back to position string (e.g., "F7")
            col_char = COLUMNS[col]
            row_num = row + 1  # Convert from 0-indexed to 1-indexed
            position_str = f"{col_char}{row_num}"
            result[channel_name].append(position_str)
    
    # Sort for consistent output
    for channel_name in channel_names:
        result[channel_name].sort()
    
    return result

