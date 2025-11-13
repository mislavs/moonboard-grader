"""
Grid to Moves Converter for Moonboard Problems

This module converts a 3-channel tensor grid back into a moves array format
that matches the original data structure.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from .position_parser import COLUMNS, ROWS, COLS

# Validation thresholds
MAX_HOLDS = 20
MAX_START_HOLDS = 4
MAX_END_HOLDS = 2
MIN_MIDDLE_HOLDS = 3


def _move_sort_key(move: Dict) -> Tuple[int, int]:
    """
    Generate sort key for a move dictionary.
    
    Sorts by row (ascending), then by column (ascending).
    
    Args:
        move: Move dictionary with 'description' field
        
    Returns:
        Tuple of (row_num, col_idx) for sorting
    """
    pos = move['description']
    col_char = pos[0]
    row_num = int(pos[1:])
    col_idx = COLUMNS.index(col_char)
    return (row_num, col_idx)


def grid_to_moves(tensor: np.ndarray, threshold: float = 0.5) -> List[Dict]:
    """
    Convert a 3-channel grid tensor back to a moves array.
    
    This is the inverse operation of create_grid_tensor from grid_builder.py.
    Converts the binary grid representation back into the list of move dictionaries
    that the original dataset uses.
    
    Args:
        tensor: Grid tensor of shape (3, 18, 11) where:
            - Channel 0: Start holds
            - Channel 1: Middle holds  
            - Channel 2: End holds
        threshold: Threshold for considering a value as "on" (default: 0.5)
            Values above threshold are treated as 1, below as 0.
    
    Returns:
        List of move dictionaries with format:
        [
            {"description": "A5", "isStart": True, "isEnd": False},
            {"description": "F7", "isStart": False, "isEnd": False},
            {"description": "K18", "isStart": False, "isEnd": True}
        ]
        
    Raises:
        ValueError: If tensor has invalid shape
        
    Examples:
        >>> tensor = np.zeros((3, 18, 11))
        >>> tensor[0, 0, 0] = 1.0  # A1 start
        >>> tensor[1, 6, 5] = 1.0  # F7 middle
        >>> tensor[2, 17, 10] = 1.0  # K18 end
        >>> moves = grid_to_moves(tensor)
        >>> len(moves)
        3
        >>> moves[0]['description']
        'A1'
        >>> moves[0]['isStart']
        True
    """
    if not isinstance(tensor, np.ndarray):
        raise ValueError(f"tensor must be a numpy array, got {type(tensor).__name__}")
    
    expected_shape = (3, ROWS, COLS)
    if tensor.shape != expected_shape:
        raise ValueError(f"tensor must have shape {expected_shape}, got {tensor.shape}")
    
    moves = []
    
    # Dictionary to track which holds we've already added
    # Key: position string (e.g., "F7"), Value: move dict
    position_map = {}
    
    # Process each channel
    channel_names = ['start', 'middle', 'end']
    
    for channel_idx, channel_name in enumerate(channel_names):
        # Find all positions where the channel has a value above threshold
        positions = np.argwhere(tensor[channel_idx] > threshold)
        
        for row, col in positions:
            # Convert back to position string (e.g., "F7")
            col_char = COLUMNS[col]
            row_num = row + 1  # Convert from 0-indexed to 1-indexed
            position_str = f"{col_char}{row_num}"
            
            # Check if we've already added this position
            if position_str in position_map:
                # Update existing move
                move = position_map[position_str]
                if channel_name == 'start':
                    move['isStart'] = True
                elif channel_name == 'end':
                    move['isEnd'] = True
                # Middle doesn't need updating - it's the default state
            else:
                # Create new move
                move = {
                    'description': position_str,
                    'isStart': channel_name == 'start',
                    'isEnd': channel_name == 'end'
                }
                position_map[position_str] = move
                moves.append(move)
    
    # Sort moves for consistent ordering (by row, then by column)
    moves.sort(key=_move_sort_key)
    
    return moves


def validate_moves(moves: List[Dict]) -> Dict[str, Any]:
    """
    Validate that a moves array meets basic requirements for a valid problem.
    
    Args:
        moves: List of move dictionaries
        
    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'errors': List[str],
            'warnings': List[str],
            'stats': {
                'total_holds': int,
                'start_holds': int,
                'middle_holds': int,
                'end_holds': int
            }
        }
        
    Examples:
        >>> moves = [
        ...     {"description": "A1", "isStart": True, "isEnd": False},
        ...     {"description": "F7", "isStart": False, "isEnd": False},
        ...     {"description": "F10", "isStart": False, "isEnd": False},
        ...     {"description": "F12", "isStart": False, "isEnd": False},
        ...     {"description": "K18", "isStart": False, "isEnd": True}
        ... ]
        >>> result = validate_moves(moves)
        >>> result['valid']
        True
    """
    errors = []
    warnings = []
    
    # Count holds by type
    start_holds = sum(1 for m in moves if m.get('isStart', False))
    end_holds = sum(1 for m in moves if m.get('isEnd', False))
    middle_holds = sum(1 for m in moves if not m.get('isStart', False) and not m.get('isEnd', False))
    total_holds = len(moves)
    
    # Validation rules
    if start_holds == 0:
        errors.append("Problem must have at least one start hold")
    
    if end_holds == 0:
        errors.append("Problem must have at least one end hold")
    
    if middle_holds < MIN_MIDDLE_HOLDS:
        errors.append(f"Problem must have at least {MIN_MIDDLE_HOLDS} middle holds (found {middle_holds})")
    
    # Warnings for unusual problems
    if total_holds > MAX_HOLDS:
        warnings.append(f"Problem has many holds ({total_holds})")
    
    if start_holds > MAX_START_HOLDS:
        warnings.append(f"Problem has many start holds ({start_holds})")
    
    if end_holds > MAX_END_HOLDS:
        warnings.append(f"Problem has many end holds ({end_holds})")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'stats': {
            'total_holds': total_holds,
            'start_holds': start_holds,
            'middle_holds': middle_holds,
            'end_holds': end_holds
        }
    }

