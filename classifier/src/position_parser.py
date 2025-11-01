"""
Position Parser for Moonboard Hold Positions

This module parses hold position strings (like "F7") into row/column indices.
The moonboard has:
- Columns A-K (11 columns total, indexed 0-10)
- Rows 1-18 (18 rows total, indexed 0-17)
"""

# Constants
ROWS = 18
COLS = 11
COLUMNS = 'ABCDEFGHIJK'


def parse_position(position_str: str) -> tuple[int, int]:
    """
    Parse a hold position string into (row, column) indices.
    
    Args:
        position_str: Position string like "F7", "A1", "K18"
        
    Returns:
        Tuple of (row_index, col_index) where:
        - row_index is 0-17 (corresponding to rows 1-18)
        - col_index is 0-10 (corresponding to columns A-K)
        
    Raises:
        ValueError: If position string is invalid
        
    Examples:
        >>> parse_position("F7")
        (6, 5)
        >>> parse_position("A1")
        (0, 0)
        >>> parse_position("K18")
        (17, 10)
    """
    if not isinstance(position_str, str):
        raise ValueError(f"Position must be a string, got {type(position_str).__name__}")
    
    # Normalize: strip whitespace and convert to uppercase
    position_str = position_str.strip().upper()
    
    if len(position_str) < 2:
        raise ValueError(f"Invalid position format: '{position_str}'. Expected format like 'F7'")
    
    # Extract column (first character) and row (remaining characters)
    col_char = position_str[0]
    row_str = position_str[1:]
    
    # Validate column
    if col_char not in COLUMNS:
        raise ValueError(
            f"Invalid column '{col_char}'. Must be one of {COLUMNS}"
        )
    
    # Validate row
    try:
        row_num = int(row_str)
    except ValueError:
        raise ValueError(
            f"Invalid row '{row_str}'. Must be a number between 1 and {ROWS}"
        )
    
    if row_num < 1 or row_num > ROWS:
        raise ValueError(
            f"Invalid row {row_num}. Must be between 1 and {ROWS}"
        )
    
    # Convert to 0-indexed
    col_index = COLUMNS.index(col_char)
    row_index = row_num - 1
    
    return (row_index, col_index)


def validate_position(position_str: str) -> bool:
    """
    Validate whether a position string is valid.
    
    Args:
        position_str: Position string to validate
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> validate_position("F7")
        True
        >>> validate_position("Z1")
        False
        >>> validate_position("A20")
        False
    """
    try:
        parse_position(position_str)
        return True
    except (ValueError, TypeError, AttributeError):
        return False

