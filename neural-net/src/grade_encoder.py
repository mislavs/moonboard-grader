"""
Grade Encoder Module

Converts Font climbing grades (e.g., "6B+", "7A") to integer labels and vice versa.
Supports Font grades from 5+ to 8C+.
"""

from typing import List

# Ordered list of all Font grades from easiest to hardest
_FONT_GRADES = [
    "5+",
    "6A", "6A+", "6B", "6B+", "6C", "6C+",
    "7A", "7A+", "7B", "7B+", "7C", "7C+",
    "8A", "8A+", "8B", "8B+", "8C", "8C+"
]

# Create mapping dictionaries for O(1) lookups
_GRADE_TO_LABEL = {grade: idx for idx, grade in enumerate(_FONT_GRADES)}
_LABEL_TO_GRADE = {idx: grade for idx, grade in enumerate(_FONT_GRADES)}


def encode_grade(grade_str: str) -> int:
    """
    Convert a Font grade string to an integer label.
    
    Args:
        grade_str: Font grade string (e.g., "6B+", "7a"). Case-insensitive.
        
    Returns:
        Integer label (0-indexed) corresponding to the grade.
        
    Raises:
        ValueError: If the grade string is invalid or not in the supported range.
        
    Examples:
        >>> encode_grade("6A")
        1
        >>> encode_grade("7b+")
        10
    """
    if not isinstance(grade_str, str):
        raise ValueError(f"Grade must be a string, got {type(grade_str).__name__}")
    
    # Normalize to uppercase and strip whitespace
    normalized = grade_str.strip().upper()
    
    if not normalized:
        raise ValueError("Grade string cannot be empty")
    
    if normalized not in _GRADE_TO_LABEL:
        raise ValueError(
            f"Invalid grade '{grade_str}'. Must be one of: {', '.join(_FONT_GRADES)}"
        )
    
    return _GRADE_TO_LABEL[normalized]


def decode_grade(label: int) -> str:
    """
    Convert an integer label to a Font grade string.
    
    Args:
        label: Integer label (0-indexed) to decode.
        
    Returns:
        Font grade string in uppercase (e.g., "6B+", "7A").
        
    Raises:
        ValueError: If the label is out of valid range.
        
    Examples:
        >>> decode_grade(1)
        '6A'
        >>> decode_grade(10)
        '7B+'
    """
    if not isinstance(label, int):
        raise ValueError(f"Label must be an integer, got {type(label).__name__}")
    
    if label not in _LABEL_TO_GRADE:
        raise ValueError(
            f"Invalid label {label}. Must be between 0 and {len(_FONT_GRADES) - 1}"
        )
    
    return _LABEL_TO_GRADE[label]


def get_all_grades() -> List[str]:
    """
    Get ordered list of all valid Font grades from easiest to hardest.
    
    Returns:
        List of grade strings in ascending difficulty order.
        
    Examples:
        >>> grades = get_all_grades()
        >>> grades[0]
        '5+'
        >>> grades[-1]
        '8C+'
    """
    return _FONT_GRADES.copy()


def get_num_grades() -> int:
    """
    Get total number of grade classes.
    
    Returns:
        Number of distinct grade levels supported.
        
    Examples:
        >>> get_num_grades()
        19
    """
    return len(_FONT_GRADES)

