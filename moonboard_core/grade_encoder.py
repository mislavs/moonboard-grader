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


def remap_label(label: int, offset: int) -> int:
    """
    Remap a grade label by subtracting an offset.
    
    Used when filtering to a subset of grades - converts original grade indices
    to consecutive model indices starting from 0.
    
    Args:
        label: Original grade label (e.g., 2 for 6A+)
        offset: Offset to subtract (typically the min_grade_index)
    
    Returns:
        Remapped label (e.g., 2 - 2 = 0 for 6A+ in filtered model)
        
    Examples:
        >>> remap_label(2, 2)  # 6A+ becomes 0
        0
        >>> remap_label(5, 2)  # 6C becomes 3
        3
    """
    if not isinstance(label, int):
        raise ValueError(f"label must be an integer, got {type(label).__name__}")
    if not isinstance(offset, int):
        raise ValueError(f"offset must be an integer, got {type(offset).__name__}")
    
    remapped = label - offset
    if remapped < 0:
        raise ValueError(f"Remapped label would be negative: {label} - {offset} = {remapped}")
    
    return remapped


def unmap_label(label: int, offset: int) -> int:
    """
    Unmap a grade label by adding an offset.
    
    Used during inference with filtered models - converts model prediction
    indices back to original grade indices.
    
    Args:
        label: Model prediction label (e.g., 0 from filtered model)
        offset: Offset to add (typically the min_grade_index)
    
    Returns:
        Original grade index (e.g., 0 + 2 = 2 for 6A+)
        
    Examples:
        >>> unmap_label(0, 2)  # Model outputs 0, maps to 6A+ (index 2)
        2
        >>> unmap_label(3, 2)  # Model outputs 3, maps to 6C (index 5)
        5
    """
    if not isinstance(label, int):
        raise ValueError(f"label must be an integer, got {type(label).__name__}")
    if not isinstance(offset, int):
        raise ValueError(f"offset must be an integer, got {type(offset).__name__}")
    
    unmapped = label + offset
    if unmapped >= len(_FONT_GRADES):
        raise ValueError(
            f"Unmapped label out of range: {label} + {offset} = {unmapped} (max: {len(_FONT_GRADES) - 1})"
        )
    
    return unmapped


def get_filtered_grade_names(min_grade_index: int, max_grade_index: int) -> List[str]:
    """
    Get a subset of grade names for a filtered range.
    
    Args:
        min_grade_index: Minimum grade index (inclusive)
        max_grade_index: Maximum grade index (inclusive)
    
    Returns:
        List of grade names in the specified range
        
    Raises:
        ValueError: If indices are invalid
        
    Examples:
        >>> get_filtered_grade_names(2, 4)  # 6A+ through 6B+
        ['6A+', '6B', '6B+']
        >>> get_filtered_grade_names(2, 12)  # 6A+ through 7C
        ['6A+', '6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C']
    """
    if not isinstance(min_grade_index, int) or not isinstance(max_grade_index, int):
        raise ValueError("min_grade_index and max_grade_index must be integers")
    
    if min_grade_index < 0:
        raise ValueError(f"min_grade_index must be >= 0, got {min_grade_index}")
    
    if max_grade_index >= len(_FONT_GRADES):
        raise ValueError(
            f"max_grade_index must be < {len(_FONT_GRADES)}, got {max_grade_index}"
        )
    
    if max_grade_index < min_grade_index:
        raise ValueError(
            f"max_grade_index ({max_grade_index}) must be >= min_grade_index ({min_grade_index})"
        )
    
    return _FONT_GRADES[min_grade_index:max_grade_index + 1]
