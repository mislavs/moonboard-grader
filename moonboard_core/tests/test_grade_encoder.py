"""
Unit tests for grade_encoder module.

Tests all functionality of Font grade encoding/decoding including:
- Valid grade conversions
- Invalid input handling
- Bidirectional conversion consistency
- Grade ordering
- Edge cases
"""

import pytest
from moonboard_core.grade_encoder import (
    encode_grade,
    decode_grade,
    get_all_grades,
    get_num_grades
)


class TestEncodeGrade:
    """Tests for encode_grade function."""
    
    def test_encode_valid_grades(self):
        """Test encoding all valid Font grades."""
        assert encode_grade("5+") == 0
        assert encode_grade("6A") == 1
        assert encode_grade("6A+") == 2
        assert encode_grade("6B") == 3
        assert encode_grade("6B+") == 4
        assert encode_grade("6C") == 5
        assert encode_grade("6C+") == 6
        assert encode_grade("7A") == 7
        assert encode_grade("7A+") == 8
        assert encode_grade("7B") == 9
        assert encode_grade("7B+") == 10
        assert encode_grade("7C") == 11
        assert encode_grade("7C+") == 12
        assert encode_grade("8A") == 13
        assert encode_grade("8A+") == 14
        assert encode_grade("8B") == 15
        assert encode_grade("8B+") == 16
        assert encode_grade("8C") == 17
        assert encode_grade("8C+") == 18
    
    def test_encode_case_insensitive(self):
        """Test that encoding is case-insensitive."""
        assert encode_grade("6b+") == encode_grade("6B+")
        assert encode_grade("7a") == encode_grade("7A")
        assert encode_grade("8C+") == encode_grade("8c+")
        assert encode_grade("5+") == encode_grade("5+")
    
    def test_encode_with_whitespace(self):
        """Test that leading/trailing whitespace is handled."""
        assert encode_grade(" 6B+ ") == encode_grade("6B+")
        assert encode_grade("\t7A\n") == encode_grade("7A")
        assert encode_grade("  8C+  ") == encode_grade("8C+")
    
    def test_encode_invalid_grade(self):
        """Test that invalid grades raise ValueError."""
        with pytest.raises(ValueError, match="Invalid grade"):
            encode_grade("9A")  # Too high
        
        with pytest.raises(ValueError, match="Invalid grade"):
            encode_grade("4A")  # Too low
        
        with pytest.raises(ValueError, match="Invalid grade"):
            encode_grade("6D")  # Invalid letter
        
        with pytest.raises(ValueError, match="Invalid grade"):
            encode_grade("6A++")  # Invalid format
        
        with pytest.raises(ValueError, match="Invalid grade"):
            encode_grade("V5")  # Wrong grading system
    
    def test_encode_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            encode_grade("")
        
        with pytest.raises(ValueError, match="cannot be empty"):
            encode_grade("   ")
    
    def test_encode_wrong_type(self):
        """Test that non-string input raises ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            encode_grade(6)
        
        with pytest.raises(ValueError, match="must be a string"):
            encode_grade(None)
        
        with pytest.raises(ValueError, match="must be a string"):
            encode_grade(["6A"])


class TestDecodeGrade:
    """Tests for decode_grade function."""
    
    def test_decode_valid_labels(self):
        """Test decoding all valid labels."""
        assert decode_grade(0) == "5+"
        assert decode_grade(1) == "6A"
        assert decode_grade(2) == "6A+"
        assert decode_grade(5) == "6C"
        assert decode_grade(10) == "7B+"
        assert decode_grade(15) == "8B"
        assert decode_grade(18) == "8C+"
    
    def test_decode_invalid_label(self):
        """Test that invalid labels raise ValueError."""
        with pytest.raises(ValueError, match="Invalid label"):
            decode_grade(-1)
        
        with pytest.raises(ValueError, match="Invalid label"):
            decode_grade(19)  # Out of range
        
        with pytest.raises(ValueError, match="Invalid label"):
            decode_grade(100)
    
    def test_decode_wrong_type(self):
        """Test that non-integer input raises ValueError."""
        with pytest.raises(ValueError, match="must be an integer"):
            decode_grade("5")
        
        with pytest.raises(ValueError, match="must be an integer"):
            decode_grade(5.0)
        
        with pytest.raises(ValueError, match="must be an integer"):
            decode_grade(None)


class TestBidirectionalConversion:
    """Tests for round-trip encode/decode consistency."""
    
    def test_encode_decode_roundtrip(self):
        """Test that encode(decode(x)) == x for all valid labels."""
        for label in range(get_num_grades()):
            grade_str = decode_grade(label)
            encoded = encode_grade(grade_str)
            assert encoded == label, f"Round-trip failed for label {label}"
    
    def test_decode_encode_roundtrip(self):
        """Test that decode(encode(x)) == x.upper() for all valid grades."""
        for grade in get_all_grades():
            label = encode_grade(grade)
            decoded = decode_grade(label)
            assert decoded == grade.upper(), f"Round-trip failed for grade {grade}"
    
    def test_case_normalization(self):
        """Test that lowercase input is normalized to uppercase output."""
        lowercase_grade = "6b+"
        label = encode_grade(lowercase_grade)
        decoded = decode_grade(label)
        assert decoded == "6B+"  # Should be uppercase


class TestGetAllGrades:
    """Tests for get_all_grades function."""
    
    def test_returns_list(self):
        """Test that get_all_grades returns a list."""
        grades = get_all_grades()
        assert isinstance(grades, list)
    
    def test_correct_length(self):
        """Test that the list has correct length."""
        grades = get_all_grades()
        assert len(grades) == 19
    
    def test_correct_ordering(self):
        """Test that grades are in ascending difficulty order."""
        grades = get_all_grades()
        assert grades[0] == "5+"
        assert grades[-1] == "8C+"
        assert grades[7] == "7A"
    
    def test_returns_copy(self):
        """Test that get_all_grades returns a copy, not reference."""
        grades1 = get_all_grades()
        grades2 = get_all_grades()
        assert grades1 is not grades2  # Different objects
        assert grades1 == grades2  # Same content
        
        # Modifying one shouldn't affect the other
        grades1.append("9A")
        assert len(grades2) == 19
    
    def test_all_unique(self):
        """Test that all grades are unique."""
        grades = get_all_grades()
        assert len(grades) == len(set(grades))


class TestGetNumGrades:
    """Tests for get_num_grades function."""
    
    def test_returns_int(self):
        """Test that get_num_grades returns an integer."""
        num = get_num_grades()
        assert isinstance(num, int)
    
    def test_correct_count(self):
        """Test that the count is correct."""
        assert get_num_grades() == 19
    
    def test_consistency_with_get_all_grades(self):
        """Test that get_num_grades matches len(get_all_grades())."""
        assert get_num_grades() == len(get_all_grades())


class TestGradeOrdering:
    """Tests to verify grade difficulty ordering."""
    
    def test_labels_increase_with_difficulty(self):
        """Test that harder grades have higher labels."""
        easy_label = encode_grade("6A")
        medium_label = encode_grade("7B")
        hard_label = encode_grade("8C+")
        
        assert easy_label < medium_label < hard_label
    
    def test_plus_grades_harder_than_base(self):
        """Test that plus grades are harder than their base grades."""
        assert encode_grade("6A") < encode_grade("6A+")
        assert encode_grade("7B") < encode_grade("7B+")
        assert encode_grade("8C") < encode_grade("8C+")
    
    def test_sequential_grades_differ_by_one(self):
        """Test that consecutive grades differ by exactly 1."""
        grades = get_all_grades()
        for i in range(len(grades) - 1):
            label1 = encode_grade(grades[i])
            label2 = encode_grade(grades[i + 1])
            assert label2 - label1 == 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_minimum_grade(self):
        """Test the easiest grade (5+)."""
        assert encode_grade("5+") == 0
        assert decode_grade(0) == "5+"
    
    def test_maximum_grade(self):
        """Test the hardest grade (8C+)."""
        assert encode_grade("8C+") == 18
        assert decode_grade(18) == "8C+"
    
    def test_boundary_labels(self):
        """Test labels at boundaries."""
        # First valid label
        assert decode_grade(0) == "5+"
        
        # Last valid label
        assert decode_grade(get_num_grades() - 1) == "8C+"

