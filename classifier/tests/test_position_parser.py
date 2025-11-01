"""
Unit tests for position_parser module
"""

import pytest
from src.position_parser import (
    parse_position,
    validate_position,
    ROWS,
    COLS,
    COLUMNS
)


class TestConstants:
    """Test that constants are defined correctly"""
    
    def test_rows_constant(self):
        assert ROWS == 18
    
    def test_cols_constant(self):
        assert COLS == 11
    
    def test_columns_constant(self):
        assert COLUMNS == 'ABCDEFGHIJK'
        assert len(COLUMNS) == 11


class TestParsePosition:
    """Test parse_position function"""
    
    def test_simple_position(self):
        """Test parsing a simple position"""
        row, col = parse_position("F7")
        assert row == 6  # Row 7 → index 6
        assert col == 5  # Column F → index 5
    
    def test_bottom_left_corner(self):
        """Test bottom-left corner (A1)"""
        row, col = parse_position("A1")
        assert row == 0
        assert col == 0
    
    def test_bottom_right_corner(self):
        """Test bottom-right corner (K1)"""
        row, col = parse_position("K1")
        assert row == 0
        assert col == 10
    
    def test_top_left_corner(self):
        """Test top-left corner (A18)"""
        row, col = parse_position("A18")
        assert row == 17
        assert col == 0
    
    def test_top_right_corner(self):
        """Test top-right corner (K18)"""
        row, col = parse_position("K18")
        assert row == 17
        assert col == 10
    
    def test_all_columns(self):
        """Test that all columns A-K parse correctly"""
        for i, col_char in enumerate(COLUMNS):
            row, col = parse_position(f"{col_char}5")
            assert col == i, f"Column {col_char} should map to index {i}"
            assert row == 4  # Row 5 → index 4
    
    def test_all_rows(self):
        """Test that all rows 1-18 parse correctly"""
        for row_num in range(1, ROWS + 1):
            row, col = parse_position(f"F{row_num}")
            assert row == row_num - 1, f"Row {row_num} should map to index {row_num - 1}"
            assert col == 5  # Column F → index 5
    
    def test_lowercase_column(self):
        """Test that lowercase columns are handled"""
        row, col = parse_position("f7")
        assert row == 6
        assert col == 5
    
    def test_mixed_case(self):
        """Test mixed case handling"""
        row, col = parse_position("F7")
        assert row == 6
        assert col == 5
    
    def test_whitespace_handling(self):
        """Test that leading/trailing whitespace is stripped"""
        row, col = parse_position("  F7  ")
        assert row == 6
        assert col == 5
    
    def test_whitespace_and_lowercase(self):
        """Test whitespace with lowercase"""
        row, col = parse_position("  f7  ")
        assert row == 6
        assert col == 5
    
    def test_double_digit_row(self):
        """Test parsing double-digit rows"""
        row, col = parse_position("A10")
        assert row == 9
        assert col == 0
    
    def test_invalid_column_z(self):
        """Test that invalid column Z raises error"""
        with pytest.raises(ValueError, match="Invalid column 'Z'"):
            parse_position("Z7")
    
    def test_invalid_column_l(self):
        """Test that column L (after K) raises error"""
        with pytest.raises(ValueError, match="Invalid column 'L'"):
            parse_position("L7")
    
    def test_invalid_row_zero(self):
        """Test that row 0 raises error"""
        with pytest.raises(ValueError, match="Invalid row 0"):
            parse_position("F0")
    
    def test_invalid_row_nineteen(self):
        """Test that row 19 raises error"""
        with pytest.raises(ValueError, match="Invalid row 19"):
            parse_position("F19")
    
    def test_invalid_row_twenty(self):
        """Test that row 20 raises error"""
        with pytest.raises(ValueError, match="Invalid row 20"):
            parse_position("A20")
    
    def test_invalid_row_negative(self):
        """Test that negative row raises error"""
        with pytest.raises(ValueError, match="Invalid row -1"):
            parse_position("F-1")
    
    def test_invalid_format_empty_string(self):
        """Test that empty string raises error"""
        with pytest.raises(ValueError, match="Invalid position format"):
            parse_position("")
    
    def test_invalid_format_single_char(self):
        """Test that single character raises error"""
        with pytest.raises(ValueError, match="Invalid position format"):
            parse_position("F")
    
    def test_invalid_format_just_number(self):
        """Test that just a number raises error"""
        with pytest.raises(ValueError, match="Invalid position format"):
            parse_position("7")
    
    def test_invalid_format_no_number(self):
        """Test that no number in row position raises error"""
        with pytest.raises(ValueError, match="Invalid row"):
            parse_position("FF")
    
    def test_invalid_format_special_chars(self):
        """Test that special characters raise error"""
        with pytest.raises(ValueError, match="Invalid column"):
            parse_position("@7")
    
    def test_invalid_type_integer(self):
        """Test that integer input raises error"""
        with pytest.raises(ValueError, match="Position must be a string"):
            parse_position(7)
    
    def test_invalid_type_none(self):
        """Test that None input raises error"""
        with pytest.raises(ValueError, match="Position must be a string"):
            parse_position(None)
    
    def test_invalid_type_list(self):
        """Test that list input raises error"""
        with pytest.raises(ValueError, match="Position must be a string"):
            parse_position(["F", "7"])
    
    def test_return_type(self):
        """Test that return type is a tuple"""
        result = parse_position("F7")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)


class TestValidatePosition:
    """Test validate_position function"""
    
    def test_valid_position_returns_true(self):
        """Test that valid positions return True"""
        assert validate_position("F7") is True
    
    def test_all_corners_valid(self):
        """Test that all corner positions are valid"""
        assert validate_position("A1") is True
        assert validate_position("K1") is True
        assert validate_position("A18") is True
        assert validate_position("K18") is True
    
    def test_lowercase_valid(self):
        """Test that lowercase is valid"""
        assert validate_position("f7") is True
    
    def test_whitespace_valid(self):
        """Test that whitespace is handled"""
        assert validate_position("  F7  ") is True
    
    def test_invalid_column_returns_false(self):
        """Test that invalid column returns False"""
        assert validate_position("Z7") is False
    
    def test_invalid_row_returns_false(self):
        """Test that invalid row returns False"""
        assert validate_position("F0") is False
        assert validate_position("F19") is False
        assert validate_position("A20") is False
    
    def test_invalid_format_returns_false(self):
        """Test that invalid format returns False"""
        assert validate_position("") is False
        assert validate_position("F") is False
        assert validate_position("7") is False
        assert validate_position("FF") is False
    
    def test_invalid_type_returns_false(self):
        """Test that invalid types return False"""
        assert validate_position(7) is False
        assert validate_position(None) is False
        assert validate_position(["F", "7"]) is False
    
    def test_special_chars_returns_false(self):
        """Test that special characters return False"""
        assert validate_position("@7") is False
        assert validate_position("F$") is False


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_boundary_row_1(self):
        """Test minimum row boundary"""
        row, col = parse_position("F1")
        assert row == 0
    
    def test_boundary_row_18(self):
        """Test maximum row boundary"""
        row, col = parse_position("F18")
        assert row == 17
    
    def test_boundary_col_a(self):
        """Test minimum column boundary"""
        row, col = parse_position("A7")
        assert col == 0
    
    def test_boundary_col_k(self):
        """Test maximum column boundary"""
        row, col = parse_position("K7")
        assert col == 10
    
    def test_multiple_spaces(self):
        """Test multiple spaces are handled"""
        row, col = parse_position("   F7   ")
        assert row == 6
        assert col == 5
    
    def test_tabs_and_spaces(self):
        """Test that tabs are handled"""
        row, col = parse_position("\tF7\t")
        assert row == 6
        assert col == 5

