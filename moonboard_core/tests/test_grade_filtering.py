"""
Tests for grade filtering functionality.

Tests the new functions added for filtered grade classifiers:
- remap_label
- unmap_label
- get_filtered_grade_names
- filter_dataset_by_grades
"""

import pytest
import numpy as np
from moonboard_core.grade_encoder import (
    remap_label, 
    unmap_label, 
    get_filtered_grade_names,
    encode_grade,
    decode_grade
)
from moonboard_core.data_processor import filter_dataset_by_grades


class TestRemapLabel:
    """Tests for remap_label function."""
    
    def test_remap_with_offset_2(self):
        """Test remapping with offset of 2."""
        assert remap_label(2, 2) == 0  # 6A+ -> 0
        assert remap_label(3, 2) == 1  # 6B -> 1
        assert remap_label(11, 2) == 9  # 7C -> 9
    
    def test_remap_with_offset_0(self):
        """Test that offset 0 returns same label."""
        assert remap_label(5, 0) == 5
        assert remap_label(10, 0) == 10
    
    def test_remap_preserves_ordering(self):
        """Test that remapping preserves relative ordering."""
        labels = [2, 3, 4, 5]
        offset = 2
        remapped = [remap_label(l, offset) for l in labels]
        assert remapped == [0, 1, 2, 3]
    
    def test_remap_invalid_negative_result(self):
        """Test that remapping that would result in negative raises error."""
        with pytest.raises(ValueError, match="negative"):
            remap_label(1, 2)  # 1 - 2 = -1
    
    def test_remap_invalid_label_type(self):
        """Test that non-integer label raises error."""
        with pytest.raises(ValueError, match="must be an integer"):
            remap_label("2", 2)
    
    def test_remap_invalid_offset_type(self):
        """Test that non-integer offset raises error."""
        with pytest.raises(ValueError, match="must be an integer"):
            remap_label(2, 2.5)


class TestUnmapLabel:
    """Tests for unmap_label function."""
    
    def test_unmap_with_offset_2(self):
        """Test unmapping with offset of 2."""
        assert unmap_label(0, 2) == 2  # 0 -> 6A+
        assert unmap_label(1, 2) == 3  # 1 -> 6B
        assert unmap_label(9, 2) == 11  # 9 -> 7C
    
    def test_unmap_with_offset_0(self):
        """Test that offset 0 returns same label."""
        assert unmap_label(5, 0) == 5
        assert unmap_label(10, 0) == 10
    
    def test_unmap_reverse_of_remap(self):
        """Test that unmap is the reverse of remap."""
        original = 5
        offset = 2
        remapped = remap_label(original, offset)
        unmapped = unmap_label(remapped, offset)
        assert unmapped == original
    
    def test_unmap_multiple_roundtrips(self):
        """Test multiple remap/unmap cycles."""
        offset = 3
        for label in range(3, 15):
            remapped = remap_label(label, offset)
            unmapped = unmap_label(remapped, offset)
            assert unmapped == label
    
    def test_unmap_out_of_range(self):
        """Test that unmapping beyond valid range raises error."""
        with pytest.raises(ValueError, match="out of range"):
            unmap_label(17, 2)  # 17 + 2 = 19, which is out of range
    
    def test_unmap_invalid_label_type(self):
        """Test that non-integer label raises error."""
        with pytest.raises(ValueError, match="must be an integer"):
            unmap_label(1.5, 2)
    
    def test_unmap_invalid_offset_type(self):
        """Test that non-integer offset raises error."""
        with pytest.raises(ValueError, match="must be an integer"):
            unmap_label(1, "2")


class TestGetFilteredGradeNames:
    """Tests for get_filtered_grade_names function."""
    
    def test_basic_range(self):
        """Test getting a basic range of grades."""
        result = get_filtered_grade_names(2, 4)
        assert result == ['6A+', '6B', '6B+']
    
    def test_6a_plus_through_7c(self):
        """Test the main use case: 6A+ through 7C."""
        result = get_filtered_grade_names(2, 11)
        expected = ['6A+', '6B', '6B+', '6C', '6C+', '7A', '7A+', '7B', '7B+', '7C']
        assert result == expected
        assert len(result) == 10
    
    def test_single_grade(self):
        """Test getting a single grade."""
        result = get_filtered_grade_names(5, 5)
        assert result == ['6C']
    
    def test_full_range(self):
        """Test getting all grades."""
        result = get_filtered_grade_names(0, 18)
        assert len(result) == 19
        assert result[0] == '5+'
        assert result[-1] == '8C+'
    
    def test_length_matches_range(self):
        """Test that length matches the specified range."""
        for min_idx in range(0, 15):
            for max_idx in range(min_idx, 18):
                result = get_filtered_grade_names(min_idx, max_idx)
                expected_length = max_idx - min_idx + 1
                assert len(result) == expected_length
    
    def test_invalid_min_negative(self):
        """Test that negative min_grade_index raises error."""
        with pytest.raises(ValueError, match="must be >= 0"):
            get_filtered_grade_names(-1, 5)
    
    def test_invalid_max_too_large(self):
        """Test that max beyond range raises error."""
        with pytest.raises(ValueError, match="must be <"):
            get_filtered_grade_names(0, 20)
    
    def test_invalid_max_less_than_min(self):
        """Test that max < min raises error."""
        with pytest.raises(ValueError, match="must be >="):
            get_filtered_grade_names(5, 3)
    
    def test_invalid_types(self):
        """Test that non-integer indices raise error."""
        with pytest.raises(ValueError, match="must be integers"):
            get_filtered_grade_names(2.5, 5)
        with pytest.raises(ValueError, match="must be integers"):
            get_filtered_grade_names(2, 5.5)


class TestFilterDatasetByGrades:
    """Tests for filter_dataset_by_grades function."""
    
    def test_filter_basic(self):
        """Test basic filtering of dataset."""
        # Create mock dataset with grades 0-9
        dataset = [(np.zeros((3, 18, 11)), i) for i in range(10)]
        
        # Filter to grades 2-5
        filtered = filter_dataset_by_grades(dataset, 2, 5)
        
        assert len(filtered) == 4
        labels = [label for _, label in filtered]
        assert labels == [2, 3, 4, 5]
    
    def test_filter_removes_outside_range(self):
        """Test that grades outside range are removed."""
        dataset = [
            (np.zeros((3, 18, 11)), 1),  # Should be removed
            (np.zeros((3, 18, 11)), 2),  # Should be kept
            (np.zeros((3, 18, 11)), 3),  # Should be kept
            (np.zeros((3, 18, 11)), 10), # Should be removed
        ]
        
        filtered = filter_dataset_by_grades(dataset, 2, 3)
        
        assert len(filtered) == 2
        labels = [label for _, label in filtered]
        assert labels == [2, 3]
    
    def test_filter_6a_plus_through_7c(self):
        """Test filtering to common range: 6A+ through 7C."""
        # Create dataset with all 19 grades
        dataset = [(np.zeros((3, 18, 11)), i) for i in range(19)]
        
        filtered = filter_dataset_by_grades(dataset, 2, 11)
        
        assert len(filtered) == 10
        labels = [label for _, label in filtered]
        assert labels == list(range(2, 12))
    
    def test_filter_preserves_tensors(self):
        """Test that filtering preserves tensor data."""
        tensor1 = np.ones((3, 18, 11)) * 1.0
        tensor2 = np.ones((3, 18, 11)) * 2.0
        tensor3 = np.ones((3, 18, 11)) * 3.0
        
        dataset = [
            (tensor1, 1),
            (tensor2, 2),
            (tensor3, 3),
        ]
        
        filtered = filter_dataset_by_grades(dataset, 2, 3)
        
        assert len(filtered) == 2
        assert np.array_equal(filtered[0][0], tensor2)
        assert np.array_equal(filtered[1][0], tensor3)
    
    def test_filter_empty_result(self):
        """Test filtering that results in empty dataset."""
        dataset = [(np.zeros((3, 18, 11)), i) for i in range(5)]
        
        filtered = filter_dataset_by_grades(dataset, 10, 15)
        
        assert len(filtered) == 0
    
    def test_filter_all_in_range(self):
        """Test when all items are in range."""
        dataset = [(np.zeros((3, 18, 11)), i) for i in range(5, 10)]
        
        filtered = filter_dataset_by_grades(dataset, 0, 20)
        
        assert len(filtered) == len(dataset)
    
    def test_filter_invalid_dataset_type(self):
        """Test that non-list dataset raises error."""
        with pytest.raises(ValueError, match="must be a list"):
            filter_dataset_by_grades("not a list", 2, 5)
    
    def test_filter_invalid_indices_type(self):
        """Test that non-integer indices raise error."""
        dataset = [(np.zeros((3, 18, 11)), 2)]
        
        with pytest.raises(ValueError, match="must be integers"):
            filter_dataset_by_grades(dataset, 2.5, 5)
    
    def test_filter_invalid_min_negative(self):
        """Test that negative min raises error."""
        dataset = [(np.zeros((3, 18, 11)), 2)]
        
        with pytest.raises(ValueError, match="must be >= 0"):
            filter_dataset_by_grades(dataset, -1, 5)
    
    def test_filter_invalid_max_less_than_min(self):
        """Test that max < min raises error."""
        dataset = [(np.zeros((3, 18, 11)), 2)]
        
        with pytest.raises(ValueError, match="must be >="):
            filter_dataset_by_grades(dataset, 5, 3)


class TestIntegrationWorkflow:
    """Integration tests for complete filtering workflow."""
    
    def test_complete_workflow(self):
        """Test complete filtering, remapping, and unmapping workflow."""
        # Create dataset
        dataset = [(np.zeros((3, 18, 11)), i) for i in range(19)]
        
        # Filter to 6A+ through 7C
        min_idx = 2
        max_idx = 11
        offset = min_idx
        
        filtered = filter_dataset_by_grades(dataset, min_idx, max_idx)
        assert len(filtered) == 10
        
        # Remap labels
        remapped = [(tensor, remap_label(label, offset)) for tensor, label in filtered]
        
        # Check remapped labels are 0-9
        remapped_labels = [label for _, label in remapped]
        assert remapped_labels == list(range(10))
        
        # Unmap predictions back
        for i, (_, remapped_label) in enumerate(remapped):
            original_label = unmap_label(remapped_label, offset)
            expected_original = min_idx + i
            assert original_label == expected_original
    
    def test_grade_name_consistency(self):
        """Test that filtered grade names match the filtered indices."""
        min_idx = 2
        max_idx = 11
        
        grade_names = get_filtered_grade_names(min_idx, max_idx)
        
        for i, name in enumerate(grade_names):
            original_idx = min_idx + i
            assert decode_grade(original_idx) == name
    
    def test_encode_decode_with_filtering(self):
        """Test that encoding and decoding work with filtered grades."""
        grades = ['6A+', '6B', '6C', '7A', '7B', '7C']
        offset = 2
        
        for grade in grades:
            # Encode to original index
            original_idx = encode_grade(grade)
            
            # Remap to model space
            model_idx = remap_label(original_idx, offset)
            
            # Unmap back
            unmapped_idx = unmap_label(model_idx, offset)
            
            # Decode to grade
            result_grade = decode_grade(unmapped_idx)
            
            assert result_grade == grade


