"""
Unit tests for data_processor module
"""

import pytest
import json
import numpy as np
from pathlib import Path
import tempfile
import shutil

from moonboard_core.data_processor import (
    process_problem,
    load_dataset,
    get_dataset_stats,
    save_processed_dataset,
    load_processed_dataset
)
from moonboard_core.grade_encoder import encode_grade


class TestProcessProblem:
    """Test process_problem function"""
    
    def test_simple_problem(self):
        """Test processing a simple problem"""
        problem = {
            "grade": "6B+",
            "moves": [
                {"description": "A1", "isStart": True, "isEnd": False},
                {"description": "F7", "isStart": False, "isEnd": False},
                {"description": "K18", "isStart": False, "isEnd": True}
            ]
        }
        
        tensor, label = process_problem(problem)
        
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (3, 18, 11)
        assert isinstance(label, int)
        assert label == encode_grade("6B+")
    
    def test_empty_moves(self):
        """Test problem with no moves"""
        problem = {
            "grade": "6A",
            "moves": []
        }
        
        tensor, label = process_problem(problem)
        
        assert tensor.shape == (3, 18, 11)
        assert np.sum(tensor) == 0  # No holds
        assert label == encode_grade("6A")
    
    def test_invalid_problem_type(self):
        """Test that non-dict raises error"""
        with pytest.raises(ValueError, match="problem_dict must be a dict"):
            process_problem("not a dict")
    
    def test_missing_grade_field(self):
        """Test that missing grade raises error"""
        problem = {
            "moves": [
                {"description": "F7", "isStart": True, "isEnd": False}
            ]
        }
        with pytest.raises(ValueError, match="missing 'grade' field"):
            process_problem(problem)


class TestLoadDataset:
    """Test load_dataset function"""
    
    def setup_method(self):
        """Create temporary directory for test files"""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_load_single_problem(self):
        """Test loading file with single problem"""
        test_file = self.temp_dir / "single.json"
        data = {
            "data": [
                {
                    "grade": "6B+",
                    "moves": [
                        {"description": "A1", "isStart": True, "isEnd": False},
                        {"description": "K18", "isStart": False, "isEnd": True}
                    ]
                }
            ]
        }
        
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        dataset = load_dataset(test_file)
        
        assert len(dataset) == 1
        tensor, label = dataset[0]
        assert tensor.shape == (3, 18, 11)
        assert label == encode_grade("6B+")
    
    def test_empty_dataset(self):
        """Test loading file with empty data array"""
        test_file = self.temp_dir / "empty.json"
        data = {"data": []}
        
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        dataset = load_dataset(test_file)
        
        assert len(dataset) == 0
    
    def test_file_not_found(self):
        """Test that non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_dataset(self.temp_dir / "nonexistent.json")
    
    def test_missing_data_key(self):
        """Test that missing 'data' key raises error"""
        test_file = self.temp_dir / "no_data.json"
        data = {"problems": []}
        
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        with pytest.raises(ValueError, match="JSON must have a 'data' key"):
            load_dataset(test_file)
    
    def test_load_with_min_repeats_none(self):
        """Test loading with min_repeats=None (no filtering)"""
        test_file = self.temp_dir / "repeats.json"
        data = {
            "data": [
                {
                    "grade": "6B+",
                    "repeats": 0,
                    "moves": [
                        {"description": "A1", "isStart": True, "isEnd": False},
                        {"description": "K18", "isStart": False, "isEnd": True}
                    ]
                },
                {
                    "grade": "6C",
                    "repeats": 5,
                    "moves": [
                        {"description": "B2", "isStart": True, "isEnd": False},
                        {"description": "J17", "isStart": False, "isEnd": True}
                    ]
                },
                {
                    "grade": "7A",
                    "repeats": 10,
                    "moves": [
                        {"description": "C3", "isStart": True, "isEnd": False},
                        {"description": "H16", "isStart": False, "isEnd": True}
                    ]
                }
            ]
        }
        
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        dataset = load_dataset(test_file, min_repeats=None)
        
        # Should include all problems
        assert len(dataset) == 3
    
    def test_load_with_min_repeats_zero(self):
        """Test loading with min_repeats=0 (include all)"""
        test_file = self.temp_dir / "repeats_zero.json"
        data = {
            "data": [
                {
                    "grade": "6B+",
                    "repeats": 0,
                    "moves": [
                        {"description": "A1", "isStart": True, "isEnd": False},
                        {"description": "K18", "isStart": False, "isEnd": True}
                    ]
                },
                {
                    "grade": "6C",
                    "repeats": 5,
                    "moves": [
                        {"description": "B2", "isStart": True, "isEnd": False},
                        {"description": "J17", "isStart": False, "isEnd": True}
                    ]
                }
            ]
        }
        
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        dataset = load_dataset(test_file, min_repeats=0)
        
        # Should include all problems (0 repeats and above)
        assert len(dataset) == 2
    
    def test_load_with_min_repeats_one(self):
        """Test loading with min_repeats=1 (exclude zero-repeat routes)"""
        test_file = self.temp_dir / "repeats_one.json"
        data = {
            "data": [
                {
                    "grade": "6B+",
                    "repeats": 0,
                    "moves": [
                        {"description": "A1", "isStart": True, "isEnd": False},
                        {"description": "K18", "isStart": False, "isEnd": True}
                    ]
                },
                {
                    "grade": "6C",
                    "repeats": 5,
                    "moves": [
                        {"description": "B2", "isStart": True, "isEnd": False},
                        {"description": "J17", "isStart": False, "isEnd": True}
                    ]
                },
                {
                    "grade": "7A",
                    "repeats": 10,
                    "moves": [
                        {"description": "C3", "isStart": True, "isEnd": False},
                        {"description": "H16", "isStart": False, "isEnd": True}
                    ]
                }
            ]
        }
        
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        dataset = load_dataset(test_file, min_repeats=1)
        
        # Should exclude the first problem (0 repeats)
        assert len(dataset) == 2
        # Verify the remaining problems are the ones with repeats >= 1
        _, label1 = dataset[0]
        _, label2 = dataset[1]
        assert label1 == encode_grade("6C")
        assert label2 == encode_grade("7A")
    
    def test_load_with_min_repeats_high(self):
        """Test loading with higher min_repeats value"""
        test_file = self.temp_dir / "repeats_high.json"
        data = {
            "data": [
                {
                    "grade": "6B+",
                    "repeats": 0,
                    "moves": [
                        {"description": "A1", "isStart": True, "isEnd": False},
                        {"description": "K18", "isStart": False, "isEnd": True}
                    ]
                },
                {
                    "grade": "6C",
                    "repeats": 5,
                    "moves": [
                        {"description": "B2", "isStart": True, "isEnd": False},
                        {"description": "J17", "isStart": False, "isEnd": True}
                    ]
                },
                {
                    "grade": "7A",
                    "repeats": 10,
                    "moves": [
                        {"description": "C3", "isStart": True, "isEnd": False},
                        {"description": "H16", "isStart": False, "isEnd": True}
                    ]
                },
                {
                    "grade": "7B",
                    "repeats": 15,
                    "moves": [
                        {"description": "D4", "isStart": True, "isEnd": False},
                        {"description": "G15", "isStart": False, "isEnd": True}
                    ]
                }
            ]
        }
        
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        dataset = load_dataset(test_file, min_repeats=10)
        
        # Should only include problems with >= 10 repeats
        assert len(dataset) == 2
        _, label1 = dataset[0]
        _, label2 = dataset[1]
        assert label1 == encode_grade("7A")
        assert label2 == encode_grade("7B")
    
    def test_load_with_missing_repeats_field(self):
        """Test that problems without repeats field default to 0"""
        test_file = self.temp_dir / "missing_repeats.json"
        data = {
            "data": [
                {
                    "grade": "6B+",
                    # No repeats field
                    "moves": [
                        {"description": "A1", "isStart": True, "isEnd": False},
                        {"description": "K18", "isStart": False, "isEnd": True}
                    ]
                },
                {
                    "grade": "6C",
                    "repeats": 5,
                    "moves": [
                        {"description": "B2", "isStart": True, "isEnd": False},
                        {"description": "J17", "isStart": False, "isEnd": True}
                    ]
                }
            ]
        }
        
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        dataset = load_dataset(test_file, min_repeats=1)
        
        # Should exclude the first problem (missing repeats = 0)
        assert len(dataset) == 1
        _, label = dataset[0]
        assert label == encode_grade("6C")


class TestGetDatasetStats:
    """Test get_dataset_stats function"""
    
    def create_mock_dataset(self, num_problems=5):
        """Helper to create a mock dataset"""
        dataset = []
        grades = ["6A", "6A+", "6B", "6B+", "6C"]
        
        for i in range(num_problems):
            problem = {
                "grade": grades[i % len(grades)],
                "moves": [
                    {"description": "A1", "isStart": True, "isEnd": False},
                    {"description": "F7", "isStart": False, "isEnd": False},
                    {"description": "K18", "isStart": False, "isEnd": True}
                ]
            }
            tensor, label = process_problem(problem)
            dataset.append((tensor, label))
        
        return dataset
    
    def test_empty_dataset(self):
        """Test stats for empty dataset"""
        stats = get_dataset_stats([])
        
        assert stats['total_problems'] == 0
        assert stats['grade_distribution'] == {}
        assert stats['avg_holds'] == 0.0
        assert stats['min_holds'] == 0
        assert stats['max_holds'] == 0
    
    def test_single_problem_stats(self):
        """Test stats for single problem"""
        dataset = self.create_mock_dataset(1)
        stats = get_dataset_stats(dataset)
        
        assert stats['total_problems'] == 1
        assert stats['avg_holds'] == 3.0  # 1 start + 1 middle + 1 end
        assert stats['avg_start_holds'] == 1.0
        assert stats['avg_middle_holds'] == 1.0
        assert stats['avg_end_holds'] == 1.0
        assert stats['min_holds'] == 3
        assert stats['max_holds'] == 3
    
    def test_invalid_dataset_type(self):
        """Test that non-list raises error"""
        with pytest.raises(ValueError, match="dataset must be a list"):
            get_dataset_stats("not a list")


class TestSaveLoadProcessedDataset:
    """Test save_processed_dataset and load_processed_dataset functions"""
    
    def setup_method(self):
        """Create temporary directory for test files"""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_mock_dataset(self, num_problems=5):
        """Helper to create a mock dataset"""
        dataset = []
        grades = ["6A", "6A+", "6B", "6B+", "6C"]
        
        for i in range(num_problems):
            problem = {
                "grade": grades[i % len(grades)],
                "moves": [
                    {"description": "A1", "isStart": True, "isEnd": False},
                    {"description": "F7", "isStart": False, "isEnd": False},
                    {"description": "K18", "isStart": False, "isEnd": True}
                ]
            }
            tensor, label = process_problem(problem)
            dataset.append((tensor, label))
        
        return dataset
    
    def test_save_and_load_round_trip(self):
        """Test that save and load preserve data"""
        original_dataset = self.create_mock_dataset(10)
        save_path = self.temp_dir / "dataset.npz"
        
        # Save
        save_processed_dataset(original_dataset, save_path)
        assert save_path.exists()
        
        # Load
        loaded_dataset = load_processed_dataset(save_path)
        
        # Verify
        assert len(loaded_dataset) == len(original_dataset)
        
        for i, ((orig_tensor, orig_label), (load_tensor, load_label)) in enumerate(zip(original_dataset, loaded_dataset)):
            assert np.array_equal(orig_tensor, load_tensor), f"Tensor {i} mismatch"
            assert orig_label == load_label, f"Label {i} mismatch"
    
    def test_save_invalid_dataset_type(self):
        """Test that non-list raises error"""
        with pytest.raises(ValueError, match="dataset must be a list"):
            save_processed_dataset("not a list", self.temp_dir / "test.npz")
    
    def test_save_empty_dataset(self):
        """Test that empty dataset raises error"""
        with pytest.raises(ValueError, match="Cannot save empty dataset"):
            save_processed_dataset([], self.temp_dir / "test.npz")
    
    def test_load_file_not_found(self):
        """Test that loading non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_processed_dataset(self.temp_dir / "nonexistent.npz")

