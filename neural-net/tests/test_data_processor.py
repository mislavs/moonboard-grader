"""
Unit tests for data_processor module
"""

import pytest
import json
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.data_processor import (
    process_problem,
    load_dataset,
    get_dataset_stats,
    save_processed_dataset,
    load_processed_dataset
)
from src.grade_encoder import encode_grade


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
    
    def test_example_json_problem(self):
        """Test processing the example.json problem"""
        problem = {
            "grade": "6B+",
            "moves": [
                {"description": "J4", "isStart": True, "isEnd": False},
                {"description": "G6", "isStart": False, "isEnd": False},
                {"description": "F7", "isStart": False, "isEnd": False},
                {"description": "B8", "isStart": False, "isEnd": False},
                {"description": "B9", "isStart": False, "isEnd": False},
                {"description": "A11", "isStart": False, "isEnd": False},
                {"description": "D15", "isStart": False, "isEnd": False},
                {"description": "A16", "isStart": False, "isEnd": False},
                {"description": "F18", "isStart": False, "isEnd": True}
            ]
        }
        
        tensor, label = process_problem(problem)
        
        assert tensor.shape == (3, 18, 11)
        assert label == encode_grade("6B+")
        # Verify counts
        assert np.sum(tensor[0]) == 1  # 1 start hold
        assert np.sum(tensor[1]) == 7  # 7 middle holds
        assert np.sum(tensor[2]) == 1  # 1 end hold
    
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
    
    def test_different_grades(self):
        """Test problems with different grades"""
        grades = ["5+", "6A", "7B", "8C+"]
        
        for grade in grades:
            problem = {
                "grade": grade,
                "moves": [
                    {"description": "F7", "isStart": True, "isEnd": False}
                ]
            }
            tensor, label = process_problem(problem)
            assert label == encode_grade(grade)
    
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
    
    def test_missing_moves_field(self):
        """Test that missing moves raises error"""
        problem = {
            "grade": "6B+"
        }
        with pytest.raises(ValueError, match="missing 'moves' field"):
            process_problem(problem)
    
    def test_invalid_grade_type(self):
        """Test that non-string grade raises error"""
        problem = {
            "grade": 6,
            "moves": []
        }
        with pytest.raises(ValueError, match="'grade' must be a string"):
            process_problem(problem)
    
    def test_invalid_moves_type(self):
        """Test that non-list moves raises error"""
        problem = {
            "grade": "6B+",
            "moves": "not a list"
        }
        with pytest.raises(ValueError, match="'moves' must be a list"):
            process_problem(problem)
    
    def test_invalid_grade_value(self):
        """Test that invalid grade raises error"""
        problem = {
            "grade": "9Z",
            "moves": []
        }
        with pytest.raises(ValueError, match="Error encoding grade"):
            process_problem(problem)
    
    def test_invalid_move_position(self):
        """Test that invalid move position raises error"""
        problem = {
            "grade": "6B+",
            "moves": [
                {"description": "Z99", "isStart": True, "isEnd": False}
            ]
        }
        with pytest.raises(ValueError, match="Error creating grid tensor"):
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
    
    def test_load_example_json(self):
        """Test loading the example.json file"""
        # This assumes example.json exists in the project root
        example_path = Path("example.json")
        if not example_path.exists():
            pytest.skip("example.json not found")
        
        dataset = load_dataset(example_path)
        
        assert isinstance(dataset, list)
        assert len(dataset) == 1  # example.json has 1 problem
        
        tensor, label = dataset[0]
        assert tensor.shape == (3, 18, 11)
        assert label == encode_grade("6B+")
    
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
    
    def test_load_multiple_problems(self):
        """Test loading file with multiple problems"""
        test_file = self.temp_dir / "multiple.json"
        data = {
            "data": [
                {
                    "grade": "6A",
                    "moves": [
                        {"description": "A1", "isStart": True, "isEnd": False},
                        {"description": "K18", "isStart": False, "isEnd": True}
                    ]
                },
                {
                    "grade": "6B",
                    "moves": [
                        {"description": "F7", "isStart": True, "isEnd": False},
                        {"description": "G10", "isStart": False, "isEnd": True}
                    ]
                },
                {
                    "grade": "7A",
                    "moves": [
                        {"description": "B2", "isStart": True, "isEnd": False},
                        {"description": "J17", "isStart": False, "isEnd": True}
                    ]
                }
            ]
        }
        
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        dataset = load_dataset(test_file)
        
        assert len(dataset) == 3
        
        # Check each problem
        _, label0 = dataset[0]
        _, label1 = dataset[1]
        _, label2 = dataset[2]
        
        assert label0 == encode_grade("6A")
        assert label1 == encode_grade("6B")
        assert label2 == encode_grade("7A")
    
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
    
    def test_path_is_directory(self):
        """Test that directory path raises error"""
        with pytest.raises(ValueError, match="Path is not a file"):
            load_dataset(self.temp_dir)
    
    def test_invalid_json(self):
        """Test that invalid JSON raises error"""
        test_file = self.temp_dir / "invalid.json"
        
        with open(test_file, 'w') as f:
            f.write("{invalid json content")
        
        with pytest.raises(json.JSONDecodeError):
            load_dataset(test_file)
    
    def test_json_not_dict(self):
        """Test that non-dict JSON root raises error"""
        test_file = self.temp_dir / "list.json"
        
        with open(test_file, 'w') as f:
            json.dump([1, 2, 3], f)
        
        with pytest.raises(ValueError, match="JSON root must be a dictionary"):
            load_dataset(test_file)
    
    def test_missing_data_key(self):
        """Test that missing 'data' key raises error"""
        test_file = self.temp_dir / "no_data.json"
        data = {"problems": []}
        
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        with pytest.raises(ValueError, match="JSON must have a 'data' key"):
            load_dataset(test_file)
    
    def test_data_not_list(self):
        """Test that non-list 'data' raises error"""
        test_file = self.temp_dir / "bad_data.json"
        data = {"data": "not a list"}
        
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        with pytest.raises(ValueError, match="'data' must be a list"):
            load_dataset(test_file)
    
    def test_invalid_problem_in_dataset(self):
        """Test that invalid problem raises error with context"""
        test_file = self.temp_dir / "invalid_problem.json"
        data = {
            "data": [
                {
                    "grade": "6A",
                    "moves": []
                },
                {
                    "grade": "INVALID",
                    "moves": []
                }
            ]
        }
        
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        with pytest.raises(ValueError, match="Failed to process"):
            load_dataset(test_file)


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
    
    def test_multiple_problems_stats(self):
        """Test stats for multiple problems"""
        dataset = self.create_mock_dataset(10)
        stats = get_dataset_stats(dataset)
        
        assert stats['total_problems'] == 10
        assert isinstance(stats['grade_distribution'], dict)
        assert sum(stats['grade_distribution'].values()) == 10
        assert stats['avg_holds'] > 0
    
    def test_grade_distribution(self):
        """Test grade distribution counting"""
        problems = [
            {"grade": "6A", "moves": []},
            {"grade": "6A", "moves": []},
            {"grade": "6B", "moves": []},
            {"grade": "6A", "moves": []},
        ]
        
        dataset = []
        for p in problems:
            tensor, label = process_problem(p)
            dataset.append((tensor, label))
        
        stats = get_dataset_stats(dataset)
        
        label_6a = encode_grade("6A")
        label_6b = encode_grade("6B")
        
        assert stats['grade_distribution'][label_6a] == 3
        assert stats['grade_distribution'][label_6b] == 1
    
    def test_hold_count_statistics(self):
        """Test that hold counts are calculated correctly"""
        # Create problems with different hold counts
        p1 = {
            "grade": "6A",
            "moves": [
                {"description": "A1", "isStart": True, "isEnd": False},
                {"description": "K18", "isStart": False, "isEnd": True}
            ]  # 2 holds
        }
        
        p2 = {
            "grade": "6A",
            "moves": [
                {"description": "A1", "isStart": True, "isEnd": False},
                {"description": "F7", "isStart": False, "isEnd": False},
                {"description": "G10", "isStart": False, "isEnd": False},
                {"description": "K18", "isStart": False, "isEnd": True}
            ]  # 4 holds
        }
        
        dataset = [process_problem(p1), process_problem(p2)]
        stats = get_dataset_stats(dataset)
        
        assert stats['avg_holds'] == 3.0  # (2 + 4) / 2
        assert stats['min_holds'] == 2
        assert stats['max_holds'] == 4
    
    def test_invalid_dataset_type(self):
        """Test that non-list raises error"""
        with pytest.raises(ValueError, match="dataset must be a list"):
            get_dataset_stats("not a list")
    
    def test_invalid_item_type(self):
        """Test that non-tuple item raises error"""
        with pytest.raises(ValueError, match="must be a .* tuple"):
            get_dataset_stats(["not a tuple"])
    
    def test_invalid_tensor_type(self):
        """Test that non-array tensor raises error"""
        with pytest.raises(ValueError, match="tensor must be a numpy array"):
            get_dataset_stats([("not array", 0)])
    
    def test_invalid_label_type(self):
        """Test that non-int label raises error"""
        tensor = np.zeros((3, 18, 11))
        with pytest.raises(ValueError, match="label must be an integer"):
            get_dataset_stats([(tensor, "not int")])


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
    
    def test_save_creates_directory(self):
        """Test that save creates parent directories"""
        dataset = self.create_mock_dataset(5)
        save_path = self.temp_dir / "subdir" / "nested" / "dataset.npz"
        
        save_processed_dataset(dataset, save_path)
        
        assert save_path.exists()
        assert save_path.parent.exists()
    
    def test_save_single_problem(self):
        """Test saving single problem"""
        dataset = self.create_mock_dataset(1)
        save_path = self.temp_dir / "single.npz"
        
        save_processed_dataset(dataset, save_path)
        loaded_dataset = load_processed_dataset(save_path)
        
        assert len(loaded_dataset) == 1
    
    def test_save_invalid_dataset_type(self):
        """Test that non-list raises error"""
        with pytest.raises(ValueError, match="dataset must be a list"):
            save_processed_dataset("not a list", self.temp_dir / "test.npz")
    
    def test_save_empty_dataset(self):
        """Test that empty dataset raises error"""
        with pytest.raises(ValueError, match="Cannot save empty dataset"):
            save_processed_dataset([], self.temp_dir / "test.npz")
    
    def test_save_invalid_item(self):
        """Test that invalid item raises error"""
        with pytest.raises(ValueError, match="must be a .* tuple"):
            save_processed_dataset(["not tuple"], self.temp_dir / "test.npz")
    
    def test_load_file_not_found(self):
        """Test that loading non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_processed_dataset(self.temp_dir / "nonexistent.npz")
    
    def test_load_invalid_format(self):
        """Test that loading invalid file raises error"""
        bad_file = self.temp_dir / "bad.npz"
        
        # Create a text file with .npz extension
        with open(bad_file, 'w') as f:
            f.write("not a numpy file")
        
        with pytest.raises(ValueError, match="Error loading"):
            load_processed_dataset(bad_file)
    
    def test_load_missing_tensors_key(self):
        """Test that file without 'tensors' key raises error"""
        bad_file = self.temp_dir / "missing_tensors.npz"
        
        # Save file with wrong keys
        np.savez(bad_file, wrong_key=np.array([1, 2, 3]))
        
        with pytest.raises(ValueError, match="missing 'tensors' array"):
            load_processed_dataset(bad_file)
    
    def test_load_missing_labels_key(self):
        """Test that file without 'labels' key raises error"""
        bad_file = self.temp_dir / "missing_labels.npz"
        
        # Save file with only tensors
        tensors = np.zeros((5, 3, 18, 11))
        np.savez(bad_file, tensors=tensors)
        
        with pytest.raises(ValueError, match="missing 'labels' array"):
            load_processed_dataset(bad_file)
    
    def test_load_mismatched_lengths(self):
        """Test that mismatched tensor/label counts raise error"""
        bad_file = self.temp_dir / "mismatched.npz"
        
        tensors = np.zeros((5, 3, 18, 11))
        labels = np.array([1, 2, 3])  # Different length
        
        np.savez(bad_file, tensors=tensors, labels=labels)
        
        with pytest.raises(ValueError, match="doesn't match"):
            load_processed_dataset(bad_file)


class TestIntegration:
    """Integration tests combining multiple functions"""
    
    def setup_method(self):
        """Create temporary directory for test files"""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline(self):
        """Test complete pipeline: JSON -> dataset -> stats -> save -> load"""
        # Create test JSON
        test_file = self.temp_dir / "problems.json"
        data = {
            "data": [
                {
                    "grade": "6A",
                    "moves": [
                        {"description": "A1", "isStart": True, "isEnd": False},
                        {"description": "K18", "isStart": False, "isEnd": True}
                    ]
                },
                {
                    "grade": "6B",
                    "moves": [
                        {"description": "F7", "isStart": True, "isEnd": False},
                        {"description": "G10", "isStart": False, "isEnd": False},
                        {"description": "J17", "isStart": False, "isEnd": True}
                    ]
                },
                {
                    "grade": "6A",
                    "moves": [
                        {"description": "B2", "isStart": True, "isEnd": False},
                        {"description": "D10", "isStart": False, "isEnd": True}
                    ]
                }
            ]
        }
        
        with open(test_file, 'w') as f:
            json.dump(data, f)
        
        # Load dataset
        dataset = load_dataset(test_file)
        assert len(dataset) == 3
        
        # Get stats
        stats = get_dataset_stats(dataset)
        assert stats['total_problems'] == 3
        assert len(stats['grade_distribution']) == 2  # 6A and 6B
        
        # Save dataset
        save_path = self.temp_dir / "processed.npz"
        save_processed_dataset(dataset, save_path)
        assert save_path.exists()
        
        # Load processed dataset
        loaded_dataset = load_processed_dataset(save_path)
        assert len(loaded_dataset) == 3
        
        # Verify loaded data matches original
        for (orig_t, orig_l), (load_t, load_l) in zip(dataset, loaded_dataset):
            assert np.array_equal(orig_t, load_t)
            assert orig_l == load_l

