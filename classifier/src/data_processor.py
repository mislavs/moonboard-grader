"""
Data Processor Pipeline for Moonboard Problems

This module combines grade encoding and grid building into an end-to-end
processing pipeline. It handles loading JSON files, processing problems,
and providing dataset statistics.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .grade_encoder import encode_grade
from .grid_builder import create_grid_tensor


def process_problem(problem_dict: Dict) -> Tuple[np.ndarray, int]:
    """
    Process a single problem dictionary into a (tensor, label) pair.
    
    Args:
        problem_dict: Problem dictionary with keys:
            - 'grade': Font grade string (e.g., "6B+")
            - 'moves': List of move dictionaries
    
    Returns:
        Tuple of (grid_tensor, grade_label) where:
        - grid_tensor: numpy array of shape (3, 18, 11)
        - grade_label: integer label for the grade
    
    Raises:
        ValueError: If problem_dict is invalid or missing required fields
        
    Examples:
        >>> problem = {
        ...     "grade": "6B+",
        ...     "moves": [
        ...         {"description": "A1", "isStart": True, "isEnd": False},
        ...         {"description": "K18", "isStart": False, "isEnd": True}
        ...     ]
        ... }
        >>> tensor, label = process_problem(problem)
        >>> tensor.shape
        (3, 18, 11)
    """
    if not isinstance(problem_dict, dict):
        raise ValueError(f"problem_dict must be a dict, got {type(problem_dict).__name__}")
    
    # Extract grade
    if 'grade' not in problem_dict:
        raise ValueError("problem_dict missing 'grade' field")
    
    grade_str = problem_dict['grade']
    if not isinstance(grade_str, str):
        raise ValueError(f"'grade' must be a string, got {type(grade_str).__name__}")
    
    # Extract moves
    if 'moves' not in problem_dict:
        raise ValueError("problem_dict missing 'moves' field")
    
    moves_list = problem_dict['moves']
    if not isinstance(moves_list, list):
        raise ValueError(f"'moves' must be a list, got {type(moves_list).__name__}")
    
    # Process grade and grid
    try:
        grade_label = encode_grade(grade_str)
    except ValueError as e:
        raise ValueError(f"Error encoding grade '{grade_str}': {e}")
    
    try:
        grid_tensor = create_grid_tensor(moves_list)
    except ValueError as e:
        raise ValueError(f"Error creating grid tensor: {e}")
    
    return (grid_tensor, grade_label)


def load_dataset(json_path: Union[str, Path]) -> List[Tuple[np.ndarray, int]]:
    """
    Load a dataset from a JSON file and process all problems.
    
    Args:
        json_path: Path to JSON file containing moonboard problems.
            Expected format:
            {
                "data": [
                    {
                        "grade": "6B+",
                        "moves": [...]
                    },
                    ...
                ]
            }
    
    Returns:
        List of (tensor, label) tuples, one per problem
    
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValueError: If JSON is malformed or problems are invalid
        json.JSONDecodeError: If JSON is not valid
        
    Examples:
        >>> dataset = load_dataset("data/problems.json")
        >>> len(dataset)
        100
        >>> tensor, label = dataset[0]
        >>> tensor.shape
        (3, 18, 11)
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    if not json_path.is_file():
        raise ValueError(f"Path is not a file: {json_path}")
    
    # Load JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in {json_path}: {e.msg}",
            e.doc,
            e.pos
        )
    except Exception as e:
        raise ValueError(f"Error reading {json_path}: {e}")
    
    # Validate structure
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be a dictionary, got {type(data).__name__}")
    
    if 'data' not in data:
        raise ValueError("JSON must have a 'data' key containing the problems list")
    
    problems_list = data['data']
    if not isinstance(problems_list, list):
        raise ValueError(f"'data' must be a list, got {type(problems_list).__name__}")
    
    # Process all problems
    dataset = []
    errors = []
    
    for i, problem in enumerate(problems_list):
        try:
            tensor_label_pair = process_problem(problem)
            dataset.append(tensor_label_pair)
        except Exception as e:
            errors.append(f"Problem {i}: {str(e)}")
    
    # If there were errors, report them
    if errors:
        error_msg = f"Failed to process {len(errors)} problem(s):\n" + "\n".join(errors[:5])
        if len(errors) > 5:
            error_msg += f"\n... and {len(errors) - 5} more errors"
        raise ValueError(error_msg)
    
    return dataset


def get_dataset_stats(dataset: List[Tuple[np.ndarray, int]]) -> Dict:
    """
    Calculate statistics about a processed dataset.
    
    Args:
        dataset: List of (tensor, label) tuples
    
    Returns:
        Dictionary with statistics:
        - 'total_problems': Total number of problems
        - 'grade_distribution': Dict mapping grade labels to counts
        - 'avg_holds': Average number of total holds per problem
        - 'avg_start_holds': Average number of start holds
        - 'avg_middle_holds': Average number of middle holds
        - 'avg_end_holds': Average number of end holds
        - 'min_holds': Minimum number of holds
        - 'max_holds': Maximum number of holds
        
    Raises:
        ValueError: If dataset is invalid
        
    Examples:
        >>> dataset = load_dataset("data/problems.json")
        >>> stats = get_dataset_stats(dataset)
        >>> stats['total_problems']
        100
    """
    if not isinstance(dataset, list):
        raise ValueError(f"dataset must be a list, got {type(dataset).__name__}")
    
    if len(dataset) == 0:
        return {
            'total_problems': 0,
            'grade_distribution': {},
            'avg_holds': 0.0,
            'avg_start_holds': 0.0,
            'avg_middle_holds': 0.0,
            'avg_end_holds': 0.0,
            'min_holds': 0,
            'max_holds': 0
        }
    
    # Validate dataset structure
    for i, item in enumerate(dataset):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"Item {i} must be a (tensor, label) tuple")
        tensor, label = item
        if not isinstance(tensor, np.ndarray):
            raise ValueError(f"Item {i}: tensor must be a numpy array")
        if not isinstance(label, (int, np.integer)):
            raise ValueError(f"Item {i}: label must be an integer")
    
    # Calculate statistics
    total_problems = len(dataset)
    grade_distribution = {}
    total_holds = []
    start_holds = []
    middle_holds = []
    end_holds = []
    
    for tensor, label in dataset:
        # Convert numpy int to Python int for JSON serialization
        label = int(label)
        
        # Count grade distribution
        if label not in grade_distribution:
            grade_distribution[label] = 0
        grade_distribution[label] += 1
        
        # Count holds in each channel
        start_count = int(np.sum(tensor[0]))
        middle_count = int(np.sum(tensor[1]))
        end_count = int(np.sum(tensor[2]))
        total_count = start_count + middle_count + end_count
        
        start_holds.append(start_count)
        middle_holds.append(middle_count)
        end_holds.append(end_count)
        total_holds.append(total_count)
    
    stats = {
        'total_problems': total_problems,
        'grade_distribution': grade_distribution,
        'avg_holds': float(np.mean(total_holds)),
        'avg_start_holds': float(np.mean(start_holds)),
        'avg_middle_holds': float(np.mean(middle_holds)),
        'avg_end_holds': float(np.mean(end_holds)),
        'min_holds': int(np.min(total_holds)),
        'max_holds': int(np.max(total_holds))
    }
    
    return stats


def save_processed_dataset(dataset: List[Tuple[np.ndarray, int]], save_path: Union[str, Path]) -> None:
    """
    Save a processed dataset to disk in numpy format.
    
    Args:
        dataset: List of (tensor, label) tuples
        save_path: Path to save the dataset (.npz format)
    
    Raises:
        ValueError: If dataset is invalid
        
    Examples:
        >>> dataset = load_dataset("data/problems.json")
        >>> save_processed_dataset(dataset, "data/processed_dataset.npz")
    """
    if not isinstance(dataset, list):
        raise ValueError(f"dataset must be a list, got {type(dataset).__name__}")
    
    if len(dataset) == 0:
        raise ValueError("Cannot save empty dataset")
    
    save_path = Path(save_path)
    
    # Extract tensors and labels
    tensors = []
    labels = []
    
    for i, item in enumerate(dataset):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"Item {i} must be a (tensor, label) tuple")
        tensor, label = item
        tensors.append(tensor)
        labels.append(label)
    
    # Stack into arrays
    tensors_array = np.stack(tensors, axis=0)  # Shape: (N, 3, 18, 11)
    labels_array = np.array(labels, dtype=np.int64)  # Shape: (N,)
    
    # Create directory if needed
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as compressed numpy file
    np.savez_compressed(
        save_path,
        tensors=tensors_array,
        labels=labels_array
    )


def load_processed_dataset(load_path: Union[str, Path]) -> List[Tuple[np.ndarray, int]]:
    """
    Load a processed dataset from disk.
    
    Args:
        load_path: Path to the saved dataset (.npz format)
    
    Returns:
        List of (tensor, label) tuples
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
        
    Examples:
        >>> dataset = load_processed_dataset("data/processed_dataset.npz")
        >>> len(dataset)
        100
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {load_path}")
    
    # Load numpy file
    try:
        data = np.load(load_path)
    except Exception as e:
        raise ValueError(f"Error loading {load_path}: {e}")
    
    # Validate structure
    if 'tensors' not in data:
        raise ValueError("Dataset file missing 'tensors' array")
    if 'labels' not in data:
        raise ValueError("Dataset file missing 'labels' array")
    
    tensors_array = data['tensors']
    labels_array = data['labels']
    
    # Validate shapes
    if len(tensors_array.shape) != 4:
        raise ValueError(f"Expected tensors to have 4 dimensions, got {len(tensors_array.shape)}")
    if len(labels_array.shape) != 1:
        raise ValueError(f"Expected labels to have 1 dimension, got {len(labels_array.shape)}")
    if tensors_array.shape[0] != labels_array.shape[0]:
        raise ValueError(f"Number of tensors ({tensors_array.shape[0]}) doesn't match number of labels ({labels_array.shape[0]})")
    
    # Convert back to list of tuples
    dataset = []
    for i in range(len(tensors_array)):
        tensor = tensors_array[i]
        label = int(labels_array[i])
        dataset.append((tensor, label))
    
    return dataset

