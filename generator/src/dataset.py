"""
PyTorch Dataset for MoonBoard climbing problems.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from moonboard_core.data_processor import load_dataset, filter_dataset_by_grades
from moonboard_core.grade_encoder import (
    decode_grade,
    get_all_grades,
    get_filtered_grade_names,
)
from .label_space import LabelSpaceMode

logger = logging.getLogger(__name__)


def _safe_decode_grade(label: int) -> str:
    try:
        return decode_grade(label)
    except Exception:
        return str(label)


class MoonBoardDataset(Dataset):
    """
    PyTorch Dataset for MoonBoard climbing problems.
    
    Returns grid tensors (3x18x11) and grade labels for training the VAE.
    
    Args:
        data_path: Path to the problems.json file
        min_grade_index: Minimum grade index (inclusive) for filtering
        max_grade_index: Maximum grade index (inclusive) for filtering
        transform: Optional transform to apply to grids
    """
    
    def __init__(
        self, 
        data_path: str, 
        min_grade_index: Optional[int] = None, 
        max_grade_index: Optional[int] = None, 
        label_space_mode: Optional[LabelSpaceMode] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to problems.json
            min_grade_index: Minimum grade index (e.g., 2 for 6A+)
            max_grade_index: Maximum grade index (e.g., 2 for 6A+ only)
            transform: Optional transform function
        """
        self.data_path = data_path
        self.transform = transform
        self.min_grade_index = min_grade_index
        self.max_grade_index = max_grade_index
        self._validate_grade_filter_args()

        self.is_filtered = self.min_grade_index is not None and self.max_grade_index is not None
        inferred_mode = "remapped" if self.is_filtered else "global_legacy"
        self.label_space_mode = label_space_mode or inferred_mode
        if self.label_space_mode not in ("remapped", "global_legacy"):
            raise ValueError(
                f"label_space_mode must be 'remapped' or 'global_legacy', got {self.label_space_mode}"
            )
        self.grade_offset = self.min_grade_index if self.label_space_mode == "remapped" else 0

        # Load and filter dataset
        logger.info(f"Loading dataset from {data_path}")
        full_dataset = load_dataset(data_path)
        logger.info(f"Loaded {len(full_dataset)} problems")

        self.dataset = self._apply_grade_filter(full_dataset)
        if len(self.dataset) == 0:
            if self.is_filtered:
                raise ValueError(
                    f"No problems found in grade range [{self.min_grade_index}, {self.max_grade_index}]"
                )
            raise ValueError("Dataset is empty")
        self.grade_names, self.grade_to_label, self.label_to_grade = self._build_grade_mappings()
        self.model_grade_names = [
            decode_grade(self.model_to_global_label(model_label))
            for model_label in range(self.get_num_model_grades())
        ]

    def _validate_grade_filter_args(self) -> None:
        """Validate grade filter configuration."""
        min_idx = self.min_grade_index
        max_idx = self.max_grade_index
        if (min_idx is None) != (max_idx is None):
            raise ValueError("min_grade_index and max_grade_index must be provided together")
        if min_idx is not None and max_idx is not None:
            # Reuse moonboard_core validation semantics.
            get_filtered_grade_names(min_idx, max_idx)
    
    def _apply_grade_filter(self, full_dataset: List[Tuple]) -> List[Tuple]:
        """Apply grade filtering if specified."""
        if self.min_grade_index is not None and self.max_grade_index is not None:
            grade_names = get_filtered_grade_names(self.min_grade_index, self.max_grade_index)
            logger.info(f"Filtering to grades {self.min_grade_index}-{self.max_grade_index}: {grade_names}")
            filtered = filter_dataset_by_grades(full_dataset, self.min_grade_index, self.max_grade_index)
            logger.info(f"After filtering: {len(filtered)} problems")
            return filtered
        return full_dataset
    
    def _build_grade_mappings(self) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
        """
        Build grade name and label mappings using moonboard_core as source of truth.
        
        Always uses the full global grade list to ensure consistent label indices
        across training and inference, regardless of which grades are in the dataset.
        """
        # Keep global grade list as source of truth for external semantics.
        grade_names = get_all_grades()
        
        # Create bidirectional mappings using global indices
        grade_to_label = {grade: idx for idx, grade in enumerate(grade_names)}
        label_to_grade = {idx: grade for idx, grade in enumerate(grade_names)}
        
        return grade_names, grade_to_label, label_to_grade
        
    def __len__(self):
        """Return the number of problems in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single problem.
        
        Args:
            idx: Index of the problem
            
        Returns:
            grid_tensor: Tensor of shape (3, 18, 11) with hold positions
            grade_label: Integer label for the grade in model label space
        """
        # Get grid and label from processed dataset
        grid_array, global_label = self.dataset[idx]

        # Convert to PyTorch tensor
        grid_tensor = torch.from_numpy(grid_array).float()

        # Validate label is within global range
        num_global_grades = len(self.grade_to_label)
        if global_label < 0 or global_label >= num_global_grades:
            grade_str = decode_grade(global_label)
            raise ValueError(
                f"Invalid global grade label {global_label} ({grade_str}). "
                f"Must be in global range [0, {num_global_grades-1}]"
            )
        grade_label = self.global_to_model_label(int(global_label))

        # Apply transform if provided
        if self.transform is not None:
            grid_tensor = self.transform(grid_tensor)

        return grid_tensor, grade_label

    def get_num_grades(self) -> int:
        """Backward-compatible alias for get_num_model_grades."""
        return self.get_num_model_grades()

    def get_num_model_grades(self) -> int:
        """Return the number of grades in model label space."""
        if self.label_space_mode == "remapped" and self.is_filtered:
            return (self.max_grade_index - self.min_grade_index) + 1
        return len(self.grade_to_label)

    def global_to_model_label(self, global_label: int) -> int:
        """Map global moonboard_core label to model label space."""
        if self.label_space_mode == "remapped":
            if self.is_filtered:
                if not self.min_grade_index <= global_label <= self.max_grade_index:
                    raise ValueError(
                        f"Global label {global_label} ({_safe_decode_grade(global_label)}) outside "
                        f"filtered range [{self.min_grade_index}, {self.max_grade_index}]"
                    )
            return global_label - self.grade_offset

        if self.label_space_mode == "global_legacy":
            if global_label < 0 or global_label >= len(self.grade_names):
                raise ValueError(
                    f"Global label {global_label} out of range [0, {len(self.grade_names) - 1}]"
                )
            return global_label

        raise ValueError(f"Unsupported label_space_mode: {self.label_space_mode}")

    def model_to_global_label(self, model_label: int) -> int:
        """Map model label space to global moonboard_core label."""
        if model_label < 0 or model_label >= self.get_num_model_grades():
            raise ValueError(
                f"Model label {model_label} out of range [0, {self.get_num_model_grades() - 1}]"
            )
        if self.label_space_mode == "remapped":
            return model_label + self.grade_offset
        if self.label_space_mode == "global_legacy":
            return model_label
        raise ValueError(f"Unsupported label_space_mode: {self.label_space_mode}")

    def get_grade_from_label(self, label: int) -> Optional[str]:
        """
        Convert a grade label back to grade string.
        
        Args:
            label: Integer grade label
            
        Returns:
            grade: Grade string (e.g., "6B+")
        """
        try:
            global_label = self.model_to_global_label(int(label))
        except (TypeError, ValueError):
            return None
        return self.label_to_grade.get(global_label, None)
    
    def get_label_from_grade(self, grade: str) -> Optional[int]:
        """
        Convert a grade string to integer label.
        
        Args:
            grade: Grade string (e.g., "6B+")
            
        Returns:
            label: Integer grade label
        """
        global_label = self.grade_to_label.get(grade, None)
        if global_label is None:
            return None
        try:
            return self.global_to_model_label(global_label)
        except ValueError:
            return None


def create_data_loaders(
    data_path: str, 
    batch_size: int = 32, 
    train_split: float = 0.8, 
    shuffle: bool = True, 
    num_workers: int = 0, 
    min_grade_index: Optional[int] = None, 
    max_grade_index: Optional[int] = None,
    label_space_mode: Optional[LabelSpaceMode] = None,
) -> Tuple:
    """
    Create train and validation data loaders.
    
    Args:
        data_path: Path to problems.json
        batch_size: Batch size for data loaders
        train_split: Fraction of data to use for training (rest for validation)
        shuffle: Whether to shuffle the training data
        num_workers: Number of worker processes for data loading
        min_grade_index: Minimum grade index for filtering (e.g., 2 for 6A+)
        max_grade_index: Maximum grade index for filtering (e.g., 2 for 6A+ only)
        label_space_mode: Optional override for dataset label mapping mode
        
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        dataset: The full dataset (for accessing grade mappings)
    """
    from torch.utils.data import DataLoader, random_split
    
    # Create full dataset
    dataset = MoonBoardDataset(
        data_path,
        min_grade_index=min_grade_index,
        max_grade_index=max_grade_index,
        label_space_mode=label_space_mode,
    )
    
    # Split into train and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset
