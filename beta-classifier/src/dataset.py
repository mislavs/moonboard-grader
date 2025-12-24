"""
Dataset Module

Provides data loading, feature normalization, and PyTorch dataset classes
for move sequence classification.
"""

import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from moonboard_core import encode_grade

# Number of features per move
NUM_FEATURES = 15


class FeatureNormalizer:
    """
    Z-score normalizer for move sequence features.
    
    Fit on training data only to avoid data leakage, then persist for inference.
    """
    
    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self._fitted = False
    
    def fit(self, train_sequences: List[np.ndarray]) -> 'FeatureNormalizer':
        """
        Compute mean and std from training sequences.
        
        Args:
            train_sequences: List of (seq_len, 15) feature arrays
            
        Returns:
            self for method chaining
        """
        # Concatenate all moves to compute global statistics
        all_features = np.concatenate(train_sequences, axis=0)
        self.mean = all_features.mean(axis=0)  # (15,)
        self.std = all_features.std(axis=0)    # (15,)
        
        # Prevent division by zero for constant features
        self.std[self.std == 0] = 1.0
        
        self._fitted = True
        return self
    
    def transform(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply z-score normalization using fitted statistics.
        
        Args:
            sequences: List of (seq_len, 15) feature arrays
            
        Returns:
            List of normalized feature arrays
        """
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer must be fitted before transform")
        
        return [(seq - self.mean) / self.std for seq in sequences]
    
    def fit_transform(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """Fit and transform in one step."""
        return self.fit(sequences).transform(sequences)
    
    def save(self, path: str) -> None:
        """Save normalizer statistics to .npz file."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted normalizer")
        np.savez(path, mean=self.mean, std=self.std)
    
    @classmethod
    def load(cls, path: str) -> 'FeatureNormalizer':
        """Load normalizer from .npz file."""
        data = np.load(path)
        norm = cls()
        norm.mean = data['mean']
        norm.std = data['std']
        norm._fitted = True
        return norm


class MoveSequenceDataset(Dataset):
    """
    PyTorch Dataset for move sequences.
    
    Each sample is a variable-length sequence of 15-dimensional move features.
    """
    
    def __init__(self, sequences: List[np.ndarray], labels: np.ndarray):
        """
        Args:
            sequences: List of (seq_len, 15) normalized feature arrays
            labels: Integer grade labels, shape (n_samples,)
        """
        self.sequences = sequences
        self.labels = labels
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return torch.FloatTensor(self.sequences[idx]), self.labels[idx]
    
    def __len__(self) -> int:
        return len(self.sequences)


def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    
    Pads sequences to max length in batch and creates attention masks.
    
    Args:
        batch: List of (sequence, label) tuples
        
    Returns:
        Tuple of (padded_sequences, attention_masks, labels)
        - padded_sequences: (batch_size, max_len, 15)
        - attention_masks: (batch_size, max_len) - 1 for real, 0 for padding
        - labels: (batch_size,)
    """
    sequences, labels = zip(*batch)
    lengths = [len(s) for s in sequences]
    max_len = max(lengths)
    
    # Initialize padded tensor and mask
    padded = torch.zeros(len(batch), max_len, NUM_FEATURES)
    mask = torch.zeros(len(batch), max_len)
    
    for i, (seq, length) in enumerate(zip(sequences, lengths)):
        padded[i, :length] = seq
        mask[i, :length] = 1.0
    
    return padded, mask, torch.LongTensor(labels)


def extract_features(moves: List[Dict]) -> np.ndarray:
    """
    Extract 15 features from each move in a sequence.
    
    Features:
        0-1: targetX, targetY
        2-3: stationaryX, stationaryY
        4-5: originX, originY
        6-8: targetDifficulty, stationaryDifficulty, originDifficulty
        9-10: bodyStretchDx, bodyStretchDy
        11-12: travelDx, travelDy
        13: hand (0 or 1)
        14: successScore
        
    Args:
        moves: List of move dictionaries from beta solver
        
    Returns:
        Feature array of shape (n_moves, 15)
    """
    features = []
    for m in moves:
        features.append([
            m['targetX'], m['targetY'],
            m['stationaryX'], m['stationaryY'],
            m['originX'], m['originY'],
            m['targetDifficulty'], m['stationaryDifficulty'], m['originDifficulty'],
            m['bodyStretchDx'], m['bodyStretchDy'],
            m['travelDx'], m['travelDy'],
            m['hand'], m['successScore']
        ])
    return np.array(features, dtype=np.float32)


def load_data(path: str) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load and parse beta solver JSON data.
    
    Args:
        path: Path to solved_problems.json
        
    Returns:
        Tuple of (sequences, labels)
        - sequences: List of (seq_len, 15) feature arrays
        - labels: Integer grade labels array
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sequences = []
    labels = []
    
    for problem in data:
        if 'moves' not in problem or len(problem['moves']) == 0:
            continue
        
        features = extract_features(problem['moves'])
        sequences.append(features)
        labels.append(encode_grade(problem['grade']))
    
    return sequences, np.array(labels, dtype=np.int64)


def create_data_splits(
    sequences: List[np.ndarray],
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, Tuple[List[np.ndarray], np.ndarray]]:
    """
    Split data into train/val/test sets with stratification.
    
    Args:
        sequences: List of feature arrays
        labels: Integer labels
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict with 'train', 'val', 'test' keys, each containing (sequences, labels)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    indices = np.arange(len(sequences))
    
    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_seed
    )
    
    # Second split: val vs test
    temp_labels = labels[temp_idx]
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_ratio,
        stratify=temp_labels,
        random_state=random_seed
    )
    
    def gather(idx):
        return [sequences[i] for i in idx], labels[idx]
    
    return {
        'train': gather(train_idx),
        'val': gather(val_idx),
        'test': gather(test_idx)
    }


def create_dataloaders(
    splits: Dict[str, Tuple[List[np.ndarray], np.ndarray]],
    normalizer: FeatureNormalizer,
    batch_size: int = 64,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for all splits.
    
    Args:
        splits: Dict from create_data_splits
        normalizer: Fitted FeatureNormalizer
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        
    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    loaders = {}
    
    for split_name, (sequences, labels) in splits.items():
        normalized = normalizer.transform(sequences)
        dataset = MoveSequenceDataset(normalized, labels)
        
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders

