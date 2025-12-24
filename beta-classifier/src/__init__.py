"""
Beta Classifier Package

Transformer-based climbing grade classifier using move sequences from beta solver.
"""

from .dataset import FeatureNormalizer, MoveSequenceDataset, collate_fn
from .model import TransformerSequenceClassifier
from .trainer import Trainer
from .evaluator import Evaluator
from .predictor import Predictor

__version__ = "0.1.0"

__all__ = [
    "FeatureNormalizer",
    "MoveSequenceDataset",
    "collate_fn",
    "TransformerSequenceClassifier",
    "Trainer",
    "Evaluator",
    "Predictor",
]

