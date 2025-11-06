"""
CLI Command Module

Provides command-line interface commands for training, evaluation, and prediction.
"""

from .train import train_command
from .evaluate import evaluate_command
from .predict import predict_command

__all__ = [
    'train_command',
    'evaluate_command',
    'predict_command',
]

