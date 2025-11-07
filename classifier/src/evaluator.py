"""
Evaluation metrics and visualization for moonboard grade prediction models.

This module provides comprehensive evaluation tools including:
- Model evaluation on datasets
- Accuracy metrics (exact and tolerance-based)
- Confusion matrix generation and visualization
- Per-grade performance metrics
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from moonboard_core.grade_encoder import get_all_grades, get_num_grades


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                   device: str = 'cpu') -> Dict:
    """
    Evaluate a model on a dataset and return comprehensive metrics.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing:
            - exact_accuracy: Percentage of exact grade matches
            - tolerance_1_accuracy: Percentage within ±1 grade
            - tolerance_2_accuracy: Percentage within ±2 grades
            - avg_loss: Average cross-entropy loss
            - predictions: List of predicted labels
            - labels: List of true labels
            - num_samples: Total number of samples evaluated
            
    Raises:
        ValueError: If dataloader is empty or model/dataloader are invalid
    """
    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a PyTorch Module")
    
    if not hasattr(dataloader, '__iter__'):
        raise ValueError("dataloader must be iterable")
    
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            total_loss += loss.item()
            num_batches += 1
    
    if num_batches == 0:
        raise ValueError("dataloader is empty")
    
    # Convert to numpy arrays for metric calculations
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Calculate metrics
    exact_acc = calculate_exact_accuracy(predictions, labels)
    tol1_acc = calculate_tolerance_accuracy(predictions, labels, tolerance=1)
    tol2_acc = calculate_tolerance_accuracy(predictions, labels, tolerance=2)
    avg_loss = total_loss / num_batches
    
    return {
        'exact_accuracy': exact_acc,
        'tolerance_1_accuracy': tol1_acc,
        'tolerance_2_accuracy': tol2_acc,
        'avg_loss': avg_loss,
        'predictions': predictions.tolist(),
        'labels': labels.tolist(),
        'num_samples': len(labels)
    }


def calculate_exact_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the percentage of exact grade matches.
    
    Args:
        predictions: Array of predicted labels
        labels: Array of true labels
        
    Returns:
        Accuracy as percentage (0-100)
        
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    
    if len(predictions) == 0:
        raise ValueError("predictions array is empty")
    
    if len(predictions) != len(labels):
        raise ValueError(f"predictions ({len(predictions)}) and labels ({len(labels)}) must have same length")
    
    correct = np.sum(predictions == labels)
    accuracy = (correct / len(labels)) * 100.0
    
    return float(accuracy)


def calculate_tolerance_accuracy(predictions: np.ndarray, labels: np.ndarray, 
                                 tolerance: int = 1) -> float:
    """
    Calculate accuracy allowing for predictions within ±tolerance grades.
    
    This is useful for climbing grade prediction since being off by 1 grade
    is still very useful (e.g., predicting 6B when actual is 6B+ is close).
    
    Args:
        predictions: Array of predicted labels
        labels: Array of true labels
        tolerance: Number of grades to allow as margin (default 1)
        
    Returns:
        Accuracy as percentage (0-100)
        
    Raises:
        ValueError: If arrays have different lengths, are empty, or tolerance is invalid
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    
    if len(predictions) == 0:
        raise ValueError("predictions array is empty")
    
    if len(predictions) != len(labels):
        raise ValueError(f"predictions ({len(predictions)}) and labels ({len(labels)}) must have same length")
    
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    
    # Calculate absolute difference
    differences = np.abs(predictions - labels)
    
    # Count predictions within tolerance
    within_tolerance = np.sum(differences <= tolerance)
    accuracy = (within_tolerance / len(labels)) * 100.0
    
    return float(accuracy)


def generate_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, 
                              num_classes: Optional[int] = None) -> np.ndarray:
    """
    Generate confusion matrix for predictions.
    
    Args:
        predictions: Array of predicted labels
        labels: Array of true labels
        num_classes: Number of classes (default: use get_num_grades())
        
    Returns:
        Confusion matrix as numpy array of shape (num_classes, num_classes)
        Element [i, j] is count of samples with true label i predicted as j
        
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    
    if len(predictions) == 0:
        raise ValueError("predictions array is empty")
    
    if len(predictions) != len(labels):
        raise ValueError(f"predictions ({len(predictions)}) and labels ({len(labels)}) must have same length")
    
    if num_classes is None:
        num_classes = get_num_grades()
    
    # Use sklearn's confusion_matrix
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    
    return cm


def plot_confusion_matrix(cm: np.ndarray, grade_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 10),
                         normalize: bool = False) -> plt.Figure:
    """
    Create a visualization of the confusion matrix.
    
    Args:
        cm: Confusion matrix as numpy array
        grade_names: List of grade names for axis labels (default: use get_all_grades())
        save_path: Path to save figure (optional, if None shows plot)
        figsize: Figure size as (width, height) tuple
        normalize: If True, normalize by row (show percentages)
        
    Returns:
        Matplotlib figure object
        
    Raises:
        ValueError: If cm is not 2D square array or grade_names length doesn't match
    """
    cm = np.asarray(cm)
    
    if cm.ndim != 2:
        raise ValueError(f"confusion matrix must be 2D, got shape {cm.shape}")
    
    if cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion matrix must be square, got shape {cm.shape}")
    
    if grade_names is None:
        grade_names = get_all_grades()
    
    if len(grade_names) != cm.shape[0]:
        raise ValueError(f"grade_names length ({len(grade_names)}) must match cm size ({cm.shape[0]})")
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float')
        row_sums = cm.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums * 100
        fmt = '.1f'
        cbar_label = 'Percentage (%)'
    else:
        cm = cm.astype('int')
        fmt = 'd'
        cbar_label = 'Count'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=grade_names, yticklabels=grade_names,
                cbar_kws={'label': cbar_label}, ax=ax)
    
    ax.set_xlabel('Predicted Grade', fontsize=12)
    ax.set_ylabel('True Grade', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def per_grade_metrics(predictions: np.ndarray, labels: np.ndarray,
                     grade_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate precision, recall, and F1 score for each grade.
    
    Args:
        predictions: Array of predicted labels
        labels: Array of true labels
        grade_names: List of grade names (default: use get_all_grades())
        
    Returns:
        Dictionary mapping grade names to dicts with 'precision', 'recall', 'f1', and 'support'
        
    Raises:
        ValueError: If arrays have different lengths, are empty, or grade_names length doesn't match
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    
    if len(predictions) == 0:
        raise ValueError("predictions array is empty")
    
    if len(predictions) != len(labels):
        raise ValueError(f"predictions ({len(predictions)}) and labels ({len(labels)}) must have same length")
    
    if grade_names is None:
        grade_names = get_all_grades()
    
    num_classes = len(grade_names)
    
    # Calculate metrics using sklearn
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(num_classes)), zero_division=0
    )
    
    # Build result dictionary
    result = {}
    for i, grade in enumerate(grade_names):
        result[grade] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    return result


def calculate_mean_absolute_error(predictions: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate mean absolute error in terms of grade steps.
    
    This metric shows on average how many grades off the predictions are.
    For example, MAE of 0.5 means predictions are on average half a grade off.
    
    Args:
        predictions: Array of predicted labels
        labels: Array of true labels
        
    Returns:
        Mean absolute error as float
        
    Raises:
        ValueError: If arrays have different lengths or are empty
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    
    if len(predictions) == 0:
        raise ValueError("predictions array is empty")
    
    if len(predictions) != len(labels):
        raise ValueError(f"predictions ({len(predictions)}) and labels ({len(labels)}) must have same length")
    
    mae = np.mean(np.abs(predictions - labels))
    
    return float(mae)


def get_metrics_summary(predictions: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Get a comprehensive summary of all evaluation metrics.
    
    Args:
        predictions: Array of predicted labels
        labels: Array of true labels
        
    Returns:
        Dictionary with all metrics including accuracies, MAE, and per-grade metrics
    """
    return {
        'exact_accuracy': calculate_exact_accuracy(predictions, labels),
        'tolerance_1_accuracy': calculate_tolerance_accuracy(predictions, labels, tolerance=1),
        'tolerance_2_accuracy': calculate_tolerance_accuracy(predictions, labels, tolerance=2),
        'mean_absolute_error': calculate_mean_absolute_error(predictions, labels),
        'num_samples': len(labels),
        'per_grade_metrics': per_grade_metrics(predictions, labels)
    }

