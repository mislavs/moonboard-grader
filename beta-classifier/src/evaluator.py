"""
Evaluator Module

Metrics computation and evaluation for grade classification.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support
)

from moonboard_core import decode_grade, get_all_grades


class Evaluator:
    """
    Evaluator for computing classification metrics and visualizations.
    
    Metrics:
        - Exact accuracy
        - ±1 grade tolerance accuracy
        - ±2 grade tolerance accuracy
        - Mean absolute error (in grades)
        - Per-grade precision/recall/F1
        - Confusion matrix
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        grade_names: Optional[List[str]] = None
    ):
        """
        Args:
            model: Trained model to evaluate
            device: Device to run evaluation on
            grade_names: List of grade names for labeling (defaults to all grades)
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.grade_names = grade_names or get_all_grades()
    
    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions for all samples in data loader.
        
        Args:
            data_loader: DataLoader to evaluate
            
        Returns:
            Tuple of (predictions, true_labels, probabilities)
        """
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch_data, attention_mask, batch_labels in data_loader:
            batch_data = batch_data.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            outputs = self.model(batch_data, attention_mask)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_labels.numpy())
            all_probs.append(probs.cpu().numpy())
        
        return (
            np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs)
        )
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            predictions: Predicted grade indices
            true_labels: True grade indices
            
        Returns:
            Dict of metric name to value
        """
        # Exact accuracy
        exact_acc = accuracy_score(true_labels, predictions)
        
        # Tolerance accuracies
        errors = np.abs(predictions - true_labels)
        acc_1 = (errors <= 1).mean()
        acc_2 = (errors <= 2).mean()
        
        # Mean absolute error
        mae = mean_absolute_error(true_labels, predictions)
        
        # Macro F1
        macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        
        # Weighted F1
        weighted_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        return {
            'exact_accuracy': exact_acc,
            'accuracy_within_1': acc_1,
            'accuracy_within_2': acc_2,
            'mean_absolute_error': mae,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1
        }
    
    def compute_per_class_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class precision, recall, and F1.
        
        Args:
            predictions: Predicted grade indices
            true_labels: True grade indices
            
        Returns:
            Dict mapping grade name to metrics dict
        """
        # Get unique labels present in data
        present_labels = sorted(set(true_labels) | set(predictions))
        
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels,
            predictions,
            labels=present_labels,
            zero_division=0
        )
        
        per_class = {}
        for i, label in enumerate(present_labels):
            grade_name = decode_grade(int(label))
            per_class[grade_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': int(support[i])
            }
        
        return per_class
    
    def evaluate(self, data_loader: DataLoader) -> Dict:
        """
        Full evaluation on a data loader.
        
        Args:
            data_loader: DataLoader to evaluate
            
        Returns:
            Dict containing all metrics
        """
        predictions, true_labels, probabilities = self.predict(data_loader)
        
        metrics = self.compute_metrics(predictions, true_labels)
        per_class = self.compute_per_class_metrics(predictions, true_labels)
        
        return {
            'metrics': metrics,
            'per_class': per_class,
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities
        }
    
    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        normalize: bool = True
    ) -> plt.Figure:
        """
        Plot and optionally save confusion matrix.
        
        Args:
            predictions: Predicted grade indices
            true_labels: True grade indices
            save_path: Path to save figure (optional)
            figsize: Figure size
            normalize: Whether to normalize by row (true labels)
            
        Returns:
            Matplotlib figure
        """
        # Get labels present in data
        present_labels = sorted(set(true_labels) | set(predictions))
        label_names = [decode_grade(int(l)) for l in present_labels]
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=present_labels)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)  # Handle divide by zero
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax
        )
        
        ax.set_xlabel('Predicted Grade')
        ax.set_ylabel('True Grade')
        ax.set_title('Grade Classification Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
            plt.close(fig)
        
        return fig
    
    def plot_error_distribution(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot distribution of prediction errors.
        
        Args:
            predictions: Predicted grade indices
            true_labels: True grade indices
            save_path: Path to save figure (optional)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        errors = predictions - true_labels
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Histogram of errors
        unique_errors, counts = np.unique(errors, return_counts=True)
        colors = ['green' if e == 0 else 'orange' if abs(e) <= 1 else 'red' 
                  for e in unique_errors]
        
        ax.bar(unique_errors, counts, color=colors, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Prediction Error (predicted - true)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Grade Prediction Errors')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add stats text
        mae = np.mean(np.abs(errors))
        exact_acc = (errors == 0).mean() * 100
        within_1 = (np.abs(errors) <= 1).mean() * 100
        
        stats_text = f'MAE: {mae:.2f} grades\nExact: {exact_acc:.1f}%\n±1: {within_1:.1f}%'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Error distribution saved to {save_path}")
            plt.close(fig)
        
        return fig
    
    def print_report(self, results: Dict) -> None:
        """Print formatted evaluation report."""
        metrics = results['metrics']
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        
        print("\nOverall Metrics:")
        print(f"  Exact Accuracy:      {metrics['exact_accuracy']:.4f} ({metrics['exact_accuracy']*100:.2f}%)")
        print(f"  Accuracy (±1 grade): {metrics['accuracy_within_1']:.4f} ({metrics['accuracy_within_1']*100:.2f}%)")
        print(f"  Accuracy (±2 grade): {metrics['accuracy_within_2']:.4f} ({metrics['accuracy_within_2']*100:.2f}%)")
        print(f"  Mean Absolute Error: {metrics['mean_absolute_error']:.4f} grades")
        print(f"  Macro F1:            {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:         {metrics['weighted_f1']:.4f}")
        
        print("\nPer-Grade Metrics:")
        print("-" * 60)
        print(f"{'Grade':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 60)
        
        for grade, class_metrics in results['per_class'].items():
            print(f"{grade:<8} {class_metrics['precision']:>10.4f} "
                  f"{class_metrics['recall']:>10.4f} {class_metrics['f1']:>10.4f} "
                  f"{class_metrics['support']:>10}")
        
        print("=" * 60)

