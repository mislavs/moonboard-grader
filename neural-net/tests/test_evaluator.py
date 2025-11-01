"""
Tests for evaluation metrics and visualization module.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os

# Use non-interactive backend for matplotlib to avoid Tkinter issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.evaluator import (
    evaluate_model,
    calculate_exact_accuracy,
    calculate_tolerance_accuracy,
    generate_confusion_matrix,
    plot_confusion_matrix,
    per_grade_metrics,
    calculate_mean_absolute_error,
    get_metrics_summary
)
from src.grade_encoder import get_all_grades, get_num_grades


# Simple model for testing
class SimpleModel(nn.Module):
    """Simple model that returns fixed logits for testing."""
    def __init__(self, num_classes=19):
        super().__init__()
        self.fc = nn.Linear(3 * 18 * 11, num_classes)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.fc(x)


class TestCalculateExactAccuracy:
    """Tests for exact accuracy calculation."""
    
    def test_perfect_accuracy(self):
        """Test 100% accuracy when all predictions match."""
        predictions = np.array([0, 1, 2, 3, 4])
        labels = np.array([0, 1, 2, 3, 4])
        accuracy = calculate_exact_accuracy(predictions, labels)
        assert accuracy == 100.0
    
    def test_zero_accuracy(self):
        """Test 0% accuracy when no predictions match."""
        predictions = np.array([0, 0, 0, 0])
        labels = np.array([1, 2, 3, 4])
        accuracy = calculate_exact_accuracy(predictions, labels)
        assert accuracy == 0.0
    
    def test_partial_accuracy(self):
        """Test partial accuracy."""
        predictions = np.array([0, 1, 2, 3, 4])
        labels = np.array([0, 1, 2, 5, 6])
        accuracy = calculate_exact_accuracy(predictions, labels)
        assert accuracy == 60.0  # 3 out of 5 correct
    
    def test_single_sample(self):
        """Test with single sample."""
        predictions = np.array([5])
        labels = np.array([5])
        accuracy = calculate_exact_accuracy(predictions, labels)
        assert accuracy == 100.0
    
    def test_list_inputs(self):
        """Test that lists are converted to arrays."""
        predictions = [0, 1, 2]
        labels = [0, 1, 2]
        accuracy = calculate_exact_accuracy(predictions, labels)
        assert accuracy == 100.0
    
    def test_empty_arrays(self):
        """Test that empty arrays raise error."""
        with pytest.raises(ValueError, match="empty"):
            calculate_exact_accuracy(np.array([]), np.array([]))
    
    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise error."""
        predictions = np.array([0, 1, 2])
        labels = np.array([0, 1])
        with pytest.raises(ValueError, match="same length"):
            calculate_exact_accuracy(predictions, labels)


class TestCalculateToleranceAccuracy:
    """Tests for tolerance-based accuracy calculation."""
    
    def test_tolerance_1_all_exact(self):
        """Test tolerance=1 when all predictions are exact."""
        predictions = np.array([0, 1, 2, 3])
        labels = np.array([0, 1, 2, 3])
        accuracy = calculate_tolerance_accuracy(predictions, labels, tolerance=1)
        assert accuracy == 100.0
    
    def test_tolerance_1_off_by_one(self):
        """Test tolerance=1 with predictions off by 1."""
        predictions = np.array([0, 2, 3, 5])
        labels = np.array([1, 1, 2, 4])
        # All are within ±1: |0-1|=1, |2-1|=1, |3-2|=1, |5-4|=1
        accuracy = calculate_tolerance_accuracy(predictions, labels, tolerance=1)
        assert accuracy == 100.0
    
    def test_tolerance_1_some_off_by_two(self):
        """Test tolerance=1 with some predictions off by 2."""
        predictions = np.array([0, 1, 5, 3])
        labels = np.array([0, 1, 3, 1])
        # Within tolerance: 0==0 ✓, 1==1 ✓, |5-3|=2 ✗, |3-1|=2 ✗
        accuracy = calculate_tolerance_accuracy(predictions, labels, tolerance=1)
        assert accuracy == 50.0
    
    def test_tolerance_2(self):
        """Test tolerance=2."""
        predictions = np.array([0, 1, 5, 3])
        labels = np.array([0, 1, 3, 1])
        # Within tolerance: 0==0 ✓, 1==1 ✓, |5-3|=2 ✓, |3-1|=2 ✓
        accuracy = calculate_tolerance_accuracy(predictions, labels, tolerance=2)
        assert accuracy == 100.0
    
    def test_tolerance_0_same_as_exact(self):
        """Test that tolerance=0 is same as exact accuracy."""
        predictions = np.array([0, 1, 2, 3, 4])
        labels = np.array([0, 1, 2, 5, 6])
        exact = calculate_exact_accuracy(predictions, labels)
        tol0 = calculate_tolerance_accuracy(predictions, labels, tolerance=0)
        assert exact == tol0
    
    def test_negative_tolerance(self):
        """Test that negative tolerance raises error."""
        predictions = np.array([0, 1, 2])
        labels = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="non-negative"):
            calculate_tolerance_accuracy(predictions, labels, tolerance=-1)
    
    def test_empty_arrays(self):
        """Test that empty arrays raise error."""
        with pytest.raises(ValueError, match="empty"):
            calculate_tolerance_accuracy(np.array([]), np.array([]), tolerance=1)


class TestGenerateConfusionMatrix:
    """Tests for confusion matrix generation."""
    
    def test_perfect_predictions(self):
        """Test confusion matrix with perfect predictions."""
        predictions = np.array([0, 1, 2, 0, 1, 2])
        labels = np.array([0, 1, 2, 0, 1, 2])
        cm = generate_confusion_matrix(predictions, labels, num_classes=3)
        
        expected = np.array([
            [2, 0, 0],
            [0, 2, 0],
            [0, 0, 2]
        ])
        np.testing.assert_array_equal(cm, expected)
    
    def test_all_wrong_same_prediction(self):
        """Test when all predictions are the same but wrong."""
        predictions = np.array([1, 1, 1, 1])
        labels = np.array([0, 2, 3, 4])
        cm = generate_confusion_matrix(predictions, labels, num_classes=5)
        
        assert cm[0, 1] == 1  # True 0, predicted 1
        assert cm[2, 1] == 1  # True 2, predicted 1
        assert cm[3, 1] == 1  # True 3, predicted 1
        assert cm[4, 1] == 1  # True 4, predicted 1
        assert cm.sum() == 4
    
    def test_mixed_predictions(self):
        """Test with mixed correct/incorrect predictions."""
        predictions = np.array([0, 0, 1, 1, 2, 3])
        labels = np.array([0, 1, 1, 2, 2, 3])
        cm = generate_confusion_matrix(predictions, labels, num_classes=4)
        
        assert cm[0, 0] == 1  # True 0, predicted 0 (correct)
        assert cm[1, 0] == 1  # True 1, predicted 0 (wrong)
        assert cm[1, 1] == 1  # True 1, predicted 1 (correct)
        assert cm[2, 1] == 1  # True 2, predicted 1 (wrong)
        assert cm[2, 2] == 1  # True 2, predicted 2 (correct)
        assert cm[3, 3] == 1  # True 3, predicted 3 (correct)
    
    def test_default_num_classes(self):
        """Test that default num_classes uses get_num_grades()."""
        predictions = np.array([0, 1, 2])
        labels = np.array([0, 1, 2])
        cm = generate_confusion_matrix(predictions, labels)
        
        assert cm.shape == (get_num_grades(), get_num_grades())
    
    def test_empty_arrays(self):
        """Test that empty arrays raise error."""
        with pytest.raises(ValueError, match="empty"):
            generate_confusion_matrix(np.array([]), np.array([]), num_classes=3)


class TestPlotConfusionMatrix:
    """Tests for confusion matrix visualization."""
    
    def test_basic_plot(self):
        """Test basic confusion matrix plotting."""
        cm = np.array([
            [10, 2, 0],
            [1, 15, 3],
            [0, 2, 8]
        ])
        grade_names = ['6A', '6A+', '6B']
        
        fig = plot_confusion_matrix(cm, grade_names)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_save_to_file(self):
        """Test saving confusion matrix to file."""
        cm = np.array([[5, 1], [2, 3]])
        grade_names = ['6A', '6B']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'confusion_matrix.png')
            fig = plot_confusion_matrix(cm, grade_names, save_path=save_path)
            
            assert os.path.exists(save_path)
            assert os.path.getsize(save_path) > 0
            plt.close(fig)
    
    def test_normalized_plot(self):
        """Test normalized confusion matrix (percentages)."""
        cm = np.array([
            [10, 5],
            [2, 8]
        ])
        grade_names = ['6A', '6B']
        
        fig = plot_confusion_matrix(cm, grade_names, normalize=True)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_default_grade_names(self):
        """Test using default grade names from get_all_grades()."""
        num_grades = get_num_grades()
        cm = np.eye(num_grades)  # Identity matrix (perfect predictions)
        
        fig = plot_confusion_matrix(cm)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_custom_figsize(self):
        """Test custom figure size."""
        cm = np.array([[5, 1], [2, 3]])
        grade_names = ['6A', '6B']
        
        fig = plot_confusion_matrix(cm, grade_names, figsize=(8, 6))
        
        assert fig.get_size_inches()[0] == 8
        assert fig.get_size_inches()[1] == 6
        plt.close(fig)
    
    def test_invalid_cm_shape(self):
        """Test that non-square matrix raises error."""
        cm = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="square"):
            plot_confusion_matrix(cm, ['A', 'B'])
    
    def test_mismatched_grade_names(self):
        """Test that mismatched grade_names length raises error."""
        cm = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="must match"):
            plot_confusion_matrix(cm, ['A', 'B', 'C'])


class TestPerGradeMetrics:
    """Tests for per-grade metrics calculation."""
    
    def test_perfect_predictions(self):
        """Test per-grade metrics with perfect predictions."""
        predictions = np.array([0, 0, 1, 1, 2, 2])
        labels = np.array([0, 0, 1, 1, 2, 2])
        grade_names = ['6A', '6A+', '6B']
        
        metrics = per_grade_metrics(predictions, labels, grade_names)
        
        assert len(metrics) == 3
        for grade in grade_names:
            assert metrics[grade]['precision'] == 1.0
            assert metrics[grade]['recall'] == 1.0
            assert metrics[grade]['f1'] == 1.0
            assert metrics[grade]['support'] > 0
    
    def test_no_predictions_for_class(self):
        """Test when a class has no predictions."""
        predictions = np.array([0, 0, 0, 0])
        labels = np.array([0, 0, 1, 2])
        grade_names = ['6A', '6A+', '6B']
        
        metrics = per_grade_metrics(predictions, labels, grade_names)
        
        # Class 0 should have good recall but imperfect precision
        assert metrics['6A']['recall'] == 1.0
        assert metrics['6A']['precision'] < 1.0
        
        # Classes 1 and 2 should have 0 recall
        assert metrics['6A+']['recall'] == 0.0
        assert metrics['6B']['recall'] == 0.0
    
    def test_mixed_performance(self):
        """Test with mixed prediction performance."""
        predictions = np.array([0, 1, 1, 2, 2, 2])
        labels = np.array([0, 0, 1, 1, 2, 2])
        grade_names = ['6A', '6A+', '6B']
        
        metrics = per_grade_metrics(predictions, labels, grade_names)
        
        # Class 0: 1 TP, 1 FN, 0 FP -> precision=1.0, recall=0.5
        assert metrics['6A']['precision'] == 1.0
        assert metrics['6A']['recall'] == 0.5
        
        # Class 1: 1 TP, 1 FN, 1 FP -> precision=0.5, recall=0.5
        assert metrics['6A+']['precision'] == 0.5
        assert metrics['6A+']['recall'] == 0.5
    
    def test_support_counts(self):
        """Test that support counts are correct."""
        predictions = np.array([0, 0, 1, 1, 2, 2, 2])
        labels = np.array([0, 0, 0, 1, 1, 2, 2])
        grade_names = ['6A', '6A+', '6B']
        
        metrics = per_grade_metrics(predictions, labels, grade_names)
        
        assert metrics['6A']['support'] == 3
        assert metrics['6A+']['support'] == 2
        assert metrics['6B']['support'] == 2
    
    def test_default_grade_names(self):
        """Test using default grade names."""
        predictions = np.array([0, 1, 2])
        labels = np.array([0, 1, 2])
        
        metrics = per_grade_metrics(predictions, labels)
        
        assert len(metrics) == get_num_grades()
        assert '5+' in metrics
        assert '6A' in metrics
    
    def test_empty_arrays(self):
        """Test that empty arrays raise error."""
        with pytest.raises(ValueError, match="empty"):
            per_grade_metrics(np.array([]), np.array([]))


class TestCalculateMeanAbsoluteError:
    """Tests for mean absolute error calculation."""
    
    def test_zero_error(self):
        """Test MAE when all predictions are exact."""
        predictions = np.array([0, 1, 2, 3, 4])
        labels = np.array([0, 1, 2, 3, 4])
        mae = calculate_mean_absolute_error(predictions, labels)
        assert mae == 0.0
    
    def test_all_off_by_one(self):
        """Test MAE when all predictions are off by 1."""
        predictions = np.array([1, 2, 3, 4, 5])
        labels = np.array([0, 1, 2, 3, 4])
        mae = calculate_mean_absolute_error(predictions, labels)
        assert mae == 1.0
    
    def test_mixed_errors(self):
        """Test MAE with mixed errors."""
        predictions = np.array([0, 1, 4, 3])
        labels = np.array([0, 1, 2, 1])
        # Errors: 0, 0, 2, 2 -> mean = 1.0
        mae = calculate_mean_absolute_error(predictions, labels)
        assert mae == 1.0
    
    def test_fractional_mae(self):
        """Test MAE with fractional result."""
        predictions = np.array([0, 1, 2, 5])
        labels = np.array([0, 1, 2, 3])
        # Errors: 0, 0, 0, 2 -> mean = 0.5
        mae = calculate_mean_absolute_error(predictions, labels)
        assert mae == 0.5
    
    def test_empty_arrays(self):
        """Test that empty arrays raise error."""
        with pytest.raises(ValueError, match="empty"):
            calculate_mean_absolute_error(np.array([]), np.array([]))


class TestGetMetricsSummary:
    """Tests for comprehensive metrics summary."""
    
    def test_summary_contains_all_metrics(self):
        """Test that summary contains all expected metrics."""
        predictions = np.array([0, 1, 2, 3, 4])
        labels = np.array([0, 1, 2, 3, 4])
        
        summary = get_metrics_summary(predictions, labels)
        
        assert 'exact_accuracy' in summary
        assert 'tolerance_1_accuracy' in summary
        assert 'tolerance_2_accuracy' in summary
        assert 'mean_absolute_error' in summary
        assert 'num_samples' in summary
        assert 'per_grade_metrics' in summary
    
    def test_summary_values(self):
        """Test that summary values are correct."""
        predictions = np.array([0, 1, 2, 3, 4])
        labels = np.array([0, 1, 2, 3, 4])
        
        summary = get_metrics_summary(predictions, labels)
        
        assert summary['exact_accuracy'] == 100.0
        assert summary['tolerance_1_accuracy'] == 100.0
        assert summary['mean_absolute_error'] == 0.0
        assert summary['num_samples'] == 5
        assert isinstance(summary['per_grade_metrics'], dict)


class TestEvaluateModel:
    """Tests for full model evaluation."""
    
    def test_basic_evaluation(self):
        """Test basic model evaluation."""
        # Create simple dataset
        num_samples = 20
        data = torch.randn(num_samples, 3, 18, 11)
        labels = torch.randint(0, 19, (num_samples,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=4)
        
        # Create and evaluate model
        model = SimpleModel(num_classes=19)
        results = evaluate_model(model, dataloader, device='cpu')
        
        assert 'exact_accuracy' in results
        assert 'tolerance_1_accuracy' in results
        assert 'tolerance_2_accuracy' in results
        assert 'avg_loss' in results
        assert 'predictions' in results
        assert 'labels' in results
        assert 'num_samples' in results
        
        assert results['num_samples'] == num_samples
        assert len(results['predictions']) == num_samples
        assert len(results['labels']) == num_samples
    
    def test_evaluation_with_known_predictions(self):
        """Test evaluation where we can predict the output."""
        # Create dataset where all samples have same label
        num_samples = 10
        data = torch.randn(num_samples, 3, 18, 11)
        labels = torch.zeros(num_samples, dtype=torch.long)
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=5)
        
        model = SimpleModel(num_classes=19)
        results = evaluate_model(model, dataloader, device='cpu')
        
        # All true labels should be 0
        assert all(label == 0 for label in results['labels'])
    
    def test_tolerance_accuracy_ordering(self):
        """Test that tolerance accuracies are ordered correctly."""
        num_samples = 20
        data = torch.randn(num_samples, 3, 18, 11)
        labels = torch.randint(0, 19, (num_samples,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=4)
        
        model = SimpleModel(num_classes=19)
        results = evaluate_model(model, dataloader, device='cpu')
        
        # Tolerance accuracies should be ordered: exact <= tol1 <= tol2
        assert results['exact_accuracy'] <= results['tolerance_1_accuracy']
        assert results['tolerance_1_accuracy'] <= results['tolerance_2_accuracy']
    
    def test_evaluation_single_batch(self):
        """Test evaluation with single batch."""
        num_samples = 5
        data = torch.randn(num_samples, 3, 18, 11)
        labels = torch.randint(0, 19, (num_samples,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=10)  # Larger than dataset
        
        model = SimpleModel(num_classes=19)
        results = evaluate_model(model, dataloader, device='cpu')
        
        assert results['num_samples'] == num_samples
    
    def test_invalid_model(self):
        """Test that invalid model raises error."""
        dataloader = DataLoader(TensorDataset(torch.randn(5, 3, 18, 11), torch.zeros(5, dtype=torch.long)))
        
        with pytest.raises(ValueError, match="PyTorch Module"):
            evaluate_model("not a model", dataloader)
    
    def test_empty_dataloader(self):
        """Test that empty dataloader raises error."""
        model = SimpleModel(num_classes=19)
        empty_dataloader = DataLoader(TensorDataset(torch.randn(0, 3, 18, 11), torch.zeros(0, dtype=torch.long)))
        
        with pytest.raises(ValueError, match="empty"):
            evaluate_model(model, empty_dataloader)
    
    def test_model_in_eval_mode(self):
        """Test that model is set to eval mode during evaluation."""
        num_samples = 10
        data = torch.randn(num_samples, 3, 18, 11)
        labels = torch.randint(0, 19, (num_samples,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=5)
        
        model = SimpleModel(num_classes=19)
        model.train()  # Start in training mode
        
        evaluate_model(model, dataloader, device='cpu')
        
        # Model should be in eval mode after evaluation
        assert not model.training


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

