"""
Evaluate Command

Handles model evaluation on test datasets with comprehensive metrics reporting.
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from .utils import setup_device, print_section_header, print_completion_message
from src import (
    load_dataset,
    Predictor,
    evaluate_model,
    get_metrics_summary,
    generate_confusion_matrix,
    plot_confusion_matrix,
    get_all_grades,
    MoonboardDataset,
)


def setup_evaluate_parser(subparsers):
    """
    Setup argument parser for evaluate command.
    
    Args:
        subparsers: ArgumentParser subparsers object
        
    Returns:
        Configured evaluate parser
    """
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    eval_parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to evaluation data JSON file'
    )
    eval_parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage (default: use CUDA if available)'
    )
    eval_parser.add_argument(
        '--save-confusion-matrix',
        action='store_true',
        help='Save confusion matrix plot'
    )
    eval_parser.add_argument(
        '--output',
        type=str,
        help='Output path for confusion matrix (default: confusion_matrix.png)'
    )
    eval_parser.set_defaults(func=evaluate_command)
    return eval_parser


def evaluate_command(args):
    """
    Execute evaluation command.
    
    Args:
        args: Parsed command-line arguments
    """
    print_section_header("MOONBOARD GRADE PREDICTION - EVALUATION")
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[ERROR] Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"\n* Loading model from: {checkpoint_path}")
    
    # Load predictor (handles model loading)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"* Using device: {device}")
    
    predictor = Predictor(str(checkpoint_path), device=device)
    model_info = predictor.get_model_info()
    
    print(f"\n>> Model Information:")
    print(f"   Type: {model_info['model_type']}")
    print(f"   Parameters: {model_info['num_parameters']:,}")
    print(f"   Classes: {model_info['num_classes']}")
    
    # Load evaluation data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Error: Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"\n>> Loading evaluation data from: {data_path}")
    dataset = load_dataset(str(data_path))
    
    if len(dataset) == 0:
        print("[ERROR] Error: No problems found in dataset")
        sys.exit(1)
    
    print(f"   Total problems: {len(dataset)}")
    
    # Check if model uses filtered grades (from checkpoint metadata)
    # Load checkpoint to get filtering info
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    grade_offset = checkpoint.get('grade_offset', 0)
    min_grade_idx = checkpoint.get('min_grade_index', 0)
    max_grade_idx = checkpoint.get('max_grade_index', 18)
    
    if grade_offset > 0:
        from src import filter_dataset_by_grades, remap_label, decode_grade
        
        print(f"\n>> Detected filtered model:")
        print(f"   Grade range: {decode_grade(min_grade_idx)} - {decode_grade(max_grade_idx)}")
        print(f"   Label offset: {grade_offset}")
        
        # Filter dataset to same range as model
        original_count = len(dataset)
        dataset = filter_dataset_by_grades(dataset, min_grade_idx, max_grade_idx)
        filtered_count = len(dataset)
        
        print(f"   Filtered evaluation data: {original_count} -> {filtered_count} problems")
        
        # Remap labels to match model's expected range
        dataset = [(tensor, remap_label(label, grade_offset)) for tensor, label in dataset]
    
    # Create data loader
    tensors = np.array([x[0] for x in dataset])
    labels = np.array([x[1] for x in dataset])
    eval_dataset = MoonboardDataset(tensors, labels)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    print(f"\n>> Evaluating model...")
    metrics = evaluate_model(predictor.model, eval_loader, device)
    
    print(f"\n>> Evaluation Results:")
    print(f"   Exact Accuracy:    {metrics['exact_accuracy']:.2f}%")
    print(f"   Macro Accuracy:    {metrics['macro_accuracy']:.2f}%")
    print(f"   +-1 Grade Accuracy: {metrics['tolerance_1_accuracy']:.2f}%")
    print(f"   +-2 Grade Accuracy: {metrics['tolerance_2_accuracy']:.2f}%")
    print(f"   Loss:              {metrics['avg_loss']:.4f}")
    
    # Get detailed metrics
    print(f"\n>> Detailed Metrics:")
    # Use filtered grade names if model is filtered
    if grade_offset > 0:
        from src import get_filtered_grade_names
        grade_names = get_filtered_grade_names(min_grade_idx, max_grade_idx)
    else:
        grade_names = get_all_grades()
    
    metrics_summary = get_metrics_summary(
        metrics['predictions'],
        metrics['labels'],
        grade_names=grade_names
    )
    
    print(f"   Mean Absolute Error: {metrics_summary['mean_absolute_error']:.2f} grades")
    
    # Per-grade metrics
    print(f"\n>> Per-Grade Performance:")
    per_grade = metrics_summary['per_grade_metrics']
    print(f"   {'Grade':<6} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print(f"   {'-'*50}")
    
    for grade in grade_names:
        if grade in per_grade:
            p = per_grade[grade]['precision']
            r = per_grade[grade]['recall']
            f = per_grade[grade]['f1']
            s = per_grade[grade]['support']
            print(f"   {grade:<6} {p:>9.2f} {r:>9.2f} {f:>9.2f} {s:>9}")
    
    # Save confusion matrix if requested
    if args.save_confusion_matrix:
        output_path = Path(args.output) if args.output else Path("confusion_matrix.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use filtered grade names if model is filtered
        if grade_offset > 0:
            from src import get_filtered_grade_names
            cm_grade_names = get_filtered_grade_names(min_grade_idx, max_grade_idx)
            cm_num_classes = len(cm_grade_names)
        else:
            cm_grade_names = get_all_grades()
            cm_num_classes = len(cm_grade_names)
        
        cm = generate_confusion_matrix(
            metrics['predictions'],
            metrics['labels'],
            num_classes=cm_num_classes
        )
        
        plot_confusion_matrix(
            cm,
            cm_grade_names,
            str(output_path),
            normalize=True
        )
        print(f"\n* Saved confusion matrix to: {output_path}")
    
    print_completion_message("Evaluation completed successfully!")

