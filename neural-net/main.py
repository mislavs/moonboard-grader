#!/usr/bin/env python3
"""
Main CLI script for Moonboard Grade Prediction

Usage:
    python main.py train --config config.yaml
    python main.py evaluate --checkpoint models/best_model.pth --data data/problems.json
    python main.py predict --checkpoint models/best_model.pth --input problem.json
"""

import argparse
import json
import sys
from pathlib import Path
import yaml
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

# Import our modules
from src import (
    load_dataset,
    get_dataset_stats,
    create_data_splits,
    create_model,
    count_parameters,
    Trainer,
    Predictor,
    evaluate_model,
    get_metrics_summary,
    plot_confusion_matrix,
    decode_grade,
    get_all_grades,
)


def load_config(config_path):
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def train_command(args):
    """Execute training command."""
    print("=" * 70)
    print("MOONBOARD GRADE PREDICTION - TRAINING")
    print("=" * 70)
    
    # Load configuration
    config = load_config(args.config)
    print(f"\n✓ Loaded configuration from: {args.config}")
    
    # Set device
    device_name = config.get('device', 'cpu')
    if device_name == 'cuda' and not torch.cuda.is_available():
        print(f"⚠ CUDA requested but not available, falling back to CPU")
        device_name = 'cpu'
    device = torch.device(device_name)
    print(f"✓ Using device: {device}")
    
    # Load dataset
    data_path = config['data']['path']
    print(f"\n📂 Loading dataset from: {data_path}")
    dataset = load_dataset(data_path)
    
    if len(dataset) == 0:
        print("❌ Error: No problems found in dataset")
        sys.exit(1)
    
    # Get dataset statistics
    stats = get_dataset_stats(dataset)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total problems: {stats['total_problems']}")
    print(f"   Grade distribution:")
    for grade, count in sorted(stats['grade_distribution'].items()):
        print(f"      {grade}: {count}")
    
    # Create data splits
    print(f"\n🔀 Creating train/val/test splits...")
    tensors = np.array([x[0] for x in dataset])
    labels = np.array([x[1] for x in dataset])
    
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    random_seed = config['data'].get('random_seed', 42)
    
    train_dataset, val_dataset, test_dataset = create_data_splits(
        tensors, labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_seed
    )
    
    print(f"   Train: {len(train_dataset)} ({train_ratio*100:.0f}%)")
    print(f"   Val:   {len(val_dataset)} ({val_ratio*100:.0f}%)")
    print(f"   Test:  {len(test_dataset)} ({test_ratio*100:.0f}%)")
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model_type = config['model']['type']
    num_classes = config['model']['num_classes']
    print(f"\n🧠 Creating model: {model_type.upper()}")
    model = create_model(model_type, num_classes=num_classes)
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"   Parameters: {num_params:,}")
    
    # Create optimizer
    optimizer_type = config['training'].get('optimizer', 'adam').lower()
    learning_rate = config['training']['learning_rate']
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    print(f"   Optimizer: {optimizer_type.upper()} (lr={learning_rate})")
    
    # Create loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint']['dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=str(checkpoint_dir)
    )
    
    # Train model
    num_epochs = config['training']['num_epochs']
    early_stopping_patience = config['training'].get('early_stopping_patience')
    
    print(f"\n🏋️  Training for {num_epochs} epochs...")
    if early_stopping_patience:
        print(f"   Early stopping: patience={early_stopping_patience}")
    
    history = trainer.fit(
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        verbose=True
    )
    
    # Save training history
    trainer.save_history("training_history.json")
    history_path = checkpoint_dir / "training_history.json"
    print(f"\n✓ Saved training history to: {history_path}")
    
    # Evaluate on test set
    print(f"\n📈 Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    
    print(f"\n🎯 Test Set Results:")
    print(f"   Exact Accuracy:  {test_metrics['exact_accuracy']*100:.2f}%")
    print(f"   ±1 Grade Accuracy: {test_metrics['tolerance_1_accuracy']*100:.2f}%")
    print(f"   ±2 Grade Accuracy: {test_metrics['tolerance_2_accuracy']*100:.2f}%")
    print(f"   Loss: {test_metrics['avg_loss']:.4f}")
    
    # Save confusion matrix if requested
    if config.get('evaluation', {}).get('save_confusion_matrix', False):
        cm_path = config['evaluation'].get('confusion_matrix_path', 'models/confusion_matrix.png')
        cm_path = Path(cm_path)
        cm_path.parent.mkdir(parents=True, exist_ok=True)
        
        metrics_summary = get_metrics_summary(
            test_metrics['predictions'],
            test_metrics['labels']
        )
        
        plot_confusion_matrix(
            metrics_summary['confusion_matrix'],
            get_all_grades(),
            str(cm_path),
            normalize=True
        )
        print(f"\n✓ Saved confusion matrix to: {cm_path}")
    
    print("\n" + "=" * 70)
    print("✅ Training completed successfully!")
    print("=" * 70)


def evaluate_command(args):
    """Execute evaluation command."""
    print("=" * 70)
    print("MOONBOARD GRADE PREDICTION - EVALUATION")
    print("=" * 70)
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"\n✓ Loading model from: {checkpoint_path}")
    
    # Load predictor (handles model loading)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"✓ Using device: {device}")
    
    predictor = Predictor(str(checkpoint_path), device=device)
    model_info = predictor.get_model_info()
    
    print(f"\n🧠 Model Information:")
    print(f"   Type: {model_info['model_type']}")
    print(f"   Parameters: {model_info['num_parameters']:,}")
    print(f"   Classes: {model_info['num_classes']}")
    
    # Load evaluation data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"\n📂 Loading evaluation data from: {data_path}")
    dataset = load_dataset(str(data_path))
    
    if len(dataset) == 0:
        print("❌ Error: No problems found in dataset")
        sys.exit(1)
    
    print(f"   Total problems: {len(dataset)}")
    
    # Create data loader
    from src.dataset import MoonboardDataset
    tensors = np.array([x[0] for x in dataset])
    labels = np.array([x[1] for x in dataset])
    eval_dataset = MoonboardDataset(tensors, labels)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    print(f"\n📈 Evaluating model...")
    metrics = evaluate_model(predictor.model, eval_loader, device)
    
    print(f"\n🎯 Evaluation Results:")
    print(f"   Exact Accuracy:     {metrics['exact_accuracy']*100:.2f}%")
    print(f"   ±1 Grade Accuracy:  {metrics['tolerance_1_accuracy']*100:.2f}%")
    print(f"   ±2 Grade Accuracy:  {metrics['tolerance_2_accuracy']*100:.2f}%")
    print(f"   Loss:               {metrics['loss']:.4f}")
    
    # Get detailed metrics
    print(f"\n📊 Detailed Metrics:")
    metrics_summary = get_metrics_summary(metrics['predictions'], metrics['labels'])
    
    print(f"   Mean Absolute Error: {metrics_summary['mean_absolute_error']:.2f} grades")
    
    # Per-grade metrics
    print(f"\n📋 Per-Grade Performance:")
    per_grade = metrics_summary['per_grade_metrics']
    print(f"   {'Grade':<6} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print(f"   {'-'*50}")
    
    for grade in get_all_grades():
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
        
        plot_confusion_matrix(
            metrics_summary['confusion_matrix'],
            get_all_grades(),
            str(output_path),
            normalize=True
        )
        print(f"\n✓ Saved confusion matrix to: {output_path}")
    
    print("\n" + "=" * 70)
    print("✅ Evaluation completed successfully!")
    print("=" * 70)


def predict_command(args):
    """Execute prediction command."""
    print("=" * 70)
    print("MOONBOARD GRADE PREDICTION - PREDICTION")
    print("=" * 70)
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"\n✓ Loading model from: {checkpoint_path}")
    
    # Load predictor
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"✓ Using device: {device}")
    
    predictor = Predictor(str(checkpoint_path), device=device)
    model_info = predictor.get_model_info()
    
    print(f"\n🧠 Model: {model_info['model_type']} ({model_info['num_parameters']:,} parameters)")
    
    # Load input problem
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"\n📂 Loading problem from: {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Handle both single problem and problem with 'data' array
    if 'data' in data and isinstance(data['data'], list):
        problems = data['data']
        batch_mode = True
    elif 'moves' in data:
        problems = [data]
        batch_mode = False
    else:
        print("❌ Error: Invalid input format. Expected 'moves' field or 'data' array.")
        sys.exit(1)
    
    print(f"   Found {len(problems)} problem(s)")
    
    # Make predictions
    top_k = args.top_k if args.top_k else 3
    
    if batch_mode:
        print(f"\n🔮 Making predictions (top-{top_k})...")
        predictions = predictor.predict_batch(problems, return_top_k=top_k)
        
        for i, pred in enumerate(predictions):
            if 'error' in pred:
                print(f"\n❌ Problem {i+1}: {pred['error']}")
            else:
                print(f"\n📊 Problem {i+1}:")
                print(f"   Predicted Grade: {pred['predicted_grade']}")
                print(f"   Confidence: {pred['confidence']*100:.2f}%")
                print(f"\n   Top {top_k} Predictions:")
                for j, (grade, prob) in enumerate(pred['top_k_predictions'], 1):
                    print(f"      {j}. {grade:<4} ({prob*100:.2f}%)")
    else:
        print(f"\n🔮 Making prediction (top-{top_k})...")
        pred = predictor.predict(problems[0], return_top_k=top_k)
        
        print(f"\n📊 Results:")
        print(f"   Predicted Grade: {pred['predicted_grade']}")
        print(f"   Confidence: {pred['confidence']*100:.2f}%")
        print(f"\n   Top {top_k} Predictions:")
        for j, (grade, prob) in enumerate(pred['top_k_predictions'], 1):
            print(f"      {j}. {grade:<4} ({prob*100:.2f}%)")
        
        # Show actual grade if present
        if 'grade' in problems[0]:
            actual_grade = problems[0]['grade']
            print(f"\n   Actual Grade: {actual_grade}")
            
            # Check if prediction was correct
            if pred['predicted_grade'] == actual_grade:
                print(f"   ✅ Exact match!")
            else:
                # Check if within tolerance
                from src.grade_encoder import encode_grade
                pred_idx = encode_grade(pred['predicted_grade'])
                actual_idx = encode_grade(actual_grade)
                diff = abs(pred_idx - actual_idx)
                
                if diff == 1:
                    print(f"   ⚠️  Off by 1 grade")
                elif diff == 2:
                    print(f"   ⚠️  Off by 2 grades")
                else:
                    print(f"   ❌ Off by {diff} grades")
    
    # Save output if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if batch_mode:
                json.dump(predictions, f, indent=2)
            else:
                json.dump(pred, f, indent=2)
        
        print(f"\n✓ Saved predictions to: {output_path}")
    
    print("\n" + "=" * 70)
    print("✅ Prediction completed successfully!")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Moonboard Grade Prediction Neural Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train --config config.yaml
  
  # Evaluate a trained model
  python main.py evaluate --checkpoint models/best_model.pth --data data/test.json
  
  # Make predictions
  python main.py predict --checkpoint models/best_model.pth --input problem.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    subparsers.required = True
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    train_parser.set_defaults(func=train_command)
    
    # Evaluate command
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
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions on new problems')
    predict_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    predict_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input problem JSON file'
    )
    predict_parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage (default: use CUDA if available)'
    )
    predict_parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Return top K predictions (default: 3)'
    )
    predict_parser.add_argument(
        '--output',
        type=str,
        help='Save predictions to JSON file'
    )
    predict_parser.set_defaults(func=predict_command)
    
    # Parse arguments and execute
    args = parser.parse_args()
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

