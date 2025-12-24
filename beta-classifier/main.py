"""
Beta Classifier CLI

Command-line interface for training, evaluating, and predicting with
the transformer-based grade classifier.

Usage:
    py main.py train --config config.yaml
    py main.py evaluate --checkpoint models/best_model.pth --normalizer models/normalizer.npz --data ../data/solved_problems.json
    py main.py predict --checkpoint models/best_model.pth --normalizer models/normalizer.npz --input problem.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from moonboard_core import get_num_grades

from src.dataset import (
    load_data,
    create_data_splits,
    create_dataloaders,
    FeatureNormalizer
)
from src.model import TransformerSequenceClassifier
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.predictor import Predictor


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_command(args: argparse.Namespace) -> None:
    """Execute training command."""
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Determine device
    device = config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {config['data']['path']}")
    sequences, labels = load_data(config['data']['path'])
    print(f"Loaded {len(sequences)} problems")
    
    # Create splits
    print("Creating train/val/test splits...")
    splits = create_data_splits(
        sequences,
        labels,
        train_ratio=config['data']['train_ratio'],
        val_ratio=config['data']['val_ratio'],
        test_ratio=config['data']['test_ratio'],
        random_seed=config['data']['random_seed']
    )
    
    print(f"  Train: {len(splits['train'][0])} samples")
    print(f"  Val:   {len(splits['val'][0])} samples")
    print(f"  Test:  {len(splits['test'][0])} samples")
    
    # Fit normalizer on training data only
    print("Fitting normalizer on training data...")
    normalizer = FeatureNormalizer()
    normalizer.fit(splits['train'][0])
    
    # Save normalizer
    checkpoint_dir = Path(config['checkpoint']['dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    normalizer_path = checkpoint_dir / 'normalizer.npz'
    normalizer.save(str(normalizer_path))
    print(f"Normalizer saved to {normalizer_path}")
    
    # Create data loaders
    print("Creating data loaders...")
    loaders = create_dataloaders(
        splits,
        normalizer,
        batch_size=config['training']['batch_size']
    )
    
    # Create model
    print("Creating model...")
    model_config = config['model']
    model = TransformerSequenceClassifier(
        input_dim=15,
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_layers=model_config['n_layers'],
        num_classes=model_config['num_classes'],
        dropout=model_config['dropout'],
        max_seq_len=model_config['max_seq_len']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create trainer
    print("Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        checkpoint_dir=config['checkpoint']['dir'],
        use_class_weights=config['training']['use_class_weights'],
        label_smoothing=config['training']['label_smoothing'],
        num_classes=model_config['num_classes']
    )
    
    # Train
    history = trainer.train()
    
    print("\nTraining complete!")
    print(f"  Epochs trained: {history['epochs_trained']}")
    print(f"  Best val loss: {history['best_val_loss']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluator = Evaluator(model, device=device)
    results = evaluator.evaluate(loaders['test'])
    evaluator.print_report(results)
    
    # Save confusion matrix
    cm_path = checkpoint_dir / 'confusion_matrix.png'
    evaluator.plot_confusion_matrix(
        results['predictions'],
        results['true_labels'],
        save_path=str(cm_path)
    )
    
    # Save error distribution
    err_path = checkpoint_dir / 'error_distribution.png'
    evaluator.plot_error_distribution(
        results['predictions'],
        results['true_labels'],
        save_path=str(err_path)
    )


def evaluate_command(args: argparse.Namespace) -> None:
    """Execute evaluation command."""
    print(f"Loading model from {args.checkpoint}")
    print(f"Loading normalizer from {args.normalizer}")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TransformerSequenceClassifier(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load normalizer
    normalizer = FeatureNormalizer.load(args.normalizer)
    
    # Load data
    print(f"Loading data from {args.data}")
    sequences, labels = load_data(args.data)
    print(f"Loaded {len(sequences)} problems")
    
    # Normalize and create dataloader
    from src.dataset import MoveSequenceDataset, collate_fn
    from torch.utils.data import DataLoader
    
    normalized = normalizer.transform(sequences)
    dataset = MoveSequenceDataset(normalized, labels)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Evaluate
    evaluator = Evaluator(model, device=device)
    results = evaluator.evaluate(loader)
    evaluator.print_report(results)
    
    # Save visualizations if output dir specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        evaluator.plot_confusion_matrix(
            results['predictions'],
            results['true_labels'],
            save_path=str(output_dir / 'confusion_matrix.png')
        )
        
        evaluator.plot_error_distribution(
            results['predictions'],
            results['true_labels'],
            save_path=str(output_dir / 'error_distribution.png')
        )


def predict_command(args: argparse.Namespace) -> None:
    """Execute prediction command."""
    print(f"Loading model from {args.checkpoint}")
    print(f"Loading normalizer from {args.normalizer}")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create predictor
    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        normalizer_path=args.normalizer,
        device=device
    )
    
    # Load input
    print(f"Loading input from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle single problem or list
    if isinstance(data, list):
        problems = data
    else:
        problems = [data]
    
    print(f"Predicting grades for {len(problems)} problems...")
    
    for i, problem in enumerate(problems):
        if args.compare:
            result = predictor.compare_with_actual(problem)
        else:
            result = predictor.predict_with_alternatives(problem, top_k=3)
        
        print(f"\nProblem {i + 1}:")
        if 'name' in problem:
            print(f"  Name: {problem['name']}")
        
        print(f"  Predicted: {result['predicted_grade']} (confidence: {result['confidence']:.3f})")
        
        if 'actual_grade' in result:
            print(f"  Actual: {result['actual_grade']}")
            print(f"  Error: {result['error']:+d} grades")
            print(f"  Correct: {result['correct']}, Within ±1: {result['within_1']}")
        elif 'alternatives' in result:
            print("  Alternatives:")
            for alt in result['alternatives']:
                print(f"    {alt['grade']}: {alt['probability']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Beta Classifier - Climbing grade prediction from move sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a model:
    py main.py train --config config.yaml
    
  Evaluate on test data:
    py main.py evaluate --checkpoint models/best_model.pth --normalizer models/normalizer.npz --data ../data/solved_problems.json
    
  Predict grades:
    py main.py predict --checkpoint models/best_model.pth --normalizer models/normalizer.npz --input problem.json
    py main.py predict --checkpoint models/best_model.pth --normalizer models/normalizer.npz --input problem.json --compare
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to config YAML file (default: config.yaml)'
    )
    train_parser.set_defaults(func=train_command)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument(
        '--checkpoint', '-m',
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    eval_parser.add_argument(
        '--normalizer', '-n',
        required=True,
        help='Path to normalizer (.npz file)'
    )
    eval_parser.add_argument(
        '--data', '-d',
        required=True,
        help='Path to data JSON file'
    )
    eval_parser.add_argument(
        '--output', '-o',
        help='Output directory for visualizations'
    )
    eval_parser.set_defaults(func=evaluate_command)
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict grades for problems')
    predict_parser.add_argument(
        '--checkpoint', '-m',
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    predict_parser.add_argument(
        '--normalizer', '-n',
        required=True,
        help='Path to normalizer (.npz file)'
    )
    predict_parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to input JSON (single problem or list)'
    )
    predict_parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare predictions with actual grades if available'
    )
    predict_parser.set_defaults(func=predict_command)
    
    # Parse and execute
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()

