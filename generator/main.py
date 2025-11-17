"""
Command-line interface for the MoonBoard Generator.

Provides commands for training the VAE and generating new climbing problems.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import torch

from src.vae import ConditionalVAE
from src.dataset import create_data_loaders
from src.vae_trainer import VAETrainer
from src.generator import ProblemGenerator, format_problem_output
from src.evaluator import run_evaluation, get_metrics
from moonboard_core import get_filtered_grade_names, encode_grade

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        config: Dictionary with configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file: {e}")


def build_training_config(config: Dict) -> Dict:
    """
    Extract and validate training configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        training_config: Configuration for VAETrainer
    """
    return {
        'learning_rate': config['training']['learning_rate'],
        'num_epochs': config['training']['num_epochs'],
        'kl_weight': config['training']['kl_weight'],
        'kl_annealing': config['training']['kl_annealing'],
        'kl_annealing_epochs': config['training']['kl_annealing_epochs'],
        'checkpoint_dir': config['checkpoint']['checkpoint_dir'],
        'log_dir': config['logging']['log_dir'],
        'log_interval': config['logging'].get('log_interval', 100),
    }


def train_command(args):
    """
    Train the VAE model.
    
    Args:
        args: Command-line arguments
    """
    try:
        # Load configuration
        config = load_config(args.config)
        print(f"\n📋 Configuration loaded from: {args.config}")
        
        # Set device
        device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        # Create data loaders
        print(f"\n📊 Loading dataset...")
        data_config = config['data']
        train_loader, val_loader, dataset = create_data_loaders(
            data_path=data_config['data_path'],
            batch_size=data_config['batch_size'],
            train_split=data_config['train_split'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 0),
            min_grade_index=data_config.get('min_grade_index', None),
            max_grade_index=data_config.get('max_grade_index', None)
        )
        
        num_grades = dataset.get_num_grades()
        print(f"   Total problems: {len(dataset)}")
        print(f"   Unique grades: {num_grades}")
        print(f"   Grade range: {', '.join(dataset.grade_names)}")
        
        # Create model
        print(f"\n🧠 Creating model...")
        model_config = config['model']
        model = ConditionalVAE(
            latent_dim=model_config['latent_dim'],
            num_grades=num_grades,  # Use actual number from dataset
            grade_embedding_dim=model_config['grade_embedding_dim']
        )
        
        print(f"   Latent dim: {model_config['latent_dim']}")
        print(f"   Grade embedding dim: {model_config['grade_embedding_dim']}")
        
        # Prepare training config
        training_config = build_training_config(config)
        
        # Create trainer
        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=device
        )
        
        # Load checkpoint if resuming
        start_epoch = 0
        if args.resume:
            resume_path = Path(args.resume)
            if resume_path.exists():
                print(f"\n↻  Resuming from checkpoint: {resume_path}")
                trainer.load_checkpoint(str(resume_path))
                start_epoch = trainer.current_epoch + 1
                print(f"   Starting from epoch {start_epoch}")
            else:
                print(f"\n⚠️  Checkpoint {args.resume} not found. Starting from scratch.")
        
        # Train
        trainer.train(start_epoch=start_epoch)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        trainer.save_checkpoint(trainer.current_epoch, 'interrupted_checkpoint.pth')
        print("   Saved interrupted checkpoint")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


def _determine_grade_labels(
    args,
    model_num_grades: int
) -> Tuple[List[int], Optional[str]]:
    """
    Determine grade labels from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        model_num_grades: Number of grades the model supports
        
    Returns:
        Tuple of (grade_labels list, grade_name for display or None)
        
    Raises:
        SystemExit: If arguments are invalid
    """
    if args.grade_labels:
        # Use provided grade labels
        grade_labels = [int(x) for x in args.grade_labels.split(',')]
        
        # Validate grade labels
        for label in grade_labels:
            if label >= model_num_grades or label < 0:
                print(
                    f"\n[!] Grade label {label} is out of range for model "
                    f"(valid: 0-{model_num_grades-1})"
                )
                sys.exit(1)
        
        return grade_labels, None
        
    elif args.grade:
        # For models trained on single grade, always use label 0
        if model_num_grades == 1:
            grade_labels = [0] * args.num_samples
            return grade_labels, args.grade
        else:
            # Convert grade string to label
            try:
                grade_label = encode_grade(args.grade)
                if grade_label >= model_num_grades or grade_label < 0:
                    print(
                        f"\n[!] Grade '{args.grade}' (label {grade_label}) "
                        f"is out of range for model (valid: 0-{model_num_grades-1})"
                    )
                    sys.exit(1)
                grade_labels = [grade_label] * args.num_samples
                return grade_labels, args.grade
            except ValueError as e:
                print(f"\n[!] Invalid grade '{args.grade}': {e}")
                sys.exit(1)
    else:
        print(f"\n[!] Must specify either --grade or --grade-labels")
        sys.exit(1)


def _format_output_problems(
    problems: List[Dict],
    include_grade: bool
) -> Tuple[List[Dict], int]:
    """
    Format generated problems for output.
    
    Args:
        problems: List of generated problem dictionaries
        include_grade: Whether to include grade information
        
    Returns:
        Tuple of (formatted problems list, count of valid problems)
    """
    # Get grade names for output
    grade_names = None
    if include_grade:
        try:
            grade_names = get_filtered_grade_names(None, None)
        except Exception:
            pass
    
    # Format output
    output_problems = []
    valid_count = 0
    for problem in problems:
        formatted = format_problem_output(
            problem,
            include_grade=include_grade,
            grade_names=grade_names
        )
        output_problems.append(formatted)
        
        if problem.get('validation', {}).get('valid', True):
            valid_count += 1
    
    return output_problems, valid_count


def generate_command(args):
    """
    Generate new climbing problems using a trained model.
    
    Args:
        args: Command-line arguments
    """
    try:
        # Set device
        device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
        print(f"\n[*] Generating problems...")
        print(f"    Device: {device}")
        print(f"    Checkpoint: {args.checkpoint}")
        
        # Load generator
        generator = ProblemGenerator.from_checkpoint(
            checkpoint_path=args.checkpoint,
            device=device,
            threshold=args.threshold
        )
        
        # Determine grade labels to generate
        model_num_grades = generator.model.num_grades
        print(f"    Model num_grades: {model_num_grades}")
        
        grade_labels, grade_name = _determine_grade_labels(args, model_num_grades)
        
        if grade_name:
            if model_num_grades == 1:
                print(f"   Grade: {grade_name} (model trained on single grade, using label 0)")
            else:
                print(f"   Grade: {grade_name} (label {grade_labels[0]})")
        else:
            print(f"   Generating for grade labels: {grade_labels}")
        
        print(f"   Number of samples: {len(grade_labels)}")
        print(f"   Temperature: {args.temperature}")
        print(f"   Threshold: {args.threshold}")
        
        # Generate problems
        if args.retry:
            print(f"   Using retry logic (max attempts: {args.max_attempts})")
            # Use retry for single grade only
            if len(set(grade_labels)) == 1:
                problems = generator.generate_with_retry(
                    grade_label=grade_labels[0],
                    num_samples=len(grade_labels),
                    max_attempts=args.max_attempts,
                    temperature=args.temperature
                )
            else:
                print("    [!] Retry mode only works with single grade, using batch mode")
                problems = generator.generate_batch(
                    grade_labels=grade_labels,
                    temperature=args.temperature,
                    validate=True
                )
        else:
            problems = generator.generate_batch(
                grade_labels=grade_labels,
                temperature=args.temperature,
                validate=True
            )
        
        # Format output
        output_problems, valid_count = _format_output_problems(
            problems,
            args.include_grade
        )
        
        # Print summary
        print(f"\n[+] Generated {len(problems)} problems")
        print(f"    Valid: {valid_count}/{len(problems)}")
        
        if valid_count < len(problems):
            invalid_count = len(problems) - valid_count
            print(f"    [!] {invalid_count} problems failed validation")
        
        # Save or print output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(output_problems, f, indent=2)
            
            print(f"\n[+] Saved to: {output_path}")
        else:
            # Print to stdout
            print(f"\n[*] Generated problems:\n")
            print(json.dumps(output_problems, indent=2))
        
    except FileNotFoundError as e:
        print(f"\n[!] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Generation failed: {e}")
        logger.error(f"Generation failed: {e}", exc_info=True)
        sys.exit(1)


def evaluate_command(args):
    """
    Evaluate the trained model with quality metrics.
    
    Args:
        args: Command-line arguments
    """
    try:
        # Set device
        device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
        
        # Print header
        print("\n" + "=" * 50)
        print("=== GENERATOR EVALUATION ===")
        print("=" * 50)
        print(f"Model: {args.checkpoint}")
        print(f"Device: {device}")
        print()
        
        # Check what's available
        available_metrics = get_metrics()
        
        if available_metrics:
            print(f"Available metrics: {', '.join(available_metrics)}")
        else:
            print("No metrics available yet.")
        print()
        
        # Determine which metrics to run
        if args.metrics:
            requested_metrics = [m.strip() for m in args.metrics.split(',')]
            metrics_to_run = [m for m in requested_metrics if m in available_metrics]
        else:
            metrics_to_run = available_metrics
        
        if not metrics_to_run:
            if args.metrics and available_metrics:
                print(f"Requested metrics not ready yet.")
                print(f"Available: {', '.join(available_metrics)}")
            else:
                print("No metrics to evaluate yet.")
            print()
            return
        
        # Load model
        print(f"Loading model from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Create model with checkpoint parameters
        model = ConditionalVAE(
            latent_dim=checkpoint['latent_dim'],
            num_grades=checkpoint['num_grades'],
            grade_embedding_dim=checkpoint['grade_embedding_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully")
        print(f"Running metrics: {', '.join(metrics_to_run)}")
        print()
        
        # Run evaluation
        results = run_evaluation(
            model=model,
            checkpoint_path=args.checkpoint,
            data_path=args.data,
            classifier_checkpoint=args.classifier_checkpoint,
            metrics=metrics_to_run,
            num_samples=args.num_samples,
            device=device
        )
        
        # Display results
        print("=" * 50)
        print("=== RESULTS ===")
        print("=" * 50)
        
        for metric_name, metric_results in results.get('metrics', {}).items():
            print(f"\n{metric_name.upper().replace('_', ' ')}:")
            print("-" * 40)
            _print_metric_results(metric_results, indent=2)
        
        print()
        
        # Save to JSON if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to: {output_path}")
            print()
        
    except FileNotFoundError as e:
        print(f"\n[!] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


def _print_metric_results(results: Dict, indent: int = 0):
    """
    Pretty print metric results recursively.
    
    Args:
        results: Dictionary of results
        indent: Indentation level
    """
    prefix = " " * indent
    
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_metric_results(value, indent + 2)
        elif isinstance(value, (list, tuple)):
            print(f"{prefix}{key}: {value}")
        elif isinstance(value, float):
            print(f"{prefix}{key}: {value:.4f}")
        else:
            print(f"{prefix}{key}: {value}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='MoonBoard Generator - VAE-based climbing problem generation'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the VAE model')
    train_parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    train_parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    train_parser.set_defaults(func=train_command)
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate new climbing problems')
    generate_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )
    generate_parser.add_argument(
        '--grade',
        type=str,
        help='Grade to generate (e.g., "6B+", "7A")'
    )
    generate_parser.add_argument(
        '--grade-labels',
        type=str,
        help='Comma-separated grade labels (e.g., "0,1,2")'
    )
    generate_parser.add_argument(
        '--num-samples',
        type=int,
        default=1,
        help='Number of problems to generate (default: 1)'
    )
    generate_parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature - higher = more random (default: 1.0)'
    )
    generate_parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for binary hold detection (default: 0.5)'
    )
    generate_parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path (if not specified, prints to stdout)'
    )
    generate_parser.add_argument(
        '--include-grade',
        action='store_true',
        help='Include grade information in output'
    )
    generate_parser.add_argument(
        '--retry',
        action='store_true',
        help='Use retry logic to ensure valid problems'
    )
    generate_parser.add_argument(
        '--max-attempts',
        type=int,
        default=10,
        help='Maximum retry attempts (default: 10)'
    )
    generate_parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage even if CUDA is available'
    )
    generate_parser.set_defaults(func=generate_command)
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model quality')
    evaluate_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint file'
    )
    evaluate_parser.add_argument(
        '--data',
        type=str,
        default='../data/processed/moonboard_2016.json',
        help='Path to dataset JSON file (default: ../data/processed/moonboard_2016.json)'
    )
    evaluate_parser.add_argument(
        '--classifier-checkpoint',
        type=str,
        default=None,
        help='Path to classifier checkpoint (required for grade_conditioning metric)'
    )
    evaluate_parser.add_argument(
        '--metrics',
        type=str,
        default=None,
        help='Comma-separated list of metrics to run (default: all implemented)'
    )
    evaluate_parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples per grade for generation-based metrics (default: 100)'
    )
    evaluate_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save JSON results (optional)'
    )
    evaluate_parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage even if CUDA is available'
    )
    evaluate_parser.set_defaults(func=evaluate_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()

