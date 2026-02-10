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
from src.label_space import (
    EvaluationLabelContext,
    build_label_context,
    infer_num_model_grades,
)
from moonboard_core import decode_grade, encode_grade

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
    trainer = None
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
        
        num_grades = dataset.get_num_model_grades()
        print(f"   Total problems: {len(dataset)}")
        print(f"   Model grades: {num_grades}")
        print(f"   Grade range: {', '.join(dataset.model_grade_names)}")

        # Extract filtering metadata from dataset
        label_space_mode = dataset.label_space_mode
        grade_offset = dataset.grade_offset
        min_grade_index = dataset.min_grade_index
        max_grade_index = dataset.max_grade_index
        
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
            device=device,
            label_space_mode=label_space_mode,
            grade_offset=grade_offset,
            min_grade_index=min_grade_index,
            max_grade_index=max_grade_index
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
        if trainer is not None:
            trainer.save_checkpoint(trainer.current_epoch, 'interrupted_checkpoint.pth')
            print("   Saved interrupted checkpoint")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


def _determine_grade_labels(
    args,
    label_context: EvaluationLabelContext
) -> Tuple[List[int], Optional[str]]:
    """
    Determine grade labels from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        label_context: Checkpoint label-space context
        
    Returns:
        Tuple of (grade_labels list, grade_name for display or None)
        
    Raises:
        SystemExit: If arguments are invalid
    """
    if args.grade_labels:
        # grade-labels are treated as global labels for stable CLI semantics.
        try:
            requested_global_labels = [int(x.strip()) for x in args.grade_labels.split(',') if x.strip()]
        except ValueError:
            print("\n[!] --grade-labels must be a comma-separated list of integers")
            sys.exit(1)
        if not requested_global_labels:
            print("\n[!] --grade-labels must contain at least one integer value")
            sys.exit(1)
        try:
            grade_labels = [
                label_context.global_to_model_label(global_label)
                for global_label in requested_global_labels
            ]
        except ValueError as e:
            min_idx, max_idx = label_context.get_global_grade_bounds()
            print(
                f"\n[!] {e}\n"
                f"    Checkpoint supports global labels "
                f"{min_idx} ({decode_grade(min_idx)}) through {max_idx} ({decode_grade(max_idx)})."
            )
            sys.exit(1)
        return grade_labels, None

    elif args.grade:
        # Convert global grade string to model label using checkpoint context.
        try:
            global_grade_label = encode_grade(args.grade)
            model_grade_label = label_context.global_to_model_label(global_grade_label)
            grade_labels = [model_grade_label] * args.num_samples
            return grade_labels, args.grade
        except ValueError as e:
            min_idx, max_idx = label_context.get_global_grade_bounds()
            print(
                f"\n[!] Invalid grade '{args.grade}': {e}\n"
                f"    Checkpoint supports grades {decode_grade(min_idx)} to {decode_grade(max_idx)}."
            )
            sys.exit(1)
    else:
        print(f"\n[!] Must specify either --grade or --grade-labels")
        sys.exit(1)


def _format_output_problems(
    problems: List[Dict],
    include_grade: bool,
    label_context: EvaluationLabelContext
) -> Tuple[List[Dict], int]:
    """
    Format generated problems for output.
    
    Args:
        problems: List of generated problem dictionaries
        include_grade: Whether to include grade information
        label_context: Checkpoint label-space context
        
    Returns:
        Tuple of (formatted problems list, count of valid problems)
    """
    # Format output
    output_problems = []
    valid_count = 0
    for problem in problems:
        formatted = format_problem_output(
            problem,
            include_grade=include_grade,
            label_context=label_context
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
        label_context = generator.label_context
        model_num_grades = label_context.num_model_grades
        min_idx, max_idx = label_context.get_global_grade_bounds()
        print(f"    Model num_grades: {model_num_grades}")
        print(f"    Label space mode: {label_context.label_space_mode}")
        print(f"    Global grade range: {decode_grade(min_idx)} to {decode_grade(max_idx)}")

        grade_labels, grade_name = _determine_grade_labels(args, label_context)

        if grade_name:
            resolved_global = label_context.model_to_global_label(grade_labels[0])
            print(
                f"   Grade: {grade_name} "
                f"(global {resolved_global}, model label {grade_labels[0]})"
            )
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
            args.include_grade,
            label_context
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
        
        # Suppress generator and evaluator logging during evaluation for cleaner output
        generator_logger = logging.getLogger('src.generator')
        evaluator_logger = logging.getLogger('src.evaluator')
        dataset_logger = logging.getLogger('src.dataset')
        
        original_generator_level = generator_logger.level
        original_evaluator_level = evaluator_logger.level
        original_dataset_level = dataset_logger.level
        
        generator_logger.setLevel(logging.ERROR)
        evaluator_logger.setLevel(logging.WARNING)
        dataset_logger.setLevel(logging.WARNING)
        
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
            requested_metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
            unknown_metrics = [m for m in requested_metrics if m not in available_metrics]
            if unknown_metrics:
                print(f"Unknown metric name(s): {', '.join(unknown_metrics)}")
                print(f"Available metrics: {', '.join(available_metrics)}")
                sys.exit(1)
            metrics_to_run = requested_metrics
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
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        num_model_grades = infer_num_model_grades(checkpoint)
        label_context = build_label_context(checkpoint, num_model_grades=num_model_grades)

        # Create model with checkpoint parameters
        model_config = checkpoint.get('model_config', {})
        model = ConditionalVAE(
            latent_dim=model_config.get('latent_dim', 128),
            num_grades=num_model_grades,
            grade_embedding_dim=model_config.get('grade_embedding_dim', 32)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"Model loaded successfully")
        min_idx, max_idx = label_context.get_global_grade_bounds()
        print(f"Label space mode: {label_context.label_space_mode}")
        print(f"Global grade range: {decode_grade(min_idx)} to {decode_grade(max_idx)}")
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
            label_context=label_context,
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
        
        # Restore original logging levels
        generator_logger.setLevel(original_generator_level)
        evaluator_logger.setLevel(original_evaluator_level)
        dataset_logger.setLevel(original_dataset_level)
        
    except FileNotFoundError as e:
        print(f"\n[!] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Ensure logging levels are restored even if there's an exception
        try:
            generator_logger.setLevel(original_generator_level)
            evaluator_logger.setLevel(original_evaluator_level)
            dataset_logger.setLevel(original_dataset_level)
        except:
            pass


def _print_metric_results(results: Dict, indent: int = 0):
    """
    Pretty print metric results recursively.
    
    Args:
        results: Dictionary of results
        indent: Indentation level
    """
    prefix = " " * indent
    
    for key, value in results.items():
        if key == 'per_grade_iou' and isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_grade_iou_table(value, indent + 2)
        elif key == 'per_grade_centroids' and isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_grade_centroids_table(value, indent + 2)
        elif key == 'per_grade' and isinstance(value, dict):
            print(f"{prefix}{key}:")
            # Detect which type of per_grade data we have
            # Check first non-empty value to determine type
            sample_value = next((v for v in value.values() if v), None)
            if sample_value:
                if 'mean_diversity' in sample_value or 'uniqueness_ratio' in sample_value:
                    # Diversity metric
                    _print_grade_diversity_table(value, indent + 2)
                elif 'wasserstein_distances' in sample_value or 'mean_distance' in sample_value:
                    # Statistical metric
                    _print_grade_statistical_table(value, indent + 2)
                elif 'exact_match_percent' in sample_value or 'off_by_one_percent' in sample_value:
                    # Classifier check metric
                    _print_classifier_check_table(value, indent + 2)
                else:
                    # Unknown format, print as nested dict
                    _print_metric_results(value, indent + 2)
            else:
                # Empty data
                _print_metric_results(value, indent + 2)
        elif key == 'per_statistic' and isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_per_statistic_table(value, indent + 2)
        elif isinstance(value, dict):
            print(f"{prefix}{key}:")
            _print_metric_results(value, indent + 2)
        elif isinstance(value, (list, tuple)):
            print(f"{prefix}{key}: {value}")
        elif isinstance(value, float):
            print(f"{prefix}{key}: {value:.4f}")
        else:
            print(f"{prefix}{key}: {value}")


def _print_grade_iou_table(grade_data: Dict, indent: int = 0):
    """
    Print per-grade IoU statistics as a formatted table.
    
    Args:
        grade_data: Dictionary mapping grade labels to IoU statistics
        indent: Indentation level
    """
    prefix = " " * indent
    
    # Table header
    header = f"{prefix}{'Grade':<8} {'Mean IoU':<12} {'Std IoU':<12} {'Samples':<10}"
    separator = f"{prefix}{'-' * 42}"
    
    print(separator)
    print(header)
    print(separator)
    
    # Sort grades by their keys for consistent display
    for grade, stats in sorted(grade_data.items()):
        mean_iou = stats.get('mean_iou', 0.0)
        std_iou = stats.get('std_iou', 0.0)
        num_samples = stats.get('num_samples', 0)
        
        row = f"{prefix}{grade:<8} {mean_iou:<12.4f} {std_iou:<12.4f} {num_samples:<10}"
        print(row)
    
    print(separator)


def _print_grade_diversity_table(grade_data: Dict, indent: int = 0):
    """
    Print per-grade diversity statistics as a formatted table.
    
    Args:
        grade_data: Dictionary mapping grade names to diversity statistics
        indent: Indentation level
    """
    prefix = " " * indent
    
    # Table header
    header = f"{prefix}{'Grade':<8} {'Diversity':<12} {'Uniqueness':<12} {'Valid':<10} {'Requested':<12} {'Status':<20}"
    separator = f"{prefix}{'-' * 74}"
    
    print(separator)
    print(header)
    print(separator)
    
    # Sort grades by their keys for consistent display
    for grade, stats in sorted(grade_data.items()):
        if stats.get('skipped', False):
            # Skipped grade
            reason = stats.get('reason', 'unknown')
            num_valid = stats.get('num_valid', 0)
            num_requested = stats.get('num_requested', 0)
            status = f"SKIPPED ({reason})"
            row = f"{prefix}{grade:<8} {'-':<12} {'-':<12} {num_valid:<10} {num_requested:<12} {status:<20}"
        else:
            # Valid grade
            mean_diversity = stats.get('mean_diversity', 0.0)
            uniqueness = stats.get('uniqueness_ratio', 0.0)
            total_valid = stats.get('total_valid', 0)
            num_requested = stats.get('num_requested', 0)
            unique = stats.get('unique_problems', 0)
            
            diversity_str = f"{mean_diversity:.4f}"
            uniqueness_str = f"{uniqueness:.1%}"
            valid_str = f"{total_valid} ({unique}u)"
            status = "OK"
            
            row = f"{prefix}{grade:<8} {diversity_str:<12} {uniqueness_str:<12} {valid_str:<10} {num_requested:<12} {status:<20}"
        
        print(row)
    
    print(separator)


def _print_grade_statistical_table(grade_data: Dict, indent: int = 0):
    """
    Print per-grade statistical similarity statistics as a formatted table.
    
    Args:
        grade_data: Dictionary mapping grade names to statistical similarity metrics
        indent: Indentation level
    """
    prefix = " " * indent
    
    # Table header
    header = f"{prefix}{'Grade':<8} {'Mean Dist':<12} {'Gen/Real':<12} {'Status':<30}"
    separator = f"{prefix}{'-' * 62}"
    
    print(separator)
    print(header)
    print(separator)
    
    # Sort grades by their keys for consistent display
    for grade, stats in sorted(grade_data.items()):
        if stats.get('skipped', False):
            # Skipped grade
            reason = stats.get('reason', 'unknown')
            num_real = stats.get('num_real', 0)
            status = f"SKIPPED ({reason})"
            row = f"{prefix}{grade:<8} {'-':<12} {'-':<12} {status:<30}"
        else:
            # Valid grade
            mean_distance = stats.get('mean_distance', 0.0)
            num_generated = stats.get('num_generated', 0)
            num_real = stats.get('num_real', 0)
            
            mean_dist_str = f"{mean_distance:.4f}" if mean_distance is not None else "N/A"
            gen_real_str = f"{num_generated}/{num_real}"
            status = "OK"
            
            row = f"{prefix}{grade:<8} {mean_dist_str:<12} {gen_real_str:<12} {status:<30}"
        
        print(row)
    
    print(separator)


def _print_per_statistic_table(statistic_data: Dict, indent: int = 0):
    """
    Print per-statistic Wasserstein distances as a formatted table.
    
    Args:
        statistic_data: Dictionary mapping statistic names to their distance metrics
        indent: Indentation level
    """
    prefix = " " * indent
    
    # Table header
    header = f"{prefix}{'Statistic':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}"
    separator = f"{prefix}{'-' * 68}"
    
    print(separator)
    print(header)
    print(separator)
    
    # Define display order for statistics
    stat_order = ['num_holds', 'num_start', 'num_end', 'num_middle', 'vertical_spread']
    
    # Print in defined order, then any remaining stats
    displayed = set()
    for stat_name in stat_order:
        if stat_name in statistic_data:
            stats = statistic_data[stat_name]
            if stats is None:
                row = f"{prefix}{stat_name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}"
            else:
                mean_val = stats.get('mean', 0.0)
                std_val = stats.get('std', 0.0)
                min_val = stats.get('min', 0.0)
                max_val = stats.get('max', 0.0)
                
                row = f"{prefix}{stat_name:<20} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}"
            
            print(row)
            displayed.add(stat_name)
    
    # Print any remaining statistics not in the predefined order
    for stat_name in sorted(statistic_data.keys()):
        if stat_name not in displayed:
            stats = statistic_data[stat_name]
            if stats is None:
                row = f"{prefix}{stat_name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}"
            else:
                mean_val = stats.get('mean', 0.0)
                std_val = stats.get('std', 0.0)
                min_val = stats.get('min', 0.0)
                max_val = stats.get('max', 0.0)
                
                row = f"{prefix}{stat_name:<20} {mean_val:<12.4f} {std_val:<12.4f} {min_val:<12.4f} {max_val:<12.4f}"
            
            print(row)
    
    print(separator)


def _print_classifier_check_table(grade_data: Dict, indent: int = 0):
    """
    Print classifier check statistics as a formatted table.
    
    Args:
        grade_data: Dictionary mapping grade names to classifier check metrics
        indent: Indentation level
    """
    prefix = " " * indent
    
    # Table header
    header = f"{prefix}{'Grade':<8} {'Exact':<10} {'Off±1':<10} {'Off±2':<10} {'Off>2':<10} {'Classified':<15}"
    separator = f"{prefix}{'-' * 63}"
    
    print(separator)
    print(header)
    print(separator)
    
    # Sort grades by their keys for consistent display
    for grade, stats in sorted(grade_data.items()):
        exact = stats.get('exact_match_percent', 0.0)
        off1 = stats.get('off_by_one_percent', 0.0)
        off2 = stats.get('off_by_two_percent', 0.0)
        off_more = stats.get('off_more_percent', 0.0)
        generated = stats.get('generated_count', 0)
        classified = stats.get('classified_count', 0)
        
        exact_str = f"{exact:.1f}%"
        off1_str = f"{off1:.1f}%"
        off2_str = f"{off2:.1f}%"
        off_more_str = f"{off_more:.1f}%"
        classified_str = f"{classified}/{generated}"
        
        row = f"{prefix}{grade:<8} {exact_str:<10} {off1_str:<10} {off2_str:<10} {off_more_str:<10} {classified_str:<15}"
        print(row)
        
        # Print prediction distribution if available
        if 'prediction_distribution' in stats and stats['prediction_distribution']:
            pred_dist = stats['prediction_distribution']
            # Sort by count (descending) and format as: "Predicted: 6B(35), 6A+(28), 6C(20)"
            sorted_preds = sorted(pred_dist.items(), key=lambda x: x[1], reverse=True)
            pred_str = ", ".join([f"{pred}({count})" for pred, count in sorted_preds[:5]])  # Top 5
            print(f"{prefix}  └─ Predicted: {pred_str}")
    
    print(separator)


def _print_grade_centroids_table(centroid_data: Dict, indent: int = 0):
    """
    Print per-grade centroid statistics as a formatted table.
    
    Displays a summary of each grade's latent space centroid without
    showing the full high-dimensional mean vectors.
    
    Args:
        centroid_data: Dictionary mapping grade names to centroid statistics
        indent: Indentation level
    """
    prefix = " " * indent
    
    # Table header
    header = f"{prefix}{'Grade':<10} {'Latent Std':<15} {'Samples':<10}"
    separator = f"{prefix}{'-' * 35}"
    
    print(separator)
    print(header)
    print(separator)
    
    # Grades are already strings, just sort them alphabetically
    # (they should be in order since they come from a sorted numeric label list)
    for grade_name, stats in sorted(centroid_data.items()):
        latent_std = stats.get('std', 0.0)
        count = stats.get('count', 0)
        
        row = f"{prefix}{grade_name:<10} {latent_std:<15.4f} {count:<10}"
        print(row)
    
    print(separator)
    print(f"{prefix}Note: Full {len(next(iter(centroid_data.values()))['mean'])}-dimensional centroid vectors available in JSON output")


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
        help='Comma-separated global grade labels (e.g., "2,3,4")'
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
        default='../data/problems.json',
        help='Path to dataset JSON file (default: ../data/problems.json)'
    )
    evaluate_parser.add_argument(
        '--classifier-checkpoint',
        type=str,
        default=None,
        help='Path to classifier checkpoint (required for classifier_check metric)'
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

