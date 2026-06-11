"""
Train Command

Handles model training workflow including data loading, model creation,
training loop execution, and evaluation.
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from rich import box
from rich.panel import Panel
from rich.table import Table

from .utils import (
    console,
    load_config,
    setup_device,
    print_section_header,
    print_completion_message,
)
from src import (
    load_dataset,
    get_dataset_stats,
    create_datasets,
    create_data_loaders,
    create_model,
    count_parameters,
    Trainer,
    evaluate_model,
    generate_confusion_matrix,
    plot_confusion_matrix,
    decode_grade,
    get_all_grades,
)


def _print_key_value_table(title, rows):
    """Print a compact two-column Rich table."""
    table = Table(title=title, show_lines=False, box=box.ASCII)
    table.add_column("Setting", style="bold")
    table.add_column("Value", overflow="fold")
    for setting, value in rows:
        table.add_row(str(setting), str(value))
    console.print()
    console.print(table)


def _print_dataset_stats(stats):
    table = Table(title="Dataset Statistics", show_lines=False, box=box.ASCII)
    table.add_column("Grade", style="bold")
    table.add_column("Problems", justify="right")
    for grade_label, count in sorted(stats['grade_distribution'].items()):
        table.add_row(decode_grade(grade_label), str(count))

    console.print()
    console.print(f"[bold]Total problems:[/bold] {stats['total_problems']}")
    console.print(table)


def _print_split_summary(split_mode, splits):
    table = Table(title=f"Train/Val/Test Splits ({split_mode})", show_lines=False, box=box.ASCII)
    table.add_column("Split", style="bold")
    table.add_column("Samples", justify="right")
    table.add_column("Ratio", justify="right")
    for name, size, ratio in splits:
        table.add_row(name, str(size), f"{ratio * 100:.0f}%")
    console.print()
    console.print(table)


def _print_test_metrics(test_metrics):
    table = Table(title="Test Set Results", show_lines=False, box=box.ASCII)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Exact Accuracy", f"{test_metrics['exact_accuracy']:.2f}%")
    table.add_row("Macro Accuracy", f"{test_metrics['macro_accuracy']:.2f}%")
    table.add_row("+-1 Grade Accuracy", f"{test_metrics['tolerance_1_accuracy']:.2f}%")
    table.add_row("+-2 Grade Accuracy", f"{test_metrics['tolerance_2_accuracy']:.2f}%")
    table.add_row("Loss", f"{test_metrics['avg_loss']:.4f}")
    console.print()
    console.print(table)


def setup_train_parser(subparsers):
    """
    Setup argument parser for train command.
    
    Args:
        subparsers: ArgumentParser subparsers object
        
    Returns:
        Configured train parser
    """
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    train_parser.set_defaults(func=train_command)
    return train_parser


def train_command(args):
    """
    Execute training command.
    
    Args:
        args: Parsed command-line arguments
    """
    print_section_header("MOONBOARD GRADE PREDICTION - TRAINING")
    
    # Load configuration
    config = load_config(args.config)
    console.print(
        Panel.fit(
            f"[bold]Config:[/bold] {args.config}",
            title="Training Setup",
            border_style="cyan",
            box=box.ASCII,
        )
    )

    # Validate filtered grade configuration early to fail fast on class-space mismatch.
    num_classes = config['model']['num_classes']
    if config.get('data', {}).get('filter_grades', False):
        min_grade_idx = config['data']['min_grade_index']
        max_grade_idx = config['data']['max_grade_index']
        expected_classes = max_grade_idx - min_grade_idx + 1

        if expected_classes <= 0:
            raise ValueError(
                "Invalid filtered-grade config: max_grade_index must be >= min_grade_index "
                f"(got min_grade_index={min_grade_idx}, max_grade_index={max_grade_idx})."
            )

        if expected_classes != num_classes:
            raise ValueError(
                "Invalid filtered-grade config: when data.filter_grades=true, "
                "model.num_classes must equal (max_grade_index - min_grade_index + 1). "
                f"Got model.num_classes={num_classes}, min_grade_index={min_grade_idx}, "
                f"max_grade_index={max_grade_idx}, expected_classes={expected_classes}."
            )
    
    # Set up reproducibility if configured
    repro_seed = config.get('training', {}).get('reproducibility_seed', None)
    if repro_seed is not None:
        import random
        random.seed(repro_seed)
        np.random.seed(repro_seed)
        torch.manual_seed(repro_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(repro_seed)
        deterministic = config.get('training', {}).get('deterministic', False)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
        console.print(
            f"Reproducibility seed: {repro_seed} (deterministic={deterministic})",
            style="cyan",
        )

    # Set device
    device_name = config.get('device', 'cpu')
    device, device_name = setup_device(device_name)
    console.print(f"Using device: {device}", style="cyan")
    
    # Load dataset
    data_path = config['data']['path']
    console.print()
    console.print(f"Loading dataset from: {data_path}", style="cyan")
    
    # Check if repeats filtering is enabled
    filter_repeats_enabled = config.get('data', {}).get('filter_repeats', False)
    min_repeats = None
    
    if filter_repeats_enabled:
        min_repeats = config['data'].get('min_repeats', 1)
        console.print(f"Filtering routes with minimum {min_repeats} repeat(s)", style="cyan")
    
    dataset = load_dataset(data_path, min_repeats=min_repeats)
    
    if len(dataset) == 0:
        console.print("[ERROR] Error: No problems found in dataset", style="bold red")
        sys.exit(1)
    
    # Get dataset statistics
    stats = get_dataset_stats(dataset)
    _print_dataset_stats(stats)
    
    # Check if grade filtering is enabled
    filter_enabled = config.get('data', {}).get('filter_grades', False)
    grade_offset = 0
    min_grade_idx = 0
    max_grade_idx = 18
    
    if filter_enabled:
        from src import filter_dataset_by_grades, remap_label
        
        min_grade_idx = config['data']['min_grade_index']
        max_grade_idx = config['data']['max_grade_index']
        grade_offset = min_grade_idx
        
        # Filter dataset to specified grade range
        original_count = len(dataset)
        dataset = filter_dataset_by_grades(dataset, min_grade_idx, max_grade_idx)
        filtered_count = len(dataset)
        
        _print_key_value_table(
            "Grade Filtering",
            [
                ("Grade range", f"{decode_grade(min_grade_idx)} - {decode_grade(max_grade_idx)}"),
                ("Original problems", original_count),
                ("Filtered problems", filtered_count),
                ("Removed", original_count - filtered_count),
                ("Label offset", grade_offset),
            ],
        )
        
        # Remap labels to start from 0
        dataset = [(tensor, remap_label(label, grade_offset)) for tensor, label in dataset]
    
    # Create data splits
    group_by_layout = config.get('data', {}).get('group_by_layout', False)
    split_mode = "grouped (layout-aware)" if group_by_layout else "stratified"
    console.print()
    console.print(f"Creating train/val/test splits ({split_mode})...", style="cyan")
    tensors = np.array([x[0] for x in dataset])
    labels = np.array([x[1] for x in dataset])
    
    train_ratio = config['data']['train_ratio']
    val_ratio = config['data']['val_ratio']
    test_ratio = config['data']['test_ratio']
    random_seed = config['data'].get('random_seed', 42)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        tensors, labels, config, train_ratio, val_ratio, test_ratio, random_seed
    )
    
    _print_split_summary(
        split_mode,
        [
            ("Train", len(train_dataset), train_ratio),
            ("Val", len(val_dataset), val_ratio),
            ("Test", len(test_dataset), test_ratio),
        ],
    )
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    
    # Check if balanced sampling is enabled
    use_balanced_sampling = config['training'].get('use_balanced_sampling', False)
    
    if use_balanced_sampling:
        # Calculate sample weights for balanced sampling
        train_labels = train_dataset.labels
        class_counts = np.bincount(train_labels, minlength=num_classes)
        
        # Avoid division by zero for classes not present
        class_counts = np.maximum(class_counts, 1)
        
        # Use sqrt balancing by default (less aggressive than full inverse)
        # Full inverse: 1/count, Sqrt: 1/sqrt(count)
        sampling_strategy = config['training'].get('balanced_sampling_strategy', 'sqrt')
        
        if sampling_strategy == 'sqrt':
            class_weights_sampling = 1.0 / np.sqrt(class_counts)
        elif sampling_strategy == 'inverse':
            class_weights_sampling = 1.0 / class_counts
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        # Assign weight to each sample based on its class
        sample_weights = class_weights_sampling[train_labels]
        sample_weights = torch.DoubleTensor(sample_weights)
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        # Create train loader with sampler (can't use shuffle with sampler)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        
        # Create val/test loaders normally
        _, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size
        )
        
        _print_key_value_table(
            "Data Loader",
            [
                ("Batch size", batch_size),
                ("Balanced sampling", f"{sampling_strategy} strategy"),
                (
                    "Sampling weight range",
                    f"{class_weights_sampling.min():.3f} - {class_weights_sampling.max():.3f}",
                ),
            ],
        )
    else:
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size
        )
        _print_key_value_table(
            "Data Loader",
            [
                ("Batch size", batch_size),
                ("Balanced sampling", "off"),
            ],
        )
    
    # Create model
    model_type = config['model']['type']
    console.print()
    console.print(f"Creating model: {model_type.upper()}", style="cyan")
    
    # Extract model-specific parameters from config
    model_params = {
        'use_attention': config['model'].get('use_attention', True),
        'dropout_conv': config['model'].get('dropout_conv', 0.1),
        'dropout_fc1': config['model'].get('dropout_fc1', 0.3),
        'dropout_fc2': config['model'].get('dropout_fc2', 0.4)
    }
    
    # Create model using unified factory (handles all model types)
    model = create_model(
        model_type=model_type,
        num_classes=num_classes,
        **model_params
    )
    
    # Print model-specific info
    model_notes = []
    if model_type in ['residual_cnn', 'deep_residual_cnn']:
        model_notes.append(f"Advanced model with attention: {model_params['use_attention']}")
    if model_type in ['cnn', 'residual_cnn', 'deep_residual_cnn']:
        model_notes.append("CoordConv coordinate channels")
    
    model = model.to(device)
    
    num_params = count_parameters(model)
    
    # Create optimizer
    optimizer_type = config['training'].get('optimizer', 'adam').lower()
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training'].get('weight_decay', 0.0001)
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    # Calculate class weights for imbalanced dataset
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    
    if config['training'].get('use_class_weights', True):
        # Calculate class weights using sklearn's balanced approach
        # This handles class imbalance without extreme weights
        # Get labels from training dataset
        train_labels = train_dataset.labels
        unique_classes = np.unique(train_labels)
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=train_labels
        )
        
        # Create full weight array for all classes (including those not in training set)
        class_weights = np.ones(num_classes)
        class_weights[unique_classes] = class_weights_array
        
        # Cap weights to reasonable range to prevent extreme values
        # Max weight of 5.0 means rare classes get at most 5x importance
        max_weight = config['training'].get('max_class_weight', 5.0)
        class_weights = np.clip(class_weights, 0.1, max_weight)
        
        class_weights = torch.FloatTensor(class_weights).to(device)
        class_weight_summary = (
            f"balanced, cap={max_weight}, "
            f"range={class_weights.min().item():.2f} - {class_weights.max().item():.2f}"
        )
    else:
        class_weights = None
        class_weight_summary = "off"
    
    # Create loss function (support advanced loss functions)
    loss_type = config['training'].get('loss_type', 'focal_ordinal')
    if loss_type != 'ce':
        from src.losses import create_loss_function
        criterion = create_loss_function(
            loss_type=loss_type,
            num_classes=num_classes,
            class_weights=class_weights,
            gamma=config['training'].get('focal_gamma', 2.0),
            ordinal_weight=config['training'].get('ordinal_weight', 0.5),
            ordinal_alpha=config['training'].get('ordinal_alpha', 2.0),
            ordinal_smoothing_kernel=config['training'].get('ordinal_smoothing_kernel'),
            smoothing=label_smoothing
        )
        loss_notes = [f"Loss: {loss_type}"]
        if loss_type in ['focal', 'focal_ordinal']:
            loss_notes.append(f"Focal gamma: {config['training'].get('focal_gamma', 2.0)}")
        if loss_type in ['ordinal', 'focal_ordinal']:
            loss_notes.append(f"Ordinal alpha: {config['training'].get('ordinal_alpha', 2.0)}")
        if loss_type == 'ordinal_smoothing':
            loss_notes.append(
                "Ordinal smoothing kernel: "
                f"{config['training'].get('ordinal_smoothing_kernel', [0.025, 0.075, 0.8, 0.075, 0.025])}"
            )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        loss_notes = ["Loss: cross entropy"]
    
    if label_smoothing > 0 and loss_type in ['ce', 'label_smoothing']:
        loss_notes.append(f"Label smoothing: {label_smoothing}")
    
    # Create learning rate scheduler
    use_scheduler = config['training'].get('use_scheduler', True)
    scheduler = None
    if use_scheduler:
        scheduler_factor = config['training'].get('scheduler_factor', 0.3)
        scheduler_patience = config['training'].get('scheduler_patience', 3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=1e-7
        )
        scheduler_summary = (
            f"ReduceLROnPlateau, factor={scheduler_factor}, "
            f"patience={scheduler_patience}"
        )
    else:
        scheduler_summary = "off"
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint']['dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get gradient clipping value
    gradient_clip = config['training'].get('gradient_clip', None)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        scheduler=scheduler,
        gradient_clip=gradient_clip,
        grade_offset=grade_offset,
        min_grade_index=min_grade_idx,
        max_grade_index=max_grade_idx,
        console=console,
    )

    training_rows = [
        ("Model type", model_type.upper()),
        ("Parameters", f"{num_params:,}"),
        ("Optimizer", f"{optimizer_type.upper()} (lr={learning_rate}, weight_decay={weight_decay})"),
        ("Class weights", class_weight_summary),
        ("Scheduler", scheduler_summary),
    ]
    if model_notes:
        training_rows.append(("Model notes", "; ".join(model_notes)))
    if loss_notes:
        training_rows.append(("Loss", "; ".join(loss_notes)))
    
    if gradient_clip is not None:
        training_rows.append(("Gradient clipping", f"max_norm={gradient_clip}"))

    _print_key_value_table("Model and Training Configuration", training_rows)
    
    # Train model
    num_epochs = config['training']['num_epochs']
    early_stopping_patience = config['training'].get('early_stopping_patience')
    
    console.print()
    console.print(
        Panel.fit(
            f"[bold]Epochs:[/bold] {num_epochs}\n"
            f"[bold]Early stopping:[/bold] "
            f"{early_stopping_patience if early_stopping_patience else 'off'}",
            title="Training",
            border_style="cyan",
            box=box.ASCII,
        )
    )
    
    # Record start time
    training_start_time = datetime.now()
    
    history, final_metrics = trainer.fit(
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        verbose=True
    )

    # Persist training history for post-run analysis.
    history_path = trainer.save_history('training_history.json')
    console.print()
    console.print(f"Saved training history to: {history_path}", style="green")
    
    # Calculate training duration
    training_end_time = datetime.now()
    training_duration = training_end_time - training_start_time
    total_seconds = int(training_duration.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    if hours > 0:
        duration_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        duration_str = f"{minutes}m {seconds}s"
    else:
        duration_str = f"{seconds}s"
    
    console.print(f"Training duration: {duration_str}", style="cyan")
    
    # Evaluate the same checkpoint artifact that will be saved with metrics.
    best_model_path = checkpoint_dir / "best_model.pth"
    final_model_path = checkpoint_dir / "final_model.pth"
    eval_checkpoint_path = None

    if best_model_path.exists():
        eval_checkpoint_path = best_model_path
    elif final_model_path.exists():
        eval_checkpoint_path = final_model_path

    if eval_checkpoint_path is not None:
        checkpoint = torch.load(eval_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        console.print()
        console.print(
            f"Evaluating on test set using checkpoint: {eval_checkpoint_path.name}",
            style="cyan",
        )
    else:
        console.print()
        console.print(
            "Evaluating on test set using in-memory final model (no checkpoint found)",
            style="cyan",
        )

    test_metrics = evaluate_model(model, test_loader, device)
    _print_test_metrics(test_metrics)
    
    # Generate unique timestamp for this training session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exact_acc = int(test_metrics['exact_accuracy'])
    tol1_acc = int(test_metrics['tolerance_1_accuracy'])
    tol2_acc = int(test_metrics['tolerance_2_accuracy'])
    
    # Save confusion matrix if requested (using the same timestamp)
    cm_path = None
    if config.get('evaluation', {}).get('save_confusion_matrix', False):
        cm_filename = f"confusion_matrix_{timestamp}.png"
        cm_path = checkpoint_dir / cm_filename
        
        # Use filtered grade names if model is filtered
        if grade_offset > 0:
            from src import get_filtered_grade_names
            cm_grade_names = get_filtered_grade_names(min_grade_idx, max_grade_idx)
            cm_num_classes = len(cm_grade_names)
        else:
            cm_grade_names = get_all_grades()
            cm_num_classes = len(cm_grade_names)
        
        cm = generate_confusion_matrix(
            test_metrics['predictions'],
            test_metrics['labels'],
            num_classes=cm_num_classes
        )
        
        plot_confusion_matrix(
            cm,
            cm_grade_names,
            str(cm_path),
            normalize=True
        )
        console.print()
        console.print(f"Saved confusion matrix to: {cm_path}", style="green")
    
    # Generate unique model filename with timestamp and accuracy metrics
    unique_model_filename = f"model_{timestamp}_acc{exact_acc}_tol1-{tol1_acc}_tol2-{tol2_acc}.pth"
    unique_model_path = checkpoint_dir / unique_model_filename
    
    # Copy the evaluated checkpoint artifact to the unique filename.
    if eval_checkpoint_path is not None and eval_checkpoint_path.exists():
        shutil.copy2(eval_checkpoint_path, unique_model_path)
        _print_key_value_table(
            "Saved Model Artifact",
            [
                ("Unique model", unique_model_path),
                ("Source checkpoint", eval_checkpoint_path.name),
            ],
        )
    
    # Log final test results to TensorBoard
    trainer.log_test_results(config, test_metrics, str(cm_path) if cm_path and cm_path.exists() else None)
    
    console.print()
    console.print(
        "TensorBoard logs saved. View with: py -m tensorboard.main --logdir=runs",
        style="green",
    )
    print_completion_message("Training completed successfully!")
