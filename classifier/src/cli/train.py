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
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from .utils import load_config, setup_device, print_section_header, print_completion_message
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
    print(f"\n✓ Loaded configuration from: {args.config}")
    
    # Set device
    device_name = config.get('device', 'cpu')
    device, device_name = setup_device(device_name)
    print(f"✓ Using device: {device}")
    
    # Load dataset
    data_path = config['data']['path']
    print(f"\n📂 Loading dataset from: {data_path}")
    
    # Check if repeats filtering is enabled
    filter_repeats_enabled = config.get('data', {}).get('filter_repeats', False)
    min_repeats = None
    
    if filter_repeats_enabled:
        min_repeats = config['data'].get('min_repeats', 1)
        print(f"   Filtering routes with minimum {min_repeats} repeat(s)")
    
    dataset = load_dataset(data_path, min_repeats=min_repeats)
    
    if len(dataset) == 0:
        print("❌ Error: No problems found in dataset")
        sys.exit(1)
    
    # Get dataset statistics
    stats = get_dataset_stats(dataset)
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total problems: {stats['total_problems']}")
    print(f"   Grade distribution:")
    for grade_label, count in sorted(stats['grade_distribution'].items()):
        grade_name = decode_grade(grade_label)
        print(f"      {grade_name}: {count}")
    
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
        
        print(f"\n🔍 Grade Filtering:")
        print(f"   Filtering to grades {decode_grade(min_grade_idx)} - {decode_grade(max_grade_idx)}")
        print(f"   Original problems: {original_count}")
        print(f"   Filtered problems: {filtered_count}")
        print(f"   Removed: {original_count - filtered_count}")
        print(f"   Using label offset: {grade_offset}")
        
        # Remap labels to start from 0
        dataset = [(tensor, remap_label(label, grade_offset)) for tensor, label in dataset]
    
    # Create data splits
    print(f"\n🔀 Creating train/val/test splits...")
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
    
    print(f"   Train: {len(train_dataset)} ({train_ratio*100:.0f}%)")
    print(f"   Val:   {len(val_dataset)} ({val_ratio*100:.0f}%)")
    print(f"   Test:  {len(test_dataset)} ({test_ratio*100:.0f}%)")
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )
    
    # Create model
    model_type = config['model']['type']
    num_classes = config['model']['num_classes']
    print(f"\n🧠 Creating model: {model_type.upper()}")
    
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
    if model_type in ['residual_cnn', 'deep_residual_cnn']:
        print(f"   Using advanced model with attention: {model_params['use_attention']}")
    
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"   Parameters: {num_params:,}")
    
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
    
    print(f"   Optimizer: {optimizer_type.upper()} (lr={learning_rate}, weight_decay={weight_decay})")
    
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
        
        print(f"   Using balanced class weights (capped at {max_weight})")
        print(f"   Weight range: {class_weights.min():.2f} - {class_weights.max():.2f}")
    else:
        class_weights = None
    
    # Create loss function (support advanced loss functions)
    loss_type = config['training'].get('loss_type', 'ce')
    if loss_type != 'ce':
        from src.losses import create_loss_function
        criterion = create_loss_function(
            loss_type=loss_type,
            num_classes=num_classes,
            class_weights=class_weights,
            gamma=config['training'].get('focal_gamma', 2.0),
            ordinal_weight=config['training'].get('ordinal_weight', 0.5),
            ordinal_alpha=config['training'].get('ordinal_alpha', 2.0),
            smoothing=label_smoothing
        )
        print(f"   Using {loss_type} loss")
        if loss_type in ['focal', 'focal_ordinal']:
            print(f"   Focal gamma: {config['training'].get('focal_gamma', 2.0)}")
        if loss_type in ['ordinal', 'focal_ordinal']:
            print(f"   Ordinal alpha: {config['training'].get('ordinal_alpha', 2.0)}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    if label_smoothing > 0:
        print(f"   Using label smoothing: {label_smoothing}")
    
    # Create learning rate scheduler
    use_scheduler = config['training'].get('use_scheduler', True)
    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.3,      # More aggressive reduction (was 0.5)
            patience=3,      # Faster response to plateaus (was 5)
            min_lr=1e-7      # Lower minimum (was 1e-6)
        )
        print(f"   Using ReduceLROnPlateau scheduler (factor=0.3, patience=3)")
    
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
        max_grade_index=max_grade_idx
    )
    
    if gradient_clip is not None:
        print(f"   Using gradient clipping: max_norm={gradient_clip}")
    
    # Train model
    num_epochs = config['training']['num_epochs']
    early_stopping_patience = config['training'].get('early_stopping_patience')
    
    print(f"\n🏋️  Training for {num_epochs} epochs...")
    if early_stopping_patience:
        print(f"   Early stopping: patience={early_stopping_patience}")
    
    # Record start time
    training_start_time = datetime.now()
    
    history, final_metrics = trainer.fit(
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        verbose=True
    )
    
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
    
    print(f"\n⏱️  Training duration: {duration_str}")
    
    # Evaluate on test set
    print(f"\n📈 Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    
    print(f"\n🎯 Test Set Results:")
    print(f"   Exact Accuracy:  {test_metrics['exact_accuracy']:.2f}%")
    print(f"   ±1 Grade Accuracy: {test_metrics['tolerance_1_accuracy']:.2f}%")
    print(f"   ±2 Grade Accuracy: {test_metrics['tolerance_2_accuracy']:.2f}%")
    print(f"   Loss: {test_metrics['avg_loss']:.4f}")
    
    # Generate unique timestamp for this training session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exact_acc = int(test_metrics['exact_accuracy'])
    tol1_acc = int(test_metrics['tolerance_1_accuracy'])
    tol2_acc = int(test_metrics['tolerance_2_accuracy'])
    
    # Save confusion matrix if requested (using the same timestamp)
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
        print(f"\n✓ Saved confusion matrix to: {cm_path}")
    
    # Generate unique model filename with timestamp and accuracy metrics
    unique_model_filename = f"model_{timestamp}_acc{exact_acc}_tol1-{tol1_acc}_tol2-{tol2_acc}.pth"
    unique_model_path = checkpoint_dir / unique_model_filename
    
    # Copy the best model to the unique filename
    best_model_path = checkpoint_dir / "best_model.pth"
    if best_model_path.exists():
        shutil.copy2(best_model_path, unique_model_path)
        print(f"\n✓ Saved unique model to: {unique_model_path}")
    
    # Log final test results to TensorBoard
    trainer.log_test_results(config, test_metrics, str(cm_path) if cm_path.exists() else None)
    
    print(f"\n✓ TensorBoard logs saved. View with: py -m tensorboard.main --logdir=runs")
    print_completion_message("✅ Training completed successfully!")

