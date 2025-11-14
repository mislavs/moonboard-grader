"""
Unit tests for the Trainer module.

Tests the Trainer class for training loops, validation, checkpointing,
early stopping, and metrics tracking.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import json
from pathlib import Path

from src.trainer import Trainer
from src.models import FullyConnectedModel, ConvolutionalModel


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_dataset():
    """Create a small dataset for testing (10 samples)."""
    # Create random data: 10 samples, 3 channels, 18 rows, 11 columns
    data = torch.randn(10, 3, 18, 11)
    labels = torch.randint(0, 5, (10,))  # 5 classes
    return TensorDataset(data, labels)


@pytest.fixture
def tiny_dataloaders(small_dataset):
    """Create tiny train and validation dataloaders."""
    # Split into train (6) and val (4)
    train_dataset = TensorDataset(
        small_dataset.tensors[0][:6],
        small_dataset.tensors[1][:6]
    )
    val_dataset = TensorDataset(
        small_dataset.tensors[0][6:],
        small_dataset.tensors[1][6:]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    return train_loader, val_loader


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return FullyConnectedModel(num_classes=5)


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Test Initialization
# ============================================================================

def test_trainer_initialization(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test basic trainer initialization."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device='cpu',
        checkpoint_dir=temp_checkpoint_dir
    )
    
    assert trainer.model is not None
    assert trainer.train_loader is train_loader
    assert trainer.val_loader is val_loader
    assert trainer.device == 'cpu'
    assert trainer.best_val_loss == float('inf')
    assert trainer.epochs_without_improvement == 0
    assert trainer.current_epoch == 0
    assert Path(temp_checkpoint_dir).exists()


def test_trainer_initialization_without_val_loader(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test trainer initialization without validation loader."""
    train_loader, _ = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=None,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    assert trainer.val_loader is None


def test_trainer_creates_checkpoint_dir(simple_model, tiny_dataloaders):
    """Test that trainer creates checkpoint directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / 'models' / 'checkpoints'
        assert not checkpoint_dir.exists()
        
        train_loader, val_loader = tiny_dataloaders
        optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        trainer = Trainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint_dir=str(checkpoint_dir)
        )
        
        assert checkpoint_dir.exists()


def test_trainer_initialization_invalid_train_loader(simple_model, temp_checkpoint_dir):
    """Test trainer initialization with invalid train_loader."""
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # None train_loader
    with pytest.raises(ValueError, match="train_loader cannot be None"):
        Trainer(
            model=simple_model,
            train_loader=None,
            val_loader=None,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint_dir=temp_checkpoint_dir
        )


def test_trainer_initialization_empty_train_loader(simple_model, temp_checkpoint_dir):
    """Test trainer initialization with empty train_loader."""
    empty_dataset = TensorDataset(
        torch.empty(0, 3, 18, 11),
        torch.empty(0, dtype=torch.long)
    )
    empty_loader = DataLoader(empty_dataset, batch_size=2)
    
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    with pytest.raises(ValueError, match="train_loader cannot be empty"):
        Trainer(
            model=simple_model,
            train_loader=empty_loader,
            val_loader=None,
            optimizer=optimizer,
            criterion=criterion,
            checkpoint_dir=temp_checkpoint_dir
        )


def test_trainer_initialization_invalid_types(tiny_dataloaders, temp_checkpoint_dir):
    """Test trainer initialization with invalid argument types."""
    train_loader, val_loader = tiny_dataloaders
    
    # Invalid model type
    with pytest.raises(TypeError, match="model must be a PyTorch nn.Module"):
        Trainer(
            model="not a model",
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=None,
            criterion=None,
            checkpoint_dir=temp_checkpoint_dir
        )


# ============================================================================
# Test Training Epoch
# ============================================================================

def test_train_epoch(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test training for one epoch."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    loss = trainer.train_epoch()
    
    assert isinstance(loss, float)
    assert loss > 0  # Loss should be positive
    assert not torch.isnan(torch.tensor(loss))  # Loss should not be NaN


def test_train_epoch_model_in_train_mode(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test that model is in training mode during train_epoch."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Set to eval mode first
    trainer.model.eval()
    assert not trainer.model.training
    
    # Train epoch should set to train mode
    trainer.train_epoch()
    assert trainer.model.training


def test_train_epoch_gradients_computed(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test that gradients are computed during training."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Check that parameters have no gradients initially
    for param in trainer.model.parameters():
        assert param.grad is None or torch.all(param.grad == 0)
    
    # Train one epoch
    trainer.train_epoch()
    
    # Check that at least some parameters have gradients
    has_gradients = any(
        param.grad is not None and torch.any(param.grad != 0)
        for param in trainer.model.parameters()
    )
    assert has_gradients


# ============================================================================
# Test Validation Epoch
# ============================================================================

def test_validate_epoch(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test validation for one epoch."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    val_loss, val_accuracy, val_tolerance_1_accuracy = trainer.validate_epoch()
    
    assert isinstance(val_loss, float)
    assert isinstance(val_accuracy, float)
    assert isinstance(val_tolerance_1_accuracy, float)
    assert val_loss >= 0
    assert 0 <= val_accuracy <= 1
    assert 0 <= val_tolerance_1_accuracy <= 1


def test_validate_epoch_without_val_loader(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test validation when val_loader is None."""
    train_loader, _ = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=None,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    val_loss, val_accuracy, val_tolerance_1_accuracy = trainer.validate_epoch()
    
    assert val_loss == 0.0
    assert val_accuracy == 0.0
    assert val_tolerance_1_accuracy == 0.0


def test_validate_epoch_model_in_eval_mode(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test that model is in eval mode during validation."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Set to train mode first
    trainer.model.train()
    assert trainer.model.training
    
    # Validate should set to eval mode
    trainer.validate_epoch()
    assert not trainer.model.training


def test_validate_epoch_no_gradients(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test that gradients are not computed during validation."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Zero out any existing gradients
    trainer.optimizer.zero_grad()
    
    # Validate
    trainer.validate_epoch()
    
    # Check that parameters still have no gradients
    for param in trainer.model.parameters():
        assert param.grad is None or torch.all(param.grad == 0)


# ============================================================================
# Test Full Training (fit)
# ============================================================================

def test_fit_basic(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test basic training with fit method."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    history, final_metrics = trainer.fit(num_epochs=3, verbose=False)
    
    assert 'train_loss' in history
    assert 'val_loss' in history
    assert 'val_accuracy' in history
    assert len(history['train_loss']) == 3
    assert len(history['val_loss']) == 3
    assert len(history['val_accuracy']) == 3
    assert 'final_val_loss' in final_metrics
    assert 'final_val_accuracy' in final_metrics


def test_fit_loss_decreases(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test that training loss generally decreases over epochs."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.01)  # Higher LR
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    history, _ = trainer.fit(num_epochs=10, verbose=False)
    
    # Loss should generally decrease (allow some fluctuation)
    first_loss = history['train_loss'][0]
    last_loss = history['train_loss'][-1]
    assert last_loss < first_loss


def test_fit_saves_best_model(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test that best model is saved during training."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    trainer.fit(num_epochs=5, verbose=False)
    
    # Check that best_model.pth exists
    best_model_path = Path(temp_checkpoint_dir) / 'best_model.pth'
    assert best_model_path.exists()


def test_fit_saves_final_model(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test that final model is saved after training."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    trainer.fit(num_epochs=3, verbose=False)
    
    # Check that final_model.pth exists
    final_model_path = Path(temp_checkpoint_dir) / 'final_model.pth'
    assert final_model_path.exists()


def test_fit_without_validation(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test training without validation data."""
    train_loader, _ = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=None,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    history, _ = trainer.fit(num_epochs=3, verbose=False)
    
    assert len(history['train_loss']) == 3
    assert all(loss == 0.0 for loss in history['val_loss'])
    assert all(acc == 0.0 for acc in history['val_accuracy'])


def test_fit_invalid_num_epochs(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test fit with invalid num_epochs."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    with pytest.raises(ValueError, match="num_epochs must be positive"):
        trainer.fit(num_epochs=0)
    
    with pytest.raises(ValueError, match="num_epochs must be positive"):
        trainer.fit(num_epochs=-5)


# ============================================================================
# Test Early Stopping
# ============================================================================

def test_early_stopping_triggers(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test that early stopping mechanism tracks epochs without improvement."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Train with early stopping patience
    history, _ = trainer.fit(num_epochs=20, early_stopping_patience=3, verbose=False)
    
    # Verify early stopping attributes exist and are tracked
    assert hasattr(trainer, 'epochs_without_improvement')
    assert hasattr(trainer, 'best_val_loss')
    assert trainer.best_val_loss < float('inf')  # Should have been updated
    
    # If early stopping triggered, epochs_without_improvement should equal patience
    # If not, it should be less than patience
    if len(history['train_loss']) < 20:
        assert trainer.epochs_without_improvement >= 3
    else:
        assert trainer.epochs_without_improvement < 3


def test_early_stopping_without_val_loader(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test that early stopping doesn't trigger without validation data."""
    train_loader, _ = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=None,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Should complete all epochs even with early stopping patience
    history, _ = trainer.fit(num_epochs=5, early_stopping_patience=2, verbose=False)
    
    assert len(history['train_loss']) == 5


def test_no_early_stopping_when_patience_none(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test that early stopping doesn't occur when patience is None."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Should complete all epochs without early stopping
    history, _ = trainer.fit(num_epochs=5, early_stopping_patience=None, verbose=False)
    
    assert len(history['train_loss']) == 5


def test_best_val_loss_tracking(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test that best validation loss is properly tracked."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Train for a few epochs
    history, _ = trainer.fit(num_epochs=5, verbose=False)
    
    # Best val loss should be the minimum of all validation losses
    assert trainer.best_val_loss == min(history['val_loss'])
    
    # Best val loss should be less than infinity
    assert trainer.best_val_loss < float('inf')


# ============================================================================
# Test Checkpointing
# ============================================================================

def test_save_checkpoint(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test saving a checkpoint."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Train for one epoch
    trainer.train_epoch()
    trainer.current_epoch = 1
    
    # Save checkpoint
    trainer.save_checkpoint('test_checkpoint.pth')
    
    # Check file exists
    checkpoint_path = Path(temp_checkpoint_dir) / 'test_checkpoint.pth'
    assert checkpoint_path.exists()
    
    # Check checkpoint contents
    checkpoint = torch.load(checkpoint_path)
    assert 'model_state_dict' in checkpoint
    assert 'optimizer_state_dict' in checkpoint
    assert 'epoch' in checkpoint
    assert 'best_val_loss' in checkpoint
    assert 'history' in checkpoint


def test_load_checkpoint(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test loading a checkpoint."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create and train first trainer
    trainer1 = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    trainer1.fit(num_epochs=2, verbose=False)
    trainer1.save_checkpoint('test_checkpoint.pth')
    
    # Create second trainer and load checkpoint
    model2 = FullyConnectedModel(num_classes=5)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    
    trainer2 = Trainer(
        model=model2,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer2,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    trainer2.load_checkpoint('test_checkpoint.pth')
    
    # Check that states match
    assert trainer2.current_epoch == trainer1.current_epoch
    assert trainer2.best_val_loss == trainer1.best_val_loss
    assert trainer2.history == trainer1.history


def test_load_checkpoint_file_not_found(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test loading a non-existent checkpoint."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        trainer.load_checkpoint('nonexistent.pth')


def test_checkpoint_state_dict_restoration(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test that model weights are correctly restored from checkpoint."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Train and save
    trainer.fit(num_epochs=2, verbose=False)
    trainer.save_checkpoint('weights_test.pth')
    
    # Get a weight tensor value
    original_weight = trainer.model.network[1].weight.data.clone()
    
    # Create new model with random weights
    model2 = FullyConnectedModel(num_classes=5)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
    
    trainer2 = Trainer(
        model=model2,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer2,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    # Load checkpoint
    trainer2.load_checkpoint('weights_test.pth')
    
    # Weights should match
    restored_weight = trainer2.model.network[1].weight.data
    assert torch.allclose(original_weight, restored_weight)


# ============================================================================
# Test History Tracking
# ============================================================================

def test_history_tracking(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test that training history is correctly tracked."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    num_epochs = 5
    history, _ = trainer.fit(num_epochs=num_epochs, verbose=False)
    
    # Check history structure
    assert isinstance(history, dict)
    assert 'train_loss' in history
    assert 'val_loss' in history
    assert 'val_accuracy' in history
    assert 'val_tolerance_1_accuracy' in history
    
    # Check lengths
    assert len(history['train_loss']) == num_epochs
    assert len(history['val_loss']) == num_epochs
    assert len(history['val_accuracy']) == num_epochs
    assert len(history['val_tolerance_1_accuracy']) == num_epochs
    
    # Check all values are numbers
    for loss in history['train_loss']:
        assert isinstance(loss, float)
    for loss in history['val_loss']:
        assert isinstance(loss, float)
    for acc in history['val_accuracy']:
        assert isinstance(acc, float)
    for acc in history['val_tolerance_1_accuracy']:
        assert isinstance(acc, float)


def test_get_history(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test getting training history."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    trainer.fit(num_epochs=3, verbose=False)
    history = trainer.get_history()
    
    assert isinstance(history, dict)
    assert len(history['train_loss']) == 3
    
    # Should be a copy, not a reference
    history['train_loss'].append(999.0)
    assert len(trainer.history['train_loss']) == 3


def test_save_history(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test saving training history to JSON."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    trainer.fit(num_epochs=3, verbose=False)
    trainer.save_history('test_history.json')
    
    # Check file exists
    history_path = Path(temp_checkpoint_dir) / 'test_history.json'
    assert history_path.exists()
    
    # Load and verify contents
    with open(history_path, 'r') as f:
        loaded_history = json.load(f)
    
    assert loaded_history == trainer.history


# ============================================================================
# Test with Different Models
# ============================================================================

def test_trainer_with_cnn_model(tiny_dataloaders, temp_checkpoint_dir):
    """Test trainer with convolutional model."""
    train_loader, val_loader = tiny_dataloaders
    model = ConvolutionalModel(num_classes=5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    history, _ = trainer.fit(num_epochs=3, verbose=False)
    
    assert len(history['train_loss']) == 3
    assert all(isinstance(loss, float) for loss in history['train_loss'])


# ============================================================================
# Test with Different Optimizers
# ============================================================================

def test_trainer_with_sgd_optimizer(simple_model, tiny_dataloaders, temp_checkpoint_dir):
    """Test trainer with SGD optimizer."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.SGD(simple_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    history, _ = trainer.fit(num_epochs=3, verbose=False)
    
    assert len(history['train_loss']) == 3


# ============================================================================
# Test Verbose Output
# ============================================================================

def test_fit_verbose_output(simple_model, tiny_dataloaders, temp_checkpoint_dir, capsys):
    """Test that verbose output is printed when verbose=True."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    trainer.fit(num_epochs=2, verbose=True)
    
    captured = capsys.readouterr()
    assert "Training for" in captured.out
    assert "Epoch" in captured.out
    assert "Train Loss" in captured.out


def test_fit_no_verbose_output(simple_model, tiny_dataloaders, temp_checkpoint_dir, capsys):
    """Test that no output is printed when verbose=False."""
    train_loader, val_loader = tiny_dataloaders
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=simple_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint_dir=temp_checkpoint_dir
    )
    
    trainer.fit(num_epochs=2, verbose=False)
    
    captured = capsys.readouterr()
    assert captured.out == ""

