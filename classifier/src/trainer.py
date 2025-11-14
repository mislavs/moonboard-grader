"""
Training Loop Module

Manages the training process including training epochs, validation,
checkpointing, early stopping, and metrics tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
from datetime import datetime


class Trainer:
    """
    Trainer class for managing the model training process.
    
    Handles training loops, validation, checkpointing, early stopping,
    and metrics tracking.
    
    Attributes:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to train on (cpu or cuda)
        checkpoint_dir: Directory to save checkpoints
        history: Dictionary tracking training and validation metrics
        best_val_loss: Best validation loss seen so far
        epochs_without_improvement: Counter for early stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        checkpoint_dir: str = 'models',
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        gradient_clip: Optional[float] = None,
        grade_offset: int = 0,
        min_grade_index: int = 0,
        max_grade_index: int = 18
    ):
        """
        Initialize the Trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (can be None)
            optimizer: PyTorch optimizer
            criterion: Loss function (e.g., nn.CrossEntropyLoss())
            device: Device to train on ('cpu' or 'cuda')
            checkpoint_dir: Directory to save model checkpoints
            scheduler: Optional learning rate scheduler
            gradient_clip: Optional gradient clipping max norm value
            grade_offset: Offset for grade label remapping (0 if not using filtering)
            min_grade_index: Minimum grade index in filtered range
            max_grade_index: Maximum grade index in filtered range
            
        Raises:
            ValueError: If train_loader is None or empty
            TypeError: If arguments have incorrect types
        """
        # Validate inputs
        if train_loader is None:
            raise ValueError("train_loader cannot be None")
        if len(train_loader) == 0:
            raise ValueError("train_loader cannot be empty")
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a PyTorch nn.Module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a PyTorch optimizer")
        if not isinstance(criterion, nn.Module):
            raise TypeError("criterion must be a PyTorch nn.Module")
            
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.grade_offset = grade_offset
        self.min_grade_index = min_grade_index
        self.max_grade_index = max_grade_index
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(f'runs/{run_name}')
        
        # Initialize tracking variables
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_tolerance_1_accuracy': []
        }
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.current_epoch = 0
    
    def train_epoch(self) -> float:
        """
        Train the model for one epoch.
        
        Returns:
            Average training loss for the epoch
            
        Examples:
            >>> trainer = Trainer(model, train_loader, val_loader, optimizer, criterion)
            >>> avg_loss = trainer.train_epoch()
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_data, batch_labels in self.train_loader:
            # Move data to device
            batch_data = batch_data.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_data)
            loss = self.criterion(outputs, batch_labels)
            
            # Backward pass and optimization
            loss.backward()
            
            # Apply gradient clipping if specified
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate_epoch(self) -> Tuple[float, float, float]:
        """
        Validate the model for one epoch.
        
        Returns:
            Tuple of (average validation loss, validation accuracy, tolerance 1 accuracy)
            Returns (0.0, 0.0, 0.0) if val_loader is None
            
        Examples:
            >>> trainer = Trainer(model, train_loader, val_loader, optimizer, criterion)
            >>> val_loss, val_acc, val_tol1_acc = trainer.validate_epoch()
        """
        if self.val_loader is None:
            return 0.0, 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        tolerance_1_correct = 0
        total_samples = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in self.val_loader:
                # Move data to device
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == batch_labels).sum().item()
                
                # Calculate tolerance 1 accuracy (within ±1 grade)
                differences = torch.abs(predicted - batch_labels)
                tolerance_1_correct += (differences <= 1).sum().item()
                
                total_samples += batch_labels.size(0)
                
                # Track loss
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        tolerance_1_accuracy = tolerance_1_correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy, tolerance_1_accuracy
    
    def fit(
        self,
        num_epochs: int,
        early_stopping_patience: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[Dict[str, List[float]], Dict[str, float]]:
        """
        Train the model for multiple epochs with optional early stopping.
        
        Args:
            num_epochs: Number of epochs to train for
            early_stopping_patience: Number of epochs to wait for improvement
                                    before stopping. If None, no early stopping.
            verbose: Whether to print training progress
            
        Returns:
            Tuple containing:
                - Dictionary with training history (train_loss, val_loss, val_accuracy)
                - Dictionary with final metrics (final_val_loss, final_val_accuracy)
                
        Raises:
            ValueError: If num_epochs is not positive
            
        Examples:
            >>> trainer = Trainer(model, train_loader, val_loader, optimizer, criterion)
            >>> history, final_metrics = trainer.fit(num_epochs=50, early_stopping_patience=5)
        """
        if num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        if verbose:
            print(f"Training for {num_epochs} epochs...")
            print(f"Device: {self.device}")
            print(f"Checkpoint directory: {self.checkpoint_dir}")
            if early_stopping_patience is not None:
                print(f"Early stopping patience: {early_stopping_patience}")
            print("-" * 60)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_accuracy, val_tolerance_1_accuracy = self.validate_epoch()
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['val_tolerance_1_accuracy'].append(val_tolerance_1_accuracy)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            if self.val_loader is not None:
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/val_exact', val_accuracy, epoch)
                self.writer.add_scalar('Accuracy/val_tolerance_1', val_tolerance_1_accuracy, epoch)
            
            # Print progress
            if verbose:
                if self.val_loader is not None:
                    print(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {train_loss:.4f} - "
                          f"Val Loss: {val_loss:.4f} - "
                          f"Val Acc: {val_accuracy:.4f} - "
                          f"Val ±1 Acc: {val_tolerance_1_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {train_loss:.4f}")
            
            # Save best model based on validation loss
            if self.val_loader is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pth')
                if verbose:
                    print(f"  → New best validation loss: {val_loss:.4f} - Saved checkpoint")
            else:
                self.epochs_without_improvement += 1
            
            # Step the learning rate scheduler if using one
            if self.scheduler is not None and self.val_loader is not None:
                self.scheduler.step(val_loss)
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Early stopping check
            if early_stopping_patience is not None and self.val_loader is not None:
                if self.epochs_without_improvement >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping triggered after {epoch+1} epochs")
                        print(f"Best validation loss: {self.best_val_loss:.4f}")
                    break
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        if verbose:
            print("-" * 60)
            print("Training completed!")
            if self.val_loader is not None:
                print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Prepare final metrics
        final_metrics = {
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else 0.0,
            'final_val_accuracy': self.history['val_accuracy'][-1] if self.history['val_accuracy'] else 0.0
        }
        
        return self.history, final_metrics
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            filename: Name of the checkpoint file
            
        Examples:
            >>> trainer.save_checkpoint('model_epoch_10.pth')
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'grade_offset': self.grade_offset,
            'min_grade_index': self.min_grade_index,
            'max_grade_index': self.max_grade_index
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, filename: str) -> None:
        """
        Load model checkpoint.
        
        Args:
            filename: Name of the checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            
        Examples:
            >>> trainer.load_checkpoint('best_model.pth')
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_tolerance_1_accuracy': []
        })
    
    def log_test_results(self, config: Dict, test_metrics: Dict, confusion_matrix_path: Optional[str] = None) -> None:
        """
        Log final test results and hyperparameters to TensorBoard, then close the writer.
        
        Args:
            config: Configuration dictionary with hyperparameters
            test_metrics: Dictionary containing test metrics (exact_accuracy, tolerance_1_accuracy, etc.)
            confusion_matrix_path: Optional path to confusion matrix image
        """
        # Log test metrics as scalars
        self.writer.add_scalar('Test/exact_accuracy', test_metrics['exact_accuracy'], 0)
        self.writer.add_scalar('Test/tolerance_1_accuracy', test_metrics['tolerance_1_accuracy'], 0)
        self.writer.add_scalar('Test/tolerance_2_accuracy', test_metrics['tolerance_2_accuracy'], 0)
        self.writer.add_scalar('Test/loss', test_metrics['avg_loss'], 0)
        
        # Log hyperparameters as text for easy viewing
        hparam_text = f"""
**Model Configuration:**
- Type: {config['model']['type']}
- Learning Rate: {config['training']['learning_rate']}
- Batch Size: {config['training']['batch_size']}
- Optimizer: {config['training'].get('optimizer', 'adam')}
- Weight Decay: {config['training'].get('weight_decay', 0.001)}
- Loss Type: {config['training'].get('loss_type', 'ce')}
- Label Smoothing: {config['training'].get('label_smoothing', 0.0)}
- Early Stopping Patience: {config['training'].get('early_stopping_patience', 'None')}

**Test Results:**
- Exact Accuracy: {test_metrics['exact_accuracy']:.2f}%
- ±1 Grade Accuracy: {test_metrics['tolerance_1_accuracy']:.2f}%
- ±2 Grade Accuracy: {test_metrics['tolerance_2_accuracy']:.2f}%
- Loss: {test_metrics['avg_loss']:.4f}
"""
        self.writer.add_text('Hyperparameters_and_Results', hparam_text, 0)
        
        # Log confusion matrix if provided
        if confusion_matrix_path and Path(confusion_matrix_path).exists():
            from PIL import Image
            img = Image.open(confusion_matrix_path)
            img_array = np.array(img)
            # Convert to format TensorBoard expects: (C, H, W)
            if len(img_array.shape) == 3:
                img_array = img_array.transpose(2, 0, 1)
            self.writer.add_image('Confusion_Matrix', img_array, 0)
        
        # Close the writer
        self.writer.close()
    

