"""
Training Loop Module

Manages the training process including training epochs, validation,
checkpointing, early stopping, and metrics tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import os
import json
from pathlib import Path


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
        gradient_clip: Optional[float] = None
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
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking variables
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
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
    
    def validate_epoch(self) -> Tuple[float, float]:
        """
        Validate the model for one epoch.
        
        Returns:
            Tuple of (average validation loss, validation accuracy)
            Returns (0.0, 0.0) if val_loader is None
            
        Examples:
            >>> trainer = Trainer(model, train_loader, val_loader, optimizer, criterion)
            >>> val_loss, val_acc = trainer.validate_epoch()
        """
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
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
                total_samples += batch_labels.size(0)
                
                # Track loss
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, accuracy
    
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
            val_loss, val_accuracy = self.validate_epoch()
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            if verbose:
                if self.val_loader is not None:
                    print(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {train_loss:.4f} - "
                          f"Val Loss: {val_loss:.4f} - "
                          f"Val Acc: {val_accuracy:.4f}")
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
            'history': self.history
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
            'val_accuracy': []
        })
    
    def save_history(self, filename: str = 'training_history.json') -> None:
        """
        Save training history to JSON file.
        
        Args:
            filename: Name of the JSON file to save
            
        Examples:
            >>> trainer.save_history('history.json')
        """
        history_path = self.checkpoint_dir / filename
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_history(self) -> Dict[str, List[float]]:
        """
        Get the training history.
        
        Returns:
            Dictionary containing training metrics history
            
        Examples:
            >>> history = trainer.get_history()
            >>> print(history['train_loss'])
        """
        return {
            key: list(value) for key, value in self.history.items()
        }

