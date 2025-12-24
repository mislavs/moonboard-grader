"""
Trainer Module

Training loop with early stopping, checkpointing, and TensorBoard logging.
"""

import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from moonboard_core import decode_grade


class Trainer:
    """
    Trainer for TransformerSequenceClassifier.
    
    Features:
        - Class-weighted loss for imbalanced data
        - Label smoothing
        - Early stopping
        - Learning rate scheduling
        - TensorBoard logging
        - Checkpoint saving/loading
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        num_epochs: int = 150,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = 'models',
        use_class_weights: bool = True,
        label_smoothing: float = 0.1,
        num_classes: int = 19,
        run_dir: Optional[str] = None
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            num_epochs: Maximum number of epochs
            early_stopping_patience: Epochs without improvement before stopping
            checkpoint_dir: Directory to save checkpoints
            use_class_weights: Whether to use inverse frequency class weights
            label_smoothing: Label smoothing factor (0 = no smoothing)
            num_classes: Number of output classes
            run_dir: Directory for TensorBoard logs
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss function with class weights and label smoothing
        class_weights = None
        if use_class_weights:
            class_weights = self._compute_class_weights(train_loader, num_classes)
            class_weights = class_weights.to(device)
        
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        
        # TensorBoard writer
        if run_dir is None:
            run_dir = f"runs/run_{int(time.time())}"
        self.writer = SummaryWriter(run_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []
    
    def _compute_class_weights(self, train_loader: DataLoader, num_classes: int) -> torch.Tensor:
        """Compute inverse frequency class weights from training data."""
        class_counts = Counter()
        
        for _, _, labels in train_loader:
            class_counts.update(labels.numpy())
        
        total = sum(class_counts.values())
        weights = torch.ones(num_classes)
        
        print("\nClass distribution:")
        for cls in sorted(class_counts.keys()):
            count = class_counts[cls]
            # Balanced class weighting (less aggressive than pure inverse frequency)
            weights[cls] = total / (num_classes * count)
            print(f"  {decode_grade(int(cls)):>4}: {count:>5} samples, weight={weights[cls]:.3f}")
        
        # Clip extreme weights to prevent instability
        max_weight = 10.0
        weights = torch.clamp(weights, min=0.1, max=max_weight)
        
        # Normalize weights so they average to 1
        weights = weights / weights.mean()
        
        print(f"\nWeights after normalization (clipped to max {max_weight}):")
        print(f"  Range: [{weights.min():.3f}, {weights.max():.3f}]")
        
        return weights
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_data, attention_mask, batch_labels in pbar:
            batch_data = batch_data.to(self.device)
            attention_mask = attention_mask.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_data, attention_mask)
            loss = self.criterion(outputs, batch_labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Run validation.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_data, attention_mask, batch_labels in self.val_loader:
            batch_data = batch_data.to(self.device)
            attention_mask = attention_mask.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            outputs = self.model(batch_data, attention_mask)
            loss = self.criterion(outputs, batch_labels)
            
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, list]:
        """
        Full training loop.
        
        Returns:
            Dict with training history
        """
        print(f"Training on {self.device}")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Log to TensorBoard
            self.writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            self.writer.add_scalars('Accuracy', {
                'train': train_acc,
                'val': val_acc
            }, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pth')
                print(f"  New best model saved!")
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epochs")
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        self.writer.close()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': self.current_epoch + 1
        }
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'model_config': self.model.get_config(),
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

