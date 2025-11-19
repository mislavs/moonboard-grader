"""
Training loop for the Conditional VAE.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from .vae import ConditionalVAE, vae_loss

logger = logging.getLogger(__name__)


class VAETrainer:
    """
    Trainer for the Conditional VAE.
    
    Handles training loop, validation, checkpointing, and logging.
    
    Args:
        model: ConditionalVAE model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Dictionary with training configuration
        device: Device to train on ('cpu' or 'cuda')
    """
    
    def __init__(
        self, 
        model: ConditionalVAE, 
        train_loader, 
        val_loader, 
        config: Dict, 
        device: str = 'cpu',
        grade_offset: int = 0,
        min_grade_index: Optional[int] = None,
        max_grade_index: Optional[int] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Grade filtering metadata (for checkpoint saving)
        self.grade_offset = grade_offset
        self.min_grade_index = min_grade_index
        self.max_grade_index = max_grade_index
        
        # Training hyperparameters
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.num_epochs = config.get('num_epochs', 50)
        self.kl_weight = config.get('kl_weight', 1.0)
        self.kl_annealing = config.get('kl_annealing', False)
        self.kl_annealing_epochs = config.get('kl_annealing_epochs', 10)
        
        # Logging configuration
        self.log_interval = config.get('log_interval', 100)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'models'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # TensorBoard logging
        self.log_dir = Path(config.get('log_dir', 'runs'))
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def get_kl_weight(self, epoch: int) -> float:
        """
        Get KL divergence weight for current epoch (with optional annealing).
        
        Args:
            epoch: Current epoch number
            
        Returns:
            kl_weight: Weight for KL divergence loss
        """
        if not self.kl_annealing:
            return self.kl_weight
        
        # Start at 10% of target, anneal to 100% over kl_annealing_epochs
        # This prevents posterior collapse by ensuring KL is never completely ignored
        min_weight = 0.1 * self.kl_weight
        if epoch < self.kl_annealing_epochs:
            progress = epoch / self.kl_annealing_epochs
            return min_weight + (self.kl_weight - min_weight) * progress
        return self.kl_weight
    
    def _compute_losses(self, data_loader, kl_weight: float, log_progress: bool = False) -> Tuple[float, float, float]:
        """
        Compute losses for a data loader (shared between train/val).
        
        Args:
            data_loader: DataLoader to iterate over
            kl_weight: Weight for KL divergence
            log_progress: Whether to log batch progress
            
        Returns:
            avg_loss: Average total loss
            avg_recon_loss: Average reconstruction loss
            avg_kl_loss: Average KL divergence loss
        """
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        for batch_idx, (grids, grades) in enumerate(data_loader):
            grids = grids.to(self.device)
            grades = grades.to(self.device)
            
            # Forward pass
            x_recon, mu, logvar = self.model(grids, grades)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(x_recon, grids, mu, logvar, kl_weight)
            
            # Normalize by batch size
            batch_size = grids.size(0)
            loss = loss / batch_size
            recon_loss = recon_loss / batch_size
            kl_loss = kl_loss / batch_size
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
            
            # Log batch progress (less verbose)
            if log_progress and batch_idx % self.log_interval == 0:
                logger.debug(f'Batch [{batch_idx}/{len(data_loader)}] '
                            f'Loss: {loss.item():.4f} '
                            f'Recon: {recon_loss.item():.4f} '
                            f'KL: {kl_loss.item():.4f}')
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average total loss for the epoch
            avg_recon_loss: Average reconstruction loss
            avg_kl_loss: Average KL divergence loss
        """
        self.model.train()
        kl_weight = self.get_kl_weight(epoch)
        
        # Training with gradient updates
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        for batch_idx, (grids, grades) in enumerate(self.train_loader):
            grids = grids.to(self.device)
            grades = grades.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            x_recon, mu, logvar = self.model(grids, grades)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(x_recon, grids, mu, logvar, kl_weight)
            
            # Normalize by batch size
            batch_size = grids.size(0)
            loss = loss / batch_size
            recon_loss = recon_loss / batch_size
            kl_loss = kl_loss / batch_size
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
            
            # Log batch progress (less verbose)
            if batch_idx % self.log_interval == 0:
                logger.debug(f'Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] '
                            f'Loss: {loss.item():.4f} '
                            f'Recon: {recon_loss.item():.4f} '
                            f'KL: {kl_loss.item():.4f}')
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_recon_loss = total_recon_loss / num_batches
        avg_kl_loss = total_kl_loss / num_batches
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def validate(self, epoch: int) -> Tuple[float, float, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            avg_loss: Average total loss for validation set
            avg_recon_loss: Average reconstruction loss
            avg_kl_loss: Average KL divergence loss
        """
        self.model.eval()
        kl_weight = self.get_kl_weight(epoch)
        
        with torch.no_grad():
            return self._compute_losses(self.val_loader, kl_weight, log_progress=False)
    
    def save_checkpoint(self, epoch: int, filename: str, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            filename: Checkpoint filename
            is_best: If True, also save a copy as 'best_vae.pth'
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'latent_dim': self.model.latent_dim,
                'num_grades': self.model.num_grades,
                'grade_embedding_dim': self.model.grade_embedding_dim,
            },
            'grade_offset': self.grade_offset,
            'min_grade_index': self.min_grade_index,
            'max_grade_index': self.max_grade_index
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as best checkpoint if requested
        if is_best:
            best_path = self.checkpoint_dir / 'best_vae.pth'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
    
    def train(self, start_epoch: int = 0):
        """
        Run the full training loop.
        
        Args:
            start_epoch: Epoch to start training from (for resuming)
        """
        print(f'\n🏋️  Training for {self.num_epochs} epochs...')
        print(f'   Device: {self.device}')
        print(f'   Training samples: {len(self.train_loader.dataset)}')
        print(f'   Validation samples: {len(self.val_loader.dataset)}')
        
        for epoch in range(start_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_recon, train_kl = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_recon, val_kl = self.validate(epoch)
            
            # Update learning rate
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Print progress (classifier-style format)
            print(f"Epoch {epoch+1}/{self.num_epochs} - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Recon: {val_recon:.4f} - "
                  f"KL: {val_kl:.4f}")
            
            # Show LR reduction
            if old_lr != new_lr:
                print(f"  → Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Loss/train_recon', train_recon, epoch)
            self.writer.add_scalar('Loss/val_recon', val_recon, epoch)
            self.writer.add_scalar('Loss/train_kl', train_kl, epoch)
            self.writer.add_scalar('Loss/val_kl', val_kl, epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Save best model based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth', is_best=True)
                print(f"  → New best validation loss: {val_loss:.4f} - Saved checkpoint")
        
        # Save final model
        self.save_checkpoint(self.current_epoch, 'final_vae.pth')
        
        print(f'\n✓ Training completed!')
        print(f'  Best validation loss: {self.best_val_loss:.4f}')
        
        # Close tensorboard writer
        self.writer.close()

