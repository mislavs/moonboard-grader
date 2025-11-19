"""
Tests for the VAE Trainer.
"""

import pytest
import torch
from pathlib import Path
import tempfile
import shutil

from src.vae import ConditionalVAE
from src.vae_trainer import VAETrainer
from src.dataset import create_data_loaders


class TestVAETrainer:
    """Test suite for VAETrainer."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for checkpoints."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        # Cleanup
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def small_dataset_config(self, temp_dir):
        """Create a small dataset configuration for testing."""
        return {
            'learning_rate': 0.001,
            'num_epochs': 2,
            'kl_weight': 1.0,
            'kl_annealing': False,
            'checkpoint_dir': temp_dir,
            'log_dir': str(Path(temp_dir) / 'logs')
        }
    
    @pytest.fixture
    def model_and_loaders(self):
        """Create a small model and data loaders for testing."""
        # Create small model
        model = ConditionalVAE(latent_dim=32, num_grades=17, grade_embedding_dim=16)
        
        # Create data loaders (use small batch size and subset of data)
        train_loader, val_loader, dataset = create_data_loaders(
            data_path='../data/problems.json',
            batch_size=16,
            train_split=0.8,
            num_workers=0
        )
        
        return model, train_loader, val_loader
    
    def test_trainer_initialization(self, model_and_loaders, small_dataset_config):
        """Test that trainer initializes correctly."""
        model, train_loader, val_loader = model_and_loaders
        
        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_dataset_config,
            device='cpu'
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.current_epoch == 0
        assert trainer.best_val_loss == float('inf')
    
    def test_train_epoch(self, model_and_loaders, small_dataset_config):
        """Test training for one epoch."""
        model, train_loader, val_loader = model_and_loaders
        
        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_dataset_config,
            device='cpu'
        )
        
        # Train for one epoch
        avg_loss, avg_recon, avg_kl = trainer.train_epoch(0)
        
        # Check that losses are valid numbers
        assert isinstance(avg_loss, float)
        assert isinstance(avg_recon, float)
        assert isinstance(avg_kl, float)
        assert avg_loss >= 0
        assert avg_recon >= 0
    
    def test_validate(self, model_and_loaders, small_dataset_config):
        """Test validation."""
        model, train_loader, val_loader = model_and_loaders
        
        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_dataset_config,
            device='cpu'
        )
        
        # Validate
        avg_loss, avg_recon, avg_kl = trainer.validate(0)
        
        # Check that losses are valid numbers
        assert isinstance(avg_loss, float)
        assert isinstance(avg_recon, float)
        assert isinstance(avg_kl, float)
        assert avg_loss >= 0
        assert avg_recon >= 0
    
    def test_save_and_load_checkpoint(self, model_and_loaders, small_dataset_config):
        """Test saving and loading checkpoints."""
        model, train_loader, val_loader = model_and_loaders
        
        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_dataset_config,
            device='cpu'
        )
        
        # Save checkpoint
        checkpoint_path = Path(small_dataset_config['checkpoint_dir']) / 'test_checkpoint.pth'
        trainer.save_checkpoint(epoch=5, filename='test_checkpoint.pth')
        
        assert checkpoint_path.exists()
        
        # Create new trainer and load checkpoint
        new_model = ConditionalVAE(latent_dim=32, num_grades=17, grade_embedding_dim=16)
        new_trainer = VAETrainer(
            model=new_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_dataset_config,
            device='cpu'
        )
        
        new_trainer.load_checkpoint(str(checkpoint_path))
        
        # Check that state was loaded
        assert new_trainer.current_epoch == 5
        
        # Check that model weights are the same
        for p1, p2 in zip(trainer.model.parameters(), new_trainer.model.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_best_checkpoint_saved(self, model_and_loaders, small_dataset_config):
        """Test that best checkpoint is saved correctly."""
        model, train_loader, val_loader = model_and_loaders
        
        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_dataset_config,
            device='cpu'
        )
        
        # Save with is_best=True
        trainer.save_checkpoint(epoch=3, filename='checkpoint.pth', is_best=True)
        
        best_path = Path(small_dataset_config['checkpoint_dir']) / 'best_vae.pth'
        assert best_path.exists()


class TestTrainingIntegration:
    """Integration tests for the full training pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for checkpoints."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        # Cleanup
        shutil.rmtree(temp_path)
    
    def test_short_training_run(self, temp_dir):
        """Test a short training run to ensure everything works together."""
        # Create small model
        model = ConditionalVAE(latent_dim=32, num_grades=17, grade_embedding_dim=16)
        
        # Create data loaders with small subset
        train_loader, val_loader, _ = create_data_loaders(
            data_path='../data/problems.json',
            batch_size=32,
            train_split=0.8,
            num_workers=0
        )
        
        # Create trainer config
        config = {
            'learning_rate': 0.001,
            'num_epochs': 2,  # Just 2 epochs for testing
            'kl_weight': 0.5,
            'kl_annealing': False,
            'checkpoint_dir': temp_dir,
            'log_dir': str(Path(temp_dir) / 'logs')
        }
        
        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device='cpu'
        )
        
        # Train for 2 epochs
        trainer.train(start_epoch=0)
        
        # Check that training completed
        assert trainer.current_epoch == 1  # 0-indexed
        assert len(trainer.train_losses) == 2
        assert len(trainer.val_losses) == 2
        
        # Check that losses are valid
        for loss in trainer.train_losses + trainer.val_losses:
            assert isinstance(loss, float)
            assert loss >= 0

