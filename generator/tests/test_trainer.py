"""
Tests for the VAE Trainer.
"""

import pytest
import torch
from pathlib import Path
import shutil
from uuid import uuid4
from torch.utils.data import DataLoader, TensorDataset

from src.vae import ConditionalVAE
from src.vae_trainer import VAETrainer
from src.dataset import create_data_loaders

TMP_ROOT = Path(".tmp_pytest_sandbox")
TMP_ROOT.mkdir(exist_ok=True)


def _create_tiny_loaders(batch_size: int = 4, num_samples: int = 8):
    """Create deterministic in-memory loaders for trainer unit tests."""
    grids = torch.rand(num_samples, 3, 18, 11)
    grades = torch.randint(low=0, high=17, size=(num_samples,))
    dataset = TensorDataset(grids, grades)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, loader


class TestVAETrainer:
    """Test suite for VAETrainer."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for checkpoints."""
        temp_path = TMP_ROOT / f"tmp_{uuid4().hex}"
        temp_path.mkdir(parents=True, exist_ok=True)
        yield str(temp_path)
        # Cleanup
        shutil.rmtree(temp_path, ignore_errors=True)
    
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
        trainer.writer.close()
    
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
        trainer.writer.close()
    
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
        trainer.writer.close()

    def test_train_epoch_updates_model_parameters(self, small_dataset_config):
        """Training should update model parameters."""
        model = ConditionalVAE(latent_dim=32, num_grades=17, grade_embedding_dim=16)
        train_loader, val_loader = _create_tiny_loaders()

        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_dataset_config,
            device='cpu'
        )

        params_before = [param.detach().clone() for param in trainer.model.parameters()]
        trainer.train_epoch(0)
        params_after = list(trainer.model.parameters())

        assert any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )
        trainer.writer.close()

    def test_validate_does_not_update_model_parameters(self, small_dataset_config):
        """Validation should not update model parameters."""
        model = ConditionalVAE(latent_dim=32, num_grades=17, grade_embedding_dim=16)
        train_loader, val_loader = _create_tiny_loaders()

        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_dataset_config,
            device='cpu'
        )

        params_before = [param.detach().clone() for param in trainer.model.parameters()]
        trainer.validate(0)
        params_after = list(trainer.model.parameters())

        assert all(
            torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )
        trainer.writer.close()

    def test_train_epoch_routes_through_compute_losses(
        self, small_dataset_config, monkeypatch
    ):
        """train_epoch should route through _compute_losses with training mode."""
        model = ConditionalVAE(latent_dim=32, num_grades=17, grade_embedding_dim=16)
        train_loader, val_loader = _create_tiny_loaders()

        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_dataset_config,
            device='cpu'
        )

        captured = {}

        def fake_compute_losses(data_loader, kl_weight, training=False, epoch=None, log_progress=False):
            captured['data_loader'] = data_loader
            captured['kl_weight'] = kl_weight
            captured['training'] = training
            captured['epoch'] = epoch
            captured['log_progress'] = log_progress
            return 1.0, 0.5, 0.25

        monkeypatch.setattr(trainer, '_compute_losses', fake_compute_losses)

        result = trainer.train_epoch(0)

        assert result == (1.0, 0.5, 0.25)
        assert captured['data_loader'] is train_loader
        assert captured['training'] is True
        assert captured['epoch'] == 0
        assert captured['log_progress'] is True
        assert trainer.model.training is True
        trainer.writer.close()

    def test_validate_routes_through_compute_losses(
        self, small_dataset_config, monkeypatch
    ):
        """validate should route through _compute_losses in eval mode."""
        model = ConditionalVAE(latent_dim=32, num_grades=17, grade_embedding_dim=16)
        train_loader, val_loader = _create_tiny_loaders()

        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_dataset_config,
            device='cpu'
        )

        captured = {}

        def fake_compute_losses(data_loader, kl_weight, training=False, epoch=None, log_progress=False):
            captured['data_loader'] = data_loader
            captured['kl_weight'] = kl_weight
            captured['training'] = training
            captured['epoch'] = epoch
            captured['log_progress'] = log_progress
            return 1.1, 0.6, 0.3

        monkeypatch.setattr(trainer, '_compute_losses', fake_compute_losses)

        result = trainer.validate(0)

        assert result == (1.1, 0.6, 0.3)
        assert captured['data_loader'] is val_loader
        assert captured['training'] is False
        assert captured['epoch'] == 0
        assert captured['log_progress'] is False
        assert trainer.model.training is False
        trainer.writer.close()
    
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
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert checkpoint['label_space_mode'] == 'global_legacy'
        
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
        trainer.writer.close()
        new_trainer.writer.close()

    def test_load_checkpoint_legacy_encoder_shape_mismatch_has_clear_error(
        self, model_and_loaders, small_dataset_config
    ):
        """Legacy unconditioned-encoder checkpoints should fail with guidance."""
        model, train_loader, val_loader = model_and_loaders

        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_dataset_config,
            device='cpu'
        )

        checkpoint_path = Path(small_dataset_config['checkpoint_dir']) / 'legacy_checkpoint.pth'
        trainer.save_checkpoint(epoch=1, filename='legacy_checkpoint.pth')

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        legacy_state = checkpoint['model_state_dict']
        legacy_in_features = model.encoder_output_size
        legacy_state['fc_mu.weight'] = legacy_state['fc_mu.weight'][:, :legacy_in_features].clone()
        legacy_state['fc_logvar.weight'] = legacy_state['fc_logvar.weight'][:, :legacy_in_features].clone()
        checkpoint['model_state_dict'] = legacy_state
        torch.save(checkpoint, checkpoint_path)

        new_model = ConditionalVAE(latent_dim=32, num_grades=17, grade_embedding_dim=16)
        new_trainer = VAETrainer(
            model=new_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_dataset_config,
            device='cpu'
        )

        with pytest.raises(
            RuntimeError, match='legacy CVAE architecture without encoder grade conditioning'
        ):
            new_trainer.load_checkpoint(str(checkpoint_path))

        trainer.writer.close()
        new_trainer.writer.close()

    def test_load_checkpoint_legacy_decoder_shape_mismatch_has_clear_error(
        self, model_and_loaders, small_dataset_config
    ):
        """Legacy interpolating-decoder checkpoints should fail with guidance."""
        model, train_loader, val_loader = model_and_loaders

        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_dataset_config,
            device='cpu'
        )

        checkpoint_path = Path(small_dataset_config['checkpoint_dir']) / 'legacy_decoder_checkpoint.pth'
        trainer.save_checkpoint(epoch=1, filename='legacy_decoder_checkpoint.pth')

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        legacy_state = checkpoint['model_state_dict']
        legacy_state['decoder.0.weight'] = legacy_state['decoder.0.weight'][:, :, :3, :3].clone()
        legacy_state['output_adjust.weight'] = torch.randn(3, 3, 1, 1)
        legacy_state['output_adjust.bias'] = torch.randn(3)
        checkpoint['model_state_dict'] = legacy_state
        torch.save(checkpoint, checkpoint_path)

        new_model = ConditionalVAE(latent_dim=32, num_grades=17, grade_embedding_dim=16)
        new_trainer = VAETrainer(
            model=new_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_dataset_config,
            device='cpu'
        )

        with pytest.raises(
            RuntimeError, match='legacy CVAE decoder architecture'
        ):
            new_trainer.load_checkpoint(str(checkpoint_path))

        trainer.writer.close()
        new_trainer.writer.close()
    
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
        trainer.writer.close()


class TestTrainingIntegration:
    """Integration tests for the full training pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for checkpoints."""
        temp_path = TMP_ROOT / f"tmp_{uuid4().hex}"
        temp_path.mkdir(parents=True, exist_ok=True)
        yield str(temp_path)
        # Cleanup
        shutil.rmtree(temp_path, ignore_errors=True)
    
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

