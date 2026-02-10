"""
Tests for the Conditional VAE architecture.
"""

import pytest
import torch

from src.vae import ConditionalVAE, vae_loss


class TestConditionalVAE:
    """Test suite for ConditionalVAE model."""
    
    @pytest.fixture
    def model(self):
        """Create a VAE model for testing."""
        return ConditionalVAE(latent_dim=128, num_grades=17, grade_embedding_dim=32)
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input data."""
        batch_size = 4
        grids = torch.randn(batch_size, 3, 18, 11)
        grades = torch.randint(0, 17, (batch_size,))
        return grids, grades
    
    def test_model_initialization(self, model):
        """Test that the model initializes correctly."""
        assert model.latent_dim == 128
        assert model.num_grades == 17
        assert model.grade_embedding_dim == 32
        assert isinstance(model, torch.nn.Module)
    
    def test_encode_shape(self, model, sample_input):
        """Test that encoding produces correct output shapes."""
        grids, grades = sample_input
        mu, logvar = model.encode(grids, grades)
        
        assert mu.shape == (4, 128)
        assert logvar.shape == (4, 128)

    def test_encode_conditioned_on_grade(self, model):
        """Same grid with different grades should produce different latent params."""
        single_grid = torch.randn(1, 3, 18, 11)
        grids = single_grid.repeat(2, 1, 1, 1)
        grades = torch.tensor([0, 1], dtype=torch.long)

        mu, logvar = model.encode(grids, grades)

        same_mu = torch.allclose(mu[0], mu[1])
        same_logvar = torch.allclose(logvar[0], logvar[1])
        assert not (same_mu and same_logvar)
    
    def test_reparameterize_shape(self, model):
        """Test that reparameterization produces correct output shape."""
        mu = torch.randn(4, 128)
        logvar = torch.randn(4, 128)
        z = model.reparameterize(mu, logvar)
        
        assert z.shape == (4, 128)
    
    def test_decode_shape(self, model, sample_input):
        """Test that decoding produces correct output shape."""
        _, grades = sample_input
        z = torch.randn(4, 128)
        x_recon = model.decode(z, grades)
        
        assert x_recon.shape == (4, 3, 18, 11)

    def test_decode_does_not_call_interpolate(self, model, sample_input, monkeypatch):
        """Decoder should output native 18x11 without interpolation."""
        _, grades = sample_input
        z = torch.randn(4, 128)

        def fail_interpolate(*args, **kwargs):
            raise AssertionError("F.interpolate should not be called in decode()")

        monkeypatch.setattr("src.vae.F.interpolate", fail_interpolate)

        x_recon = model.decode(z, grades)

        assert x_recon.shape == (4, 3, 18, 11)
    
    def test_forward_pass(self, model, sample_input):
        """Test full forward pass through the model."""
        grids, grades = sample_input
        x_recon, mu, logvar = model(grids, grades)
        
        # Check output shapes
        assert x_recon.shape == (4, 3, 18, 11)
        assert mu.shape == (4, 128)
        assert logvar.shape == (4, 128)
    
    def test_sample_generation(self, model):
        """Test generating samples from the model."""
        num_samples = 5
        grades = torch.randint(0, 17, (num_samples,))
        
        samples = model.sample(num_samples, grades, device='cpu')
        
        assert samples.shape == (5, 3, 18, 11)
    
    def test_grade_embedding(self, model):
        """Test that grade embedding works correctly."""
        grades = torch.tensor([0, 5, 10, 16])
        embeddings = model.grade_embedding(grades)
        
        assert embeddings.shape == (4, 32)
        # Check that different grades produce different embeddings
        assert not torch.allclose(embeddings[0], embeddings[1])
    
    def test_model_on_different_batch_sizes(self, model):
        """Test that the model works with different batch sizes."""
        for batch_size in [1, 8, 16]:
            grids = torch.randn(batch_size, 3, 18, 11)
            grades = torch.randint(0, 17, (batch_size,))
            
            x_recon, mu, logvar = model(grids, grades)
            
            assert x_recon.shape == (batch_size, 3, 18, 11)
            assert mu.shape == (batch_size, 128)
            assert logvar.shape == (batch_size, 128)


class TestVAELoss:
    """Test suite for VAE loss function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for loss testing."""
        batch_size = 4
        x = torch.rand(batch_size, 3, 18, 11)
        x_recon = torch.randn(batch_size, 3, 18, 11)
        mu = torch.randn(batch_size, 128)
        logvar = torch.randn(batch_size, 128)
        return x, x_recon, mu, logvar
    
    def test_loss_computation(self, sample_data):
        """Test that loss is computed without errors."""
        x, x_recon, mu, logvar = sample_data
        total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
        
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(recon_loss, torch.Tensor)
        assert isinstance(kl_loss, torch.Tensor)
    
    def test_loss_positive(self, sample_data):
        """Test that all loss components are positive."""
        x, x_recon, mu, logvar = sample_data
        total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)
        
        assert total_loss.item() >= 0
        assert recon_loss.item() >= 0
        # KL loss can be negative for individual samples but should be >= 0 on average
    
    def test_kl_weight(self, sample_data):
        """Test that KL weight affects total loss correctly."""
        x, x_recon, mu, logvar = sample_data
        
        loss1, recon1, kl1 = vae_loss(x_recon, x, mu, logvar, kl_weight=1.0)
        loss2, recon2, kl2 = vae_loss(x_recon, x, mu, logvar, kl_weight=0.5)
        
        # Reconstruction loss should be the same
        assert torch.allclose(recon1, recon2)
        
        # Total loss should be different due to KL weight
        # loss1 should be larger (assuming KL loss is positive)
        if kl1.item() > 0:
            assert loss1.item() > loss2.item()
    
    def test_loss_zero_kl_weight(self, sample_data):
        """Test loss with zero KL weight."""
        x, x_recon, mu, logvar = sample_data
        total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, kl_weight=0.0)
        
        # Total loss should equal reconstruction loss when KL weight is 0
        assert torch.allclose(total_loss, recon_loss)


class TestModelTrainingStep:
    """Test that the model can perform a training step."""
    
    def test_training_step(self):
        """Test a complete training step."""
        # Create model
        model = ConditionalVAE(latent_dim=64, num_grades=10, grade_embedding_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create sample data
        grids = torch.rand(8, 3, 18, 11)
        grades = torch.randint(0, 10, (8,))
        
        # Forward pass
        model.train()
        optimizer.zero_grad()
        x_recon, mu, logvar = model(grids, grades)
        
        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(x_recon, grids, mu, logvar)
        loss = loss / grids.size(0)  # Normalize by batch size
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Check that parameters were updated
        assert loss.item() >= 0
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestModelDifferentConfigurations:
    """Test the model with different configurations."""
    
    @pytest.mark.parametrize("latent_dim", [64, 128, 256])
    def test_different_latent_dims(self, latent_dim):
        """Test model with different latent dimensions."""
        model = ConditionalVAE(latent_dim=latent_dim, num_grades=17, grade_embedding_dim=32)
        grids = torch.randn(4, 3, 18, 11)
        grades = torch.randint(0, 17, (4,))
        
        x_recon, mu, logvar = model(grids, grades)
        
        assert mu.shape == (4, latent_dim)
        assert logvar.shape == (4, latent_dim)
    
    @pytest.mark.parametrize("num_grades", [10, 17, 25])
    def test_different_num_grades(self, num_grades):
        """Test model with different numbers of grades."""
        model = ConditionalVAE(latent_dim=128, num_grades=num_grades, grade_embedding_dim=32)
        grids = torch.randn(4, 3, 18, 11)
        grades = torch.randint(0, num_grades, (4,))
        
        x_recon, mu, logvar = model(grids, grades)
        
        assert x_recon.shape == (4, 3, 18, 11)
    
    @pytest.mark.parametrize("grade_embedding_dim", [16, 32, 64])
    def test_different_grade_embedding_dims(self, grade_embedding_dim):
        """Test model with different grade embedding dimensions."""
        model = ConditionalVAE(latent_dim=128, num_grades=17, grade_embedding_dim=grade_embedding_dim)
        grades = torch.tensor([0, 5, 10, 16])
        
        embeddings = model.grade_embedding(grades)
        
        assert embeddings.shape == (4, grade_embedding_dim)

