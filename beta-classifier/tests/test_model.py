"""Tests for model module."""

import pytest
import torch

from src.model import TransformerSequenceClassifier, PositionalEncoding


class TestPositionalEncoding:
    """Tests for PositionalEncoding class."""
    
    def test_output_shape(self):
        """Output should have same shape as input."""
        pe = PositionalEncoding(d_model=64, max_seq_len=50)
        x = torch.randn(4, 20, 64)  # batch=4, seq=20, dim=64
        
        output = pe(x)
        
        assert output.shape == x.shape
    
    def test_adds_positional_info(self):
        """Output should differ from input (positional encoding added)."""
        pe = PositionalEncoding(d_model=64, max_seq_len=50, dropout=0.0)
        x = torch.zeros(1, 10, 64)
        
        output = pe(x)
        
        # Output should not be all zeros
        assert not torch.all(output == 0)
    
    def test_different_positions_different_encodings(self):
        """Different positions should have different encodings."""
        pe = PositionalEncoding(d_model=64, max_seq_len=50, dropout=0.0)
        x = torch.zeros(1, 10, 64)
        
        output = pe(x)
        
        # Position 0 and position 5 should have different encodings
        assert not torch.allclose(output[0, 0], output[0, 5])


class TestTransformerSequenceClassifier:
    """Tests for TransformerSequenceClassifier class."""
    
    @pytest.fixture
    def model(self):
        """Create a small model for testing."""
        return TransformerSequenceClassifier(
            input_dim=15,
            d_model=32,
            n_heads=2,
            n_layers=1,
            num_classes=19,
            dropout=0.0,
            max_seq_len=20
        )
    
    def test_forward_output_shape(self, model):
        """Forward pass should return correct output shape."""
        batch_size = 4
        seq_len = 10
        
        x = torch.randn(batch_size, seq_len, 15)
        mask = torch.ones(batch_size, seq_len)
        
        output = model(x, mask)
        
        assert output.shape == (batch_size, 19)  # 19 classes
    
    def test_forward_with_variable_lengths(self, model):
        """Model should handle different sequence lengths in batch."""
        batch_size = 3
        max_len = 15
        
        x = torch.randn(batch_size, max_len, 15)
        mask = torch.zeros(batch_size, max_len)
        
        # Different lengths: 5, 10, 15
        mask[0, :5] = 1
        mask[1, :10] = 1
        mask[2, :15] = 1
        
        output = model(x, mask)
        
        assert output.shape == (batch_size, 19)
    
    def test_masking_affects_output(self, model):
        """Different masks should produce different outputs."""
        x = torch.randn(1, 10, 15)
        
        mask_full = torch.ones(1, 10)
        mask_partial = torch.zeros(1, 10)
        mask_partial[0, :5] = 1
        
        output_full = model(x, mask_full)
        output_partial = model(x, mask_partial)
        
        # Outputs should be different
        assert not torch.allclose(output_full, output_partial)
    
    def test_get_config(self, model):
        """get_config should return correct configuration."""
        config = model.get_config()
        
        assert config['input_dim'] == 15
        assert config['d_model'] == 32
        assert config['n_heads'] == 2
        assert config['n_layers'] == 1
        assert config['num_classes'] == 19
        assert config['max_seq_len'] == 20
    
    def test_from_config(self):
        """from_config should create model with same architecture."""
        config = {
            'input_dim': 15,
            'd_model': 64,
            'n_heads': 4,
            'n_layers': 2,
            'num_classes': 19,
            'dropout': 0.1,
            'max_seq_len': 50
        }
        
        model = TransformerSequenceClassifier.from_config(config)
        
        assert model.d_model == 64
        assert model.n_heads == 4
        assert model.n_layers == 2
        assert model.num_classes == 19
    
    def test_gradient_flow(self, model):
        """Gradients should flow through entire model."""
        x = torch.randn(2, 8, 15, requires_grad=True)
        mask = torch.ones(2, 8)
        
        output = model(x, mask)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
    
    def test_batch_independence(self, model):
        """Samples in batch should be processed independently."""
        torch.manual_seed(42)
        model.eval()  # Disable dropout
        
        x1 = torch.randn(1, 5, 15)
        x2 = torch.randn(1, 5, 15)
        x_batch = torch.cat([x1, x2], dim=0)
        
        mask1 = torch.ones(1, 5)
        mask2 = torch.ones(1, 5)
        mask_batch = torch.cat([mask1, mask2], dim=0)
        
        # Process separately
        out1 = model(x1, mask1)
        out2 = model(x2, mask2)
        
        # Process as batch
        out_batch = model(x_batch, mask_batch)
        
        torch.testing.assert_close(out1, out_batch[0:1], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out2, out_batch[1:2], rtol=1e-4, atol=1e-4)
    
    def test_single_move_sequence(self, model):
        """Model should handle sequences with just one move."""
        x = torch.randn(1, 1, 15)
        mask = torch.ones(1, 1)
        
        output = model(x, mask)
        
        assert output.shape == (1, 19)
        assert not torch.isnan(output).any()
    
    def test_long_sequence(self):
        """Model should handle sequences up to max_seq_len."""
        model = TransformerSequenceClassifier(max_seq_len=100)
        
        x = torch.randn(1, 100, 15)
        mask = torch.ones(1, 100)
        
        output = model(x, mask)
        
        assert output.shape == (1, 19)


class TestModelCheckpointing:
    """Tests for model save/load via config."""
    
    def test_checkpoint_roundtrip(self, tmp_path):
        """Model should be restorable from checkpoint."""
        # Create and save
        model1 = TransformerSequenceClassifier(
            d_model=32, n_heads=2, n_layers=1
        )
        
        checkpoint = {
            'model_state_dict': model1.state_dict(),
            'model_config': model1.get_config()
        }
        
        path = tmp_path / "model.pth"
        torch.save(checkpoint, path)
        
        # Load
        loaded = torch.load(path)
        model2 = TransformerSequenceClassifier(**loaded['model_config'])
        model2.load_state_dict(loaded['model_state_dict'])
        
        # Compare outputs
        model1.eval()
        model2.eval()
        
        x = torch.randn(1, 5, 15)
        mask = torch.ones(1, 5)
        
        out1 = model1(x, mask)
        out2 = model2(x, mask)
        
        torch.testing.assert_close(out1, out2)

