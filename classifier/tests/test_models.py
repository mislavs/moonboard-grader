"""
Unit Tests for Model Architectures

Tests for FullyConnectedModel, ConvolutionalModel, and helper functions.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path

from src.models import (
    FullyConnectedModel,
    ConvolutionalModel,
    create_model,
    count_parameters
)
from moonboard_core.grade_encoder import get_num_grades


# Test Model Initialization

def test_fc_model_initialization_default():
    """Test FC model initializes with default num_classes."""
    model = FullyConnectedModel()
    assert model.num_classes == get_num_grades()
    assert model.input_size == 594


def test_fc_model_initialization_custom_classes():
    """Test FC model initializes with custom num_classes."""
    model = FullyConnectedModel(num_classes=10)
    assert model.num_classes == 10


def test_cnn_model_initialization_default():
    """Test CNN model initializes with default num_classes."""
    model = ConvolutionalModel()
    assert model.num_classes == get_num_grades()
    assert model.flattened_size == 1024  # 128 channels * 4 height * 2 width


def test_cnn_model_initialization_custom_classes():
    """Test CNN model initializes with custom num_classes."""
    model = ConvolutionalModel(num_classes=10)
    assert model.num_classes == 10


# Test Forward Pass

def test_fc_forward_pass_shape():
    """Test FC model forward pass produces correct output shape."""
    model = FullyConnectedModel(num_classes=19)
    batch_size = 4
    x = torch.randn(batch_size, 3, 18, 11)
    output = model(x)
    
    assert output.shape == (batch_size, 19)


def test_fc_forward_pass_single_sample():
    """Test FC model forward pass with single sample."""
    model = FullyConnectedModel(num_classes=19)
    x = torch.randn(1, 3, 18, 11)
    output = model(x)
    
    assert output.shape == (1, 19)


def test_fc_forward_pass_large_batch():
    """Test FC model forward pass with larger batch."""
    model = FullyConnectedModel(num_classes=19)
    batch_size = 64
    x = torch.randn(batch_size, 3, 18, 11)
    output = model(x)
    
    assert output.shape == (batch_size, 19)


def test_cnn_forward_pass_shape():
    """Test CNN model forward pass produces correct output shape."""
    model = ConvolutionalModel(num_classes=19)
    batch_size = 4
    x = torch.randn(batch_size, 3, 18, 11)
    output = model(x)
    
    assert output.shape == (batch_size, 19)


def test_cnn_forward_pass_single_sample():
    """Test CNN model forward pass with single sample."""
    model = ConvolutionalModel(num_classes=19)
    x = torch.randn(1, 3, 18, 11)
    output = model(x)
    
    assert output.shape == (1, 19)


def test_cnn_forward_pass_large_batch():
    """Test CNN model forward pass with larger batch."""
    model = ConvolutionalModel(num_classes=19)
    batch_size = 64
    x = torch.randn(batch_size, 3, 18, 11)
    output = model(x)
    
    assert output.shape == (batch_size, 19)


# Test Output Properties

def test_fc_output_is_logits():
    """Test FC model output is not bounded (logits, not probabilities)."""
    model = FullyConnectedModel(num_classes=19)
    x = torch.randn(2, 3, 18, 11)
    output = model(x)
    
    # Logits can be any value (not bounded to [0,1])
    assert not torch.all((output >= 0) & (output <= 1))


def test_cnn_output_is_logits():
    """Test CNN model output is not bounded (logits, not probabilities)."""
    model = ConvolutionalModel(num_classes=19)
    x = torch.randn(2, 3, 18, 11)
    output = model(x)
    
    # Logits can be any value (not bounded to [0,1])
    assert not torch.all((output >= 0) & (output <= 1))


# Test Backward Pass

def test_fc_backward_pass():
    """Test FC model backward pass works correctly."""
    model = FullyConnectedModel(num_classes=19)
    x = torch.randn(4, 3, 18, 11)
    output = model(x)
    
    # Create dummy loss and backpropagate
    loss = output.sum()
    loss.backward()
    
    # Check gradients are computed
    for param in model.parameters():
        assert param.grad is not None


def test_cnn_backward_pass():
    """Test CNN model backward pass works correctly."""
    model = ConvolutionalModel(num_classes=19)
    x = torch.randn(4, 3, 18, 11)
    output = model(x)
    
    # Create dummy loss and backpropagate
    loss = output.sum()
    loss.backward()
    
    # Check gradients are computed
    for param in model.parameters():
        assert param.grad is not None


# Test Gradient Flow

def test_fc_gradient_flow():
    """Test gradients flow through all layers of FC model."""
    model = FullyConnectedModel(num_classes=19)
    x = torch.randn(2, 3, 18, 11)
    target = torch.tensor([0, 1])
    
    criterion = nn.CrossEntropyLoss()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    
    # All parameters should have gradients
    has_grad = [param.grad is not None for param in model.parameters()]
    assert all(has_grad)


def test_cnn_gradient_flow():
    """Test gradients flow through all layers of CNN model."""
    model = ConvolutionalModel(num_classes=19)
    x = torch.randn(2, 3, 18, 11)
    target = torch.tensor([0, 1])
    
    criterion = nn.CrossEntropyLoss()
    output = model(x)
    loss = criterion(output, target)
    loss.backward()
    
    # All parameters should have gradients
    has_grad = [param.grad is not None for param in model.parameters()]
    assert all(has_grad)


# Test Model Saving and Loading

def test_fc_save_load_state_dict():
    """Test FC model state dict can be saved and loaded."""
    model1 = FullyConnectedModel(num_classes=19)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pth")
        torch.save(model1.state_dict(), path)
        
        model2 = FullyConnectedModel(num_classes=19)
        model2.load_state_dict(torch.load(path))
        
        # Compare parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


def test_cnn_save_load_state_dict():
    """Test CNN model state dict can be saved and loaded."""
    model1 = ConvolutionalModel(num_classes=19)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pth")
        torch.save(model1.state_dict(), path)
        
        model2 = ConvolutionalModel(num_classes=19)
        model2.load_state_dict(torch.load(path))
        
        # Compare parameters
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


def test_fc_save_load_entire_model():
    """Test FC model can be saved and loaded entirely."""
    model1 = FullyConnectedModel(num_classes=19)
    model1.eval()
    x = torch.randn(1, 3, 18, 11)
    
    with torch.no_grad():
        output1 = model1(x)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pth")
        torch.save(model1, path)
        
        model2 = torch.load(path, weights_only=False)
        model2.eval()
        
        with torch.no_grad():
            output2 = model2(x)
        
        assert torch.allclose(output1, output2)


def test_cnn_save_load_entire_model():
    """Test CNN model can be saved and loaded entirely."""
    model1 = ConvolutionalModel(num_classes=19)
    model1.eval()
    x = torch.randn(1, 3, 18, 11)
    
    with torch.no_grad():
        output1 = model1(x)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pth")
        torch.save(model1, path)
        
        model2 = torch.load(path, weights_only=False)
        model2.eval()
        
        with torch.no_grad():
            output2 = model2(x)
        
        assert torch.allclose(output1, output2)


# Test Training Mode vs Eval Mode

def test_fc_train_eval_modes():
    """Test FC model dropout behaves differently in train vs eval mode."""
    model = FullyConnectedModel(num_classes=19)
    x = torch.randn(100, 3, 18, 11)
    
    # Training mode - dropout active
    model.train()
    output_train1 = model(x)
    output_train2 = model(x)
    # Outputs should differ due to dropout randomness
    assert not torch.allclose(output_train1, output_train2)
    
    # Eval mode - dropout inactive
    model.eval()
    with torch.no_grad():
        output_eval1 = model(x)
        output_eval2 = model(x)
    # Outputs should be identical
    assert torch.allclose(output_eval1, output_eval2)


def test_cnn_train_eval_modes():
    """Test CNN model dropout behaves differently in train vs eval mode."""
    model = ConvolutionalModel(num_classes=19)
    x = torch.randn(100, 3, 18, 11)
    
    # Training mode - dropout active
    model.train()
    output_train1 = model(x)
    output_train2 = model(x)
    # Outputs should differ due to dropout randomness
    assert not torch.allclose(output_train1, output_train2)
    
    # Eval mode - dropout inactive
    model.eval()
    with torch.no_grad():
        output_eval1 = model(x)
        output_eval2 = model(x)
    # Outputs should be identical
    assert torch.allclose(output_eval1, output_eval2)


# Test Factory Function

def test_create_model_fc():
    """Test create_model factory function creates FC model."""
    model = create_model("fc", num_classes=19)
    assert isinstance(model, FullyConnectedModel)
    assert model.num_classes == 19


def test_create_model_cnn():
    """Test create_model factory function creates CNN model."""
    model = create_model("cnn", num_classes=19)
    assert isinstance(model, ConvolutionalModel)
    assert model.num_classes == 19


def test_create_model_default_classes():
    """Test create_model uses default num_classes if not specified."""
    model = create_model("fc")
    assert model.num_classes == get_num_grades()


def test_create_model_invalid_type():
    """Test create_model raises error for invalid model type."""
    with pytest.raises(ValueError, match="Invalid model_type"):
        create_model("invalid_type")


# Test Model Device Movement

def test_fc_model_cpu():
    """Test FC model works on CPU."""
    model = FullyConnectedModel(num_classes=19).to('cpu')
    x = torch.randn(2, 3, 18, 11).to('cpu')
    output = model(x)
    
    assert output.device.type == 'cpu'


def test_cnn_model_cpu():
    """Test CNN model works on CPU."""
    model = ConvolutionalModel(num_classes=19).to('cpu')
    x = torch.randn(2, 3, 18, 11).to('cpu')
    output = model(x)
    
    assert output.device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fc_model_cuda():
    """Test FC model works on CUDA."""
    model = FullyConnectedModel(num_classes=19).to('cuda')
    x = torch.randn(2, 3, 18, 11).to('cuda')
    output = model(x)
    
    assert output.device.type == 'cuda'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cnn_model_cuda():
    """Test CNN model works on CUDA."""
    model = ConvolutionalModel(num_classes=19).to('cuda')
    x = torch.randn(2, 3, 18, 11).to('cuda')
    output = model(x)
    
    assert output.device.type == 'cuda'


# Test Model with Real-World Data Format

def test_fc_with_binary_input():
    """Test FC model with binary input (like actual hold positions)."""
    model = FullyConnectedModel(num_classes=19)
    # Create binary tensor simulating actual holds
    x = torch.zeros(4, 3, 18, 11)
    x[0, 0, 0, 0] = 1  # Start hold at A1
    x[0, 2, 17, 10] = 1  # End hold at K18
    
    output = model(x)
    assert output.shape == (4, 19)


def test_cnn_with_binary_input():
    """Test CNN model with binary input (like actual hold positions)."""
    model = ConvolutionalModel(num_classes=19)
    # Create binary tensor simulating actual holds
    x = torch.zeros(4, 3, 18, 11)
    x[0, 0, 0, 0] = 1  # Start hold at A1
    x[0, 2, 17, 10] = 1  # End hold at K18
    
    output = model(x)
    assert output.shape == (4, 19)


# Test Deterministic Behavior

def test_fc_deterministic_eval():
    """Test FC model produces deterministic output in eval mode."""
    torch.manual_seed(42)
    model = FullyConnectedModel(num_classes=19)
    model.eval()
    
    x = torch.randn(2, 3, 18, 11)
    
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)
    
    assert torch.equal(output1, output2)


def test_cnn_deterministic_eval():
    """Test CNN model produces deterministic output in eval mode."""
    torch.manual_seed(42)
    model = ConvolutionalModel(num_classes=19)
    model.eval()
    
    x = torch.randn(2, 3, 18, 11)
    
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)
    
    assert torch.equal(output1, output2)


# Edge Cases

def test_fc_with_zeros():
    """Test FC model handles all-zero input."""
    model = FullyConnectedModel(num_classes=19)
    x = torch.zeros(1, 3, 18, 11)
    output = model(x)
    
    assert output.shape == (1, 19)
    assert torch.isfinite(output).all()


def test_cnn_with_zeros():
    """Test CNN model handles all-zero input."""
    model = ConvolutionalModel(num_classes=19)
    x = torch.zeros(1, 3, 18, 11)
    output = model(x)
    
    assert output.shape == (1, 19)
    assert torch.isfinite(output).all()


def test_fc_with_ones():
    """Test FC model handles all-one input."""
    model = FullyConnectedModel(num_classes=19)
    x = torch.ones(1, 3, 18, 11)
    output = model(x)
    
    assert output.shape == (1, 19)
    assert torch.isfinite(output).all()


def test_cnn_with_ones():
    """Test CNN model handles all-one input."""
    model = ConvolutionalModel(num_classes=19)
    x = torch.ones(1, 3, 18, 11)
    output = model(x)
    
    assert output.shape == (1, 19)
    assert torch.isfinite(output).all()

