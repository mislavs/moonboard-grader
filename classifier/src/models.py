"""
Model Architectures Module

Implements PyTorch neural network models for Moonboard grade prediction.
Supports both fully connected and convolutional architectures.
"""

import torch
import torch.nn as nn
from typing import Literal
from .grade_encoder import get_num_grades


class FullyConnectedModel(nn.Module):
    """
    Fully connected neural network for grade classification.
    
    Architecture:
        - Input: (batch, 3, 18, 11) tensor
        - Flatten to (batch, 594)
        - Linear(594 → 256) → ReLU → Dropout(0.3)
        - Linear(256 → 128) → ReLU → Dropout(0.3)
        - Linear(128 → num_classes)
        - Output: (batch, num_classes) logits
    """
    
    def __init__(self, num_classes: int = None):
        """
        Initialize the fully connected model.
        
        Args:
            num_classes: Number of grade classes to predict. 
                        If None, uses get_num_grades() (19 classes).
        """
        super().__init__()
        
        if num_classes is None:
            num_classes = get_num_grades()
        
        self.num_classes = num_classes
        self.input_size = 3 * 18 * 11  # 594
        
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 3, 18, 11)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        return self.network(x)


class ConvolutionalModel(nn.Module):
    """
    Convolutional neural network for grade classification.
    
    Architecture:
        - Input: (batch, 3, 18, 11) tensor
        - Conv2d(3 → 32, kernel=3, padding=1) → BatchNorm → ReLU → MaxPool2d(2)
        - Conv2d(32 → 64, kernel=3, padding=1) → BatchNorm → ReLU → MaxPool2d(2)
        - Conv2d(64 → 128, kernel=3, padding=1) → BatchNorm → ReLU
        - Flatten to (batch, 512)
        - Linear(512 → 256) → ReLU → Dropout(0.4)
        - Linear(256 → 128) → ReLU → Dropout(0.3)
        - Linear(128 → num_classes)
        - Output: (batch, num_classes) logits
        
    The spatial dimensions change as:
        - After Conv1: (batch, 32, 18, 11)
        - After Pool1: (batch, 32, 9, 5)
        - After Conv2: (batch, 64, 9, 5)
        - After Pool2: (batch, 64, 4, 2)
        - After Conv3: (batch, 128, 4, 2)
        - After Flatten: (batch, 1024)
    """
    
    def __init__(self, num_classes: int = None):
        """
        Initialize the convolutional model.
        
        Args:
            num_classes: Number of grade classes to predict.
                        If None, uses get_num_grades() (19 classes).
        """
        super().__init__()
        
        if num_classes is None:
            num_classes = get_num_grades()
        
        self.num_classes = num_classes
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        
        # Calculate flattened size: 128 channels * 4 height * 2 width = 1024
        self.flattened_size = 128 * 4 * 2
        
        # Fully connected layers with dropout for regularization
        self.fc1 = nn.Linear(self.flattened_size, 256)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)  # Increased from 0.4 to combat overfitting
        
        self.fc2 = nn.Linear(256, 128)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)  # Increased from 0.3 to combat overfitting
        
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 3, 18, 11)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


def create_model(
    model_type: Literal["fc", "cnn"] = "fc",
    num_classes: int = None,
    **kwargs
) -> nn.Module:
    """
    Unified factory function to create any model instance.
    
    This function creates both basic models (fc, cnn) and advanced models
    (residual_cnn, deep_residual_cnn) based on the model_type parameter.
    
    Args:
        model_type: Type of model to create. Options:
                   - "fc": Fully connected network
                   - "cnn": Convolutional network
                   - "residual_cnn": Residual CNN with attention
                   - "deep_residual_cnn": Deeper residual CNN
        num_classes: Number of grade classes to predict.
                    If None, uses get_num_grades() (19 classes).
        **kwargs: Additional model-specific parameters:
                 - use_attention (bool): Enable attention mechanisms (for residual models)
                 - dropout_conv (float): Dropout in conv layers (for residual models)
                 - dropout_fc1 (float): Dropout after first FC layer (for residual models)
                 - dropout_fc2 (float): Dropout after second FC layer (for residual models)
    
    Returns:
        Initialized PyTorch model.
        
    Raises:
        ValueError: If model_type is not recognized.
        
    Examples:
        >>> # Basic models
        >>> fc_model = create_model("fc")
        >>> cnn_model = create_model("cnn", num_classes=19)
        
        >>> # Advanced models
        >>> res_model = create_model("residual_cnn", use_attention=True)
        >>> deep_model = create_model("deep_residual_cnn", dropout_fc1=0.3)
    """
    # Basic models
    if model_type == "fc":
        return FullyConnectedModel(num_classes)
    elif model_type == "cnn":
        return ConvolutionalModel(num_classes)
    
    # Advanced models - import lazily to avoid circular dependencies
    elif model_type in ["residual_cnn", "deep_residual_cnn"]:
        from .advanced_models import create_advanced_model
        return create_advanced_model(model_type=model_type, num_classes=num_classes, **kwargs)
    
    else:
        raise ValueError(
            f"Invalid model_type '{model_type}'. "
            f"Must be one of: 'fc', 'cnn', 'residual_cnn', 'deep_residual_cnn'."
        )


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model to count parameters for.
        
    Returns:
        Total number of trainable parameters.
        
    Examples:
        >>> model = create_model("fc")
        >>> num_params = count_parameters(model)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

