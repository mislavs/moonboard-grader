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
        - Conv2d(3 → 16, kernel=3, padding=1) → ReLU → MaxPool2d(2)
        - Conv2d(16 → 32, kernel=3, padding=1) → ReLU → MaxPool2d(2)
        - Flatten to (batch, 256)
        - Linear(256 → 128) → ReLU → Dropout(0.5)
        - Linear(128 → num_classes)
        - Output: (batch, num_classes) logits
        
    The spatial dimensions change as:
        - After Conv1: (batch, 16, 18, 11)
        - After Pool1: (batch, 16, 9, 5)
        - After Conv2: (batch, 32, 9, 5)
        - After Pool2: (batch, 32, 4, 2)
        - After Flatten: (batch, 256)
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
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Calculate flattened size: 32 channels * 4 height * 2 width = 256
        self.flattened_size = 32 * 4 * 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
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
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_model(
    model_type: Literal["fc", "cnn"] = "fc",
    num_classes: int = None
) -> nn.Module:
    """
    Factory function to create a model instance.
    
    Args:
        model_type: Type of model to create. Either "fc" for fully connected
                   or "cnn" for convolutional.
        num_classes: Number of grade classes to predict.
                    If None, uses get_num_grades() (19 classes).
    
    Returns:
        Initialized PyTorch model.
        
    Raises:
        ValueError: If model_type is not "fc" or "cnn".
        
    Examples:
        >>> fc_model = create_model("fc")
        >>> cnn_model = create_model("cnn", num_classes=19)
    """
    if model_type == "fc":
        return FullyConnectedModel(num_classes)
    elif model_type == "cnn":
        return ConvolutionalModel(num_classes)
    else:
        raise ValueError(f"Invalid model_type '{model_type}'. Must be 'fc' or 'cnn'.")


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

