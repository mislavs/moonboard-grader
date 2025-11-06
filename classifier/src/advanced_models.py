"""
Advanced Model Architectures for Moonboard Grade Prediction

Implements improved neural network architectures with:
- Residual connections for better gradient flow
- Attention mechanisms to focus on critical holds
- Deeper networks with better regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .grade_encoder import get_num_grades


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections.
    
    Skip connections help gradient flow and allow training deeper networks.
    
    Architecture:
        x -> Conv -> BN -> ReLU -> Conv -> BN -> (+) -> ReLU
        |___________________________________________|
                      (skip connection)
    
    Args:
        channels: Number of input/output channels
        kernel_size: Convolutional kernel size (default: 3)
        padding: Padding (default: 1)
        dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.dropout1 = nn.Dropout2d(dropout)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout2 = nn.Dropout2d(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out = out + residual
        out = F.relu(out)
        out = self.dropout2(out)
        
        return out


class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on important holds.
    
    Learns to weight different spatial positions (holds) by importance.
    Start and end holds typically more important than middle holds.
    
    Args:
        in_channels: Number of input channels
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        # Reduce channels to 1 for spatial attention map
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.
        
        Args:
            x: Input features of shape (batch, channels, height, width)
            
        Returns:
            Attention-weighted features of same shape
        """
        # Generate attention map
        att_weights = self.attention(x)  # (batch, 1, height, width)
        
        # Apply attention weights
        out = x * att_weights
        
        return out


class ChannelAttention(nn.Module):
    """
    Channel attention module (Squeeze-and-Excitation).
    
    Learns to weight different feature channels by importance.
    
    Reference: Hu et al. "Squeeze-and-Excitation Networks" (2018)
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio (default: 16)
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.
        
        Args:
            x: Input features of shape (batch, channels, height, width)
            
        Returns:
            Channel-weighted features of same shape
        """
        batch, channels, _, _ = x.size()
        
        # Squeeze: Global average pooling
        squeeze = self.squeeze(x).view(batch, channels)
        
        # Excitation: Learn channel weights
        weights = self.excitation(squeeze).view(batch, channels, 1, 1)
        
        # Apply channel weights
        out = x * weights
        
        return out


class ResidualCNN(nn.Module):
    """
    Residual CNN with attention for grade classification.
    
    Improvements over basic CNN:
    - Residual blocks for better gradient flow
    - Spatial attention to focus on critical holds
    - Channel attention (Squeeze-and-Excitation)
    - Deeper architecture with better capacity
    - Progressive dropout
    
    Architecture:
        Input (3, 18, 11)
        ├─ Conv Block 1: 3→64, ResBlock, Attention → (64, 9, 5)
        ├─ Conv Block 2: 64→128, ResBlock, Attention → (128, 4, 2)
        ├─ Conv Block 3: 128→256, Attention → (256, 4, 2)
        ├─ Global Average Pooling → (256,)
        ├─ FC1: 256→256, Dropout(0.3)
        ├─ FC2: 256→128, Dropout(0.4)
        └─ FC3: 128→num_classes
    
    Args:
        num_classes: Number of grade classes (default: 19)
        use_attention: Whether to use attention mechanisms (default: True)
        dropout_conv: Dropout in convolutional layers (default: 0.1)
        dropout_fc1: Dropout after first FC layer (default: 0.3)
        dropout_fc2: Dropout after second FC layer (default: 0.4)
    """
    
    def __init__(
        self,
        num_classes: Optional[int] = None,
        use_attention: bool = True,
        dropout_conv: float = 0.1,
        dropout_fc1: float = 0.3,
        dropout_fc2: float = 0.4
    ):
        super().__init__()
        
        if num_classes is None:
            num_classes = get_num_grades()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # First conv block: 3 → 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.res1 = ResidualBlock(64, dropout=dropout_conv)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        if use_attention:
            self.att1_spatial = SpatialAttention(64)
            self.att1_channel = ChannelAttention(64)
        
        # Second conv block: 64 → 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.res2 = ResidualBlock(128, dropout=dropout_conv)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        if use_attention:
            self.att2_spatial = SpatialAttention(128)
            self.att2_channel = ChannelAttention(128)
        
        # Third conv block: 128 → 256
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        if use_attention:
            self.att3_spatial = SpatialAttention(256)
            self.att3_channel = ChannelAttention(256)
        
        # Global pooling
        self.pool_global = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers with progressive dropout
        self.fc1 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(dropout_fc1)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_fc2)
        
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 3, 18, 11)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        if self.use_attention:
            x = self.att1_channel(x)
            x = self.att1_spatial(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res2(x)
        if self.use_attention:
            x = self.att2_channel(x)
            x = self.att2_spatial(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        if self.use_attention:
            x = self.att3_channel(x)
            x = self.att3_spatial(x)
        
        # Global pooling
        x = self.pool_global(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class DeepResidualCNN(nn.Module):
    """
    Deeper Residual CNN with more capacity.
    
    Similar to ResidualCNN but with:
    - 4 convolutional blocks (instead of 3)
    - More residual blocks per stage
    - Higher feature dimensions
    
    Best for large datasets or when more capacity is needed.
    
    Args:
        num_classes: Number of grade classes (default: 19)
        use_attention: Whether to use attention (default: True)
    """
    
    def __init__(
        self,
        num_classes: Optional[int] = None,
        use_attention: bool = True
    ):
        super().__init__()
        
        if num_classes is None:
            num_classes = get_num_grades()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Block 1: 3 → 32
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.res1 = nn.ModuleList([ResidualBlock(32) for _ in range(2)])
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2: 32 → 64
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.res2 = nn.ModuleList([ResidualBlock(64) for _ in range(2)])
        self.pool2 = nn.MaxPool2d(2)
        if use_attention:
            self.att2 = ChannelAttention(64)
        
        # Block 3: 64 → 128
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.res3 = nn.ModuleList([ResidualBlock(128) for _ in range(2)])
        if use_attention:
            self.att3 = ChannelAttention(128)
        
        # Block 4: 128 → 256
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        if use_attention:
            self.att4 = ChannelAttention(256)
        
        # Global pooling
        self.pool_global = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc1 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        for res_block in self.res1:
            x = res_block(x)
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        for res_block in self.res2:
            x = res_block(x)
        if self.use_attention:
            x = self.att2(x)
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        for res_block in self.res3:
            x = res_block(x)
        if self.use_attention:
            x = self.att3(x)
        
        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        if self.use_attention:
            x = self.att4(x)
        
        # Global pooling and classifier
        x = self.pool_global(x).flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


def create_advanced_model(
    model_type: str = 'residual_cnn',
    num_classes: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create advanced model architectures.
    
    Args:
        model_type: Type of model
            - 'residual_cnn': Residual CNN with attention (recommended)
            - 'deep_residual_cnn': Deeper version with more capacity
        num_classes: Number of grade classes (default: 19)
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized PyTorch model
        
    Examples:
        >>> # Basic residual CNN
        >>> model = create_advanced_model('residual_cnn')
        
        >>> # With custom dropout
        >>> model = create_advanced_model(
        ...     'residual_cnn',
        ...     dropout_fc1=0.2,
        ...     dropout_fc2=0.3
        ... )
        
        >>> # Deep version
        >>> model = create_advanced_model('deep_residual_cnn')
    """
    if model_type == 'residual_cnn':
        return ResidualCNN(num_classes=num_classes, **kwargs)
    
    elif model_type == 'deep_residual_cnn':
        return DeepResidualCNN(num_classes=num_classes, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Must be one of: 'residual_cnn', 'deep_residual_cnn'"
        )



