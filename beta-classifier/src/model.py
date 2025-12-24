"""
Model Module

Transformer-based sequence classifier for climbing grade prediction.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    
    Adds position information to input embeddings using sine and cosine
    functions of different frequencies.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but should be saved/loaded)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_seq_len, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerSequenceClassifier(nn.Module):
    """
    Transformer encoder for sequence classification.
    
    Architecture:
        1. Linear projection from input_dim to d_model
        2. Sinusoidal positional encoding
        3. N transformer encoder layers
        4. Masked mean pooling over sequence
        5. Classification head (FC layers)
    """
    
    def __init__(
        self,
        input_dim: int = 15,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        num_classes: int = 19,
        dropout: float = 0.1,
        max_seq_len: int = 50
    ):
        """
        Args:
            input_dim: Dimension of input features (15 for move features)
            d_model: Transformer hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer encoder layers
            num_classes: Number of grade classes to predict
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer normalization before classification
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Attention mask of shape (batch, seq_len)
                  1.0 for real tokens, 0.0 for padding
                  
        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input features to model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask for transformer (True = ignore)
        # PyTorch transformer expects: True for positions to mask out
        attn_mask = (mask == 0)  # (batch, seq_len)
        
        # Apply transformer encoder
        x = self.transformer(x, src_key_padding_mask=attn_mask)  # (batch, seq_len, d_model)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Masked mean pooling
        # Only average over non-padded positions
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        # (batch, d_model)
        
        # Classification
        return self.classifier(x)  # (batch, num_classes)
    
    def get_config(self) -> dict:
        """Return model configuration for checkpoint saving."""
        return {
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout_rate,
            'max_seq_len': self.max_seq_len
        }
    
    @classmethod
    def from_config(cls, config: dict) -> 'TransformerSequenceClassifier':
        """Create model from configuration dict."""
        return cls(**config)

