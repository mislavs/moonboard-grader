"""
Shared CLI Utilities

Common functions used across CLI commands.
"""

import sys
import torch
import yaml
from pathlib import Path


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_device(device_name, fallback_to_cpu=True):
    """
    Setup and validate device for training/inference.
    
    Args:
        device_name: Requested device name ('cuda' or 'cpu')
        fallback_to_cpu: Whether to fallback to CPU if CUDA unavailable
        
    Returns:
        Tuple of (device, actual_device_name)
    """
    if device_name == 'cuda' and not torch.cuda.is_available():
        if fallback_to_cpu:
            print(f"⚠ CUDA requested but not available, falling back to CPU")
            device_name = 'cpu'
        else:
            print(f"❌ Error: CUDA requested but not available")
            sys.exit(1)
    
    device = torch.device(device_name)
    return device, device_name


def print_section_header(title):
    """
    Print a formatted section header.
    
    Args:
        title: Title text to display
    """
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_completion_message(message):
    """
    Print a formatted completion message.
    
    Args:
        message: Completion message to display
    """
    print("\n" + "=" * 70)
    print(message)
    print("=" * 70)

