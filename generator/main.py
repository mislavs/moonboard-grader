"""
Command-line interface for the MoonBoard Generator.

Provides commands for training the VAE and generating new climbing problems.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import yaml
import torch

from src.vae import ConditionalVAE
from src.dataset import create_data_loaders
from src.vae_trainer import VAETrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('generator.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        config: Dictionary with configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file: {e}")


def build_training_config(config: Dict) -> Dict:
    """
    Extract and validate training configuration.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        training_config: Configuration for VAETrainer
    """
    return {
        'learning_rate': config['training']['learning_rate'],
        'num_epochs': config['training']['num_epochs'],
        'kl_weight': config['training']['kl_weight'],
        'kl_annealing': config['training']['kl_annealing'],
        'kl_annealing_epochs': config['training']['kl_annealing_epochs'],
        'checkpoint_dir': config['checkpoint']['checkpoint_dir'],
        'checkpoint_frequency': config['checkpoint'].get('save_every', 5),
        'log_dir': config['logging']['log_dir'],
        'log_interval': config['logging'].get('log_interval', 100),
    }


def train_command(args):
    """
    Train the VAE model.
    
    Args:
        args: Command-line arguments
    """
    try:
        # Load configuration
        config = load_config(args.config)
        print(f"\n📋 Configuration loaded from: {args.config}")
        
        # Set device
        device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        # Create data loaders
        print(f"\n📊 Loading dataset...")
        data_config = config['data']
        train_loader, val_loader, dataset = create_data_loaders(
            data_path=data_config['data_path'],
            batch_size=data_config['batch_size'],
            train_split=data_config['train_split'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 0),
            min_grade_index=data_config.get('min_grade_index', None),
            max_grade_index=data_config.get('max_grade_index', None)
        )
        
        num_grades = dataset.get_num_grades()
        print(f"   Total problems: {len(dataset)}")
        print(f"   Unique grades: {num_grades}")
        print(f"   Grade range: {', '.join(dataset.grade_names)}")
        
        # Create model
        print(f"\n🧠 Creating model...")
        model_config = config['model']
        model = ConditionalVAE(
            latent_dim=model_config['latent_dim'],
            num_grades=num_grades,  # Use actual number from dataset
            grade_embedding_dim=model_config['grade_embedding_dim']
        )
        
        print(f"   Latent dim: {model_config['latent_dim']}")
        print(f"   Grade embedding dim: {model_config['grade_embedding_dim']}")
        
        # Prepare training config
        training_config = build_training_config(config)
        
        # Create trainer
        trainer = VAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=device
        )
        
        # Load checkpoint if resuming
        start_epoch = 0
        if args.resume:
            resume_path = Path(args.resume)
            if resume_path.exists():
                print(f"\n↻  Resuming from checkpoint: {resume_path}")
                trainer.load_checkpoint(str(resume_path))
                start_epoch = trainer.current_epoch + 1
                print(f"   Starting from epoch {start_epoch}")
            else:
                print(f"\n⚠️  Checkpoint {args.resume} not found. Starting from scratch.")
        
        # Train
        trainer.train(start_epoch=start_epoch)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        trainer.save_checkpoint(trainer.current_epoch, 'interrupted_checkpoint.pth')
        print("   Saved interrupted checkpoint")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='MoonBoard Generator - VAE-based climbing problem generation'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the VAE model')
    train_parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    train_parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    train_parser.set_defaults(func=train_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()

