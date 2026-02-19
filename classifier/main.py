#!/usr/bin/env python3
"""
Main CLI script for Moonboard Grade Prediction

Usage:
    python main.py train --config config.yaml
    python main.py evaluate --checkpoint models/best_model.pth --data data/problems.json
    python main.py predict --checkpoint models/best_model.pth --input problem.json
"""

import argparse
import sys

from src.cli.commands import setup_parsers
from src.cli.train import train_command
from src.cli.evaluate import evaluate_command
from src.cli.predict import predict_command


def main():
    """Main entry point."""
    # Ensure safe output on non-UTF-8 consoles (e.g. Windows cp1252)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')

    parser = argparse.ArgumentParser(
        description="Moonboard Grade Prediction Neural Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train --config config.yaml
  
  # Evaluate a trained model
  python main.py evaluate --checkpoint models/best_model.pth --data data/test.json
  
  # Make predictions
  python main.py predict --checkpoint models/best_model.pth --input problem.json
        """
    )
    
    # Setup all command parsers
    setup_parsers(parser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'evaluate':
            evaluate_command(args)
        elif args.command == 'predict':
            predict_command(args)
    except KeyboardInterrupt:
        print("\n\n[WARN] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
