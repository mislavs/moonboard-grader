#!/usr/bin/env python3
"""
Main CLI script for Moonboard Grade Prediction

Usage:
    py main.py create-manifest --config config.yaml --output manifests/benchmark-v1.json
    py main.py cross-validate --config config.yaml --manifest manifests/benchmark-v1.json
    py main.py refit --config config.yaml --manifest manifests/benchmark-v1.json --cv-report report.json
    py main.py evaluate --checkpoint models/refit.pth --manifest manifests/benchmark-v1.json
    py main.py predict --checkpoint models/refit.pth --input problem.json
"""

import os

# CuBLAS reads this before CUDA work begins. PyTorch requires it when strict
# deterministic algorithms are enabled on CUDA 10.2 and newer.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
  # Freeze the cohort before experimentation
  py main.py create-manifest --config config.yaml --output manifests/moonboard-masters-2017-all-v1.json

  # Compare one candidate without touching the locked test set
  py main.py cross-validate --config config.yaml --manifest manifests/moonboard-masters-2017-all-v1.json

  # Make predictions
  py main.py predict --checkpoint models/best_model.pth --input problem.json
        """
    )
    
    # Setup all command parsers
    setup_parsers(parser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the function registered by the selected subcommand.
    try:
        args.func(args)
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
