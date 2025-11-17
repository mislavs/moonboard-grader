"""
Evaluation module for the MoonBoard Generator.

Provides metrics to assess the quality of the trained VAE model.
"""

from .orchestrator import run_evaluation, get_metrics

__all__ = [
    'run_evaluation',
    'get_metrics',
]

