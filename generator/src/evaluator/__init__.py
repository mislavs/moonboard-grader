"""
Evaluation module for the MoonBoard Generator.

Provides metrics to assess the quality of the trained VAE model.
"""

from .orchestrator import run_evaluation, run_evaluation_iter, get_metrics, order_metrics

__all__ = [
    'run_evaluation',
    'run_evaluation_iter',
    'get_metrics',
    'order_metrics',
]

