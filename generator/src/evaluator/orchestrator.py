"""
Orchestrator for running evaluation metrics.
"""

from typing import Dict, List, Optional, Any
from src.label_space import EvaluationLabelContext

from .reconstruction import evaluate_reconstruction
from .diversity import evaluate_diversity
from .statistical import evaluate_statistical_similarity
from .latent_space import evaluate_latent_space
from .latent_visualization import evaluate_latent_visualization
from .classifier_check import evaluate_classifier_check


# Dispatch table mapping metric names to their functions
METRIC_FUNCTIONS = {
    'reconstruction': evaluate_reconstruction,
    'diversity': evaluate_diversity,
    'statistical': evaluate_statistical_similarity,
    'latent_space': evaluate_latent_space,
    'latent_visualization': evaluate_latent_visualization,
    'classifier_check': evaluate_classifier_check
}


def get_metrics() -> List[str]:
    """
    Get the list of available metrics.
    
    Returns:
        List of metric names
    """
    return list(METRIC_FUNCTIONS.keys())


def run_evaluation(
    model,
    checkpoint_path: str,
    data_path: Optional[str] = None,
    classifier_checkpoint: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    num_samples: int = 100,
    label_context: Optional[EvaluationLabelContext] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Run evaluation metrics on the trained VAE model.
    
    Args:
        model: Trained VAE model
        checkpoint_path: Path to model checkpoint (for display)
        data_path: Path to dataset JSON file
        classifier_checkpoint: Optional path to classifier checkpoint
        metrics: List of metrics to run (None = all available)
        num_samples: Number of samples per grade for generation-based metrics
        label_context: Checkpoint label-space context
        device: Device to run on
        
    Returns:
        Dictionary containing results for each metric
    """
    # Determine which metrics to run
    if label_context is None:
        raise ValueError("label_context is required for evaluation")

    if metrics is None:
        metrics_to_run = get_metrics()
    else:
        # Validate requested metrics
        for metric in metrics:
            if metric not in METRIC_FUNCTIONS:
                raise ValueError(f"Unknown metric: {metric}. Available: {list(METRIC_FUNCTIONS.keys())}")
        metrics_to_run = metrics
    
    # Run each metric (note: metrics have different signatures, so we handle each)
    results = {'checkpoint': checkpoint_path, 'metrics': {}}
    for metric in metrics_to_run:
        if metric == 'reconstruction':
            results['metrics'][metric] = METRIC_FUNCTIONS[metric](model, data_path, label_context, device)
        elif metric == 'diversity':
            results['metrics'][metric] = METRIC_FUNCTIONS[metric](model, data_path, label_context, num_samples, device)
        elif metric == 'statistical':
            results['metrics'][metric] = METRIC_FUNCTIONS[metric](model, data_path, label_context, num_samples, device)
        elif metric == 'latent_space':
            results['metrics'][metric] = METRIC_FUNCTIONS[metric](model, data_path, label_context, device)
        elif metric == 'latent_visualization':
            results['metrics'][metric] = METRIC_FUNCTIONS[metric](model, data_path, label_context, device, checkpoint_path)
        elif metric == 'classifier_check':
            results['metrics'][metric] = METRIC_FUNCTIONS[metric](
                model,
                checkpoint_path,
                classifier_checkpoint,
                label_context,
                num_samples,
                device,
            )
    
    return results

