"""
Orchestrator for running evaluation metrics.
"""

from typing import Dict, List, Optional, Any

from .reconstruction import evaluate_reconstruction
from .diversity import evaluate_diversity
from .statistical import evaluate_statistical_similarity
from .latent_space import evaluate_latent_space
from .grade_conditioning import evaluate_grade_conditioning


# Dispatch table mapping metric names to their functions
METRIC_FUNCTIONS = {
    'reconstruction': evaluate_reconstruction,
    'diversity': evaluate_diversity,
    'statistical': evaluate_statistical_similarity,
    'latent_space': evaluate_latent_space,
    'grade_conditioning': evaluate_grade_conditioning
}


def get_metrics() -> List[str]:
    """
    Get the list of available metrics by checking which ones are ready to use.
    
    Returns:
        List of metric names
    """
    available = []
    
    # Test each metric to see if it's ready
    for metric_name, metric_func in METRIC_FUNCTIONS.items():
        try:
            # Call with None parameters - placeholder metrics return not_implemented status
            # Different metrics have different signatures, so we need to handle each
            if metric_name == 'diversity':
                result = metric_func(None, None, None, 'cpu')
            elif metric_name in ['reconstruction', 'latent_space']:
                result = metric_func(None, None, 'cpu')
            elif metric_name in ['statistical']:
                result = metric_func(None, None, None, 'cpu')
            elif metric_name == 'grade_conditioning':
                result = metric_func(None, None, None, 'cpu')
            else:
                result = metric_func(None, None, 'cpu')
            
            # If it doesn't have 'status': 'not_implemented', it's ready
            if not (isinstance(result, dict) and result.get('status') == 'not_implemented'):
                available.append(metric_name)
        except Exception:
            available.append(metric_name)
    
    return available


def run_evaluation(
    model,
    checkpoint_path: str,
    data_path: Optional[str] = None,
    classifier_checkpoint: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    num_samples: int = 100,
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
        device: Device to run on
        
    Returns:
        Dictionary containing results for each metric
    """
    # Determine which metrics to run
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
            results['metrics'][metric] = METRIC_FUNCTIONS[metric](model, data_path, device)
        elif metric == 'diversity':
            results['metrics'][metric] = METRIC_FUNCTIONS[metric](model, data_path, num_samples, device)
        elif metric == 'statistical':
            results['metrics'][metric] = METRIC_FUNCTIONS[metric](model, data_path, num_samples, device)
        elif metric == 'latent_space':
            results['metrics'][metric] = METRIC_FUNCTIONS[metric](model, data_path, device)
        elif metric == 'grade_conditioning':
            results['metrics'][metric] = METRIC_FUNCTIONS[metric](model, classifier_checkpoint, num_samples, device)
    
    return results

