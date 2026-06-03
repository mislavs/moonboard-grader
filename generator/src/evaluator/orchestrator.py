"""
Orchestrator for running evaluation metrics.
"""

import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from src.label_space import EvaluationLabelContext
from src.generator import ProblemGenerator

from .utils import build_generation_pool
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

GENERATION_METRICS = {'diversity', 'statistical', 'classifier_check'}

METRIC_ORDER = [
    'reconstruction',
    'latent_space',
    'diversity',
    'statistical',
    'classifier_check',
    'latent_visualization',
]

EvaluationEventCallback = Callable[[str, str, int, int], None]


def get_metrics() -> List[str]:
    """
    Get the list of available metrics.
    
    Returns:
        List of metric names
    """
    return list(METRIC_FUNCTIONS.keys())


def order_metrics(metrics: List[str]) -> List[str]:
    """
    Order metrics so quick feedback appears before slower generation/plotting work.
    """
    order = {metric: index for index, metric in enumerate(METRIC_ORDER)}
    return sorted(metrics, key=lambda metric: order.get(metric, len(METRIC_ORDER)))


def _resolve_metrics(metrics: Optional[List[str]]) -> List[str]:
    if metrics is None:
        return order_metrics(get_metrics())

    for metric in metrics:
        if metric not in METRIC_FUNCTIONS:
            raise ValueError(f"Unknown metric: {metric}. Available: {list(METRIC_FUNCTIONS.keys())}")
    return order_metrics(metrics)


def _dispatch_metric(
    metric: str,
    model,
    checkpoint_path: str,
    data_path: Optional[str],
    classifier_checkpoint: Optional[str],
    num_samples: int,
    label_context: EvaluationLabelContext,
    device: str,
    pool: Optional[Dict[int, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """
    Run a single metric.

    Metrics have different signatures, so dispatch is kept explicit.
    """
    if metric == 'reconstruction':
        return METRIC_FUNCTIONS[metric](model, data_path, label_context, device)
    if metric == 'diversity':
        return METRIC_FUNCTIONS[metric](model, data_path, label_context, num_samples, device, pool=pool)
    if metric == 'statistical':
        return METRIC_FUNCTIONS[metric](model, data_path, label_context, num_samples, device, pool=pool)
    if metric == 'latent_space':
        return METRIC_FUNCTIONS[metric](model, data_path, label_context, device)
    if metric == 'latent_visualization':
        return METRIC_FUNCTIONS[metric](model, data_path, label_context, device, checkpoint_path)
    if metric == 'classifier_check':
        return METRIC_FUNCTIONS[metric](
            model,
            checkpoint_path,
            classifier_checkpoint,
            label_context,
            num_samples,
            device,
            pool=pool,
        )

    raise ValueError(f"Unknown metric: {metric}. Available: {list(METRIC_FUNCTIONS.keys())}")


def run_evaluation_iter(
    model,
    checkpoint_path: str,
    data_path: Optional[str] = None,
    classifier_checkpoint: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    num_samples: int = 100,
    label_context: Optional[EvaluationLabelContext] = None,
    device: str = 'cpu',
    on_event: Optional[EvaluationEventCallback] = None,
) -> Iterator[Tuple[str, Dict[str, Any], float]]:
    """
    Run evaluation metrics and yield each result as soon as it is ready.

    Yields:
        Tuples of (metric_name, result, elapsed_seconds)
    """
    if label_context is None:
        raise ValueError("label_context is required for evaluation")

    metrics_to_run = _resolve_metrics(metrics)
    pool = None
    total_metrics = len(metrics_to_run)
    metrics_needing_pool = {
        metric
        for metric in metrics_to_run
        if metric in {'diversity', 'statistical'}
        or (metric == 'classifier_check' and classifier_checkpoint is not None)
    }

    for metric_index, metric in enumerate(metrics_to_run, 1):
        if pool is None and metric in metrics_needing_pool:
            generator_metric_count = len(metrics_needing_pool)
            pool = build_generation_pool(
                generator=ProblemGenerator(model, device=device, threshold=0.5),
                label_context=label_context,
                num_samples=num_samples,
                max_attempts=50,
                temperature=1.0,
                on_grade=(
                    lambda grade_index, total_grades, grade_name: on_event(
                        'generate',
                        grade_name,
                        grade_index,
                        total_grades,
                    )
                    if on_event
                    else None
                ),
            )
            if on_event:
                on_event('pool_ready', 'generation_pool', generator_metric_count, generator_metric_count)

        if on_event:
            on_event('start', metric, metric_index, total_metrics)

        start = time.perf_counter()
        result = _dispatch_metric(
            metric,
            model,
            checkpoint_path,
            data_path,
            classifier_checkpoint,
            num_samples,
            label_context,
            device,
            pool=pool if metric in metrics_needing_pool else None,
        )
        elapsed = time.perf_counter() - start
        yield metric, result, elapsed


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
    results = {'checkpoint': checkpoint_path, 'metrics': {}}
    for metric, metric_result, _ in run_evaluation_iter(
        model=model,
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        classifier_checkpoint=classifier_checkpoint,
        metrics=metrics,
        num_samples=num_samples,
        label_context=label_context,
        device=device,
    ):
        results['metrics'][metric] = metric_result
    
    return results

