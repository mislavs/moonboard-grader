"""
Latent space quality metric.

Evaluates the quality of learned latent representations by measuring
grade clustering and separation in the latent space.
"""

import logging
from typing import Dict, Optional, Any

import numpy as np
import torch

from .utils import load_data_loader
from src.label_space import EvaluationLabelContext
from moonboard_core import decode_grade

logger = logging.getLogger(__name__)


def evaluate_latent_space(
    model,
    data_path: Optional[str],
    label_context: EvaluationLabelContext,
    device: str,
) -> Dict[str, Any]:
    """
    Evaluate latent space quality and grade clustering.
    
    Args:
        model: Trained VAE model
        data_path: Path to dataset
        device: Device to run on
        
    Returns:
        Dictionary with latent space quality metrics
    """
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        logger.error("scikit-learn is required for latent space metric")
        return {
            'error': 'scikit-learn is required. Install with: pip install scikit-learn',
            'skipped': True
        }
    
    if data_path is None:
        return {
            'error': 'Data path is required for latent space metric',
            'skipped': True
        }
    
    logger.info("Loading validation data...")
    val_loader, _ = load_data_loader(
        data_path,
        label_context=label_context,
        batch_size=32,
    )
    
    logger.info("Encoding problems to latent space...")
    model.eval()
    all_latents = []
    all_grades = []
    
    with torch.no_grad():
        for grids, grades in val_loader:
            grids = grids.to(device)
            # Encode to latent space (get mu, not logvar)
            mu, _ = model.encode(grids)
            all_latents.append(mu.cpu().numpy())
            all_grades.extend(grades.tolist())
    
    latents = np.concatenate(all_latents, axis=0)
    grades = np.array(all_grades)
    
    logger.info(f"Encoded {len(latents)} problems to latent space of dimension {latents.shape[1]}")
    
    # Calculate silhouette score (how well grades cluster)
    unique_grades = np.unique(grades)
    if len(unique_grades) > 1:
        logger.info("Calculating silhouette score for grade clustering...")
        silhouette = silhouette_score(latents, grades)
        logger.info(f"Silhouette score: {silhouette:.4f}")
    else:
        silhouette = 0.0
        logger.warning("Only one grade in dataset, cannot calculate silhouette score")
    
    # Calculate per-grade centroids and variance
    logger.info("Calculating per-grade centroids...")
    centroids = {}
    grade_name_to_global_label = {}
    for grade in unique_grades:
        grade_latents = latents[grades == grade]
        global_label = label_context.model_to_global_label(int(grade))
        grade_str = decode_grade(global_label)
        grade_name_to_global_label[grade_str] = global_label
        centroids[grade_str] = {
            'mean': grade_latents.mean(axis=0).tolist(),
            'std': float(grade_latents.std(axis=0).mean()),
            'count': int(len(grade_latents))
        }
    
    # Measure grade separation (distance between adjacent grade centroids)
    logger.info("Calculating grade separation distances...")
    centroid_distances = []
    sorted_grade_strs = sorted(
        centroids.keys(),
        key=lambda grade_name: grade_name_to_global_label[grade_name],
    )
    
    for i in range(len(sorted_grade_strs) - 1):
        g1, g2 = sorted_grade_strs[i], sorted_grade_strs[i+1]
        c1 = np.array(centroids[g1]['mean'])
        c2 = np.array(centroids[g2]['mean'])
        dist = np.linalg.norm(c1 - c2)
        centroid_distances.append(dist)
    
    mean_separation = float(np.mean(centroid_distances)) if centroid_distances else 0.0
    
    # Calculate overall latent space statistics
    latent_mean = float(np.abs(latents).mean())
    latent_std = float(latents.std())
    
    logger.info("Latent space evaluation complete")
    
    return {
        'silhouette_score': float(silhouette),
        'interpretation': 'Silhouette: higher is better (-1 to 1, >0.3 is good clustering)',
        'latent_mean': latent_mean,
        'latent_std': latent_std,
        'grade_separation': mean_separation,
        'grade_separation_std': float(np.std(centroid_distances)) if len(centroid_distances) > 1 else 0.0,
        'num_grades': len(unique_grades),
        'num_samples': len(latents),
        'latent_dim': int(latents.shape[1]),
        'per_grade_centroids': centroids
    }

