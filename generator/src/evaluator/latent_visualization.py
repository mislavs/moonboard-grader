"""
Latent space visualization metric.

Generates t-SNE and PCA visualizations of the latent space,
colored by grade, to help understand how well the VAE has
learned to organize problems.
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from .utils import load_data_loader
from src.label_space import EvaluationLabelContext
from moonboard_core import decode_grade

logger = logging.getLogger(__name__)


def evaluate_latent_visualization(
    model,
    data_path: Optional[str],
    label_context: EvaluationLabelContext,
    device: str,
    checkpoint_path: str
) -> Dict[str, Any]:
    """
    Generate latent space visualizations using t-SNE and PCA.
    
    Encodes all problems in the dataset to the latent space and creates
    2D scatter plots colored by grade to visualize how well the model
    has learned to cluster problems by difficulty.
    
    Args:
        model: Trained VAE model
        data_path: Path to dataset
        device: Device to run on
        checkpoint_path: Path to checkpoint (used to determine save location)
        
    Returns:
        Dictionary with visualization results including:
        - tsne_plot_path: Path to saved t-SNE visualization
        - pca_plot_path: Path to saved PCA visualization
        - num_samples: Number of problems encoded
        - num_grades: Number of unique grades
        - latent_dim: Dimensionality of latent space
    """
    if data_path is None:
        return {
            'error': 'Data path is required for latent visualization metric',
            'skipped': True
        }
    
    # Load validation data
    logger.info("Loading validation data...")
    val_loader, _ = load_data_loader(
        data_path,
        label_context=label_context,
        batch_size=32,
    )
    
    # Encode all problems to latent space
    logger.info("Encoding problems to latent space...")
    model.eval()
    all_latents = []
    all_grades = []
    
    with torch.no_grad():
        for grids, grades in val_loader:
            grids = grids.to(device)
            grades_device = grades.to(device)
            # Encode to latent space (get mu, not logvar)
            mu, _ = model.encode(grids, grades_device)
            all_latents.append(mu.cpu().numpy())
            all_grades.extend(grades.tolist())
    
    latents = np.concatenate(all_latents, axis=0)
    grades = np.array(all_grades)
    
    logger.info(f"Encoded {len(latents)} problems to latent space of dimension {latents.shape[1]}")
    
    # Get unique grades and build contiguous color labels for plotting.
    unique_grades = np.unique(grades)
    n_grades = len(unique_grades)
    grade_to_plot_idx = {int(grade): idx for idx, grade in enumerate(unique_grades)}
    plot_labels = np.array([grade_to_plot_idx[int(label)] for label in grades])
    
    # Map grade labels to grade names
    grade_names = []
    for grade_label in unique_grades:
        global_label = label_context.model_to_global_label(int(grade_label))
        grade_name = decode_grade(global_label)
        grade_names.append(grade_name)
    
    # Get discrete colors from tab20/tab20b/tab20c for better distinction
    if n_grades <= 10:
        base_cmap = cm.get_cmap('tab10')
    elif n_grades <= 20:
        base_cmap = cm.get_cmap('tab20')
    else:
        # For more grades, use viridis but sample it discretely
        base_cmap = cm.get_cmap('viridis', n_grades)
    
    colors = [base_cmap(i) for i in range(n_grades)]
    discrete_cmap = ListedColormap(colors)
    
    # Determine save location from checkpoint path
    checkpoint_dir = Path(checkpoint_path).parent
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tsne_path = checkpoint_dir / f'latent_tsne_{timestamp}.png'
    pca_path = checkpoint_dir / f'latent_pca_{timestamp}.png'
    
    # Generate t-SNE visualization
    logger.info("Generating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42, verbose=0)
    latents_tsne = tsne.fit_transform(latents)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latents_tsne[:, 0],
        latents_tsne[:, 1],
        c=plot_labels,
        cmap=discrete_cmap,
        alpha=0.6,
        s=20,
        vmin=-0.5,
        vmax=n_grades - 0.5,
    )
    cbar = plt.colorbar(scatter, ticks=range(n_grades))
    cbar.set_ticklabels(grade_names)
    cbar.set_label('Grade', rotation=270, labelpad=20)
    plt.title('VAE Latent Space Visualization (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved t-SNE visualization to: {tsne_path}")
    
    # Generate PCA visualization
    logger.info("Generating PCA visualization...")
    pca = PCA(n_components=2, random_state=42)
    latents_pca = pca.fit_transform(latents)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latents_pca[:, 0],
        latents_pca[:, 1],
        c=plot_labels,
        cmap=discrete_cmap,
        alpha=0.6,
        s=20,
        vmin=-0.5,
        vmax=n_grades - 0.5,
    )
    cbar = plt.colorbar(scatter, ticks=range(n_grades))
    cbar.set_ticklabels(grade_names)
    cbar.set_label('Grade', rotation=270, labelpad=20)
    plt.title('VAE Latent Space Visualization (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.tight_layout()
    plt.savefig(pca_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved PCA visualization to: {pca_path}")
    
    logger.info("Latent space visualization complete")
    
    return {
        'tsne_plot_path': str(tsne_path),
        'pca_plot_path': str(pca_path),
        'num_samples': int(len(latents)),
        'num_grades': int(len(unique_grades)),
        'latent_dim': int(latents.shape[1]),
        'pca_variance_explained': {
            'pc1': float(pca.explained_variance_ratio_[0]),
            'pc2': float(pca.explained_variance_ratio_[1]),
            'total': float(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])
        }
    }

