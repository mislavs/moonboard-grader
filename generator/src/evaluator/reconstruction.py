"""
Reconstruction quality metric.

Evaluates how well the VAE can reconstruct input problems by measuring
Intersection over Union (IoU) between original and reconstructed grids.
"""

from typing import Dict, Optional, Any
import torch
import numpy as np

from .utils import load_data_loader


def evaluate_reconstruction(
    model, 
    data_path: Optional[str], 
    device: str,
    threshold: float = 0.5,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    Evaluate reconstruction quality (IoU between original and reconstructed).
    
    Measures how well the VAE can reconstruct input problems by computing
    Intersection over Union (IoU) between original and reconstructed grids.
    
    Args:
        model: Trained VAE model
        data_path: Path to dataset JSON file
        device: Device to run on
        threshold: Threshold for binarizing reconstructed outputs (default: 0.5)
        batch_size: Batch size for data loader (default: 32)
        
    Returns:
        Dictionary with reconstruction metrics:
        - mean_iou: Average IoU across all samples
        - std_iou: Standard deviation of IoU
        - per_channel_iou: IoU for each channel (start/middle/end)
        - per_grade_iou: IoU statistics per grade
        - num_samples: Total number of samples evaluated
    """
    # Load validation data and dataset (for grade mappings)
    val_loader, dataset = load_data_loader(data_path, batch_size=batch_size, device=device)
    
    model.eval()
    model = model.to(device)
    
    # Storage for metrics
    all_ious = []
    channel_ious = {'start': [], 'middle': [], 'end': []}
    grade_ious = {}  # grade_label -> list of ious
    
    with torch.no_grad():
        for grids, grades in val_loader:
            grids = grids.to(device)
            grades = grades.to(device)
            
            # Forward pass through VAE
            x_recon, mu, logvar = model(grids, grades)
            
            # Binarize reconstructed output
            x_recon_binary = (torch.sigmoid(x_recon) > threshold).float()
            
            # Calculate IoU per sample
            for i in range(grids.size(0)):
                original = grids[i]
                reconstructed = x_recon_binary[i]
                grade_label = grades[i].item()
                
                # Overall IoU
                intersection = (original * reconstructed).sum()
                union = ((original + reconstructed) > 0).float().sum()
                iou = (intersection / (union + 1e-8)).item()
                all_ious.append(iou)
                
                # Store per-grade IoU
                if grade_label not in grade_ious:
                    grade_ious[grade_label] = []
                grade_ious[grade_label].append(iou)
                
                # Per-channel IoU
                for ch, name in enumerate(['start', 'middle', 'end']):
                    ch_intersection = (original[ch] * reconstructed[ch]).sum()
                    ch_union = ((original[ch] + reconstructed[ch]) > 0).float().sum()
                    ch_iou = (ch_intersection / (ch_union + 1e-8)).item()
                    channel_ious[name].append(ch_iou)
    
    # Compute per-grade statistics
    per_grade_stats = {}
    for grade_label, ious in grade_ious.items():
        # Use dataset's label_to_grade mapping to get the correct grade string
        grade_str = dataset.get_grade_from_label(int(grade_label))
        if grade_str is None:
            # Fallback - shouldn't happen, but be defensive
            grade_str = f"Grade_{int(grade_label)}"
        per_grade_stats[grade_str] = {
            'mean_iou': float(np.mean(ious)),
            'std_iou': float(np.std(ious)),
            'num_samples': len(ious)
        }
    
    return {
        'mean_iou': float(np.mean(all_ious)),
        'std_iou': float(np.std(all_ious)),
        'per_channel_iou': {
            channel: float(np.mean(ious)) 
            for channel, ious in channel_ious.items()
        },
        'per_grade_iou': per_grade_stats,
        'num_samples': len(all_ious),
        'threshold': threshold,
        'interpretation': 'IoU closer to 1.0 is better (0.7+ is good)'
    }

