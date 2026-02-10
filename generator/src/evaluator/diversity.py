"""
Diversity metric.

Evaluates the diversity of generated problems by measuring uniqueness
and pairwise Hamming distances between generated samples.
"""

from typing import Dict, Any, Optional
import logging

import numpy as np
from scipy.spatial.distance import pdist

from moonboard_core import create_grid_tensor, decode_grade
from src.generator import ProblemGenerator
from src.label_space import EvaluationLabelContext

logger = logging.getLogger(__name__)


def evaluate_diversity(
    model, 
    data_path: Optional[str],
    label_context: EvaluationLabelContext,
    num_samples: int, 
    device: str
) -> Dict[str, Any]:
    """
    Evaluate diversity of generated problems.
    
    Generates problems at each grade and calculates:
    - Pairwise Hamming distances between generated grids
    - Uniqueness ratio (percentage of unique problems)
    - Per-grade diversity statistics
    
    Args:
        model: Trained VAE model
        data_path: Path to dataset JSON file (for grade name mappings)
        num_samples: Number of samples to generate per grade
        device: Device to run on
        
    Returns:
        Dictionary with diversity metrics including:
        - overall_mean_diversity: Mean Hamming distance across all grades
        - overall_uniqueness_ratio: Average uniqueness ratio across grades
        - per_grade: Per-grade statistics (keyed by grade name like "6A+")
        - num_samples_per_grade: Number of samples generated per grade
        - interpretation: How to interpret the results
    """
    logger.info(f"Evaluating diversity with {num_samples} samples per grade")
    _ = data_path  # Unused; kept for consistent metric function signature.
    
    # Create generator from model
    generator = ProblemGenerator(model, device=device, threshold=0.5)
    
    global_grade_indices = label_context.get_global_grade_indices()
    logger.info(f"Generating problems for {len(global_grade_indices)} grades")
    
    results_per_grade = {}
    
    for global_grade_label in global_grade_indices:
        model_grade_label = label_context.global_to_model_label(global_grade_label)
        grade_name = decode_grade(global_grade_label)
        logger.info(
            f"Generating samples for grade {grade_name} "
            f"(global={global_grade_label}, model={model_grade_label})..."
        )
        
        # Generate valid problems using retry logic (keeps trying until we get num_samples valid problems)
        valid_problems = generator.generate_with_retry(
            grade_label=model_grade_label,
            num_samples=num_samples,
            max_attempts=50,  # Increased from default 10 to handle difficult grades
            temperature=1.0
        )
        
        if len(valid_problems) < 2:
            logger.warning(
                f"Grade {grade_name}: Insufficient valid problems "
                f"({len(valid_problems)}/{num_samples}), skipping (likely too few training examples)"
            )
            results_per_grade[grade_name] = {
                'skipped': True,
                'reason': 'insufficient_valid_problems',
                'num_valid': len(valid_problems),
                'num_requested': num_samples
            }
            continue
        
        # Convert problems to grid representations
        grids = []
        for problem in valid_problems:
            try:
                grid = create_grid_tensor(problem['moves'])
                # Flatten grid for distance calculation (3x18x11 -> 594)
                grids.append(grid.flatten())
            except Exception as e:
                logger.warning(f"Failed to build grid: {e}")
                continue
        
        if len(grids) < 2:
            logger.warning(f"Grade {grade_name}: Failed to build grids")
            results_per_grade[grade_name] = {
                'skipped': True,
                'reason': 'grid_conversion_failed',
                'num_valid': len(valid_problems),
                'num_requested': num_samples
            }
            continue
        
        grids = np.array(grids)
        
        # Calculate pairwise Hamming distances
        # Hamming distance: fraction of positions that differ
        distances = pdist(grids, metric='hamming')
        mean_diversity = float(np.mean(distances))
        std_diversity = float(np.std(distances))
        
        # Count unique problems (exact duplicates)
        unique_grids = np.unique(grids, axis=0)
        num_unique = len(unique_grids)
        uniqueness_ratio = num_unique / len(grids)
        
        results_per_grade[grade_name] = {
            'mean_diversity': mean_diversity,
            'std_diversity': std_diversity,
            'unique_problems': num_unique,
            'total_valid': len(grids),
            'uniqueness_ratio': uniqueness_ratio,
            'num_requested': num_samples
        }
        
        logger.info(
            f"Grade {grade_name}: diversity={mean_diversity:.3f}, "
            f"uniqueness={uniqueness_ratio:.1%} ({num_unique}/{len(grids)})"
        )
    
    # Aggregate across grades (only non-skipped grades)
    valid_grades = {k: v for k, v in results_per_grade.items() if not v.get('skipped', False)}
    
    if not valid_grades:
        return {
            'error': 'No valid problems generated across any grade',
            'per_grade': results_per_grade,
            'num_samples_per_grade': num_samples
        }
    
    all_diversities = [r['mean_diversity'] for r in valid_grades.values()]
    all_uniqueness = [r['uniqueness_ratio'] for r in valid_grades.values()]
    
    overall_mean_diversity = float(np.mean(all_diversities))
    overall_std_diversity = float(np.std(all_diversities))
    overall_uniqueness_ratio = float(np.mean(all_uniqueness))
    
    logger.info(
        f"Overall: diversity={overall_mean_diversity:.3f}, "
        f"uniqueness={overall_uniqueness_ratio:.1%}"
    )
    
    return {
        'overall_mean_diversity': overall_mean_diversity,
        'overall_std_diversity': overall_std_diversity,
        'overall_uniqueness_ratio': overall_uniqueness_ratio,
        'per_grade': results_per_grade,
        'num_samples_per_grade': num_samples,
        'num_grades_evaluated': len(valid_grades),
        'num_grades_total': len(global_grade_indices),
        'interpretation': (
            'Diversity measured by Hamming distance (0=identical, 1=completely different). '
            'Higher diversity and uniqueness indicate better generation quality. '
            'Target: >80% uniqueness, >0.3 diversity.'
        )
    }

