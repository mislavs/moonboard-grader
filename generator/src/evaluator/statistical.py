"""
Statistical similarity metric.

Evaluates how closely generated problems match the statistical distribution
of real problems using Wasserstein distance.
"""

from typing import Dict, Optional, Any
import json
import logging

import numpy as np
from scipy.stats import wasserstein_distance

from moonboard_core import encode_grade
from src.generator import ProblemGenerator
from src.dataset import MoonBoardDataset
from .utils import extract_problem_stats

logger = logging.getLogger(__name__)


def evaluate_statistical_similarity(
    model,
    data_path: Optional[str],
    num_samples: int,
    device: str
) -> Dict[str, Any]:
    """
    Evaluate statistical similarity to real problems.
    
    Generates problems at each grade and compares their statistical properties
    to real problems using Wasserstein distance. Lower distance indicates
    better similarity to real problem distributions.
    
    Statistics compared:
    - num_holds: Total number of holds in a problem
    - num_start: Number of starting holds
    - num_end: Number of ending holds
    - num_middle: Number of middle holds
    - vertical_spread: Range of row numbers used (difficulty indicator)
    
    Args:
        model: Trained VAE model
        data_path: Path to dataset JSON file
        num_samples: Number of samples to generate per grade
        device: Device to run on
        
    Returns:
        Dictionary with statistical similarity metrics including:
        - overall_mean_distance: Mean Wasserstein distance across all statistics and grades
        - overall_std_distance: Standard deviation of Wasserstein distances
        - per_grade: Per-grade statistics (keyed by grade name like "6A+")
        - per_statistic: Overall Wasserstein distances for each statistic type
        - interpretation: How to interpret the results
    """
    logger.info(f"Evaluating statistical similarity with {num_samples} samples per grade")
    
    # Load dataset
    if data_path is None:
        data_path = "../data/problems.json"
    
    # Load dataset to get grade name mappings
    dataset = MoonBoardDataset(data_path=data_path)
    
    # Load raw JSON to get real problem statistics
    logger.info(f"Loading real problems from {data_path}")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        return {
            'error': f'Failed to load dataset: {str(e)}',
            'data_path': data_path
        }
    
    if not isinstance(raw_data, dict) or 'data' not in raw_data:
        return {
            'error': 'Invalid dataset format: expected {"data": [...]}',
            'data_path': data_path
        }
    
    problems_list = raw_data['data']
    logger.info(f"Loaded {len(problems_list)} real problems")
    
    # Group real problems by grade
    real_by_grade = {}
    for problem in problems_list:
        grade_str = problem.get('grade')
        if not grade_str:
            continue
        
        try:
            # Convert grade string to label using dataset's mapping
            if grade_str not in dataset.grade_to_label:
                # Skip grades not in the dataset (e.g., filtered out)
                continue
            
            grade_label = dataset.grade_to_label[grade_str]
            
            if grade_label not in real_by_grade:
                real_by_grade[grade_label] = []
            
            # Extract statistics from real problem
            stats = extract_problem_stats(problem)
            real_by_grade[grade_label].append(stats)
        except Exception as e:
            logger.warning(f"Failed to process real problem with grade {grade_str}: {e}")
            continue
    
    logger.info(f"Grouped real problems into {len(real_by_grade)} grades")
    
    # Create generator from model
    generator = ProblemGenerator(model, device=device, threshold=0.5)
    
    # Get number of grades from model
    num_grades = model.num_grades
    logger.info(f"Generating problems for {num_grades} grades")
    
    results_per_grade = {}
    
    for grade_label in range(num_grades):
        # Skip if we don't have real data for this grade
        if grade_label not in real_by_grade:
            logger.warning(f"Grade {grade_label}: No real problems available, skipping")
            results_per_grade[grade_label] = {
                'skipped': True,
                'reason': 'no_real_problems',
                'num_real': 0
            }
            continue
        
        logger.info(f"Generating samples for grade {grade_label}...")
        
        # Generate valid problems using retry logic (keeps trying until we get num_samples valid problems)
        valid_problems = generator.generate_with_retry(
            grade_label=grade_label,
            num_samples=num_samples,
            max_attempts=50,  # Increased from default 10 to handle difficult grades
            temperature=1.0
        )
        
        if len(valid_problems) < 10:  # Need enough samples for meaningful statistics
            logger.warning(
                f"Grade {grade_label}: Insufficient valid problems "
                f"({len(valid_problems)}/{num_samples}), skipping (likely too few training examples)"
            )
            results_per_grade[grade_label] = {
                'skipped': True,
                'reason': 'insufficient_valid_problems',
                'num_valid': len(valid_problems),
                'num_requested': num_samples,
                'num_real': len(real_by_grade[grade_label])
            }
            continue
        
        # Extract statistics from generated problems
        gen_stats = []
        for problem in valid_problems:
            try:
                stats = extract_problem_stats(problem)
                gen_stats.append(stats)
            except Exception as e:
                logger.warning(f"Failed to extract stats: {e}")
                continue
        
        if len(gen_stats) < 10:
            logger.warning(f"Grade {grade_label}: Failed to extract enough stats")
            results_per_grade[grade_label] = {
                'skipped': True,
                'reason': 'stat_extraction_failed',
                'num_valid': len(valid_problems),
                'num_requested': num_samples,
                'num_real': len(real_by_grade[grade_label])
            }
            continue
        
        # Get real statistics for this grade
        real_stats = real_by_grade[grade_label]
        
        # Calculate Wasserstein distances for each statistic
        stat_names = ['num_holds', 'num_start', 'num_end', 'num_middle', 'vertical_spread']
        distances = {}
        
        for stat_name in stat_names:
            try:
                gen_values = [s[stat_name] for s in gen_stats]
                real_values = [s[stat_name] for s in real_stats]
                
                # Calculate Wasserstein distance (Earth Mover's Distance)
                distance = wasserstein_distance(gen_values, real_values)
                distances[stat_name] = float(distance)
            except Exception as e:
                logger.warning(f"Failed to calculate distance for {stat_name}: {e}")
                distances[stat_name] = None
        
        # Calculate mean distance (excluding None values)
        valid_distances = [d for d in distances.values() if d is not None]
        mean_distance = float(np.mean(valid_distances)) if valid_distances else None
        
        results_per_grade[grade_label] = {
            'wasserstein_distances': distances,
            'mean_distance': mean_distance,
            'num_generated': len(gen_stats),
            'num_real': len(real_stats)
        }
        
        logger.info(
            f"Grade {grade_label}: mean_distance={mean_distance:.3f}, "
            f"gen={len(gen_stats)}, real={len(real_stats)}"
        )
    
    # Convert numeric grade labels to human-readable grade names
    per_grade_with_names = {}
    for grade_label, stats in results_per_grade.items():
        grade_str = dataset.get_grade_from_label(int(grade_label))
        if grade_str is None:
            # Fallback - shouldn't happen, but be defensive
            grade_str = f"Grade_{int(grade_label)}"
        per_grade_with_names[grade_str] = stats
    
    # Aggregate across grades (only non-skipped grades)
    valid_grades = {k: v for k, v in per_grade_with_names.items() if not v.get('skipped', False)}
    
    if not valid_grades:
        return {
            'error': 'No valid comparisons possible (no grades with both real and generated problems)',
            'per_grade': per_grade_with_names,
            'num_samples_per_grade': num_samples
        }
    
    # Overall mean distance
    all_mean_distances = [r['mean_distance'] for r in valid_grades.values() if r['mean_distance'] is not None]
    overall_mean_distance = float(np.mean(all_mean_distances)) if all_mean_distances else None
    overall_std_distance = float(np.std(all_mean_distances)) if all_mean_distances else None
    
    # Per-statistic overall distances (aggregating across grades)
    stat_names = ['num_holds', 'num_start', 'num_end', 'num_middle', 'vertical_spread']
    per_statistic_distances = {}
    
    for stat_name in stat_names:
        stat_distances = []
        for grade_stats in valid_grades.values():
            if 'wasserstein_distances' in grade_stats:
                dist = grade_stats['wasserstein_distances'].get(stat_name)
                if dist is not None:
                    stat_distances.append(dist)
        
        if stat_distances:
            per_statistic_distances[stat_name] = {
                'mean': float(np.mean(stat_distances)),
                'std': float(np.std(stat_distances)),
                'min': float(np.min(stat_distances)),
                'max': float(np.max(stat_distances))
            }
        else:
            per_statistic_distances[stat_name] = None
    
    logger.info(
        f"Overall: mean_distance={overall_mean_distance:.3f}, "
        f"evaluated {len(valid_grades)}/{len(per_grade_with_names)} grades"
    )
    
    return {
        'overall_mean_distance': overall_mean_distance,
        'overall_std_distance': overall_std_distance,
        'per_grade': per_grade_with_names,
        'per_statistic': per_statistic_distances,
        'num_samples_per_grade': num_samples,
        'num_grades_evaluated': len(valid_grades),
        'num_grades_total': num_grades,
        'interpretation': (
            'Wasserstein distance measures similarity between generated and real problem distributions. '
            'Lower is better (0 = identical distributions). '
            'Target: <2.0 overall mean distance for good similarity.'
        )
    }

