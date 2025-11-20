"""
Classifier check metric.

Generates boulder problems for each grade and uses an external classifier
to predict their grades, reporting accuracy statistics.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Any, List
import numpy as np
import torch

logger = logging.getLogger(__name__)

def evaluate_classifier_check(
    model,
    checkpoint_path: str,
    classifier_checkpoint: Optional[str],
    num_samples: int,
    device: str
) -> Dict[str, Any]:
    """
    Evaluate generator by classifying generated problems.
    
    Args:
        model: The generator VAE model.
        checkpoint_path: Path to the generator checkpoint (for loading metadata).
        classifier_checkpoint: Path to the classifier model checkpoint.
        num_samples: Number of problems to generate per grade.
        device: 'cpu' or 'cuda'.
        
    Returns:
        Dictionary containing evaluation statistics.
    """
    if classifier_checkpoint is None:
        return {
            'error': 'Classifier checkpoint required for classifier_check metric',
            'skipped': True
        }

    # Dynamic import for classifier and moonboard_core
    try:
        project_root = Path(__file__).parents[3]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            
        from classifier.src.predictor import Predictor
        from moonboard_core import create_grid_tensor, decode_grade
    except ImportError as e:
        logger.error(f"Failed to import dependencies: {e}")
        return {
            'error': f"Failed to import dependencies: {e}",
            'skipped': True
        }

    # Load classifier
    logger.info(f"Loading classifier from {classifier_checkpoint}")
    try:
        classifier = Predictor(classifier_checkpoint, device=device)
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        return {
            'error': f"Failed to load classifier: {e}",
            'skipped': True
        }

    # Setup generator using from_checkpoint to get grade offset metadata
    from ..generator import ProblemGenerator
    generator = ProblemGenerator.from_checkpoint(checkpoint_path, device=device)
    
    # Get metadata from the generator instance
    generator_offset = generator.grade_offset
    min_grade_idx = generator.min_grade_index
    max_grade_idx = generator.max_grade_index
    
    # Determine the actual grade range to evaluate
    # Prefer using min/max_grade_index if available (more reliable)
    if min_grade_idx is not None and max_grade_idx is not None:
        # Use the explicit range from checkpoint
        grade_indices_to_eval = list(range(min_grade_idx, max_grade_idx + 1))
        logger.info(f"Using explicit grade range from checkpoint: {min_grade_idx} to {max_grade_idx}")
    else:
        # Fall back to using num_grades (unfiltered model)
        num_grades = generator.model.num_grades
        grade_indices_to_eval = list(range(generator_offset, generator_offset + num_grades))
        logger.info(f"Using num_grades={num_grades} with offset={generator_offset}")
    
    results_per_grade = {}
    
    logger.info(f"Running classifier check with {num_samples} samples per grade")
    logger.info(f"Evaluating {len(grade_indices_to_eval)} grades: {decode_grade(grade_indices_to_eval[0])} to {decode_grade(grade_indices_to_eval[-1])}")

    for global_grade_idx in grade_indices_to_eval:
        # Convert global index to generator's relative index
        generator_relative_idx = global_grade_idx - generator_offset
        grade_name = decode_grade(global_grade_idx)
        logger.info(f"Evaluating grade {grade_name} (global_idx={global_grade_idx}, generator_idx={generator_relative_idx})")
        
        # Generate problems
        try:
            problems = generator.generate_with_retry(
                grade_label=generator_relative_idx,
                num_samples=num_samples,
                max_attempts=10,
                temperature=1.0
            )
        except Exception as e:
            logger.warning(f"Generation failed for grade {grade_name}: {e}")
            continue

        if not problems:
            logger.warning(f"No valid problems generated for grade {grade_name}")
            continue

        # Classify problems (BATCH MODE)
        try:
            # Convert all problems to tensors
            grid_tensors = []
            for problem in problems:
                grid = create_grid_tensor(problem['moves'])
                grid_tensors.append(grid)
            
            # Stack into batch tensor: (batch_size, 3, 18, 11)
            batch_tensor = torch.FloatTensor(np.array(grid_tensors))
            
            # Run batch prediction
            # Note: predict_from_tensor handles both single and batch inputs
            # It returns a list of dicts when input is a batch
            batch_results = classifier.predict_from_tensor(batch_tensor)
            
            predictions = [res['predicted_label'] for res in batch_results]
            valid_count = len(predictions)
            
        except Exception as e:
            logger.error(f"Batch classification failed for grade {grade_name}: {e}")
            continue

        if not predictions:
            continue

        predictions = np.array(predictions)
        # Use global_grade_idx directly as the target
        targets = np.full(len(predictions), global_grade_idx)
        
        # Calculate stats
        diff = np.abs(predictions - targets)
        exact_match = np.mean(diff == 0) * 100
        off_by_one = np.mean(diff == 1) * 100
        off_by_two = np.mean(diff == 2) * 100
        off_more = np.mean(diff > 2) * 100
        
        # Calculate prediction distribution
        unique_preds, counts = np.unique(predictions, return_counts=True)
        prediction_distribution = {}
        for pred_label, count in zip(unique_preds, counts):
            pred_grade_name = decode_grade(int(pred_label))
            prediction_distribution[pred_grade_name] = int(count)
        
        results_per_grade[grade_name] = {
            'target_grade_idx': global_grade_idx,
            'generator_relative_idx': generator_relative_idx,
            'generated_count': len(problems),
            'classified_count': valid_count,
            'exact_match_percent': float(exact_match),
            'off_by_one_percent': float(off_by_one),
            'off_by_two_percent': float(off_by_two),
            'off_more_percent': float(off_more),
            'mean_predicted_grade': float(np.mean(predictions)),
            'std_predicted_grade': float(np.std(predictions)),
            'prediction_distribution': prediction_distribution
        }

    # Aggregate results
    if not results_per_grade:
        return {'error': 'No results generated', 'skipped': True}

    total_exact = np.mean([r['exact_match_percent'] for r in results_per_grade.values()])
    total_off_1 = np.mean([r['off_by_one_percent'] for r in results_per_grade.values()])
    total_off_2 = np.mean([r['off_by_two_percent'] for r in results_per_grade.values()])

    return {
        'overall_stats': {
            'exact_match_percent': float(total_exact),
            'off_by_one_percent': float(total_off_1),
            'off_by_two_percent': float(total_off_2)
        },
        'per_grade': results_per_grade
    }
