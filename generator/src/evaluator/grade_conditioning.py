"""
Grade conditioning metric.

Evaluates how well the generator respects grade conditioning by classifying
generated problems and comparing to the requested grade.

⚠️ WARNING: This metric has limited reliability due to classifier accuracy
(~35% exact, ~70% ±1 grade). Use for relative comparisons only.
"""

from typing import Dict, Optional, Any
import sys
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


def evaluate_grade_conditioning(
    model,
    classifier_checkpoint: Optional[str],
    num_samples: int,
    device: str
) -> Dict[str, Any]:
    """
    Evaluate grade conditioning accuracy using classifier.
    
    ⚠️ WARNING: This metric has limited reliability due to classifier accuracy.
    Current classifier baseline: ~35% exact, ~70% ±1 grade.
    Use this for RELATIVE comparison between models, not absolute quality assessment.
    
    Args:
        model: Trained VAE model
        classifier_checkpoint: Path to classifier checkpoint
        num_samples: Number of samples per grade
        device: Device to run on
        
    Returns:
        Dictionary with grade conditioning metrics and warnings
    """
    # Check if classifier checkpoint is provided
    if classifier_checkpoint is None:
        return {
            'error': 'Classifier checkpoint required for grade conditioning metric',
            'skipped': True,
            'message': 'Use --classifier-checkpoint to enable this metric'
        }
    
    # Import dependencies
    try:
        # Add parent directory to path to allow importing classifier
        project_root = Path(__file__).parents[3]  # goes up to moonboard-grader/
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from classifier.src.predictor import Predictor
    except ImportError as e:
        logger.error(f"Failed to import classifier: {e}")
        return {
            'error': f'Failed to import classifier: {e}',
            'skipped': True,
            'message': 'Ensure classifier module is available'
        }
    
    try:
        from moonboard_core import create_grid_tensor, decode_grade
    except ImportError as e:
        logger.error(f"Failed to import moonboard_core: {e}")
        return {
            'error': f'Failed to import moonboard_core: {e}',
            'skipped': True
        }
    
    # Load classifier
    logger.info(f"Loading classifier from {classifier_checkpoint}")
    try:
        classifier = Predictor(classifier_checkpoint, device=device)
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        return {
            'error': f'Failed to load classifier: {e}',
            'skipped': True
        }
    
    # Create problem generator from VAE model
    from ..generator import ProblemGenerator
    generator = ProblemGenerator(model, device=device)
    
    # Get number of grades from model
    num_grades = model.num_grades
        
    # Results per grade
    results_per_grade = {}
    
    logger.info(f"Evaluating grade conditioning with {num_samples} samples per grade")
    
    for grade_label in range(num_grades):
        logger.debug(f"Generating {num_samples} valid problems for grade {grade_label}")
        
        # Generate problems at this grade using retry logic to ensure we get valid problems
        try:
            valid_problems = generator.generate_with_retry(
                grade_label=grade_label,
                num_samples=num_samples,
                max_attempts=20,  # Increased attempts to ensure we get enough valid problems
                temperature=1.0
            )
        except Exception as e:
            logger.warning(f"Failed to generate problems for grade {grade_label}: {e}")
            results_per_grade[grade_label] = {
                'exact_accuracy': 0.0,
                'tolerance_1_accuracy': 0.0,
                'tolerance_2_accuracy': 0.0,
                'num_valid': 0,
                'num_requested': num_samples,
                'skipped': True,
                'reason': f'Generation failed: {str(e)[:50]}'
            }
            continue
        
        if len(valid_problems) == 0:
            logger.warning(f"No valid problems generated for grade {grade_label} after retries")
            results_per_grade[grade_label] = {
                'exact_accuracy': 0.0,
                'tolerance_1_accuracy': 0.0,
                'tolerance_2_accuracy': 0.0,
                'num_valid': 0,
                'num_requested': num_samples,
                'skipped': True,
                'reason': 'No valid problems generated after retries'
            }
            continue
        
        # Log if we got fewer than requested
        if len(valid_problems) < num_samples:
            logger.info(f"Grade {grade_label}: Got {len(valid_problems)}/{num_samples} valid problems after retries")
        
        # Classify each problem
        predictions = []
        for problem in valid_problems:
            try:
                # Convert moves to grid tensor
                grid = create_grid_tensor(problem['moves'])
                
                # Classify
                result = classifier.predict_from_tensor(grid)
                predictions.append(result['predicted_label'])
            except Exception as e:
                logger.warning(f"Failed to classify problem: {e}")
                continue
        
        if len(predictions) == 0:
            logger.warning(f"No successful predictions for grade {grade_label}")
            results_per_grade[grade_label] = {
                'exact_accuracy': 0.0,
                'tolerance_1_accuracy': 0.0,
                'tolerance_2_accuracy': 0.0,
                'num_valid': len(valid_problems),
                'num_generated': num_samples,
                'skipped': True,
                'reason': 'Classification failed for all problems'
            }
            continue
        
        predictions = np.array(predictions)
        labels = np.full(len(predictions), grade_label)
        
        # Calculate accuracy metrics
        exact_acc = np.mean(predictions == labels) * 100
        tolerance_1 = np.mean(np.abs(predictions - labels) <= 1) * 100
        tolerance_2 = np.mean(np.abs(predictions - labels) <= 2) * 100
        
        # Calculate prediction distribution
        unique_preds, counts = np.unique(predictions, return_counts=True)
        prediction_distribution = {}
        for pred_label, count in zip(unique_preds, counts):
            # Store as numeric label for now, will convert to grade name later
            prediction_distribution[int(pred_label)] = int(count)
        
        results_per_grade[grade_label] = {
            'exact_accuracy': float(exact_acc),
            'tolerance_1_accuracy': float(tolerance_1),
            'tolerance_2_accuracy': float(tolerance_2),
            'num_valid': len(valid_problems),
            'num_requested': num_samples,  # What we requested
            'num_classified': len(predictions),
            'prediction_distribution': prediction_distribution
        }
        
        logger.debug(
            f"Grade {grade_label}: exact={exact_acc:.1f}%, "
            f"±1={tolerance_1:.1f}%, ±2={tolerance_2:.1f}%"
        )
    
    # Convert numeric labels to grade names for per_grade results
    from moonboard_core.grade_encoder import decode_grade
    
    per_grade_with_names = {}
    for grade_label, stats in results_per_grade.items():
        # Use global decoder directly
        grade_name = decode_grade(int(grade_label))
        
        # Convert prediction_distribution numeric labels to grade names
        if 'prediction_distribution' in stats:
            pred_dist_with_names = {}
            for pred_label, count in stats['prediction_distribution'].items():
                pred_grade_name = decode_grade(int(pred_label))
                pred_dist_with_names[pred_grade_name] = count
            stats['prediction_distribution'] = pred_dist_with_names
        
        per_grade_with_names[grade_name] = stats
    
    # Calculate overall metrics
    valid_grades = [r for r in results_per_grade.values() if not r.get('skipped', False)]
    
    if len(valid_grades) == 0:
        return {
            'error': 'No valid results for any grade',
            'skipped': True,
            'per_grade': per_grade_with_names
        }
    
    all_exact = [r['exact_accuracy'] for r in valid_grades]
    all_tol1 = [r['tolerance_1_accuracy'] for r in valid_grades]
    all_tol2 = [r['tolerance_2_accuracy'] for r in valid_grades]
    
    return {
        'overall_exact_accuracy': float(np.mean(all_exact)),
        'overall_std_exact': float(np.std(all_exact)),
        'overall_tolerance_1_accuracy': float(np.mean(all_tol1)),
        'overall_std_tolerance_1': float(np.std(all_tol1)),
        'overall_tolerance_2_accuracy': float(np.mean(all_tol2)),
        'overall_std_tolerance_2': float(np.std(all_tol2)),
        'num_grades_evaluated': len(valid_grades),
        'total_grades': num_grades,
        'per_grade': per_grade_with_names,
        'warnings': [
            'Classifier baseline accuracy: ~35% exact, ~70% ±1 grade',
            'Results limited by classifier performance',
            'Use for relative comparison between models only',
            'Low absolute scores may reflect classifier weakness, not generator issues'
        ],
        'interpretation': 'CAUTION: Limited reliability metric - compare models relatively, not absolutely'
    }

