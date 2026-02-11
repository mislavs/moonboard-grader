"""
Shared utility functions for evaluation metrics.
"""

from typing import Dict

from src.dataset import create_data_loaders
from src.label_space import EvaluationLabelContext


def load_data_loader(
    data_path: str,
    label_context: EvaluationLabelContext,
    batch_size: int = 32,
):
    """
    Helper to load validation data loader and dataset.
    
    Args:
        data_path: Path to dataset JSON file
        batch_size: Batch size for data loader
    Returns:
        Tuple of (val_loader, dataset) where dataset contains grade mappings
    """
    if data_path is None:
        raise ValueError("data_path is required for this metric")

    min_grade_index, max_grade_index = label_context.get_global_grade_bounds()
    
    _, val_loader, dataset = create_data_loaders(
        data_path=data_path,
        batch_size=batch_size,
        train_split=0.8,
        shuffle=False,
        num_workers=0,
        min_grade_index=min_grade_index,
        max_grade_index=max_grade_index,
        grade_offset=label_context.grade_offset,
    )
    
    return val_loader, dataset


def extract_problem_stats(problem: Dict) -> Dict[str, float]:
    """
    Helper to extract statistics from a problem.
    
    Args:
        problem: Problem dictionary with 'moves' or 'Moves' field
        
    Returns:
        Dictionary of statistics
    """
    moves = problem.get('moves') or problem.get('Moves', [])
    
    num_holds = len(moves)
    num_start = sum(1 for m in moves if m.get('isStart') or m.get('IsStart'))
    num_end = sum(1 for m in moves if m.get('isEnd') or m.get('IsEnd'))
    num_middle = num_holds - num_start - num_end
    
    # Vertical spread (range of row numbers)
    positions = [m.get('description') or m.get('Description') for m in moves]
    rows = [int(pos[1:]) for pos in positions if pos and len(pos) > 1]
    vertical_spread = max(rows) - min(rows) if rows else 0
    
    return {
        'num_holds': num_holds,
        'num_start': num_start,
        'num_end': num_end,
        'num_middle': num_middle,
        'vertical_spread': vertical_spread
    }

