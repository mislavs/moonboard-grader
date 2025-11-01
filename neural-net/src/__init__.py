"""
Moonboard Grade Prediction Neural Network

A PyTorch-based classification system for predicting climbing grades
from Moonboard hold positions.
"""

__version__ = "0.1.0"

# Import main modules
from . import grade_encoder
from . import position_parser
from . import grid_builder
from . import data_processor

# Import commonly used functions
from .grade_encoder import (
    encode_grade,
    decode_grade,
    get_all_grades,
    get_num_grades
)

from .position_parser import (
    parse_position,
    validate_position,
    ROWS,
    COLS,
    COLUMNS
)

from .grid_builder import (
    create_grid_tensor,
    get_channel_counts,
    tensor_to_moves
)

from .data_processor import (
    process_problem,
    load_dataset,
    get_dataset_stats,
    save_processed_dataset,
    load_processed_dataset
)

__all__ = [
    # Modules
    'grade_encoder',
    'position_parser',
    'grid_builder',
    'data_processor',
    # Grade encoder functions
    'encode_grade',
    'decode_grade',
    'get_all_grades',
    'get_num_grades',
    # Position parser functions
    'parse_position',
    'validate_position',
    'ROWS',
    'COLS',
    'COLUMNS',
    # Grid builder functions
    'create_grid_tensor',
    'get_channel_counts',
    'tensor_to_moves',
    # Data processor functions
    'process_problem',
    'load_dataset',
    'get_dataset_stats',
    'save_processed_dataset',
    'load_processed_dataset',
]
