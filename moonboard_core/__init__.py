"""
Moonboard Core Package

Shared utilities for processing Moonboard climbing problems.
This package provides:
- Grade encoding/decoding (Font grades)
- Position parsing (e.g., "F7" -> (row, col))
- Grid tensor building (moves -> 3x18x11 tensors)
- Data loading and processing pipeline
"""

from .grade_encoder import (
    encode_grade,
    decode_grade,
    get_all_grades,
    get_num_grades,
    remap_label,
    unmap_label,
    get_filtered_grade_names,
)

from .position_parser import (
    parse_position,
    validate_position,
    ROWS,
    COLS,
    COLUMNS,
)

from .grid_builder import (
    create_grid_tensor,
    get_channel_counts,
    tensor_to_moves,
)

from .data_processor import (
    process_problem,
    load_dataset,
    get_dataset_stats,
    save_processed_dataset,
    load_processed_dataset,
    filter_dataset_by_grades,
)

__version__ = "0.1.0"

__all__ = [
    # Grade encoding
    "encode_grade",
    "decode_grade",
    "get_all_grades",
    "get_num_grades",
    "remap_label",
    "unmap_label",
    "get_filtered_grade_names",
    # Position parsing
    "parse_position",
    "validate_position",
    "ROWS",
    "COLS",
    "COLUMNS",
    # Grid building
    "create_grid_tensor",
    "get_channel_counts",
    "tensor_to_moves",
    # Data processing
    "process_problem",
    "load_dataset",
    "get_dataset_stats",
    "save_processed_dataset",
    "load_processed_dataset",
    "filter_dataset_by_grades",
]

