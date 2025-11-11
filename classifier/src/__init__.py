"""
Moonboard Grade Prediction Neural Network

A PyTorch-based classification system for predicting climbing grades
from Moonboard hold positions.
"""

__version__ = "0.1.0"

# Import main modules
from . import dataset
from . import data_splitter
from . import models
from . import advanced_models
from . import losses
from . import trainer
from . import evaluator
from . import predictor

# Import commonly used functions from moonboard_core
from moonboard_core.grade_encoder import (
    encode_grade,
    decode_grade,
    get_all_grades,
    get_num_grades,
    remap_label,
    unmap_label,
    get_filtered_grade_names
)

from moonboard_core.position_parser import (
    parse_position,
    validate_position,
    ROWS,
    COLS,
    COLUMNS
)

from moonboard_core.grid_builder import (
    create_grid_tensor,
    get_channel_counts,
    tensor_to_moves
)

from moonboard_core.data_processor import (
    process_problem,
    load_dataset,
    get_dataset_stats,
    save_processed_dataset,
    load_processed_dataset,
    filter_dataset_by_grades
)

from .dataset import (
    MoonboardDataset,
    create_data_splits,
    get_split_info
)

from .data_splitter import (
    create_stratified_splits,
    create_datasets,
    create_data_loaders
)

from .advanced_models import (
    ResidualCNN,
    DeepResidualCNN,
    ResidualBlock,
    SpatialAttention,
    ChannelAttention,
    create_advanced_model
)

from .losses import (
    FocalLoss,
    OrdinalCrossEntropyLoss,
    FocalOrdinalLoss,
    LabelSmoothingCrossEntropy,
    create_loss_function
)

from .models import (
    FullyConnectedModel,
    ConvolutionalModel,
    create_model,
    count_parameters
)

from .trainer import (
    Trainer
)

from .evaluator import (
    evaluate_model,
    calculate_exact_accuracy,
    calculate_tolerance_accuracy,
    generate_confusion_matrix,
    plot_confusion_matrix,
    per_grade_metrics,
    calculate_mean_absolute_error,
    get_metrics_summary
)

from .predictor import (
    Predictor
)

__all__ = [
    # Modules
    'dataset',
    'data_splitter',
    'models',
    'advanced_models',
    'losses',
    'trainer',
    'evaluator',
    'predictor',
    'cli',
    # Grade encoder functions (from moonboard_core)
    'encode_grade',
    'decode_grade',
    'get_all_grades',
    'get_num_grades',
    'remap_label',
    'unmap_label',
    'get_filtered_grade_names',
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
    'filter_dataset_by_grades',
    # Dataset functions
    'MoonboardDataset',
    'create_data_splits',
    'get_split_info',
    # Data splitter functions
    'create_stratified_splits',
    'create_datasets',
    'create_data_loaders',
    # Model functions
    'FullyConnectedModel',
    'ConvolutionalModel',
    'create_model',
    'count_parameters',
    # Advanced model functions
    'ResidualCNN',
    'DeepResidualCNN',
    'ResidualBlock',
    'SpatialAttention',
    'ChannelAttention',
    'create_advanced_model',
    # Loss functions
    'FocalLoss',
    'OrdinalCrossEntropyLoss',
    'FocalOrdinalLoss',
    'LabelSmoothingCrossEntropy',
    'create_loss_function',
    # Trainer functions
    'Trainer',
    # Evaluator functions
    'evaluate_model',
    'calculate_exact_accuracy',
    'calculate_tolerance_accuracy',
    'generate_confusion_matrix',
    'plot_confusion_matrix',
    'per_grade_metrics',
    'calculate_mean_absolute_error',
    'get_metrics_summary',
    # Predictor functions
    'Predictor',
]
