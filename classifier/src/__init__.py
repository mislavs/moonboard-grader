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
from . import dataset
from . import data_splitter
from . import augmentation
from . import advanced_augmentation
from . import models
from . import advanced_models
from . import losses
from . import trainer
from . import evaluator
from . import predictor

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

from .dataset import (
    MoonboardDataset,
    create_data_splits,
    get_split_info
)

from .data_splitter import (
    create_stratified_splits,
    create_datasets_with_augmentation,
    create_data_loaders
)

from .augmentation import (
    MoonboardAugmentation,
    create_augmentation,
    no_augmentation
)

from .advanced_augmentation import (
    AdvancedMoonboardAugmentation,
    MixUpAugmentation,
    CutMixAugmentation,
    create_augmentation_pipeline
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
    'grade_encoder',
    'position_parser',
    'grid_builder',
    'data_processor',
    'dataset',
    'data_splitter',
    'augmentation',
    'advanced_augmentation',
    'models',
    'advanced_models',
    'losses',
    'trainer',
    'evaluator',
    'predictor',
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
    # Dataset functions
    'MoonboardDataset',
    'create_data_splits',
    'get_split_info',
    # Data splitter functions
    'create_stratified_splits',
    'create_datasets_with_augmentation',
    'create_data_loaders',
    # Augmentation functions
    'MoonboardAugmentation',
    'create_augmentation',
    'no_augmentation',
    # Advanced augmentation functions
    'AdvancedMoonboardAugmentation',
    'MixUpAugmentation',
    'CutMixAugmentation',
    'create_augmentation_pipeline',
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
