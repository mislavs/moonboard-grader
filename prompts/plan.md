# Moonboard Grade Prediction Neural Network

## Overview

Build a classification neural network that takes moonboard hold positions as input and predicts the Font scale grade. We'll use Python with PyTorch, implementing each component incrementally with unit tests using pytest.

## Architecture Approach

- Represent moonboard as 11x18 grid (columns A-K, rows 1-18)
- Use 3-channel tensor: start holds, middle holds, end holds
- Classification problem: predict one of discrete Font grades (5+, 6A, 6A+, 6B, etc.)

## Tech Stack

- **Python 3.10+**, **PyTorch**, **pandas**, **scikit-learn**, **matplotlib**, **numpy**, **pytest**

## Implementation Steps

### Step 1: Project Setup ✅

**Files**:  `neural-net/README.md`, directory structure

Create `neural-net` folder inside `moonboard-grader` with all necessary subdirectories (src/, tests/, data/, models/). Install PyTorch, pytest, pandas, scikit-learn, matplotlib, numpy. Document installation in README.

**Status**: COMPLETED
- Created directory structure: src/, tests/, data/, models/
- Created README.md with comprehensive installation instructions
- Created requirements.txt with all dependencies (torch, numpy, pandas, scikit-learn, matplotlib, pyyaml, pytest, etc.)
- Created __init__.py files for Python packages
- Added .gitignore for Python/PyTorch projects
- Added data/README.md with data format documentation

---

### Step 2: Grade Encoder ✅

**Files**: `src/grade_encoder.py`, `tests/test_grade_encoder.py`

Create utility to convert Font grades (strings like "6B+") to integer labels and back. Should handle all Font grades from 5+ to 8C+, be case-insensitive, and raise errors for invalid grades.

**Functions Needed**:

- `encode_grade(grade_str)`: string → int
- `decode_grade(label)`: int → string  
- `get_all_grades()`: return ordered list of valid grades
- `get_num_grades()`: return total number of grade classes

**Unit Tests**: Test all valid grades, invalid input handling, bidirectional conversion, ordering

**Status**: COMPLETED
- Implemented grade_encoder.py with all 4 required functions
- Font grades from 5+ to 8C+ (19 grades total)
- Case-insensitive encoding with normalization
- Comprehensive error handling for invalid inputs
- Created test_grade_encoder.py with 26 tests covering:
  - Valid grade conversions
  - Case insensitivity and whitespace handling
  - Invalid input validation
  - Bidirectional conversion consistency
  - Grade ordering verification
  - Edge cases and boundaries
- All tests passing (26/26) ✓

---

### Step 3: Position Parser ✅

**Files**: `src/position_parser.py`, `tests/test_position_parser.py`

Parse hold position strings (like "F7") into row/column indices. Columns A-K map to 0-10, rows 1-18 map to 0-17.

**Functions Needed**:

- `parse_position(position_str)`: "F7" → (row, col) tuple
- `validate_position(position_str)`: return bool
- Constants: `ROWS = 18`, `COLS = 11`, `COLUMNS = 'ABCDEFGHIJK'`

**Unit Tests**: Test all valid positions, invalid positions (Z1, A20), edge cases (lowercase, whitespace)

**Status**: COMPLETED
- Implemented position_parser.py with all required functions
- Constants defined: ROWS=18, COLS=11, COLUMNS='ABCDEFGHIJK'
- `parse_position()` converts position strings to (row, col) tuples with 0-based indexing
- Handles case-insensitivity and whitespace normalization
- Comprehensive error handling for invalid inputs
- `validate_position()` returns boolean for validation checks
- Created test_position_parser.py with 45 tests covering:
  - All valid corner positions (A1, K1, A18, K18)
  - All columns A-K and rows 1-18
  - Case-insensitivity (lowercase, mixed case)
  - Whitespace handling (leading, trailing, tabs)
  - Invalid columns (Z, L), invalid rows (0, 19, 20, negative)
  - Invalid formats (empty, single char, no number)
  - Invalid types (int, None, list)
  - Edge cases and boundary conditions
- All tests passing (45/45) ✓

---

### Step 4: Grid Tensor Builder ✅

**Files**: `src/grid_builder.py`, `tests/test_grid_builder.py`

Convert a problem's moves array (from JSON) into a 3x11x18 numpy array where channel 0 = start holds, channel 1 = middle holds, channel 2 = end holds. Each position is binary (1 if hold present, 0 otherwise).

**Functions Needed**:

- `create_grid_tensor(moves_list)`: takes list of move dicts → returns (3, 11, 18) numpy array
- Helper to determine which channel based on `isStart` and `isEnd` flags

**Unit Tests**: Test with example.json moves, verify shape, verify correct channels, test empty moves, test overlapping positions

**Status**: COMPLETED
- Implemented grid_builder.py with 3 main functions
- `create_grid_tensor()` converts moves list to (3, 18, 11) numpy array
  - Channel 0: start holds (isStart=True)
  - Channel 1: middle holds (isStart=False, isEnd=False)
  - Channel 2: end holds (isEnd=True)
  - Returns float32 arrays with binary values (0.0 or 1.0)
- `get_channel_counts()` returns count of holds in each channel
- `tensor_to_moves()` converts tensor back to position strings (for debugging)
- Comprehensive error handling for invalid inputs
- Tested with example.json problem data
- Created test_grid_builder.py with 39 tests covering:
  - Empty moves, single holds, complete problems
  - All three channel types (start, middle, end)
  - Multi-start and multi-end scenarios
  - Holds that are both start and end (single-move problems)
  - Case-insensitivity and whitespace handling
  - Invalid input validation (missing fields, wrong types)
  - Invalid positions and out-of-bounds checks
  - Helper function validation
  - Round-trip conversion consistency
  - Integration tests with full workflow
- All tests passing (39/39) ✓
- Updated src/__init__.py to export grid_builder functions
- Total test suite: 110 tests passing ✓

---

### Step 5: Data Processor Pipeline ✅

**Files**: `src/data_processor.py`, `tests/test_data_processor.py`

Combine previous components into end-to-end processing. Load JSON file(s), process each problem into (tensor, label) pairs, provide dataset statistics.

**Functions Needed**:

- `process_problem(problem_dict)`: single problem JSON → (numpy array, int label)
- `load_dataset(json_path)`: load JSON file → list of (tensor, label) pairs
- `get_dataset_stats(problems)`: return dict with grade distribution, total count, etc.

**Unit Tests**: Test with example.json, test with mock multi-problem dataset, test malformed JSON handling

**Status**: COMPLETED
- Implemented data_processor.py with 5 main functions
- `process_problem()` combines grade encoding and grid building
  - Takes problem dict with 'grade' and 'moves' fields
  - Returns (tensor, label) tuple
  - Comprehensive error handling for invalid problems
- `load_dataset()` loads and processes JSON files
  - Supports standard JSON format with 'data' array
  - Processes all problems with detailed error reporting
  - Returns list of (tensor, label) tuples
- `get_dataset_stats()` calculates dataset statistics
  - Total problems and grade distribution
  - Average/min/max hold counts per channel
  - Returns JSON-serializable dictionary
- `save_processed_dataset()` saves datasets to .npz format
  - Compressed numpy format for efficient storage
  - Stacks tensors and labels into arrays
- `load_processed_dataset()` loads saved datasets
  - Validates file structure and shapes
  - Returns list of (tensor, label) tuples
- Successfully tested with example.json
- Created test_data_processor.py with 43 tests covering:
  - Single and multiple problem processing
  - Empty datasets and edge cases
  - JSON loading and error handling (invalid JSON, missing fields, wrong types)
  - Dataset statistics calculation
  - Save/load round-trip consistency
  - File system error handling
  - Complete end-to-end integration pipeline
- All tests passing (43/43) ✓
- Updated src/__init__.py to export data_processor functions
- Total test suite: 153 tests passing ✓

---

### Step 6: PyTorch Dataset Class ✅

**Files**: `src/dataset.py`, `tests/test_dataset.py`

Create PyTorch Dataset class that wraps the processed data. Implement stratified train/val/test splitting to preserve grade distribution across splits.

**Classes/Functions Needed**:

- `MoonboardDataset(data, labels)`: PyTorch Dataset with `__len__` and `__getitem__`
- `create_data_splits(data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)`: returns 3 MoonboardDataset objects
- Use scikit-learn's `StratifiedShuffleSplit` for stratification

**Unit Tests**: Test dataset length, test getitem returns (tensor, int), test DataLoader compatibility, verify split ratios, verify stratification

**Status**: COMPLETED
- Implemented dataset.py with 3 main classes/functions
- `MoonboardDataset` class inherits from PyTorch Dataset
  - Accepts numpy arrays of shape (N, 3, 18, 11) for data and (N,) for labels
  - Handles both 3D single samples and 4D batches
  - Returns (torch.FloatTensor, int) tuples from __getitem__
  - Includes get_label_distribution() helper method
  - Comprehensive validation for shapes, types, and data integrity
- `create_data_splits()` function for stratified splitting
  - Uses scikit-learn's StratifiedShuffleSplit for proper stratification
  - Supports custom train/val/test ratios (default 0.7/0.15/0.15)
  - Two-stage splitting to preserve grade distribution across all splits
  - Validates ratios and minimum class counts for stratification
  - Reproducible with random_state parameter
- `get_split_info()` helper function for split analysis
  - Returns total counts, sizes, ratios, and label distributions
  - Useful for verifying stratification quality
- Created test_dataset.py with 35 tests covering:
  - Dataset initialization (numpy arrays, lists, single samples)
  - __len__ and __getitem__ functionality
  - PyTorch DataLoader compatibility and iteration
  - Label distribution calculation
  - Input validation (shapes, types, lengths)
  - Data type conversions (float32 for data, int64 for labels)
  - Stratified split creation with various ratios
  - Reproducibility with random states
  - Stratification verification
  - Edge cases (small datasets, imbalanced classes)
  - Complete integration workflows
- All tests passing (35/35) ✓
- Updated src/__init__.py to export dataset functions
- Total test suite: 188 tests (188 passing) ✓

---

### Step 7: Model Architectures ✅

**Files**: `src/models.py`, `tests/test_models.py`

Implement two PyTorch model architectures for comparison. Both should accept (batch, 3, 18, 11) input and output (batch, num_classes) logits.

**Model 1 - Fully Connected**:

- Flatten input → Linear(594, 256) → ReLU → Dropout(0.3) → Linear(256, 128) → ReLU → Dropout(0.3) → Linear(128, num_classes)

**Model 2 - Convolutional**:

- Conv2d(3→16, kernel=3, padding=1) → ReLU → MaxPool2d(2) → Conv2d(16→32, kernel=3, padding=1) → ReLU → MaxPool2d(2) → Flatten → Linear(256→128) → ReLU → Dropout(0.5) → Linear(128, num_classes)

**Unit Tests**: Test initialization, test forward pass with dummy batch, verify output shape, test backward pass works, test save/load state dict

**Status**: COMPLETED
- Implemented models.py with two model architectures
- `FullyConnectedModel` class with 3-layer MLP architecture
  - Input flattening from (batch, 3, 18, 11) to (batch, 594)
  - Hidden layers: 594→256→128→num_classes
  - ReLU activations and 0.3 dropout between layers
  - Total parameters: 187,667 (for 19 classes)
- `ConvolutionalModel` class with CNN architecture
  - Two conv layers with ReLU and max pooling
  - Spatial dimensions: 18×11 → 9×5 → 4×2
  - Flattened size: 32×4×2 = 256
  - FC layers: 256→128→num_classes
  - Total parameters: 40,435 (for 19 classes)
- `create_model()` factory function for easy model creation
- `count_parameters()` helper function for model analysis
- Created test_models.py with 41 comprehensive tests covering:
  - Model initialization (default and custom num_classes)
  - Forward pass shapes (single, batch, large batch)
  - Output properties (logits, not probabilities)
  - Backward pass and gradient flow
  - Model save/load (state_dict and entire model)
  - Training vs eval modes (dropout behavior)
  - Factory function and parameter counting
  - Device movement (CPU/CUDA)
  - Binary input (realistic hold positions)
  - Deterministic behavior in eval mode
  - Edge cases (all zeros, all ones)
- All tests passing (39 passed, 2 skipped for CUDA) ✓
- Updated src/__init__.py to export model functions
- Total test suite: 229 tests (227 passing, 2 skipped) ✓

---

### Step 8: Training Loop ✅

**Files**: `src/trainer.py`, `tests/test_trainer.py`

Create Trainer class to manage training loop, validation, checkpointing, and early stopping.

**Trainer Class Should Have**:

- `__init__(model, train_loader, val_loader, optimizer, criterion, device, checkpoint_dir)`
- `train_epoch()`: one epoch of training, return avg loss
- `validate_epoch()`: one epoch of validation, return avg loss and accuracy
- `fit(num_epochs, early_stopping_patience)`: full training loop
- Save best model based on validation loss
- Track metrics history (losses, accuracies)

**Configuration**: Use CrossEntropyLoss, Adam optimizer (lr=0.001), batch size=32

**Unit Tests**: Test training on tiny dataset (10 samples), verify loss decreases, test checkpoint saving/loading, test early stopping logic

**Status**: COMPLETED
- Implemented trainer.py with comprehensive Trainer class
- `Trainer.__init__()` initializes model, data loaders, optimizer, criterion, device, and checkpoint directory
  - Automatically creates checkpoint directory if it doesn't exist
  - Initializes tracking variables (history, best_val_loss, epochs_without_improvement)
  - Comprehensive input validation for all parameters
- `train_epoch()` performs one epoch of training
  - Sets model to training mode
  - Iterates through batches, computes loss, backpropagates, updates weights
  - Returns average training loss for the epoch
- `validate_epoch()` performs one epoch of validation
  - Sets model to evaluation mode
  - Computes loss and accuracy without gradients
  - Returns tuple of (average validation loss, validation accuracy)
  - Handles case when val_loader is None
- `fit()` manages full training loop
  - Trains for specified number of epochs
  - Tracks metrics in history dictionary
  - Implements early stopping based on validation loss
  - Saves best model checkpoint when validation loss improves
  - Saves final model checkpoint after training completes
  - Optional verbose output for training progress
  - Returns history dictionary with all metrics
- Checkpoint management methods:
  - `save_checkpoint()` saves model state, optimizer state, epoch, and history
  - `load_checkpoint()` restores complete training state from checkpoint
- History management methods:
  - `get_history()` returns deep copy of training history
  - `save_history()` exports history to JSON file
- Early stopping features:
  - Tracks epochs without improvement in validation loss
  - Stops training if no improvement for specified patience epochs
  - Only triggers when validation data is provided
  - Configurable patience parameter (None disables early stopping)
- Created test_trainer.py with 34 comprehensive tests covering:
  - Initialization (with/without validation, directory creation, input validation)
  - Training epoch (loss computation, train mode, gradient computation)
  - Validation epoch (metrics calculation, eval mode, no gradients)
  - Full training with fit() (basic training, loss decrease, model saving)
  - Early stopping logic (triggering, tracking, disabling)
  - Checkpoint save/load (file creation, state restoration, weight preservation)
  - History tracking (structure, accuracy, deep copying, JSON export)
  - Different models (FC, CNN) and optimizers (Adam, SGD)
  - Verbose output control
- All tests passing (34/34) ✓
- Updated src/__init__.py to export Trainer class
- Total test suite: 263 tests (261 passing, 2 skipped for CUDA) ✓

---

### Step 9: Evaluation Metrics ✅

**Files**: `src/evaluator.py`, `tests/test_evaluator.py`

Implement comprehensive evaluation metrics and visualization for model performance.

**Functions Needed**:

- `evaluate_model(model, dataloader, device)`: returns dict with metrics
- `calculate_exact_accuracy(predictions, labels)`: percentage of exact matches
- `calculate_tolerance_accuracy(predictions, labels, tolerance=1)`: percentage within ±tolerance grades
- `generate_confusion_matrix(predictions, labels)`: confusion matrix as numpy array
- `plot_confusion_matrix(cm, grade_names, save_path)`: matplotlib visualization
- `per_grade_metrics(predictions, labels)`: precision/recall/f1 per grade

**Unit Tests**: Test metric calculations with known predictions/labels, test confusion matrix generation, test tolerance accuracy logic

**Status**: COMPLETED
- Implemented evaluator.py with 8 comprehensive evaluation functions
- `evaluate_model()` performs full model evaluation on datasets
  - Returns exact accuracy, tolerance accuracies (±1, ±2), loss, predictions, labels
  - Automatically sets model to eval mode
  - Works with any PyTorch DataLoader
- `calculate_exact_accuracy()` measures perfect grade predictions
- `calculate_tolerance_accuracy()` measures predictions within ±N grades
  - Useful for climbing since ±1 grade is still very valuable
- `calculate_mean_absolute_error()` shows average grade offset
- `generate_confusion_matrix()` creates confusion matrices using sklearn
- `plot_confusion_matrix()` generates beautiful visualizations
  - Supports normalized (percentage) and count modes
  - Customizable figure size and save paths
  - Uses seaborn heatmaps for clarity
- `per_grade_metrics()` calculates precision/recall/F1 per grade
  - Uses sklearn's precision_recall_fscore_support
  - Includes support counts for each grade
- `get_metrics_summary()` provides comprehensive evaluation in one call
- Created test_evaluator.py with 46 tests covering:
  - Exact and tolerance accuracy calculations
  - Confusion matrix generation and visualization
  - Per-grade metrics (precision, recall, F1, support)
  - Mean absolute error calculations
  - Full model evaluation on PyTorch DataLoaders
  - Edge cases (empty data, mismatched lengths, invalid inputs)
  - Matplotlib plotting with non-interactive backend (Agg) for Windows compatibility
- All tests passing (46/46) ✓
- Updated src/__init__.py to export evaluator functions
- Total test suite: 309 tests (307 passing, 2 skipped for CUDA) ✓

---

### Step 10: Inference Interface ✅

**Files**: `src/predictor.py`, `tests/test_predictor.py`

Create easy-to-use prediction interface that loads a trained model and makes predictions on new problems.

**Predictor Class Should Have**:

- `__init__(checkpoint_path, device='cpu')`: load trained model
- `predict(problem_json)`: single problem dict → return dict with predicted_grade, confidence, all_probabilities
- `predict_batch(problems_list)`: list of problems → list of prediction dicts
- Handle CPU/GPU device management

**Unit Tests**: Test loading model, test prediction on example.json, test batch predictions, verify output format

**Status**: COMPLETED
- Implemented predictor.py with comprehensive Predictor class
- `Predictor.__init__()` loads trained models from checkpoint files
  - Automatically infers model architecture (FC or CNN) from state dict keys
  - Supports both CPU and CUDA devices with validation
  - Sets model to eval mode automatically
  - Validates checkpoint structure and file existence
- `predict()` makes predictions on single problems
  - Takes problem dict with 'moves' field
  - Returns predicted grade, confidence, all probabilities, and top-k predictions
  - Handles custom top_k parameter for returning multiple predictions
  - Comprehensive error handling for invalid inputs
- `predict_batch()` makes predictions on multiple problems
  - Processes list of problem dicts
  - Returns list of prediction dictionaries
  - Individual error reporting for failed problems
- `predict_from_tensor()` makes predictions from pre-processed tensors
  - Accepts both numpy arrays and torch tensors
  - Handles single samples and batches
  - Useful when tensor is already available
- `get_model_info()` returns model metadata
  - Model type (FullyConnected or Convolutional)
  - Number of parameters and classes
  - Device and checkpoint path information
- Architecture inference from state dict keys:
  - FC model: checks for 'network.7.weight' (final Sequential layer)
  - CNN model: checks for 'fc2.weight' (final linear layer)
- Created test_predictor.py with 35 comprehensive tests covering:
  - Initialization (FC/CNN models, Path objects, device validation)
  - Checkpoint validation (file existence, structure, malformed files)
  - Single predictions (basic usage, different top_k, consistency)
  - Batch predictions (single/multiple problems, custom top_k)
  - Tensor predictions (numpy/torch, batches, consistency with dict input)
  - Model info retrieval (metadata, parameter counts)
  - Edge cases (empty problems, single holds, many holds, probability validation)
  - CUDA device support (when available)
- All tests passing (35/35, 2 skipped for CUDA) ✓
- Updated src/__init__.py to export Predictor class
- Total test suite: 346 tests (342 passing, 4 skipped) ✓

---

### Step 11: Main CLI Script ✅

**Files**: `main.py`, `config.yaml`

Create command-line interface for training, evaluation, and prediction.

**Commands**:

- `python main.py train --config config.yaml`: train new model
- `python main.py evaluate --checkpoint path/to/model.pth --data data/problems.json`: evaluate model
- `python main.py predict --checkpoint path/to/model.pth --input problem.json`: predict single problem

**config.yaml Should Include**: model type (fc/cnn), learning rate, batch size, epochs, early stopping patience, train/val/test ratios, data path, checkpoint directory

**Status**: COMPLETED
- Implemented main.py with comprehensive CLI interface
- Three main commands: train, evaluate, predict
- `train` command:
  - Loads config from YAML file
  - Creates train/val/test splits with stratification
  - Trains model with early stopping
  - Saves best and final checkpoints
  - Saves training history and confusion matrix
  - Supports both FC and CNN models
  - Supports Adam and SGD optimizers
  - Verbose output with emojis and progress tracking
- `evaluate` command:
  - Loads trained model from checkpoint
  - Evaluates on provided dataset
  - Shows exact, ±1, and ±2 grade accuracy
  - Displays per-grade precision/recall/F1
  - Optional confusion matrix visualization
- `predict` command:
  - Makes predictions on single or multiple problems
  - Returns top-K predictions with probabilities
  - Shows comparison with actual grade if available
  - Optional JSON output export
  - Supports both single problem and batch mode
- Created config.yaml with sensible defaults
  - Model configuration (type, num_classes)
  - Training hyperparameters (lr, batch_size, epochs, optimizer)
  - Data configuration (paths, split ratios, random seed)
  - Checkpoint settings (directory, save options)
  - Device configuration (cuda/cpu with auto-fallback)
  - Evaluation settings (tolerance levels, confusion matrix)
  - Prediction settings (top_k)
- Created test_main.py with 18 comprehensive tests covering:
  - Configuration loading (valid, missing, full config)
  - Train command (config loading, missing data, actual training)
  - Evaluate command (missing checkpoint, missing data)
  - Predict command (missing files, invalid formats, actual predictions)
  - CLI integration (help messages, subcommands, argument parsing)
  - Device handling (CUDA fallback to CPU)
  - Output generation (JSON export, file creation)
- All tests passing (18/18) ✓
- Total test suite: 364 tests (360 passing, 4 skipped for CUDA) ✓

---

### Step 12: Technical Specification ✅

**File**: `spec.md`

Document the complete system with technical details for future reference.

**Sections**:

- Problem statement and objectives
- Data representation (grid encoding, channels, grade labels)
- Model architectures (detailed layer specs for both models)
- Training methodology (loss function, optimizer, hyperparameters, early stopping)
- Evaluation metrics (accuracy types, confusion matrix interpretation)
- API specification (input/output formats for prediction)
- Performance benchmarks (results on your dataset)
- Known limitations and future improvements

**Status**: COMPLETED
- Created comprehensive spec.md (450+ lines)
- **Section 1: Problem Statement** - Overview, objectives, and use cases
- **Section 2: Data Representation** - Moonboard grid (11×18), 3-channel tensor encoding, Font grade labels (19 classes)
- **Section 3: Model Architectures** - Detailed FC and CNN specifications with parameter counts (187,667 and 40,435 params)
- **Section 4: Training Methodology** - Cross-entropy loss, Adam optimizer, hyperparameters, early stopping, checkpointing
- **Section 5: Evaluation Metrics** - Exact accuracy, tolerance accuracy (±1, ±2), MAE, per-grade metrics, confusion matrix interpretation
- **Section 6: API Specification** - Complete Predictor class API, input/output formats, CLI commands with examples
- **Section 7: Performance Benchmarks** - Expected accuracy ranges, training times, computational requirements
- **Section 8: Limitations & Improvements** - 4 current limitations, 15+ future enhancement ideas including:
  - Enhanced input features (hold types, wall angle, sequences)
  - Advanced architectures (GNN, attention, multi-task learning)
  - Data augmentation and uncertainty quantification
  - Explainability and deployment enhancements
- **Section 9: Implementation Notes** - Code organization, dependencies, reproducibility, best practices
- **Section 10: Conclusion** - System summary and key takeaways

---

## Project Structure

```
moonboard-grader/neural-net
├── README.md
├── spec.md
├── config.yaml
├── main.py
├── data/
│   └── problems.json
├── src/
│   ├── __init__.py
│   ├── grade_encoder.py
│   ├── position_parser.py
│   ├── grid_builder.py
│   ├── data_processor.py
│   ├── dataset.py
│   ├── models.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── predictor.py
├── tests/
│   ├── __init__.py
│   └── test_*.py (one per module)
└── models/
    └── (checkpoints saved here)
```

## Key Design Decisions

**Why 3-channel representation**: Separating start/middle/end holds gives the network explicit information about problem structure

**Why two model types**: FC is simpler baseline, CNN can learn spatial patterns (hold proximity matters in climbing)

**Why tolerance metrics**: Predicting exactly right is hard; ±1 grade is still very useful

**Why stratified splits**: Preserves grade distribution in train/val/test, important for imbalanced data

**Why pytest**: Standard in Python ML projects, good fixture support, easy to run subsets of tests