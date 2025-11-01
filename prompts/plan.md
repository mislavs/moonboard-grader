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

### Step 6: PyTorch Dataset Class

**Files**: `src/dataset.py`, `tests/test_dataset.py`

Create PyTorch Dataset class that wraps the processed data. Implement stratified train/val/test splitting to preserve grade distribution across splits.

**Classes/Functions Needed**:

- `MoonboardDataset(data, labels)`: PyTorch Dataset with `__len__` and `__getitem__`
- `create_data_splits(data, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)`: returns 3 MoonboardDataset objects
- Use scikit-learn's `StratifiedShuffleSplit` for stratification

**Unit Tests**: Test dataset length, test getitem returns (tensor, int), test DataLoader compatibility, verify split ratios, verify stratification

---

### Step 7: Model Architectures

**Files**: `src/models.py`, `tests/test_models.py`

Implement two PyTorch model architectures for comparison. Both should accept (batch, 3, 11, 18) input and output (batch, num_classes) logits.

**Model 1 - Fully Connected**:

- Flatten input → Linear(594, 256) → ReLU → Dropout(0.3) → Linear(256, 128) → ReLU → Dropout(0.3) → Linear(128, num_classes)

**Model 2 - Convolutional**:

- Conv2d(3→16, kernel=3, padding=1) → ReLU → MaxPool2d(2) → Conv2d(16→32, kernel=3, padding=1) → ReLU → MaxPool2d(2) → Flatten → Linear(→128) → ReLU → Dropout(0.5) → Linear(128, num_classes)

**Unit Tests**: Test initialization, test forward pass with dummy batch, verify output shape, test backward pass works, test save/load state dict

---

### Step 8: Training Loop

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

---

### Step 9: Evaluation Metrics

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

---

### Step 10: Inference Interface

**Files**: `src/predictor.py`, `tests/test_predictor.py`

Create easy-to-use prediction interface that loads a trained model and makes predictions on new problems.

**Predictor Class Should Have**:

- `__init__(checkpoint_path, device='cpu')`: load trained model
- `predict(problem_json)`: single problem dict → return dict with predicted_grade, confidence, all_probabilities
- `predict_batch(problems_list)`: list of problems → list of prediction dicts
- Handle CPU/GPU device management

**Unit Tests**: Test loading model, test prediction on example.json, test batch predictions, verify output format

---

### Step 11: Main CLI Script

**Files**: `main.py`, `config.yaml`

Create command-line interface for training, evaluation, and prediction.

**Commands**:

- `python main.py train --config config.yaml`: train new model
- `python main.py evaluate --checkpoint path/to/model.pth --data data/problems.json`: evaluate model
- `python main.py predict --checkpoint path/to/model.pth --input problem.json`: predict single problem

**config.yaml Should Include**: model type (fc/cnn), learning rate, batch size, epochs, early stopping patience, train/val/test ratios, data path, checkpoint directory

---

### Step 12: Technical Specification

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