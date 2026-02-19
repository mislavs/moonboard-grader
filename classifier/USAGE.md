# Moonboard Grade Predictor - Usage Guide

## Quick Start

### 1. Training a Model

Create or edit `config.yaml` to set your preferences, then run:

```bash
python main.py train --config config.yaml
```

This will:
- Load and split your dataset (70% train, 15% val, 15% test by default)
- Train the model with early stopping
- Save the best model to `models/best_model.pth`
- Save the final model to `models/final_model.pth`
- Save training history to `models/training_history.json`
- Generate a confusion matrix visualization

**Example config.yaml:**
```yaml
model:
  type: "cnn"  # or "fc" for fully connected
  num_classes: 19

training:
  learning_rate: 0.001
  batch_size: 32
  num_epochs: 100
  early_stopping_patience: 10
  optimizer: "adam"

data:
  path: "data/problems.json"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  random_seed: 42

checkpoint:
  dir: "models"

device: "cuda"  # or "cpu"
```

See the [Configuration Reference](#configuration-reference) section below for detailed explanations of all available parameters.

### 2. Evaluating a Model

Evaluate a trained model on a test dataset:

```bash
python main.py evaluate \
  --checkpoint models/best_model.pth \
  --data data/test_problems.json \
  --save-confusion-matrix \
  --output confusion_matrix.png
```

This will show:
- Exact accuracy (perfect grade match)
- ±1 grade accuracy (within 1 grade)
- ±2 grade accuracy (within 2 grades)
- Mean absolute error
- Per-grade precision, recall, and F1 scores
- Optional confusion matrix visualization

### 3. Making Predictions

Predict the grade of a new problem:

```bash
python main.py predict \
  --checkpoint models/best_model.pth \
  --input my_problem.json \
  --top-k 5 \
  --output predictions.json
```

**Input format (single problem):**
```json
{
  "moves": [
    {"description": "F7", "isStart": true, "isEnd": false},
    {"description": "G10", "isStart": false, "isEnd": false},
    {"description": "E15", "isStart": false, "isEnd": true}
  ]
}
```

**Input format (batch of problems):**
```json
{
  "data": [
    {
      "moves": [...]
    },
    {
      "moves": [...]
    }
  ]
}
```

**Output format:**
```json
{
  "predicted_grade": "6B+",
  "predicted_label": 7,
  "confidence": 0.45,
  "all_probabilities": {
    "5+": 0.01,
    "6A": 0.05,
    "6B": 0.30,
    "6B+": 0.45,
    "6C": 0.15,
    ...
  },
  "top_k_predictions": [
    ["6B+", 0.45],
    ["6C", 0.30],
    ["6B", 0.15],
    ["6C+", 0.05],
    ["7A", 0.03]
  ]
}
```

## Command Reference

### Train Command

```bash
python main.py train [--config CONFIG_FILE]
```

**Options:**
- `--config`: Path to YAML configuration file (default: `config.yaml`)

### Evaluate Command

```bash
python main.py evaluate --checkpoint CHECKPOINT --data DATA_FILE [OPTIONS]
```

**Required:**
- `--checkpoint`: Path to model checkpoint (.pth file)
- `--data`: Path to evaluation data JSON file

**Options:**
- `--cpu`: Force CPU usage (default: use CUDA if available)
- `--save-confusion-matrix`: Save confusion matrix plot
- `--output`: Output path for confusion matrix (default: `confusion_matrix.png`)

### Predict Command

```bash
python main.py predict --checkpoint CHECKPOINT --input INPUT_FILE [OPTIONS]
```

**Required:**
- `--checkpoint`: Path to model checkpoint (.pth file)
- `--input`: Path to input problem JSON file

**Options:**
- `--cpu`: Force CPU usage (default: use CUDA if available)
- `--top-k`: Return top K predictions (default: 3)
- `--output`: Save predictions to JSON file

## Data Format

Your training data should be a JSON file with the following structure:

```json
{
  "data": [
    {
      "grade": "6B+",
      "moves": [
        {
          "description": "A1",
          "isStart": true,
          "isEnd": false
        },
        {
          "description": "F7",
          "isStart": false,
          "isEnd": false
        },
        {
          "description": "K18",
          "isStart": false,
          "isEnd": true
        }
      ]
    },
    ...
  ]
}
```

**Field descriptions:**
- `grade`: Font grade string (5+ to 8C+)
- `moves`: Array of hold positions
  - `description`: Position string (column A-K, row 1-18, e.g., "F7")
  - `isStart`: Boolean indicating start hold
  - `isEnd`: Boolean indicating end/finish hold

## Recommended Configuration

The default `config.yaml` uses 19 output classes covering the full Font grade
range (5+ to 8C+). However, the current Moonboard dataset contains approximately
15 observed grades, with the majority concentrated in the 6A+ to 7C range.
Training with extra never-seen classes wastes model capacity and can hurt
calibration.

**For best results, use `config_improved.yaml`** which filters to the 10 most
populated grades (6A+ through 7C, indices 2-11) and sets `num_classes: 10`:

```bash
python main.py train --config config_improved.yaml
```

If you need the full grade range (e.g., after adding data for extreme grades),
use `config.yaml` with `num_classes: 19` and `filter_grades: false`.

## Tips

1. **Start with CNN model**: Generally performs better than fully connected for spatial patterns
2. **Use early stopping**: Prevents overfitting, patience=10 is a good default
3. **Monitor validation accuracy**: If training accuracy is much higher than validation, reduce model complexity or add regularization
4. **Stratified splits**: Automatically preserves grade distribution across train/val/test sets
5. **Tolerance metrics**: ±1 grade accuracy is very useful for climbing - being off by one grade is still valuable
6. **CUDA acceleration**: Training is much faster on GPU if available

## Troubleshooting

**Q: Training is very slow**
- A: Use `--device cuda` if you have a GPU, or reduce `batch_size` in config

**Q: Model predicts the same grade for everything**
- A: Dataset might be imbalanced - check grade distribution with `get_dataset_stats()`

**Q: Validation accuracy much lower than training**
- A: Model is overfitting - increase dropout, reduce model size, or get more data

**Q: "CUDA out of memory" error**
- A: Reduce `batch_size` in config.yaml or use `--device cpu`

## Example Workflow

```bash
# 1. Prepare your data
# Place moonboard problems in data/problems.json

# 2. Train a model
python main.py train --config config.yaml

# 3. Evaluate the trained model
python main.py evaluate \
  --checkpoint models/best_model.pth \
  --data data/problems.json \
  --save-confusion-matrix

# 4. Make predictions on new problems
python main.py predict \
  --checkpoint models/best_model.pth \
  --input new_problem.json \
  --top-k 5
```

## Python API

You can also use the components directly in Python:

```python
from src import (
    load_dataset,
    create_data_splits,
    create_model,
    Trainer,
    Predictor
)

# Load and split data
dataset = load_dataset("data/problems.json")
tensors = [x[0] for x in dataset]
labels = [x[1] for x in dataset]
train_ds, val_ds, test_ds = create_data_splits(tensors, labels)

# Create and train model
model = create_model("cnn", num_classes=19)
trainer = Trainer(model, train_loader, val_loader, optimizer, criterion)
history = trainer.fit(num_epochs=100, early_stopping_patience=10)

# Make predictions
predictor = Predictor("models/best_model.pth")
result = predictor.predict({"moves": [...]})
print(f"Predicted: {result['predicted_grade']}")
```

## Configuration Reference

This section provides detailed explanations for all available parameters in the configuration YAML file.

### Model Configuration

```yaml
model:
  type: "cnn"
  num_classes: 19
  use_attention: true
  dropout_conv: 0.15
  dropout_fc1: 0.4
  dropout_fc2: 0.5
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `type` | string | `"cnn"` | Model architecture type. Options: `"fc"` (fully connected), `"cnn"` (convolutional), `"residual_cnn"` (residual CNN with attention), `"deep_residual_cnn"` (deeper residual variant) |
| `num_classes` | int | `19` | Number of output grade classes. Use `19` for full range (5+ to 8C+), or fewer when filtering grades |
| `use_attention` | bool | `false` | Enable spatial and channel attention mechanisms. Only applicable to `residual_cnn` and `deep_residual_cnn` models |
| `dropout_conv` | float | `0.1` | Dropout rate applied in convolutional layers. Range: 0.0-1.0. Higher values reduce overfitting |
| `dropout_fc1` | float | `0.3` | Dropout rate after the first fully connected layer. Range: 0.0-1.0 |
| `dropout_fc2` | float | `0.4` | Dropout rate after the second fully connected layer. Range: 0.0-1.0 |

### Training Configuration

```yaml
training:
  learning_rate: 0.0003
  batch_size: 64
  num_epochs: 150
  early_stopping_patience: 8
  optimizer: "adam"
  weight_decay: 0.001
  use_class_weights: true
  max_class_weight: 5.0
  use_scheduler: true
  scheduler_factor: 0.3
  scheduler_patience: 3
  label_smoothing: 0.1
  gradient_clip: 1.0
  loss_type: "focal_ordinal"
  focal_gamma: 2.0
  ordinal_weight: 0.5
  ordinal_alpha: 2.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | `0.001` | Initial learning rate for the optimizer. Lower values (e.g., 0.0003) often work better for fine-grained classification |
| `batch_size` | int | `32` | Number of samples per training batch. Larger batches train faster but use more memory |
| `num_epochs` | int | `100` | Maximum number of training epochs. Training may stop earlier due to early stopping |
| `early_stopping_patience` | int | `10` | Number of epochs to wait for validation improvement before stopping training |
| `optimizer` | string | `"adam"` | Optimizer algorithm. Options: `"adam"` (adaptive learning rates), `"sgd"` (stochastic gradient descent) |
| `weight_decay` | float | `0.0` | L2 regularization coefficient. Higher values (e.g., 0.001-0.002) help prevent overfitting |
| `use_class_weights` | bool | `false` | Apply class weights to handle imbalanced grade distribution. Recommended for Moonboard data |
| `max_class_weight` | float | `5.0` | Maximum class weight to prevent extreme weighting of rare grades. Only used when `use_class_weights` is true |
| `use_scheduler` | bool | `false` | Enable learning rate scheduler that reduces LR when validation loss plateaus |
| `scheduler_factor` | float | `0.5` | Factor by which LR is reduced when scheduler triggers. New LR = old LR × factor |
| `scheduler_patience` | int | `5` | Number of epochs to wait before reducing LR if no improvement |
| `label_smoothing` | float | `0.0` | Label smoothing coefficient (0.0-1.0). Values like 0.1 help prevent overconfident predictions |
| `gradient_clip` | float | `null` | Maximum gradient norm for gradient clipping. Values like 1.0 help stabilize training |
| `loss_type` | string | `"ce"` | Loss function type. Options: `"ce"` (cross-entropy), `"focal"` (focal loss for class imbalance), `"ordinal"` (ordinal regression), `"focal_ordinal"` (combined), `"label_smoothing"` |
| `focal_gamma` | float | `2.0` | Focusing parameter for focal loss. Higher values focus more on hard examples. Range: 1.5-3.0 |
| `ordinal_weight` | float | `0.5` | Weight for the ordinal component when using combined losses. Range: 0.0-1.0 |
| `ordinal_alpha` | float | `2.0` | Distance penalty for ordinal loss. Controls how much to penalize predictions far from the true grade |
| `reproducibility_seed` | int | `null` | Seed all RNGs (Python, NumPy, PyTorch) for repeatable training runs. Omit to use default random state |
| `deterministic` | bool | `false` | Force deterministic algorithms in PyTorch. May reduce performance. Only applies when `reproducibility_seed` is set |

### Data Configuration

```yaml
data:
  path: "../data/problems.json"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  random_seed: 42
  filter_grades: true
  min_grade_index: 2
  max_grade_index: 11
  filter_repeats: false
  min_repeats: 1
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | string | required | Path to the training data JSON file |
| `train_ratio` | float | `0.7` | Proportion of data used for training. Must sum to 1.0 with val_ratio and test_ratio |
| `val_ratio` | float | `0.15` | Proportion of data used for validation during training |
| `test_ratio` | float | `0.15` | Proportion of data used for final evaluation |
| `random_seed` | int | `42` | Seed for reproducible data splits. Change to get different train/val/test splits |
| `filter_grades` | bool | `false` | Enable grade filtering to train on a subset of grades |
| `min_grade_index` | int | `0` | Minimum grade index to include (0=5+, 1=6A, 2=6A+, ...). Only used when `filter_grades` is true |
| `max_grade_index` | int | `18` | Maximum grade index to include (18=8C+). Only used when `filter_grades` is true |
| `filter_repeats` | bool | `false` | Enable filtering by number of times a problem has been repeated/logged |
| `min_repeats` | int | `0` | Minimum number of repeats required. 0 = include all, 1 = exclude zero-repeat routes |
| `group_by_layout` | bool | `false` | Group-aware splitting: ensures identical hold layouts stay in the same split. Prevents data leakage from duplicate problems. Disables stratified splitting in favour of group-based splitting |

### Checkpoint Configuration

```yaml
checkpoint:
  dir: "models"
  save_best: true
  save_final: true
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dir` | string | `"models"` | Directory to save model checkpoints and artifacts |
| `save_best` | bool | `true` | Save the best model (based on validation loss) as `best_model.pth` |
| `save_final` | bool | `true` | Save the final model after training completes as `final_model.pth` |

### Device Configuration

```yaml
device: "cuda"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | string | `"cuda"` | Computation device. Options: `"cuda"` (GPU, recommended), `"cpu"` (slower but always available) |

### Evaluation Configuration

```yaml
evaluation:
  tolerance_levels: [1, 2]
  save_confusion_matrix: true
  confusion_matrix_path: "models/confusion_matrix.png"
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tolerance_levels` | list[int] | `[1, 2]` | List of tolerance levels for accuracy calculation. `[1, 2]` means compute ±1 and ±2 grade accuracy |
| `save_confusion_matrix` | bool | `false` | Automatically save confusion matrix visualization after training |
| `confusion_matrix_path` | string | `"confusion_matrix.png"` | File path for saving the confusion matrix image |

### Prediction Configuration

```yaml
prediction:
  top_k: 3
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | `3` | Number of top predictions to return with probabilities |

### Example Configurations

**Basic CNN Training:**
```yaml
model:
  type: "cnn"
  num_classes: 19

training:
  learning_rate: 0.001
  batch_size: 32
  num_epochs: 100
  early_stopping_patience: 10

data:
  path: "../data/problems.json"

device: "cuda"
```

**Advanced Training with Focal Loss and Attention:**
```yaml
model:
  type: "residual_cnn"
  num_classes: 10
  use_attention: true
  dropout_conv: 0.15
  dropout_fc1: 0.4
  dropout_fc2: 0.5

training:
  learning_rate: 0.0003
  batch_size: 64
  num_epochs: 150
  early_stopping_patience: 10
  optimizer: "adam"
  weight_decay: 0.002
  loss_type: "focal_ordinal"
  focal_gamma: 2.0
  ordinal_weight: 0.5
  use_class_weights: true
  max_class_weight: 5.0
  use_scheduler: true
  gradient_clip: 1.0

data:
  path: "../data/problems.json"
  filter_grades: true
  min_grade_index: 2   # 6A+
  max_grade_index: 11  # 7C

device: "cuda"
```

**Training on Filtered Popular Problems:**
```yaml
model:
  type: "cnn"
  num_classes: 10

training:
  learning_rate: 0.0003
  batch_size: 64
  num_epochs: 150
  use_class_weights: true
  max_class_weight: 10.0

data:
  path: "../data/problems.json"
  filter_grades: true
  min_grade_index: 2
  max_grade_index: 11
  filter_repeats: true
  min_repeats: 1  # Only include problems that have been logged at least once

device: "cuda"
```

## Next Steps

- Collect more training data to improve accuracy
- Experiment with different model architectures
- Try different hyperparameters (learning rate, batch size, etc.)
- Analyze per-grade performance to identify which grades are hardest to predict

