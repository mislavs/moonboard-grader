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

## Next Steps

- Collect more training data to improve accuracy
- Experiment with different model architectures
- Try different hyperparameters (learning rate, batch size, etc.)
- Analyze per-grade performance to identify which grades are hardest to predict

