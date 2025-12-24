# Beta Classifier

Transformer-based climbing grade classifier using move sequences from the beta solver.

## Overview

This project trains a sequence classification model to predict Moonboard climbing problem grades from the move sequences produced by the beta solver. Instead of using static hold positions like the grid-based classifier, this model uses rich per-move features including:

- Position coordinates (target, stationary, origin)
- Hold difficulty scores
- Body stretch and travel distances
- Hand assignment
- Success probability scores

## Architecture

```
Input (N x 15 features)
    │
    ▼
Linear Projection → d_model
    │
    ▼
Sinusoidal Positional Encoding
    │
    ▼
Transformer Encoder (N layers)
    │
    ▼
Masked Mean Pooling
    │
    ▼
Classification Head → 19 grades
```

## Installation

```bash
cd beta-classifier
uv sync
```

## Usage

### Training

```bash
py main.py train --config config.yaml
```

This will:
1. Load data from `../data/solved_problems.json`
2. Split into train/val/test sets (70/15/15)
3. Fit a feature normalizer on training data
4. Train the model with early stopping
5. Save best model to `models/best_model.pth`
6. Save normalizer to `models/normalizer.npz`
7. Generate confusion matrix and error distribution plots

### Evaluation

```bash
py main.py evaluate \
    --checkpoint models/best_model.pth \
    --normalizer models/normalizer.npz \
    --data ../data/solved_problems.json \
    --output eval_output/
```

### Prediction

Single problem or batch prediction:

```bash
# Show predictions with alternatives
py main.py predict \
    --checkpoint models/best_model.pth \
    --normalizer models/normalizer.npz \
    --input problem.json

# Compare with actual grades
py main.py predict \
    --checkpoint models/best_model.pth \
    --normalizer models/normalizer.npz \
    --input problem.json \
    --compare
```

## Input Format

The model expects beta solver output with this structure:

```json
{
  "name": "Problem Name",
  "grade": "7A",
  "moves": [
    {
      "targetX": 6,
      "targetY": 5,
      "stationaryX": 9,
      "stationaryY": 3,
      "originX": 9,
      "originY": 3,
      "targetDifficulty": 5.5,
      "stationaryDifficulty": 4.5,
      "originDifficulty": 4.5,
      "bodyStretchDx": -3,
      "bodyStretchDy": 2,
      "travelDx": -3,
      "travelDy": 2,
      "hand": 0,
      "successScore": 0.677
    }
  ]
}
```

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  d_model: 64        # Transformer hidden dimension
  n_heads: 4         # Attention heads
  n_layers: 2        # Encoder layers
  dropout: 0.1       # Dropout rate
  max_seq_len: 50    # Max sequence length
  num_classes: 19    # Number of grade classes (5+ to 8C+)

training:
  learning_rate: 0.0001
  batch_size: 64
  num_epochs: 150
  early_stopping_patience: 10
  use_class_weights: true    # Handle class imbalance
  label_smoothing: 0.1       # Regularization
```

## Features (15 per move)

| # | Feature | Description |
|---|---------|-------------|
| 0-1 | targetX, targetY | Position of target hold |
| 2-3 | stationaryX, stationaryY | Position of stationary hand |
| 4-5 | originX, originY | Position of moving hand before move |
| 6-8 | *Difficulty | Hold difficulty scores |
| 9-10 | bodyStretchDx, bodyStretchDy | Distance between hands after move |
| 11-12 | travelDx, travelDy | Distance traveled by moving hand |
| 13 | hand | 0=left, 1=right |
| 14 | successScore | Beta solver success probability |

## Metrics

The model is evaluated on:
- **Exact accuracy**: Predictions matching true grade
- **±1 accuracy**: Within one grade level
- **±2 accuracy**: Within two grade levels
- **MAE**: Mean absolute error in grade levels
- **Macro/Weighted F1**: Per-class performance

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
beta-classifier/
├── main.py              # CLI entry point
├── config.yaml          # Training configuration
├── pyproject.toml       # Dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── dataset.py       # Data loading, normalization
│   ├── model.py         # Transformer architecture
│   ├── trainer.py       # Training loop
│   ├── evaluator.py     # Metrics and visualization
│   └── predictor.py     # Inference interface
├── models/              # Saved checkpoints
│   ├── best_model.pth
│   └── normalizer.npz
├── runs/                # TensorBoard logs
└── tests/
    ├── test_dataset.py
    ├── test_model.py
    └── test_predictor.py
```

## Dependencies

- PyTorch >= 2.5.0
- NumPy >= 1.26.0
- scikit-learn >= 1.6.0
- PyYAML >= 6.0.2
- matplotlib >= 3.9.0
- seaborn >= 0.13.0
- tensorboard >= 2.18.0
- tqdm >= 4.67.0
- moonboard_core (local package)

