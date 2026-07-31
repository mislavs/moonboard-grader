# Moonboard Grade Prediction Neural Network

A PyTorch-based classification neural network that predicts Font scale climbing grades from Moonboard hold positions.

## Overview

This project implements a deep learning system to predict the difficulty grade of Moonboard climbing problems. The model takes hold positions as input (represented as a 3-channel grid) and outputs a predicted Font grade (5+ to 8C+).

## Features

- **Multi-channel representation**: Separates start holds, middle holds, and end holds
- **Multiple model architectures**: Fully connected baseline and convolutional neural network
- **Comprehensive evaluation**: Exact accuracy, tolerance-based accuracy, confusion matrices
- **Frozen benchmark manifests**: Dataset hashes, filters, problem IDs, and split membership
- **Repeated grouped validation**: Five layout-aware folds across three fixed seeds
- **Production-ready inference**: Easy-to-use predictor interface for new problems
- **CLI interface**: Command-line tools for training, evaluation, and prediction

## Architecture

- **Input**: 3x11x18 tensor representing Moonboard holds
  - Channel 0: Start holds
  - Channel 1: Middle holds  
  - Channel 2: End holds
- **Output**: Classification over Font grades (5+ through 8C+)
- **Models**: 
  - Fully Connected: Simple baseline for quick experimentation
  - Convolutional: Learns spatial patterns in hold placement

## Installation

### Prerequisites

- Python 3.9 or higher
- uv package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))

### Setup

1. Clone the repository and navigate to the classifier directory:
```bash
cd moonboard-grader/classifier
```

2. Install dependencies:

```bash
uv sync
```

### Verify Installation

Run the test suite to verify everything is installed correctly:
```bash
uv run pytest
```

## Usage

### 1. Freeze the Benchmark Cohort

```bash
py main.py create-manifest --config config.yaml --output manifests/moonboard-masters-2017-all-v1.json
```

The manifest is immutable. Any dataset or filter change requires a new manifest version.

### 2. Compare a Candidate

```bash
py main.py cross-validate --config config.yaml --manifest manifests/moonboard-masters-2017-all-v1.json
```

This requires a clean committed revision, performs five grouped folds for each seed in `[42, 43, 44]`, and never loads the locked test membership.
The CLI configures `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing PyTorch so strict deterministic CUDA training works on CUDA 10.2 and newer.

### 3. Refit the Selected Candidate

Run this only after comparing validation reports and from a clean committed revision:

```bash
py main.py refit --config config.yaml --manifest manifests/moonboard-masters-2017-all-v1.json --cv-report models/experiments/<candidate>/cv/cv_report.json
```

The refit uses the complete development pool and the median CV-selected epoch.

### 4. Evaluate the Promoted Refit

```bash
py main.py evaluate --checkpoint models/experiments/<candidate>/refit_<candidate>_seed-42.pth --manifest manifests/moonboard-masters-2017-all-v1.json
```

Official evaluation accepts only provenance-complete refit checkpoints. For arbitrary data, use the explicitly non-comparable diagnostic command:

```bash
py main.py evaluate-diagnostic --checkpoint models/legacy.pth --data ../data/external.json --output diagnostic.json
```

### Making Predictions

```bash
py main.py predict --checkpoint models/best_model.pth --input problem.json
```

## Experiment Artifacts

CV writes one atomic checkpoint/result pair per seed and fold plus `cv_report.json`. Re-running the command resumes only hash-compatible completed folds. Refit checkpoints embed the resolved configuration, manifest membership, dataset/filter hashes, seeds, code revision, and runtime versions. Locked test metrics live only in a separate `.test.json` report and never in model filenames.

## Data Format

Manifest-backed datasets must include the same `problemId` on every move:

```json
{
  "grade": "6B+",
  "repeats": 10,
  "moves": [
    {"problemId": 123, "description": "A5", "isStart": true, "isEnd": false},
    {"problemId": 123, "description": "F7", "isStart": false, "isEnd": false},
    {"problemId": 123, "description": "K12", "isStart": false, "isEnd": true}
  ]
}
```

## Development

### Running Tests

Run all tests:
```bash
uv run pytest
```

Run tests for a specific module:
```bash
uv run pytest tests/test_models.py
```

Run with coverage:
```bash
uv run pytest --cov=src tests/
```

### Code Organization

Each module in `src/` has a corresponding test file in `tests/`. We follow test-driven development practices, writing tests before implementation.

## Performance

- **Exact Accuracy**: Percentage of predictions matching true grade exactly
- **±1 Grade Accuracy**: Predictions within one grade of true grade
- **Per-Grade Metrics**: Precision, recall, and F1-score for each grade class

See `spec.md` for detailed performance benchmarks.

## Contributing

When adding new features:

1. Write unit tests first
2. Implement the feature
3. Ensure all tests pass
4. Update documentation

## License

MIT License

## Acknowledgments

Built for analyzing Moonboard climbing problems and predicting difficulty grades using deep learning.

