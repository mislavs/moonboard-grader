# Moonboard Grade Prediction Neural Network

A PyTorch-based classification neural network that predicts Font scale climbing grades from Moonboard hold positions.

## Overview

This project implements a deep learning system to predict the difficulty grade of Moonboard climbing problems. The model takes hold positions as input (represented as a 3-channel grid) and outputs a predicted Font grade (5+ to 8C+).

## Features

- **Multi-channel representation**: Separates start holds, middle holds, and end holds
- **Multiple model architectures**: Fully connected baseline and convolutional neural network
- **Comprehensive evaluation**: Exact accuracy, tolerance-based accuracy, confusion matrices
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

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository and navigate to the classifier directory:
```bash
cd moonboard-grader/classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install the moonboard_core shared utilities package:
```bash
pip install -e ../moonboard_core
```

4. Install classifier dependencies:
```bash
pip install -r requirements.txt
```

### Verify Installation

Run the test suite to verify everything is installed correctly:
```bash
pytest tests/
```

## Usage

### Training a Model

```bash
python main.py train --config config.yaml
```

### Evaluating a Model

```bash
python main.py evaluate --checkpoint models/best_model.pth --data ../data/problems.json
```

### Making Predictions

```bash
python main.py predict --checkpoint models/best_model.pth --input problem.json
```

## Data Format

Input problems should be in JSON format with the following structure:

```json
{
  "Grade": "6B+",
  "Moves": [
    {"Description": "A5", "IsStart": true, "IsEnd": false},
    {"Description": "F7", "IsStart": false, "IsEnd": false},
    {"Description": "K12", "IsStart": false, "IsEnd": true}
  ]
}
```

## Development

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run tests for a specific module:
```bash
pytest tests/test_models.py
```

Run with coverage:
```bash
pytest --cov=src tests/
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

