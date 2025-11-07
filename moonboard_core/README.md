# Moonboard Core

Shared utilities for processing Moonboard climbing problems.

## Features

This package provides core functionality for working with Moonboard boulder problems:

- **Grade Encoding**: Convert between Font grade strings (e.g., "6B+") and integer labels
- **Position Parsing**: Parse hold positions (e.g., "F7") into row/column indices
- **Grid Building**: Convert problem moves into 3-channel tensor representations (start/middle/end holds)
- **Data Processing**: Load and process JSON problem datasets

## Installation

Install in editable mode for development:

```bash
py -m pip install -e .
```

For development with testing tools:

```bash
py -m pip install -e .[dev]
```

## Usage

```python
from moonboard_core import encode_grade, parse_position, create_grid_tensor, load_dataset

# Grade encoding
label = encode_grade("6B+")  # Returns: 4
grade = decode_grade(4)      # Returns: "6B+"

# Position parsing
row, col = parse_position("F7")  # Returns: (6, 5)

# Grid tensor creation
moves = [
    {"description": "A1", "isStart": True, "isEnd": False},
    {"description": "F7", "isStart": False, "isEnd": False},
    {"description": "K18", "isStart": False, "isEnd": True}
]
tensor = create_grid_tensor(moves)  # Returns: (3, 18, 11) numpy array

# Load dataset
dataset = load_dataset("../data/problems.json")
```

## Grid Tensor Format

The grid tensors are 3-channel representations of the Moonboard (3 × 18 × 11):

- **Channel 0**: Start holds
- **Channel 1**: Middle holds (neither start nor end)
- **Channel 2**: End holds

Each channel is an 18×11 binary grid representing the Moonboard wall:
- 18 rows (bottom to top)
- 11 columns (A-K, left to right)

## Testing

Run tests with pytest:

```bash
py -m pytest
```

## Package Structure

```
moonboard_core/
├── __init__.py           # Package exports
├── grade_encoder.py      # Font grade encoding/decoding
├── position_parser.py    # Hold position parsing
├── grid_builder.py       # Tensor grid creation
├── data_processor.py     # Dataset loading and processing
├── setup.py              # Package configuration
├── pytest.ini            # Pytest configuration
└── tests/                # Unit tests
    ├── test_grade_encoder.py
    ├── test_position_parser.py
    ├── test_grid_builder.py
    └── test_data_processor.py
```

## Development

This package is designed to be shared between multiple projects in the moonboard-grader ecosystem:

- **classifier**: CNN-based grade prediction
- **generator**: VAE-based problem generation
- **backend**: API service

All projects should use this package for consistent data processing.

