# Agent Instructions

## Python Command

On this system, use `py` instead of `python` for Python commands. For example:
- `py -m pytest` instead of `python -m pytest`
- `py setup.py --version` instead of `python setup.py --version`
- `py -m pip install -e .` instead of `python -m pip install -e .`

## Running Tests

To run tests for this project, use:

```bash
py -m pytest
```

This will discover and run all tests in the test suite.

## Documenting changes

All notable changes (such as model optimizations) should be documented in docs/changelog.md

