# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Moonboard Grader is a machine learning system that predicts difficulty grades of Moonboard climbing problems using deep learning. It uses the Font climbing grade scale (5+ to 8C+, 19 classes).

## Commands

### Python Projects (use `uv` package manager)

```bash
# Install dependencies
uv sync

# Run CLI tools (use 'py' not 'python')
py main.py train --config config.yaml
py main.py evaluate --checkpoint models/best_model.pth
py main.py predict --checkpoint models/best_model.pth --input problem.json

# Run tests
pytest
```

### Backend (FastAPI)

```bash
cd backend
uv sync
uvicorn app.main:app --reload --host localhost --port 8000
# API docs at http://localhost:8000/docs
```

### Frontend (React/Vite)

```bash
cd frontend
npm install
npm run dev      # Development server at http://localhost:5173
npm run build    # Production build
npm run lint     # ESLint
```

### Beta Solver (.NET)

```bash
cd beta-solver/BetaSolver
dotnet build
dotnet test     # ALWAYS run ALL tests, never a subset
dotnet run --project BetaSolver.Cli
```

### TensorBoard

```bash
tensorboard --logdir=runs
```

## Architecture

### Components

- **classifier/** - CNN-based grid classifier (PyTorch). Converts 3-channel 18×11 hold grid to grade prediction.
- **beta-classifier/** - Transformer-based sequence classifier (PyTorch). Uses move sequences from beta solver.
- **generator/** - Conditional VAE for synthetic problem generation (PyTorch).
- **backend/** - FastAPI REST API serving model predictions.
- **frontend/** - React/TypeScript/Vite web interface for visualization.
- **beta-solver/** - C# .NET algorithm finding optimal climbing sequences via dynamic programming.
- **moonboard_core/** - Shared Python utilities (grade encoding, grid building, position parsing).
- **aspire-app/** - .NET Aspire orchestration for running full stack.

### Data Representation

**Grid format** (classifier/generator): 3-channel 18×11 tensor
- Channel 0: Start holds
- Channel 1: Middle holds
- Channel 2: End holds
- Rows: 1-18 (bottom to top), Columns: A-K (left to right)

**Problem JSON format**:
```json
{
  "Grade": "6B+",
  "Moves": [
    {"Description": "A5", "IsStart": true, "IsEnd": false},
    {"Description": "K12", "IsStart": false, "IsEnd": true}
  ]
}
```

**Font grades**: 5+, 6A, 6A+, 6B, 6B+, 6C, 6C+, 7A, 7A+, 7B, 7B+, 7C, 7C+, 8A, 8A+, 8B, 8B+, 8C, 8C+ (labels 0-18)

### Key Files

- `moonboard_core/grade_encoder.py` - Grade ↔ label conversion
- `moonboard_core/grid_builder.py` - Problem → tensor conversion
- `classifier/src/models.py` - CNN architecture
- `beta-classifier/src/model.py` - Transformer architecture
- `generator/src/vae.py` - VAE architecture
- `backend/app/main.py` - FastAPI entry point
- `data/problems.json` - Grid-based training data (~130MB)
- `data/solved_problems.json` - Beta solver output for sequence classifier

## Conventions

### General
- Use `py` instead of `python` for Python commands
- Don't run model training without asking first
- Don't create summary documents unless explicitly asked

### .NET (Beta Solver)
- Use primary constructors
- Each class in its own file, classes should be `sealed` unless designed for inheritance
- Each file should have one empty line at the end
- Tests follow arrange-act-assert pattern
- Don't modify `SolveTestCases` in `DpBetaSolverTests.cs` unless explicitly asked
