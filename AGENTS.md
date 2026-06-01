# Moonboard Grader Agent Guide

This file is the short entry point for coding agents and contributors.

Moonboard Grader predicts Moonboard climbing problem difficulty on the Font scale using shared board-processing utilities, ML models, a FastAPI backend, a React frontend, and .NET beta-solving tools.

## Repository Layout

- `moonboard_core/` - shared Python utilities for grade encoding, position parsing, grid building, data processing, and board configuration.
- `classifier/` - PyTorch grid-based grade classifier for 3-channel 18x11 hold tensors.
- `beta-classifier/` - PyTorch sequence classifier that predicts grades from beta-solver move sequences.
- `generator/` - PyTorch conditional VAE for synthetic Moonboard problem generation.
- `backend/` - FastAPI REST API for predictions, generation, analytics, board setups, and problem data.
- `frontend/` - React, TypeScript, and Vite UI for visualizing boards, predictions, beta, analytics, and generated problems.
- `beta-solver/` - C#/.NET dynamic-programming solver for climbing sequences, with API, CLI, and unit tests.
- `aspire-app/` - .NET Aspire orchestration for running the full stack.
- `data/` - local JSON datasets and generated solver/model data.

## Docs Structure

- `docs/architecture.md` - stable codemap, API boundaries, architecture invariants, and cross-cutting concerns.
- `docs/features.md` - shipped capabilities, planned capabilities, and user-facing workflows.
- `docs/adr/` - accepted architecture decisions and long-lived trade-offs.
- `docs/exec-plans/active/` - active implementation plans for unfinished work.
- `docs/exec-plans/completed/` - historical plans kept for context.
- `docs/exec-plans/tech-debt-tracker.md` - documentation debt and gardening checklist.
- `docs/generated/` - generated reference material; do not hand-edit generated outputs.

## Build And Test

Python projects use `uv` for dependency management. Use `py` instead of `python` when invoking Python commands.

```powershell
# From a Python project directory
uv sync
py -m pytest
```

```powershell
# Backend
cd backend
uv sync
py -m uvicorn app.main:app --reload --host localhost --port 8000
```

```powershell
# Frontend
cd frontend
npm install
npm run lint
npm run build
```

```powershell
# Beta Solver
cd beta-solver/BetaSolver
dotnet build
dotnet test
```

## Local Run

```powershell
# Classifier CLI examples
py main.py train --config config.yaml
py main.py evaluate --checkpoint models/best_model.pth
py main.py predict --checkpoint models/best_model.pth --input problem.json
```

```powershell
# Backend API, with docs at http://localhost:8000/docs
cd backend
py -m uvicorn app.main:app --reload --host localhost --port 8000
```

```powershell
# Frontend dev server at http://localhost:5173
cd frontend
npm run dev
```

```powershell
# Beta Solver CLI
cd beta-solver/BetaSolver
dotnet run --project BetaSolver.Cli
```

```powershell
# TensorBoard
tensorboard --logdir=runs
```

## Agent Safety

- Do not commit secrets or local credentials.
- Use `py` instead of `python` for Python commands.
- Do not run model training without asking first.
- Do not create summary documents unless explicitly asked.
- Do not edit local config unless the task explicitly asks for config changes.
- Do not edit generated run artifacts or generated docs.
- For dependency rules, ownership boundaries, and architecture invariants, read `docs/architecture.md`.
- For beta-solver-specific build, test, and C# conventions, read `beta-solver/BetaSolver/AGENTS.md`.

## Documentation Maintenance

- Keep `docs/architecture.md` short and stable. It should be a map, not an atlas.
- Put volatile details, exact command examples, payload shapes, and step-by-step procedures in feature docs, execution plans, generated docs, or code comments.
- Add an ADR when a decision changes ownership boundaries, dependency direction, persistence shape, external integration behavior, or agent safety rules.
- Move finished plans from `docs/exec-plans/active/` to `docs/exec-plans/completed/`.