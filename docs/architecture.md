# Architecture

## Bird's Eye View

Moonboard Grader is a multi-component system for predicting, generating, analyzing, and solving Moonboard climbing problems. Shared Python code in `moonboard_core/` defines the board, grade, and dataset primitives used by the ML projects and backend. The backend exposes those capabilities through FastAPI, the frontend renders interactive board workflows, the beta solver computes move sequences in .NET, and Aspire can orchestrate the stack during local development.

## Code Map

### `moonboard_core/`

Owns shared Python domain primitives: Font grade encoding, position parsing, 3-channel board tensor construction, tensor-to-move conversion, dataset loading, grade filtering, and board setup configuration. Python projects should depend on this package for Moonboard semantics instead of copying local helpers.

### `classifier/`

Owns the grid-based PyTorch classifier. It trains and evaluates models that convert 3-channel 18x11 hold tensors into Font grade predictions. CLI entry points live under `classifier/src/cli/`, model definitions live in `classifier/src/models.py` and `classifier/src/advanced_models.py`, and inference logic lives in `classifier/src/predictor.py`.

### `beta-classifier/`

Owns the transformer-based sequence classifier. It consumes move sequences produced by the beta solver and predicts grades from movement features rather than raw hold-grid tensors.

### `generator/`

Owns synthetic problem generation through a conditional VAE. It uses Moonboard grids and grade labels to train, evaluate, and sample grade-conditioned climbing problems.

### `backend/`

Owns the FastAPI application. Routes live in `backend/app/api/routes/`, service wrappers live in `backend/app/services/`, request and response schemas live in `backend/app/models/`, and runtime configuration, logging, and telemetry live in `backend/app/core/`.

The backend uses `backend/app/services/service_registry.py` to resolve predictor, problem, generator, and analytics services by hold setup and angle.

### `frontend/`

Owns the React/Vite user interface. Components live in `frontend/src/components/`, API calls in `frontend/src/services/api.ts`, board setup state in `frontend/src/contexts/BoardSetupContext.tsx`, typed data contracts in `frontend/src/types/`, and browser telemetry in `frontend/src/telemetry.ts`.

### `beta-solver/BetaSolver/`

Owns the C# beta solver. `BetaSolver.Core/` contains domain models, scoring, and dynamic-programming solver logic; `BetaSolver.Api/` exposes solver behavior over HTTP; `BetaSolver.Cli/` provides command-line batch processing; and `BetaSolver.Core.UnitTests/` verifies solver and model behavior.

### `aspire-app/`

Owns local orchestration for the full stack through .NET Aspire. It wires services together and supports dashboard-based health and telemetry inspection.

### `data/`

Stores local datasets and generated artifacts, including grid-based problem data and beta-solver output used by sequence models. Large data and model checkpoints should be treated as runtime artifacts, not hand-authored source.

### `config/`

Stores shared application configuration such as `config/board_setups.json`, which describes hold setups, angles, data files, and model file locations.

## Architecture Invariants

### Shared Moonboard Semantics

`moonboard_core/` is the single source of truth for Moonboard domain rules. Other Python components (classifier, beta-classifier, generator, backend) must import its primitives instead of reimplementing grade, position, grid, or dataset logic. The active `moonboard_core` review plans exist to remove existing duplication.

`moonboard_core/` is the home for shared Moonboard domain logic (grades, positions, grids, problems, board configuration, datasets). When domain logic is needed by two or more components, add or extract it here rather than duplicating it. Code that is merely shared but not part of the Moonboard domain does not belong here.

### Grid Representation

Grid-based models use a 3-channel 18x11 tensor:

- Channel 0: start holds.
- Channel 1: middle holds.
- Channel 2: end holds.
- Rows are 1 through 18 from bottom to top.
- Columns are A through K from left to right.

### Grade Labels

The Font scale spans 19 classes: `5+`, `6A`, `6A+`, `6B`, `6B+`, `6C`, `6C+`, `7A`, `7A+`, `7B`, `7B+`, `7C`, `7C+`, `8A`, `8A+`, `8B`, `8B+`, `8C`, and `8C+`. Labels are zero-based, from 0 to 18.

### Config-Driven Backend Services

Backend routes should resolve model, data, generation, and analytics capabilities through the service registry and board setup configuration. Hardcoded single-model assumptions should not leak into route handlers.

### Beta Solver Test Stability

`SolveTestCases` in `beta-solver/BetaSolver/BetaSolver.Core.UnitTests/Solver/DpBetaSolverTests.cs` documents expected solver behavior. Do not modify those cases unless explicitly asked.

## API Boundaries

### Frontend to Backend

The frontend talks to the FastAPI backend through typed service functions in `frontend/src/services/api.ts`. Backend route handlers under `backend/app/api/routes/` own HTTP request parsing, dependency resolution, and response shaping.

### Backend to ML Services

Backend service classes adapt model-specific predictors, generators, problem datasets, and analytics files into API-ready operations. Route handlers should depend on these services instead of importing ML model internals directly.

### Beta Solver API and CLI

The C# solver core is shared by the HTTP API and CLI. API/CLI contract types should translate external payloads into core domain models without pushing transport concerns into `BetaSolver.Core/`.

### Aspire Orchestration

`aspire-app/` owns local process orchestration. Application code should not depend on Aspire-specific runtime details except through normal configuration and telemetry environment variables.

## Cross-Cutting Concerns

### Configuration

Board and angle metadata are configured through `config/board_setups.json` and parsed by `moonboard_core/board_config.py`. Backend service loading and frontend board setup selection should treat that configuration as the product-level source of truth.

### Telemetry

Backend telemetry lives in `backend/app/core/telemetry.py`. Browser telemetry lives in `frontend/src/telemetry.ts`. Aspire dashboard integration is documented in `docs/frontend-otel-aspire-setup.md`.

### Persistence And Artifacts

Datasets, solver outputs, model checkpoints, analytics files, and generated run artifacts are runtime assets. Avoid treating generated data, checkpoints, or local outputs as hand-maintained source.

### Testing

Python projects use pytest. The frontend uses the npm scripts declared in `frontend/package.json`. The beta solver uses `dotnet build` and full `dotnet test`; do not run only a subset of beta-solver tests unless explicitly directed.
