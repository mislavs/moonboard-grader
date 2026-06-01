# ADR: Multi-Board Configuration And Service Registry

## Context

Moonboard Grader originally assumed a small number of hardcoded model, data, generator, and analytics files. Multi-board and multi-angle support requires the app to select capabilities by hold setup and wall angle without rewriting route handlers every time a new board or model artifact is added.

The frontend also needs enough board setup metadata to select images, hide unsupported beta-solving workflows, and show only capabilities available for the active board and angle.

## Decision

Use `config/board_setups.json` as the product-level source of truth for hold setups, angles, and associated artifact paths.

Parse this configuration through `moonboard_core/board_config.py`. In the backend, load configured model/data/generator/analytics capabilities into `backend/app/services/service_registry.py`, then resolve services from route handlers by `hold_setup` and `angle`.

Expose board setup capability metadata through `/board-setups`, including beta-solving support, board image names, generator availability, and analytics availability.

## Consequences

- Adding a new supported board or angle is primarily a configuration and artifact-management task.
- Backend routes stay focused on HTTP behavior and dependency resolution instead of hardcoded artifact paths.
- Frontend workflows can conditionally render board images and beta/generation/analytics controls based on setup metadata.
- Startup and tests must cover missing artifacts, unavailable capabilities, and default setup fallback behavior.

## Alternatives Considered

- Keep hardcoded default model/data paths in each route. This was simpler but would make each new board or angle require route-level changes.
- Let the frontend own board capabilities locally. This would avoid one backend response change but risks frontend/backend drift when model artifacts are added or removed.
- Create separate API routers per board setup. This would duplicate route behavior and make client-side selection harder to maintain.
