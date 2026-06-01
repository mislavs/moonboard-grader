# Technical Debt Tracker

Track documentation debt that is known but not worth fixing immediately.

## Documentation Debt

- `docs/features.md` needs periodic review to keep shipped and planned behavior separate.
- `docs/architecture.md` should stay a high-level codemap. Move implementation details to feature docs, ADRs, or execution plans.
- Generated reference docs are not yet produced for schema, OpenAPI, enum, or metadata references.

## Gardening Checklist

Before or after larger feature work:

- Move completed plans from `docs/exec-plans/active/` to `docs/exec-plans/completed/`.
- Add an ADR for decisions that change ownership boundaries, dependency direction, persistence shape, or external integration behavior.
- Update `docs/features.md` for new user-facing behavior.
- Update `docs/architecture.md` only for new modules, boundaries, or invariants.
- Prefer generated docs for exact schema, OpenAPI, enum, or metadata references.
