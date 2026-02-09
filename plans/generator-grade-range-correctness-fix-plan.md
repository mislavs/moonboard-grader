# Fix Plan: Grade-Range Correctness with Remapped Label Space (Full Pipeline)

## Summary
Implement a structural fix so filtered-grade models use a compact model label space (`0..N-1`) while preserving global grade semantics at boundaries (CLI, outputs, classifier comparisons). Propagate checkpoint grade-range metadata through evaluation so every metric respects training scope. Keep backward compatibility for existing checkpoints trained with global labels.

## Key Design Decisions
1. Internal model labels are remapped for filtered training.
2. Global labels remain the external/source-of-truth contract (`moonboard_core.encode_grade/decode_grade`).
3. New checkpoints explicitly declare label-space metadata.
4. Old checkpoints are supported via deterministic inference rules.
5. Evaluation scope always follows checkpoint range, not `model.num_grades` alone.

## Public API / Interface Changes
1. `MoonBoardDataset` behavior:
   - `__getitem__` returns `model_label` (remapped if filtered).
   - Add explicit mapping helpers:
     - `global_to_model_label(global_label: int) -> int`
     - `model_to_global_label(model_label: int) -> int`
     - `get_num_model_grades() -> int`
2. Checkpoint schema additions (generator):
   - `label_space_mode: "remapped" | "global_legacy"`
   - `grade_offset: int` (kept, formalized)
   - `min_grade_index: int | None` (kept)
   - `max_grade_index: int | None` (kept)
3. Evaluation internal interface:
   - Introduce `EvaluationLabelContext` (or equivalent) passed to all metrics, containing:
     - `label_space_mode`
     - `grade_offset`
     - `min_grade_index`
     - `max_grade_index`
     - `num_model_grades`
4. CLI behavior adjustments:
   - `generate` accepts global grade input and maps to model labels using checkpoint context.
   - `--include-grade` always emits global grade string correctly.
   - `evaluate --metrics` rejects unknown metric names with non-zero exit instead of silent no-op.

## Implementation Steps
1. Add a shared label-space utility module in generator (single source of truth).
   - Implement mapping and validation functions.
   - Implement checkpoint metadata inference for backward compatibility:
     - If `label_space_mode` exists, trust it.
     - Else infer remapped when `min/max` exist and `num_model_grades == (max-min+1)`.
     - Else treat as legacy global-label model.
2. Refactor `MoonBoardDataset` to support remapped labels cleanly.
   - Store both global range metadata and model-label mapping logic.
   - Ensure filtered datasets produce contiguous model labels starting at 0.
   - Update train-time `num_grades` source to `get_num_model_grades()`.
3. Update training pipeline (`main.py` + trainer metadata).
   - Build model using model-label class count.
   - Save explicit `label_space_mode` metadata in checkpoint.
4. Update generator loading and generation paths.
   - `ProblemGenerator.from_checkpoint` loads/infers label context.
   - Grade argument resolution maps global requested grade -> model label.
   - Validation prevents generation outside checkpoint grade range.
   - Output formatting maps model label -> global grade label/name.
5. Fix `--include-grade` formatting bug.
   - Stop calling `get_filtered_grade_names(None, None)`.
   - Use label-space mapping + `decode_grade` for robust grade strings.
6. Refactor evaluation orchestration to be context-driven.
   - Load checkpoint context once in `evaluate_command`.
   - Pass context through `run_evaluation` and every metric function.
7. Update each metric to honor checkpoint range.
   - Reconstruction/latent metrics: load data with same min/max grade filters from context.
   - Diversity/statistical: iterate model labels in valid model range, map to global grade labels for reporting/comparison.
   - Classifier-check: align with same shared context utility (remove duplicated mapping logic).
8. Harden evaluate CLI metric selection.
   - Unknown metrics should raise clear error and exit non-zero.
   - Keep metric naming consistent with documentation.
9. Documentation updates.
   - Explain internal remapped labels and external global grade semantics.
   - Update evaluate metric names and examples to match implementation.
10. Regression hardening for edge cases.
   - Single-grade models (`N=1`) must evaluate/generate without index errors.
   - Empty/invalid scope should fail fast with clear messages.

## Test Cases and Scenarios
1. Dataset mapping tests:
   - Unfiltered dataset: model labels equal global labels.
   - Filtered dataset `2..12`: model labels are `0..10`, round-trip mapping works.
   - Single-grade dataset `2..2`: only model label `0`, round-trip returns global `2`.
2. Checkpoint compatibility tests:
   - New remapped checkpoint loads with explicit mode.
   - Old checkpoint with min/max and compact classes inferred as remapped.
   - Old checkpoint without range metadata treated as legacy global.
3. Generation tests:
   - Requesting in-range global grade maps correctly and generates.
   - Requesting out-of-range global grade fails with explicit range message.
   - `--include-grade` emits correct global grade string.
4. Evaluation scope tests:
   - For filtered model `2..12`, every metric evaluates only those grades.
   - For single-grade model `2..2`, metrics do not crash and report only that grade.
   - Reconstruction does not embed out-of-range labels.
5. CLI behavior tests:
   - `evaluate --metrics unknown_metric` exits non-zero with actionable error.
   - `evaluate` with valid metric names proceeds and loads checkpoint.
6. End-to-end smoke:
   - Train filtered model -> save checkpoint -> generate/evaluate roundtrip works with consistent grade mapping.

## Acceptance Criteria
1. No evaluation metric processes grades outside checkpoint training range.
2. No index errors when evaluating/generating filtered or single-grade models.
3. Grade names in outputs always correspond to correct global grades.
4. Backward compatibility preserved for legacy checkpoints.
5. Unknown metric names no longer silently bypass evaluation.

## Assumptions and Defaults
1. Default strategy: remapped model labels for filtered training.
2. External interfaces remain global-grade oriented (user input, display, classifier comparisons).
3. Existing legacy checkpoints remain readable through inference rules.
4. `moonboard_core` stays unchanged; mapping logic is handled in generator layer.
