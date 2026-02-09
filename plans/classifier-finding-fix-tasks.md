# Classifier Finding Fix Tasks

Each section below is standalone and can be copied into a fresh prompt as-is.

## Task 01 - Fix Ordinal Loss Logic (Critical, Accuracy Impact) [FIXED]

```text
Task: Fix the ordinal loss implementation so distance from the true grade actually affects the loss.

Why this matters:
- The current OrdinalCrossEntropyLoss behaves like plain cross-entropy, so ordinal distance is ignored.
- This can reduce ±1/±2 accuracy quality and MAE behavior for ordered grade labels.

Evidence:
- classifier/src/losses.py:155
- classifier/src/losses.py:167
- classifier/src/losses.py:170

Current issue:
- The distance weights are multiplied by one-hot targets, so only the true-class term remains.
- For the true class, distance is always 0, so the weight is always 1.

Required change:
- Rework OrdinalCrossEntropyLoss so ordinal distance influences loss beyond the true-class-only CE term.
- Keep API compatibility (constructor args and call signature).
- Add/adjust tests proving alpha changes loss value and gradients.

Acceptance criteria:
1. Changing alpha changes the numeric loss for the same logits/targets.
2. Loss is still stable and differentiable.
3. Existing non-ordinal loss behavior is unaffected.
4. Tests cover at least one case where distant mistakes are penalized more than near mistakes.
```

## Task 02 - Validate Focal+Ordinal Combination Actually Uses Ordinal Signal (High, Accuracy Impact)

```text
Task: Ensure FocalOrdinalLoss gains real ordinal behavior after ordinal loss fix.

Why this matters:
- Combined loss currently inherits the broken ordinal component and therefore does not provide intended ordinal benefit.

Evidence:
- classifier/src/losses.py:217
- classifier/src/losses.py:236
- classifier/src/losses.py:238

Required change:
- After fixing OrdinalCrossEntropyLoss, verify FocalOrdinalLoss meaningfully changes with ordinal_alpha and ordinal_weight.
- Add tests demonstrating these parameters affect output loss.

Acceptance criteria:
1. Varying ordinal_alpha changes FocalOrdinalLoss on same inputs.
2. Setting ordinal_weight=0 makes behavior match focal-only component.
3. Test coverage added for both checks.
```

## Task 03 - Validate Filtered Grade Range Against num_classes (High, Accuracy/Correctness)

```text
Task: Add strict config validation when grade filtering is enabled.

Why this matters:
- Misaligned filtered range and model.num_classes can silently produce wrong class space or runtime failures.

Evidence:
- classifier/src/cli/train.py:110
- classifier/src/cli/train.py:111
- classifier/src/cli/train.py:155

Required change:
- When filter_grades=true, enforce:
  expected_classes = max_grade_index - min_grade_index + 1
  expected_classes == model.num_classes
- Fail early with a clear error if mismatched.

Acceptance criteria:
1. Invalid config fails before training starts with actionable message.
2. Valid filtered config continues unchanged.
3. Add tests for mismatch and valid path.
```

## Task 04 - Use Scheduler Parameters From Config (Medium, Accuracy/Optimization)

```text
Task: Make scheduler use config values instead of hardcoded constants.

Why this matters:
- Current config fields are ignored, making experiments non-reproducible and tuning ineffective.

Evidence:
- classifier/config_improved.yaml:40
- classifier/config_improved.yaml:41
- classifier/src/cli/train.py:303
- classifier/src/cli/train.py:307

Required change:
- Read scheduler_factor and scheduler_patience from config with sensible defaults.
- Keep current behavior as default if values are absent.

Acceptance criteria:
1. Changing config values changes instantiated ReduceLROnPlateau settings.
2. Logs reflect actual values used.
3. Backward compatibility preserved for configs without these keys.
```

## Task 05 - Align Evaluation Metrics With Saved Model Artifact (High, Accuracy Reporting Integrity) [FIXED]

```text
Task: Ensure printed/saved metrics correspond to the exact model artifact being saved.

Why this matters:
- Currently test metrics are computed on final in-memory model, but copied artifact is best_model.pth.
- This can misreport deployed model quality.

Evidence:
- classifier/src/cli/train.py:374
- classifier/src/cli/train.py:422
- classifier/src/cli/train.py:424
- classifier/src/cli/train.py:418

Required change:
- Pick one strategy and make it consistent:
  A) Evaluate best checkpoint and name artifacts from that result, or
  B) Save final model artifact and report final metrics.
- Keep behavior explicit in logs.

Acceptance criteria:
1. Reported exact/tolerance metrics match the same checkpoint file that gets named/saved.
2. No ambiguity in console output.
3. Add regression test for consistency.
```

## Task 06 - Prevent Hold-Layout Leakage Across Splits (Medium-High, Accuracy Validity)

```text
Task: Add optional group-aware splitting to avoid train/val/test leakage from duplicate layouts.

Why this matters:
- Dataset contains duplicate hold layouts and conflicting labels; row-level random split can leak near-identical samples across splits.
- This can inflate measured accuracy and reduce real-world generalization validity.

Evidence:
- data/problems.json quick scan:
  - total rows: 44,435
  - duplicate rows by canonical layout: 251
  - duplicated layouts: 238
  - layouts with conflicting grades: 94

Required change:
- Add optional grouped split mode keyed by canonical hold layout hash.
- Ensure same layout hash cannot appear in more than one split.
- Document tradeoffs and provide switch in config.

Acceptance criteria:
1. Grouped split mode available and tested.
2. Validation confirms no overlapping layout hashes across train/val/test in grouped mode.
3. Existing non-grouped split mode remains available for backward compatibility.
```

## Task 07 - Improve Default Class-Space Configuration for Current Dataset (Medium, Accuracy/Calibration)

```text
Task: Update defaults/docs so class space matches observed data, or clearly gate unsupported classes.

Why this matters:
- Current default config uses 19 classes while current dataset has 15 observed grades.
- Extra never-seen classes can hurt calibration and waste capacity.

Evidence:
- classifier/config.yaml (num_classes: 19)
- data/problems.json grade scan shows 15 observed classes in current data export.

Required change:
- Either:
  A) Keep 19 classes but explicitly document rationale and expected impact, or
  B) Provide a default filtered config path as recommended baseline.
- Make recommendation in README/USAGE explicit.

Acceptance criteria:
1. Defaults and docs are internally consistent.
2. User can understand recommended setup for current data snapshot without guessing.
```

## Task 08 - Fix train_command Crash When Confusion Matrix Saving Is Disabled (Critical, Runtime)

```text
Task: Fix UnboundLocalError for cm_path in train_command.

Why this matters:
- Training can fail at the end when evaluation.save_confusion_matrix=false.

Evidence:
- classifier/src/cli/train.py:390
- classifier/src/cli/train.py:428
- Reproducible error: UnboundLocalError: cannot access local variable 'cm_path'

Required change:
- Initialize cm_path safely before conditional, or guard its usage.
- Keep log_test_results call working regardless of confusion matrix flag.

Acceptance criteria:
1. training completes successfully with save_confusion_matrix=true and false.
2. No UnboundLocalError path remains.
3. Add regression test.
```

## Task 09 - Make CLI Output Robust on Non-UTF-8 Windows Consoles (High, Runtime/UX)

```text
Task: Remove hard dependency on Unicode glyph output in CLI prints.

Why this matters:
- On cp1252/non-UTF consoles, startup can crash with UnicodeEncodeError.

Evidence:
- classifier/src/cli/train.py:67
- classifier/src/cli/evaluate.py:83
- classifier/src/cli/predict.py:75
- classifier/src/cli/utils.py:49

Required change:
- Replace non-ASCII symbols with ASCII-safe output, or provide a safe-output mode/fallback.
- Ensure CLI does not crash from encoding issues.

Acceptance criteria:
1. Commands run on cp1252 console without setting PYTHONIOENCODING.
2. Messages remain readable and informative.
3. Tests cover at least one simulated non-UTF output path.
```

## Task 10 - Align predict_batch Behavior With Spec (Low-Medium, API Consistency)

```text
Task: Make batch prediction return per-item errors instead of aborting full batch.

Why this matters:
- Spec says failed items should return error info while successful items continue.
- Implementation currently raises on first bad problem.

Evidence:
- classifier/spec.md:483
- classifier/src/predictor.py:424

Required change:
- In predict_batch, catch per-item exceptions and append structured error result for that item.
- Do not fail entire batch unless input container itself is invalid.

Acceptance criteria:
1. Mixed valid/invalid input returns full-length result list.
2. Invalid items include clear error field with item index/context.
3. Tests updated for mixed batch behavior.
```

## Task 11 - Improve Split Error Handling for Small/Imbalanced Edge Cases (Low-Medium, Reliability)

```text
Task: Add pre-validation for stratified split feasibility with clearer error messages.

Why this matters:
- Some small datasets still fail inside sklearn with raw errors despite current checks.

Evidence:
- classifier/src/dataset.py:177
- Example failure: test_size smaller than number of classes in stratified split.

Required change:
- Validate feasibility before calling StratifiedShuffleSplit:
  - test split must contain at least one sample per class when stratifying
  - train/val second split must also be feasible
- Raise domain-friendly ValueError with remediation hints.

Acceptance criteria:
1. No raw sklearn stratification error leaks for known small-data edge cases.
2. Error explains exactly what to change (more data, fewer classes, or ratio changes).
3. Add tests for edge-case failures.
```

## Task 12 - Resolve Training History API/Doc Drift (Low, Maintainability)

```text
Task: Reconcile documentation/tests with Trainer history persistence behavior.

Why this matters:
- Docs/tests reference training_history.json and Trainer methods not currently present.
- Creates confusion and failing expectations.

Evidence:
- classifier/USAGE.md:18
- classifier/tests/test_trainer.py:819
- classifier/tests/test_trainer.py:845

Required change:
- Choose one:
  A) Implement get_history/save_history and write training_history.json in CLI flow, or
  B) Remove/update outdated docs/tests to match current behavior.
- Keep a single source of truth.

Acceptance criteria:
1. Docs, tests, and implementation agree.
2. No stale references to unsupported APIs remain.
3. If implemented, history file path is deterministic and tested.
```

## Task 13 - Add Full Reproducibility Controls for Training Runs (Medium, Experiment Quality)

```text
Task: Add optional global reproducibility setup for training.

Why this matters:
- Current setup controls split seed but not full training determinism.
- Re-run variance can mask real improvements/regressions.

Evidence:
- classifier/src/cli/train.py:137 (split seed used)
- No centralized torch/numpy/random seeding and deterministic flags in training flow.

Required change:
- Add config option(s) for reproducibility seed.
- Seed Python random, NumPy, torch CPU/CUDA, and optional deterministic backend flags.
- Log determinism settings at run start.

Acceptance criteria:
1. Same config+seed yields repeatable metrics within expected deterministic bounds.
2. Deterministic mode is opt-in if performance tradeoff exists.
3. Behavior documented in README/USAGE.
```
