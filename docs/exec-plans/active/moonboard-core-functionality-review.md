# Moonboard Core: Shared Functionality Review

Findings from reviewing `moonboard_core`, `classifier`, `generator`, and `beta-classifier` for shared functionality that should be consolidated into `moonboard_core`.

---

## Task 1: Add `safe_decode_grade()` to `moonboard_core.grade_encoder` [HIGH]

- [ ] Add `safe_decode_grade(label: int) -> str` to `moonboard_core/grade_encoder.py`
- [ ] Export from `moonboard_core/__init__.py`
- [ ] Add tests in `moonboard_core/tests/test_grade_encoder.py`
- [ ] Update `generator/src/dataset.py` to import from `moonboard_core`
- [ ] Update `generator/src/label_space.py` to import from `moonboard_core`
- [ ] Remove the local `_safe_decode_grade` definitions from both generator files

**Context:** An identical `_safe_decode_grade` helper exists in `generator/src/dataset.py` (line 21) and `generator/src/label_space.py` (line 16). It wraps `decode_grade()` to return `str(label)` on failure instead of raising. Useful for logging, error messages, and debugging.

---

## Task 2: Extract shared evaluation metrics into `moonboard_core` [HIGH]

- [ ] Create `moonboard_core/evaluation.py` with pure metric functions:
  - `calculate_exact_accuracy(predictions, labels) -> float`
  - `calculate_tolerance_accuracy(predictions, labels, tolerance) -> float`
  - `calculate_mean_absolute_error(predictions, labels) -> float`
  - `calculate_macro_accuracy(predictions, labels) -> float`
  - `per_grade_metrics(predictions, labels, grade_names) -> dict`
  - `generate_confusion_matrix(predictions, labels, num_classes) -> np.ndarray`
  - `plot_confusion_matrix(cm, grade_names, ...) -> Figure`
  - `get_metrics_summary(predictions, labels, grade_names) -> dict`
- [ ] Standardize return format (decide: percentages 0-100 vs fractions 0-1)
- [ ] Add tests in `moonboard_core/tests/test_evaluation.py`
- [ ] Export from `moonboard_core/__init__.py`
- [ ] Refactor `classifier/src/evaluator.py` to delegate to `moonboard_core.evaluation`
- [ ] Refactor `beta-classifier/src/evaluator.py` to delegate to `moonboard_core.evaluation`

**Context:** `classifier/src/evaluator.py` and `beta-classifier/src/evaluator.py` independently implement the same core metrics: exact accuracy, tolerance accuracy (+-1, +-2), MAE, per-grade precision/recall/F1, confusion matrix generation and plotting. Both use `sklearn.metrics` under the hood. The only differences are return format (percentage vs fraction) and whether functions are standalone or class methods. Each project can keep a thin model-specific layer (e.g., running inference on a DataLoader) but delegate the pure metric calculations to the shared module.

---

## Task 3: Create a shared label context abstraction in `moonboard_core` [HIGH]

- [ ] Create `moonboard_core/label_context.py` with:
  - `LabelSpaceMode` type alias (`Literal["remapped", "global_legacy"]`)
  - `LabelContext` dataclass with `global_to_model_label()`, `model_to_global_label()`, `model_label_to_grade_name()`, properties for grade bounds
  - Factory function `create_label_context(mode, grade_offset, min_grade_index, max_grade_index, num_model_grades)`
- [ ] Add tests in `moonboard_core/tests/test_label_context.py`
- [ ] Export from `moonboard_core/__init__.py`
- [ ] Refactor `generator/src/label_space.py` to use core `LabelContext` (keep checkpoint inference logic in generator)
- [ ] Refactor `generator/src/dataset.py` `MoonBoardDataset` to use core `LabelContext`
- [ ] Refactor `classifier/src/predictor.py` to use core `LabelContext` instead of ad-hoc `grade_offset` / `unmap_label` patterns

**Context:** Both the classifier and generator need to handle filtered grade ranges where a model trained on a subset of grades (e.g., 6A+ to 7C, indices 2-12) outputs labels 0-10 that must be mapped back to global indices. The classifier handles this ad-hoc in `predictor.py`, repeating the same `if self.grade_offset > 0: unmap_label(...)` block three times across `predict()`, `predict_with_attention()`, and `predict_from_tensor()`. The generator has a structured `EvaluationLabelContext` in `label_space.py` plus duplicate `global_to_model_label()`/`model_to_global_label()` methods in its dataset class. `moonboard_core` already has `remap_label`/`unmap_label` -- this would be a natural higher-level abstraction on top.

---

## Task 4: Extract stratified data splitting into `moonboard_core` [MEDIUM]

- [ ] Add `create_stratified_splits(labels, train_ratio, val_ratio, test_ratio, seed) -> (train_idx, val_idx, test_idx)` to `moonboard_core` (returns index arrays only, framework-agnostic)
- [ ] Add tests
- [ ] Export from `moonboard_core/__init__.py`
- [ ] Refactor `classifier/src/dataset.py` `create_data_splits()` to use it
- [ ] Refactor `classifier/src/data_splitter.py` `create_stratified_splits()` to use it
- [ ] Refactor `beta-classifier/src/dataset.py` `create_data_splits()` to use it

**Context:** Three independent implementations of stratified train/val/test splitting exist: `classifier/src/dataset.py`, `classifier/src/data_splitter.py` (the classifier has two!), and `beta-classifier/src/dataset.py`. All do the same two-phase split (first separate test, then split remaining into train/val). A framework-agnostic index-based splitter in core would eliminate all three duplicates while letting each project wrap indices into its own Dataset type.

---

## Task 5: Move `extract_problem_stats()` to `moonboard_core` [MEDIUM]

- [ ] Add `extract_problem_stats(problem_dict) -> dict` to `moonboard_core/data_processor.py` (or a new `problem_analysis.py`)
- [ ] Handle both camelCase (`isStart`, `description`) and PascalCase (`IsStart`, `Description`) key formats consistently
- [ ] Add tests
- [ ] Export from `moonboard_core/__init__.py`
- [ ] Update `generator/src/evaluator/utils.py` to import from `moonboard_core`

**Context:** `generator/src/evaluator/utils.py` has an `extract_problem_stats()` function that computes hold counts (total, start, middle, end) and vertical spread from a problem dictionary. This is general-purpose problem analysis that any consumer (backend, classifier, etc.) might need. Note that `moonboard_core.data_processor.get_dataset_stats()` already computes similar aggregate stats but only at the dataset level from tensors. This function operates on raw problem dicts and includes `vertical_spread`, complementing existing functionality.

---

## Not recommended for extraction (keep project-specific)

- **PyTorch Dataset classes** -- `classifier/MoonboardDataset`, `generator/MoonBoardDataset`, and `beta-classifier/MoveSequenceDataset` have fundamentally different interfaces (pre-loaded numpy arrays vs file-path-based loading vs variable-length sequences). Unifying them would add complexity without benefit.
- **DataLoader creation functions** -- Each project has different configuration needs (stratified splits, random splits, custom collate functions). These are appropriately project-specific.
