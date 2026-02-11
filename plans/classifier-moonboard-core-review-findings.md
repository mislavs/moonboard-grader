# Classifier + Moonboard Core Review Findings

Review of how the `classifier` project uses `moonboard_core` and identified areas for improvement.

**Overall:** The classifier correctly delegates core logic (grade encoding, grid building, data loading, position parsing) to `moonboard_core` and does not re-implement it. The issues below are about hardcoded constants, internal duplication, import hygiene, and DRY violations.

---

## Tasks

### 1. Replace hardcoded grid dimensions with moonboard_core constants

**Severity:** Medium

`moonboard_core` exports `ROWS = 18` and `COLS = 11` from `position_parser.py`, but the classifier hardcodes the grid shape `(3, 18, 11)` as magic numbers in multiple places:

- `classifier/src/models.py` line 41: `self.input_size = 3 * 18 * 11  # 594`
- `classifier/src/dataset.py` lines 57-66: shape validation against literal `(3, 18, 11)`
- `classifier/src/predictor.py` line 459: shape validation against literal `(3, 18, 11)`
- `ConvolutionalModel` derives `self.flattened_size = 128 * 4 * 2` from the input dimensions without referencing constants

**Action:** Define a `GRID_SHAPE = (3, ROWS, COLS)` constant in `moonboard_core` (or a `CHANNELS = 3` constant alongside `ROWS`/`COLS`) and use it throughout the classifier for shape validation and input size calculations.

---

### 2. Remove duplicate data splitting module

**Severity:** Medium

The classifier has two modules that implement the same stratified train/val/test splitting logic:

- `classifier/src/dataset.py` — `create_data_splits()` (lines 123-219)
- `classifier/src/data_splitter.py` — `create_stratified_splits()` + `create_datasets()` (lines 15-108)

Both use the exact same two-step `StratifiedShuffleSplit` algorithm (split off test first, then split remainder into train/val) and both return `MoonboardDataset` objects. The production code (`cli/train.py`) uses `data_splitter.create_datasets()`, so `dataset.py::create_data_splits()` appears unused.

**Action:**
- Verify `create_data_splits()` is not used anywhere
- Remove it from `dataset.py`
- Also remove the associated `get_split_info()` function if unused
- Keep `data_splitter.py` as the single source of truth for splitting

---

### 3. Clean up moonboard_core re-exports from classifier __init__.py

**Severity:** Low-Medium

`classifier/src/__init__.py` re-exports the entire `moonboard_core` public API (grade encoding, position parsing, grid building, data processing). This means internal code can do `from src import decode_grade`, which obscures the true origin of the function.

The import style is also inconsistent across the classifier:
- `cli/train.py` imports moonboard_core functions via `from src import ...` (indirect)
- `cli/predict.py` imports directly: `from moonboard_core.grade_encoder import encode_grade` (direct)
- `evaluator.py` imports directly: `from moonboard_core.grade_encoder import get_all_grades, get_num_grades`

**Action:**
- Remove the moonboard_core re-exports from `classifier/src/__init__.py`
- Update all internal classifier code to import directly from `moonboard_core`
- This makes dependencies explicit and follows the pattern already used by `predict.py` and `evaluator.py`

---

### 4. Extract duplicated label unmapping logic in predictor.py

**Severity:** Low-Medium

`predictor.py` contains the same label-unmapping + grade-decoding block copy-pasted across three methods: `predict()`, `predict_with_attention()`, and `predict_from_tensor()`. The repeated pattern:

```python
all_probs = {}
for model_label, prob in enumerate(probs_array):
    if self.grade_offset > 0:
        original_label = unmap_label(model_label, self.grade_offset)
        grade = decode_grade(original_label)
    else:
        grade = decode_grade(model_label)
    all_probs[grade] = float(prob)

top_k_indices = np.argsort(probs_array)[-return_top_k:][::-1]
top_k_predictions = []
for idx in top_k_indices:
    if self.grade_offset > 0:
        original_label = unmap_label(int(idx), self.grade_offset)
        grade = decode_grade(original_label)
    else:
        grade = decode_grade(int(idx))
    top_k_predictions.append((grade, float(probs_array[idx])))
```

This appears at lines ~218-236, ~357-376, and ~491-510.

**Action:** Extract a private helper method like `_decode_probabilities(self, probs_array, return_top_k)` that encapsulates the unmapping, decoding, and top-k logic. All three methods should call this helper.

---

### 5. Minor: MoonboardDataset.get_label_distribution() overlaps with get_dataset_stats()

**Severity:** Low

`MoonboardDataset.get_label_distribution()` computes a `{label: count}` dict, which overlaps conceptually with `moonboard_core.data_processor.get_dataset_stats()['grade_distribution']`. The difference is the data representation (PyTorch Dataset's numpy arrays vs. list of tuples).

**Action:** This is acceptable as-is since the Dataset class needs to operate on its own internal representation. No change strictly required, but worth being aware of for future refactoring if data representations are unified.
