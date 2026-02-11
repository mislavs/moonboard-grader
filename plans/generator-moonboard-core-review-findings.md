# Generator + Moonboard Core Review Findings

Review of how the `generator` project uses `moonboard_core` and identified areas where functionality is duplicated rather than reused.

**Overall:** The generator correctly uses `moonboard_core` for fundamental operations (data loading via `load_dataset`, grid building via `create_grid_tensor`, grade encoding/decoding, grid-to-moves conversion). However, there is significant duplicated label-mapping logic and several places where `moonboard_core` utilities are bypassed in favor of hand-rolled equivalents.

---

## Tasks

### 1. Refactor label remapping to use moonboard_core's `remap_label` / `unmap_label`

**Status:** Done

**Severity:** High

`moonboard_core` exports `remap_label(label, offset)` and `unmap_label(label, offset)` specifically for converting between global grade labels and model-local labels. The classifier project uses them directly. The generator has built a parallel system that reimplements the same `label - offset` / `label + offset` arithmetic in three places:

- `MoonBoardDataset.global_to_model_label()` / `model_to_global_label()` in `generator/src/dataset.py` (lines 174-204)
- `EvaluationLabelContext.global_to_model_label()` / `model_to_global_label()` in `generator/src/label_space.py` (lines 147-167)
- `ProblemGenerator.global_to_model_label()` / `model_to_global_label()` in `generator/src/generator.py` (lines 128-134) — delegates to `EvaluationLabelContext`

The generator also introduces a `label_space_mode` concept (`"remapped"` vs `"global_legacy"`) that adds conditional branching throughout the codebase. This distinction is unnecessary because `remap_label` with `offset=0` already handles the global case naturally.

**Action:**
- Replace the custom `global_to_model_label` / `model_to_global_label` methods with calls to `remap_label` / `unmap_label`
- Eliminate the `label_space_mode` branching — use offset=0 for unfiltered models and offset=min_grade_index for filtered models, matching the classifier's approach
- Simplify `EvaluationLabelContext` to store only the offset and range, delegating arithmetic to `moonboard_core`

---

### 2. Replace `_build_grade_mappings` with direct `encode_grade` / `decode_grade` calls

**Severity:** Medium

`MoonBoardDataset._build_grade_mappings()` (`generator/src/dataset.py` lines 111-125) reconstructs `grade_to_label` and `label_to_grade` dictionaries from `get_all_grades()`. These are functionally identical to the internal `_GRADE_TO_LABEL` / `_LABEL_TO_GRADE` dicts in `moonboard_core/grade_encoder.py`, exposed through `encode_grade()` and `decode_grade()`.

The downstream methods `get_grade_from_label()` (line 206) and `get_label_from_grade()` (line 222) that consume these dicts could call `decode_grade()` / `encode_grade()` directly (wrapped with the model-to-global conversion).

**Action:**
- Remove `_build_grade_mappings()` and the `grade_names`, `grade_to_label`, `label_to_grade` instance attributes
- Replace `get_grade_from_label()` with `decode_grade(self.model_to_global_label(label))`
- Replace `get_label_from_grade()` with `remap_label(encode_grade(grade), self.grade_offset)`
- Use `get_all_grades()` directly where `self.grade_names` is currently used

---

### 3. Consolidate duplicated `_safe_decode_grade` helper

**Severity:** Medium

The same helper function is copy-pasted in two files:

- `generator/src/dataset.py` (lines 21-25)
- `generator/src/label_space.py` (lines 16-20)

```python
def _safe_decode_grade(label: int) -> str:
    try:
        return decode_grade(label)
    except Exception:
        return str(label)
```

**Action:**
- Remove the duplicate from `dataset.py`
- Import it from `label_space.py`, or move it to a shared utility within the generator
- Alternatively, consider adding a `safe_decode_grade` (or `try_decode_grade`) to `moonboard_core` since other projects may benefit from the same pattern

---

### 4. Refactor `extract_problem_stats` to use moonboard_core position parsing

**Severity:** Medium

`generator/src/evaluator/utils.py` `extract_problem_stats()` (lines 44-72) manually parses position strings and counts holds:

- Row extraction uses `int(pos[1:])` which is a fragile reimplementation of what `moonboard_core.parse_position()` handles canonically (with validation, multi-digit row support, edge case handling)
- Hold counting (`isStart`/`IsStart`, `isEnd`/`IsEnd` dual-casing checks) reimplements logic that `moonboard_core.get_channel_counts()` provides for tensors
- The dual-casing defensive check suggests data with inconsistent key casing — if data always flows through `moonboard_core.grid_to_moves()`, casing would be consistent

**Action:**
- Replace `int(pos[1:])` with `parse_position(pos)` from `moonboard_core` to get the row index
- Consider converting moves to a tensor via `create_grid_tensor()` and using `get_channel_counts()` for hold counting, or at minimum standardize on a single key-casing convention

---

### 5. Extract raw JSON loading into moonboard_core

**Severity:** Low-Medium

`generator/src/evaluator/statistical.py` (lines 64-81) manually loads and validates the JSON dataset structure (opens file, parses JSON, checks for `"data"` key). This duplicates the validation logic in `moonboard_core.data_processor.load_dataset()` (lines 119-149).

The reason for the duplication is that `statistical.py` needs raw problem dicts (to extract statistics) rather than processed `(tensor, label)` tuples. `load_dataset()` only returns the processed form.

**Action:**
- Add a `load_raw_problems(json_path)` function to `moonboard_core.data_processor` that returns the validated list of problem dicts before tensor conversion
- Refactor `load_dataset()` to call `load_raw_problems()` internally, then process each problem
- Update `statistical.py` to use `load_raw_problems()` instead of hand-rolling JSON loading

---

### 6. Consider unifying checkpoint label context across classifier and generator

**Severity:** Low

The `EvaluationLabelContext` dataclass and its checkpoint inference functions (`infer_num_model_grades`, `infer_label_space_mode`, `build_label_context`) in `generator/src/label_space.py` are generator-specific in their checkpoint format knowledge. However, the classifier stores similar metadata in its checkpoints (`grade_offset`, `min_grade_index`, `max_grade_index`).

If the checkpoint metadata format were standardized across both projects, the label context logic could live in `moonboard_core` as a shared `CheckpointLabelContext` utility, reducing the per-project label management burden.

**Action:** This is a longer-term consideration. After completing task 1 (which simplifies the generator's label space handling), evaluate whether a shared checkpoint label context abstraction in `moonboard_core` would be beneficial or overengineered given the current codebase size.

---

## Open Questions

- Task 1 is the highest-value change but also the most invasive — it touches `dataset.py`, `label_space.py`, `generator.py`, and all evaluator modules that use `EvaluationLabelContext`. Consider whether to simplify incrementally (e.g., first replace the arithmetic, then remove the mode concept).
- Task 5 requires a new function in `moonboard_core`, which means coordinating changes across two packages. Assess whether `statistical.py` is the only consumer or if other projects would benefit from raw problem loading.
- The `EvaluationLabelContext` checkpoint inference logic (task 6) has legitimate backward-compatibility concerns for old checkpoints. Any unification needs to preserve that compatibility.
