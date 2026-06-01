# Moonboard Core Review Findings

Code review of `moonboard_core/` — shared Python utilities for processing Moonboard climbing problems.

## Tasks

### Medium Priority

- [ ] **Add missing field validation in `_parse_hold_setup`**
  - **Type**: Bug
  - **File**: `moonboard_core/board_config.py` (lines 102–108)
  - **Issue**: `angle_data['angle']` and `angle_data['dataFile']` are accessed without checking existence first. A config file missing these keys raises an opaque `KeyError` instead of the descriptive `ValueError` used by all other validation in the class.
  - **Fix**: Validate required keys explicitly, or wrap in try/except converting `KeyError` to `ValueError` with a message like `"Angle configuration missing required field 'angle'"`.

- [ ] **Add `max_grade_index` upper bound check in `filter_dataset_by_grades`**
  - **Type**: Risk (API inconsistency)
  - **File**: `moonboard_core/data_processor.py` (lines 206–219)
  - **Issue**: `get_filtered_grade_names(0, 100)` correctly raises `ValueError`, but `filter_dataset_by_grades(dataset, 0, 100)` silently accepts it. A caller filtering with an out-of-range index gets no warning and may include labels the system can't decode.
  - **Fix**: Add bounds check: `if max_grade_index >= get_num_grades(): raise ValueError(...)`.

- [ ] **Warn or raise when `get_registry()` is called with a different `config_path` after initialization**
  - **Type**: Questionable Approach
  - **File**: `moonboard_core/board_config.py` (lines 192–207)
  - **Issue**: The function accepts `config_path` on every call but only uses it on the first. Subsequent calls with a different path silently return the cached registry. The backend's `problem_service.py` uses path manipulation to find the config — if the module gets imported in a different order, the wrong config could silently be used.
  - **Fix**: Either (a) warn/raise when a `config_path` is provided but the registry already exists with a different path, or (b) remove the `config_path` parameter and require explicit `BoardConfigRegistry(path)` for custom paths.


### Low Priority


- [ ] **Suppress chained exceptions in `position_parser.py`**
  - **Type**: Improvement
  - **File**: `moonboard_core/position_parser.py` (lines 59–64)
  - **Issue**: When `int(row_str)` fails, a new `ValueError` is raised inside the `except ValueError` handler. Python 3 attaches the original as `__context__`, producing double-stacked tracebacks.
  - **Fix**: Add `from None`: `raise ValueError(...) from None`.

- [ ] **Add position string validation to `validate_moves`**
  - **Type**: Risk (incomplete validation)
  - **File**: `moonboard_core/grid_to_moves.py` (lines 129–201)
  - **Issue**: The function checks hold counts (start/middle/end) but doesn't validate that `description` fields are valid Moonboard positions. Invalid positions would be caught downstream by `create_grid_tensor`, but callers might assume `validate_moves` provides complete validation.
  - **Fix**: Optionally validate each move's `description` via `validate_position()` and add position errors to the error list.

### Cleanup (do last)

- [ ] **Remove dead code: 6 unused exports across `moonboard_core`**
  - **Note**: Before starting, re-check that these are still dead — earlier refactoring tasks may have introduced new callers.
  - **Type**: Dead code removal
  - **Files**: `moonboard_core/data_processor.py`, `moonboard_core/grid_builder.py`, `moonboard_core/position_parser.py`, `moonboard_core/__init__.py`, `classifier/src/__init__.py`
  - **Issue**: The following public exports have zero production callers (only re-exported by `classifier/src/__init__.py` but never invoked):
    - `save_processed_dataset` — also has `NpzFile` handle leak and `.npz` extension mismatch bugs
    - `load_processed_dataset` — also has `NpzFile` handle leak and `.npz` extension mismatch bugs
    - `process_problem` — only called in its own test
    - `get_channel_counts` — only called in its own test
    - `tensor_to_moves` — only called in its own test; also duplicates `grid_to_moves` logic
    - `validate_position` — only called in its own test
  - **Fix**: Remove all six functions, their exports from `moonboard_core/__init__.py` and `classifier/src/__init__.py`, and their corresponding tests. Removing `tensor_to_moves` also resolves the duplicate logic finding.

## Open Questions / Assumptions

- 6 of 20 public exports are dead code (only re-exported by `classifier/__init__.py`, never called): `save_processed_dataset`, `load_processed_dataset`, `process_problem`, `get_channel_counts`, `tensor_to_moves`, `validate_position`. Recommended to remove entirely rather than maintain/fix.
- The production `board_setups.json` has only one setup with one angle, so the missing-key validation bug is unlikely to trigger in production but likely during development with malformed configs.
- `load_dataset` in `data_processor.py` fails the entire load if any single problem has an error. This is a deliberate data-quality gate. Whether partial loads are ever desired is an open question.
