# Multi-Board Support: Backend

Implement the backend infrastructure for multi-board and multi-angle support so that adding new boards is a matter of dropping data/model files and updating `config/board_setups.json`.

Backend and frontend are independent -- the backend ships new fields in `/board-setups` that the frontend currently ignores, and the frontend already sends `hold_setup`/`angle` params that the backend currently ignores. Either side can be deployed first.

---

## Step 1: Config Schema

Extend `HoldSetup` and `AngleConfig` in `moonboard_core/board_config.py` with new optional fields (backward-compatible with existing config):

**`HoldSetup`** additions:
- `board_image: Optional[str] = None` -- filename in `frontend/public/boards/`
- `beta_solving_supported: bool = True`

**`AngleConfig`** additions:
- `generator_model_file: Optional[str] = None` -- path to generator model
- `analytics_file: Optional[str] = None` -- path to pre-computed hold_stats.json

Update `_parse_hold_setup` to read these with defaults so the current config (which lacks them) keeps working. Update tests in `moonboard_core/tests/test_board_config.py`.

## Step 2: Backend API Schema + Board Setups Route

**`backend/app/models/schemas.py`**:
- `HoldSetupResponse`: add `betaSolvingSupported: bool`, `boardImage: Optional[str]`
- `AngleConfigResponse`: add `hasGenerator: bool`, `hasAnalytics: bool`

**`backend/app/api/routes/board_setups.py`**:
- Pass through new fields, check file existence for `hasGenerator` / `hasAnalytics`

## Step 3: Service Registry

New file: `backend/app/services/service_registry.py`

```python
class ServiceRegistry:
    def get_predictor(setup_id, angle) -> PredictorService
    def get_problem_service(setup_id, angle) -> ProblemService
    def get_generator(setup_id, angle) -> GeneratorService
    def get_analytics_path(setup_id, angle) -> Path | None
```

Lookup logic: if `setup_id`/`angle` are None, fall back to the default config. If a specific combo doesn't have a service (e.g., no model file), raise HTTP 503 with a clear message.

## Step 4: Startup

Update `backend/app/main.py` -- replace the current hardcoded single-model loading with:

1. Load `BoardConfigRegistry` from `settings.board_config_path`
2. For each `(setup, angle_config)` in the registry:
   - Create `PredictorService` from `angle_config.model_file` (if file exists)
   - Create `ProblemService` from `angle_config.data_file`
   - Create `GeneratorService` from `angle_config.generator_model_file` (if file exists)
   - Note `angle_config.analytics_file` for analytics routing
3. **Backward compat for default combo**: if `generator_model_file` is null, try `{model_dir}/generator_model.pth`. If `analytics_file` is null, try `data/hold_stats.json`.
4. Store everything in a `ServiceRegistry` instance

## Step 5: Dependencies

Update `backend/app/api/dependencies.py`:
- Replace the three global service variables with a single `_service_registry: ServiceRegistry`
- Add `get_service_registry() -> ServiceRegistry`
- Keep existing `get_predictor_service()` / `get_problem_service()` / etc. but have them delegate to the registry's default

## Step 6: Route Handlers

Each handler already has `hold_setup` and `angle` query params. Change from using `Depends(get_loaded_predictor)` to resolving from the registry:

- **`backend/app/api/routes/prediction.py`**: `registry.get_predictor(hold_setup, angle)`
- **`backend/app/api/routes/problems.py`**: `registry.get_problem_service(hold_setup, angle)`
- **`backend/app/api/routes/generation.py`**: `registry.get_generator(hold_setup, angle)`
- **`backend/app/api/routes/analytics.py`**: `registry.get_analytics_path(hold_setup, angle)` -- replace hardcoded `ANALYTICS_DATA_PATH`. Return 404 if analytics unavailable for the combo.

## Step 7: Backend Tests

- `backend/tests/conftest.py`: create registry fixtures
- `backend/tests/test_board_setups.py`: verify new response fields
- Existing route tests in `backend/tests/test_api.py` and `backend/tests/test_service.py`: ensure they still pass with the registry pattern
- Run `pytest` to verify all tests pass
