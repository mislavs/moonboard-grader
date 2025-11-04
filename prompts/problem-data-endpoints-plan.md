# Add Problem Data Endpoints

## Overview

Create two new FastAPI endpoints to serve problem data from a JSON file: a list endpoint returning basic info (ID, name, grade) for dropdowns, and a detail endpoint returning complete problem data including moves.

## Implementation Steps

### 1. Add Configuration and Schema Models ✅ COMPLETED

**Files:**

- `backend/app/core/config.py`
- `backend/app/models/schemas.py`

Add `problems_data_path` setting to the Settings class (default: `../classifier/data/problems.json`).

Add new Pydantic models for problem data:

- `ProblemListItem`: Basic info (apiId, name, grade) for the list endpoint
- `ProblemMove`: Move data structure matching the JSON format
- `ProblemDetail`: Complete problem data for the detail endpoint

These schemas will mirror the structure in `classifier/data/problems.json` and the frontend's `problem.ts` types.

**Status:** ✅ Added config setting and all three schema models

### 2. Create Problem Service with Tests

**Files:**

- `backend/app/services/problem_service.py` (new file)
- `backend/tests/test_problem_service.py` (new file)

Create a service class `ProblemService` that:

- Loads and caches the problems JSON file on initialization using the config setting
- Provides `get_all_problems()` method returning list of basic info (uses `ProblemListItem`)
- Provides `get_problem_by_id(api_id)` method returning full problem details (uses `ProblemDetail`)
- Handles file reading errors gracefully
- Uses singleton pattern similar to `PredictorService`

Add unit tests for the service:

- Test loading problems from JSON file
- Test `get_all_problems()` returns correct basic info
- Test `get_problem_by_id()` returns full problem details
- Test error handling for missing file and invalid IDs

### 3. Create Dependency Injection

**File:** `backend/app/api/dependencies.py`

Add `get_problem_service()` dependency function that returns the singleton `ProblemService` instance, following the pattern used for `get_predictor_service()`.

### 4. Add API Routes with Tests

**Files:**

- `backend/app/api/routes.py`
- `backend/tests/test_api.py` (update existing)

Add two new endpoints:

- `GET /problems` - Returns list of all problems with apiId, name, and grade
- `GET /problems/{api_id}` - Returns full problem details including moves array

Both endpoints will use the `ProblemService` via dependency injection and include proper error handling (404 for missing problems, 500 for file errors).

Add unit tests for the endpoints:

- Test `GET /problems` returns list with correct schema
- Test `GET /problems/{api_id}` returns full problem with correct schema
- Test 404 error for invalid problem IDs
- Use pytest fixtures for test data

### 5. Verify Implementation

- Run unit tests with `py -m pytest`
- Start the backend server with `py -m uvicorn app.main:app --reload`
- Visit http://localhost:8000/docs to see new endpoints
- Manually test `GET /problems` returns the list
- Manually test `GET /problems/{api_id}` with a valid apiId returns full problem details

## Key Files

- `backend/app/core/config.py` - Add config setting
- `backend/app/models/schemas.py` - Add problem schemas
- `backend/app/services/problem_service.py` - New service (create)
- `backend/app/api/dependencies.py` - Add dependency
- `backend/app/api/routes.py` - Add routes
- `backend/tests/test_problem_service.py` - Unit tests for service (create)
- `backend/tests/test_api.py` - Unit tests for endpoints (update)