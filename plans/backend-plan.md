# FastAPI Backend for Moonboard Grade Prediction

## Overview

Create a FastAPI backend that uses the classifier package as an installed dependency. The backend imports Predictor from classifier.src and provides a REST API for grade predictions. Each implementation step leaves the project in a fully working, committable state.

## Project Structure

```
moonboard-grader/
├── classifier/              # Existing - will become installable package
│   ├── src/                # Core inference code (Predictor, models, etc.)
│   ├── setup.py            # NEW - Makes classifier installable
│   └── ...
└── backend/                # NEW - FastAPI application
    ├── app/
    │   ├── __init__.py
    │   └── main.py         # FastAPI app (imports from classifier.src)
    ├── models/
    │   └── model_for_inference.pth  # User will add this
    ├── requirements.txt
    └── README.md
```

## Implementation Steps

### 1. Make Classifier Installable

Create `classifier/setup.py` to make the classifier package installable:

- Package name: `moonboard-classifier`
- Version: 0.1.0
- Include `src` package and subpackages
- Specify core dependencies: torch, numpy
- Support editable installation with `pip install -e ./classifier`

After this step: Classifier can be installed as a package

Committable with: "Make classifier package installable"

### 2. Create Backend Application

Create complete backend folder structure:

- Create `backend/` directory at root level
- Create `backend/app/` subdirectory for application code
- Create `backend/models/` subdirectory for model files
- Create `backend/app/__init__.py` (empty)
- Create `backend/models/README.md` noting where to place model_for_inference.pth

Create `backend/requirements.txt` with:

- fastapi
- uvicorn[standard]
- python-multipart
- -e ../classifier (reference to local classifier package)

Note: torch and numpy will be installed via classifier's dependencies

Create `backend/app/main.py` with:

- FastAPI app initialization with title and description
- CORS middleware configuration (allow all origins for local development)
- Import Predictor from `classifier.src`
- Model loading at startup (hardcoded path: `models/model_for_inference.pth`)
- POST `/predict` endpoint accepting problem data and optional top_k parameter
- GET `/health` endpoint for basic health check
- GET `/model-info` endpoint to retrieve loaded model information
- Pydantic models for request/response validation
- Proper error handling with HTTPException

After this step: Backend is fully functional and runnable with `uvicorn app.main:app --reload` (will fail gracefully if model file missing)

Committable with: "Add FastAPI backend with prediction endpoint"

### 3. Create Documentation

Create `backend/README.md` with:

- Project description and architecture (explains dependency on classifier)
- Installation instructions:

  1. Install classifier package: `pip install -e ../classifier`
  2. Install backend requirements: `pip install -r requirements.txt`

- How to add the model file to `models/` directory
- How to run the server: `uvicorn app.main:app --reload --host localhost --port 8000`
- API endpoint documentation with example curl requests
- Link to interactive Swagger docs at http://localhost:8000/docs
- Testing instructions

Add section for future enhancements:

- Multi-model support (model selection via API parameter)
- Model upload/management endpoints
- Caching for faster repeated predictions
- Batch prediction endpoint for multiple problems
- Authentication/API keys for production use
- Performance monitoring and request logging
- Docker containerization with multi-stage builds
- Model versioning and A/B testing

After this step: Complete documentation in place

Committable with: "Add backend documentation and future roadmap"