# Moonboard Grade Predictor Backend

FastAPI backend service for predicting climbing problem grades using deep learning. This service provides a REST API that uses the `moonboard-classifier` package for inference.

## Architecture

The backend is built as a lightweight wrapper around the classifier package:

```
moonboard-grader/
├── classifier/              # Core ML package (installed as dependency)
│   ├── src/                # Predictor, models, preprocessing
│   └── setup.py            # Package definition
└── backend/                # FastAPI REST API
    ├── app/
    │   └── main.py         # API endpoints (imports from classifier.src)
    └── models/
        └── model_for_inference.pth  # Trained model checkpoint
```

The backend imports the `Predictor` class from `classifier.src` and exposes it via HTTP endpoints.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install the Classifier Package

The backend depends on the classifier package. Install it in editable mode:

```bash
pip install -e ../classifier
```

Or on Windows PowerShell:

```powershell
py -m pip install -e ..\classifier
```

This will install the `moonboard-classifier` package along with its dependencies (PyTorch, NumPy, etc.).

### Step 2: Install Backend Requirements

Install the FastAPI backend dependencies:

```bash
pip install -r requirements.txt
```

Or on Windows PowerShell:

```powershell
py -m pip install -r requirements.txt
```

### Step 3: Add the Model File

Copy your trained model to the `models/` directory:

```bash
cp ../classifier/models/best_model.pth models/model_for_inference.pth
```

Or on Windows PowerShell:

```powershell
Copy-Item ..\classifier\models\best_model.pth -Destination models\model_for_inference.pth
```

**Important:** The backend expects the model file to be named `model_for_inference.pth` in the `models/` directory.

## Running the Server

Start the development server with:

```bash
uvicorn app.main:app --reload --host localhost --port 8000
```

Or on Windows PowerShell:

```powershell
uvicorn app.main:app --reload --host localhost --port 8000
```

The server will start on `http://localhost:8000`.

### Server Options

- `--reload`: Auto-reload on code changes (development only)
- `--host localhost`: Listen on localhost only
- `--port 8000`: Port number (default: 8000)
- `--workers 4`: Number of worker processes (production)

## API Documentation

Once the server is running, you can access:

- **Interactive API docs (Swagger UI):** http://localhost:8000/docs
- **Alternative API docs (ReDoc):** http://localhost:8000/redoc
- **OpenAPI schema:** http://localhost:8000/openapi.json

## API Endpoints

### GET `/`

Root endpoint with basic API information.

**Example:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "Moonboard Grade Predictor API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

### GET `/health`

Health check endpoint.

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### GET `/model-info`

Get information about the loaded model.

**Example:**
```bash
curl http://localhost:8000/model-info
```

**Response:**
```json
{
  "model_path": "models/model_for_inference.pth",
  "device": "cpu",
  "model_exists": true
}
```

### POST `/predict`

Predict the grade of a climbing problem.

**Request Body:**
```json
{
  "moves": [
    {
      "description": "A1",
      "isStart": true,
      "isEnd": false
    },
    {
      "description": "B5",
      "isStart": false,
      "isEnd": false
    },
    {
      "description": "K10",
      "isStart": false,
      "isEnd": true
    }
  ],
  "top_k": 3
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "moves": [
      {"description": "A1", "isStart": true, "isEnd": false},
      {"description": "B5", "isStart": false, "isEnd": false},
      {"description": "K10", "isStart": false, "isEnd": true}
    ],
    "top_k": 3
  }'
```

**Response:**
```json
{
  "predicted_grade": "6B+",
  "confidence": 0.87,
  "top_k_predictions": [
    {"grade": "6B+", "probability": 0.87},
    {"grade": "6C", "probability": 0.09},
    {"grade": "6B", "probability": 0.03}
  ],
  "all_probabilities": {
    "6A": 0.001,
    "6A+": 0.002,
    "6B": 0.03,
    "6B+": 0.87,
    "6C": 0.09,
    ...
  }
}
```

**Parameters:**
- `moves` (required): List of holds in the problem
  - `description`: Hold position (e.g., "A1", "B5")
  - `isStart`: Whether this is a starting hold
  - `isEnd`: Whether this is a finishing hold
- `top_k` (optional): Number of top predictions to return (1-10, default: 3)

## License

Part of the Moonboard Grader project.

