# Moonboard Grade Predictor Backend

FastAPI backend service for predicting climbing problem grades using deep learning.

## Quick Start

### 1. Install Dependencies

```powershell
# Install classifier package
py -m pip install -e ..\classifier

# Install backend requirements
py -m pip install -r requirements.txt
```

### 2. Add Model File

```powershell
Copy-Item ..\classifier\models\best_model.pth -Destination models\model_for_inference.pth
```

### 3. Run the Server

```powershell
uvicorn app.main:app --reload --host localhost --port 8000
```

Server will start at: **http://localhost:8000**

API docs available at: **http://localhost:8000/docs**

## Testing

```powershell
pytest
```

## Configuration (Optional)

Edit `.env` file to configure model path, CORS origins, device (cpu/cuda), etc. See `.env.example` for options.

