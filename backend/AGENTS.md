# Agent Instructions

## Python Command

On this system, use `py` instead of `python` for Python commands. For example:
- `py -m uvicorn app.main:app --reload` instead of `python -m uvicorn app.main:app --reload`
- `uv sync` to install dependencies

## Running the Backend Server

To run the FastAPI backend server, use:

```powershell
py -m uvicorn app.main:app --reload --host localhost --port 8000
```

Or without module syntax:

```powershell
uvicorn app.main:app --reload --host localhost --port 8000
```

The server will start on http://localhost:8000 with:
- Interactive API docs at http://localhost:8000/docs
- Alternative docs at http://localhost:8000/redoc

