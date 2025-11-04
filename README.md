# Moonboard Grader

A machine learning system that predicts the difficulty grade of Moonboard climbing problems.

## What is this?

This project uses deep learning to analyze climbing hold positions on a Moonboard and predict the difficulty grade using the Font Scale.

## Components

### 1. Classifier
The machine learning model that does the grade prediction.
- Trains a neural network on climbing problem data
- Converts moonboard boulder into predictions
- **Tech**: Python, PyTorch

### 2. Backend
API server that serves predictions from the trained model.
- Provides REST API endpoints
- Loads the trained model for inference
- **Tech**: Python, FastAPI

### 3. Frontend
Web interface to visualize problems and see predictions.
- Interactive Moonboard visualization
- Display predicted grades
- **Tech**: React, TypeScript, Vite

## Tech Stack

**Classifier:**
- Python
- PyTorch
- NumPy

**Backend:**
- Python
- FastAPI
- Uvicorn

**Frontend:**
- React
- TypeScript
- Vite
- Tailwind CSS

## Quick Start

See individual README files in each component directory for detailed setup instructions:
- `classifier/README.md` - Train and evaluate models
- `backend/README.md` - Run the API server
- `frontend/README.md` - Run the web interface

