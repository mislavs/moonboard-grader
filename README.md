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

### 4. Generator
AI system that creates synthetic climbing problems using a Variational Autoencoder.
- Generates problems at specified difficulty grades
- Trains on existing problem data
- Configurable creativity and hold density
- **Tech**: Python, PyTorch, VAE

### 5. Aspire App
Orchestration layer that runs the entire application stack.
- Coordinates backend and frontend services
- Health monitoring and service dependencies
- Simplified deployment and development
- **Tech**: .NET Aspire, C#

### 6. Beta Solver
Experimental algorithm that finds the best beta for a climbing problem.
- Uses dynamic programming with memoization
- Evaluates all possible hand sequences to find the best path
- Move scoring based on multiple factors
- **Tech**: C#, .NET

### 7. Beta Classifier
Transformer-based grade classifier using move sequences from the Beta Solver.
- Predicts grades from rich per-move features (positions, difficulty, body mechanics)
- Sequence classification with attention mechanisms
- Alternative to grid-based classification using movement data
- **Tech**: Python, PyTorch, Transformer

## Quick Start

See individual README files in each component directory for detailed setup instructions:
- `classifier/README.md` - Train and evaluate grid-based models
- `backend/README.md` - Run the API server
- `frontend/README.md` - Run the web interface
- `generator/README.md` - Generate synthetic climbing problems
- `aspire-app/` - Run the full application stack with Aspire
- `beta-solver/` - Compute optimal beta sequences for problems
- `beta-classifier/README.md` - Train sequence-based grade classifier

