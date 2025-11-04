"""
FastAPI Backend for Moonboard Grade Prediction

Provides REST API endpoints for predicting climbing grades using the
moonboard-classifier package.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .core.config import settings
from .core.logging import setup_logging
from .services.predictor_service import PredictorService
from .services.problem_service import ProblemService
from .api.dependencies import set_predictor_service, set_problem_service
from .api import router

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def create_application() -> FastAPI:
    """
    Application factory pattern.
    
    Creates and configures the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description="REST API for predicting climbing problem grades using deep learning",
        version=settings.version,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    
    # Include API routes
    app.include_router(router)
    
    # Register event handlers
    app.add_event_handler("startup", startup_handler)
    app.add_event_handler("shutdown", shutdown_handler)
    
    return app


async def startup_handler():
    """
    Startup event handler.
    
    Initializes the predictor service and loads the model.
    """
    logger.info("Starting up application...")
    logger.info(f"Using model path: {settings.model_path}")
    logger.info(f"Using device: {settings.device}")
    
    # Create predictor service
    predictor_service = PredictorService(
        model_path=settings.model_path,
        device=settings.device
    )
    
    # Try to load the model
    try:
        if not settings.model_path.exists():
            logger.warning(
                f"Model file not found at {settings.model_path}. "
                "API will be available but predictions will fail. "
                "Please add model_for_inference.pth to the models/ directory."
            )
        else:
            predictor_service.load_model()
            logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("API will start but predictions will fail until model is loaded.")
    
    # Set the global predictor service
    set_predictor_service(predictor_service)
    
    # Create and set problem service
    logger.info(f"Initializing problem service with data path: {settings.problems_data_path}")
    problem_service = ProblemService()
    set_problem_service(problem_service)
    logger.info("Problem service initialized!")
    
    logger.info("Application startup complete!")


async def shutdown_handler():
    """
    Shutdown event handler.
    
    Cleanup resources on application shutdown.
    """
    logger.info("Shutting down application...")
    # Add cleanup logic here if needed
    logger.info("Shutdown complete!")


# Create the application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
