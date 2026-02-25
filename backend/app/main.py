"""
FastAPI Backend for Moonboard Grade Prediction

Provides REST API endpoints for predicting climbing grades using the
moonboard-classifier package.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
from pathlib import Path
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from .core.config import settings
from .core.logging import setup_logging
from .services.predictor_service import PredictorService
from .services.problem_service import ProblemService
from .services.generator_service import GeneratorService
from .services.service_registry import ServiceRegistry
from .api.dependencies import set_service_registry
from .api import router

# Setup logging (also initializes OpenTelemetry)
setup_logging()
logger = logging.getLogger(__name__)


def _ensure_repo_root_on_sys_path() -> None:
    """Ensure repository root is importable for moonboard_core imports."""
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _load_board_config_registry():
    """Load board configuration registry from configured path."""
    _ensure_repo_root_on_sys_path()
    from moonboard_core.board_config import BoardConfigRegistry  # noqa: E402

    return BoardConfigRegistry(settings.board_config_path)


def _resolve_project_path(project_root: Path, path_value: str) -> Path:
    """Resolve a configured path against project root."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _resolve_generator_path(
    *,
    project_root: Path,
    setup_id: str,
    angle: int,
    default_key: tuple[str, int],
    model_file: str | None,
    generator_model_file: str | None,
) -> Path | None:
    """Resolve generator model path including default fallback behavior."""
    if generator_model_file is not None:
        return _resolve_project_path(project_root, generator_model_file)

    if (setup_id, angle) != default_key:
        return None

    if model_file is not None:
        fallback_generator = Path(model_file).parent / "generator_model.pth"
        return _resolve_project_path(project_root, str(fallback_generator))

    return (project_root / "models" / "generator_model.pth").resolve()


def _register_predictor(
    *,
    registry: ServiceRegistry,
    setup_id: str,
    angle: int,
    predictor_path: Path,
) -> None:
    predictor_service = PredictorService(
        model_path=predictor_path,
        device=settings.device,
    )
    registry.register_predictor(setup_id, angle, predictor_service)

    if not predictor_path.exists():
        logger.warning(
            "Predictor model not found for setup=%s angle=%s at %s",
            setup_id,
            angle,
            predictor_path,
        )
        return

    try:
        predictor_service.load_model()
        logger.info(
            "Loaded predictor for setup=%s angle=%s from %s",
            setup_id,
            angle,
            predictor_path,
        )
    except Exception as e:
        logger.error(
            "Failed to load predictor for setup=%s angle=%s: %s",
            setup_id,
            angle,
            e,
        )


def _register_generator(
    *,
    registry: ServiceRegistry,
    setup_id: str,
    angle: int,
    generator_path: Path,
) -> None:
    generator_service = GeneratorService(
        model_path=generator_path,
        device=settings.device,
    )
    registry.register_generator(setup_id, angle, generator_service)

    if not generator_path.exists():
        logger.warning(
            "Generator model not found for setup=%s angle=%s at %s",
            setup_id,
            angle,
            generator_path,
        )
        return

    try:
        generator_service.load_model()
        logger.info(
            "Loaded generator for setup=%s angle=%s from %s",
            setup_id,
            angle,
            generator_path,
        )
    except Exception as e:
        logger.error(
            "Failed to load generator for setup=%s angle=%s: %s",
            setup_id,
            angle,
            e,
        )


def _register_analytics_path(
    *,
    registry: ServiceRegistry,
    setup_id: str,
    angle: int,
    analytics_path: Path | None,
) -> None:
    if analytics_path is None:
        return

    if analytics_path.exists():
        registry.register_analytics_path(setup_id, angle, analytics_path)
        return

    logger.warning(
        "Analytics file not found for setup=%s angle=%s at %s",
        setup_id,
        angle,
        analytics_path,
    )


def _build_service_registry(board_registry, project_root: Path) -> ServiceRegistry:
    """Build runtime service registry from board configuration."""
    setup_registry = ServiceRegistry()
    default_setup, default_angle = board_registry.get_default()
    default_key = (default_setup.id, default_angle.angle)
    setup_registry.set_default(*default_key)

    for hold_setup in board_registry.get_hold_setups():
        for angle_config in hold_setup.angles:
            setup_id = hold_setup.id
            angle = angle_config.angle
            setup_registry.register_combo(setup_id, angle)

            problems_path = _resolve_project_path(project_root, angle_config.data_file)
            setup_registry.register_problem_service(
                setup_id,
                angle,
                ProblemService(problems_path=problems_path),
            )

            if angle_config.model_file is not None:
                predictor_path = _resolve_project_path(
                    project_root, angle_config.model_file
                )
                _register_predictor(
                    registry=setup_registry,
                    setup_id=setup_id,
                    angle=angle,
                    predictor_path=predictor_path,
                )

            generator_path = _resolve_generator_path(
                project_root=project_root,
                setup_id=setup_id,
                angle=angle,
                default_key=default_key,
                model_file=angle_config.model_file,
                generator_model_file=angle_config.generator_model_file,
            )
            if generator_path is not None:
                _register_generator(
                    registry=setup_registry,
                    setup_id=setup_id,
                    angle=angle,
                    generator_path=generator_path,
                )

            analytics_path: Path | None = None
            if angle_config.analytics_file is not None:
                analytics_path = _resolve_project_path(
                    project_root, angle_config.analytics_file
                )
            elif (setup_id, angle) == default_key:
                analytics_path = project_root / "data" / "hold_stats.json"

            _register_analytics_path(
                registry=setup_registry,
                setup_id=setup_id,
                angle=angle,
                analytics_path=analytics_path,
            )

    return setup_registry


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
        description=(
            "REST API for predicting climbing problem grades "
            "using deep learning"
        ),
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

    # Instrument FastAPI for OpenTelemetry tracing
    FastAPIInstrumentor.instrument_app(app)

    return app


async def startup_handler():
    """
    Startup event handler.

    Initializes and registers services for all board setup/angle combinations.
    """
    logger.info("Starting up application...")
    logger.info(f"Using board config path: {settings.board_config_path}")
    logger.info(f"Using device: {settings.device}")

    board_registry = _load_board_config_registry()
    project_root = Path(settings.board_config_path).resolve().parent.parent
    setup_registry = _build_service_registry(board_registry, project_root)

    set_service_registry(setup_registry)
    logger.info("Service registry initialized!")

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
