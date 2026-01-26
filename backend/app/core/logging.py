"""
Logging configuration for the application.
"""

import logging
import sys
from .config import settings
from .telemetry import setup_telemetry, get_otel_logging_handler


def setup_logging() -> None:
    """
    Configure application logging.

    Sets up console handler with appropriate format and level.
    Also configures OpenTelemetry logging if OTLP endpoint is available.
    """
    setup_telemetry()

    log_level = getattr(logging, settings.log_level.upper())

    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(console_handler)

    otel_handler = get_otel_logging_handler()
    if otel_handler:
        otel_handler.setLevel(log_level)
        logger.addHandler(otel_handler)

    # Suppress verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the module/logger

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
