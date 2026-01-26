"""
OpenTelemetry configuration for the application.

Configures tracing and logging export to OTLP endpoint (e.g., Aspire Dashboard).
"""

import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry._logs import set_logger_provider

SERVICE_NAME = "moonboard-backend"

_telemetry_initialized = False


def setup_telemetry() -> None:
    """
    Configure OpenTelemetry tracing and logging.

    Reads OTLP endpoint from OTEL_EXPORTER_OTLP_ENDPOINT environment variable.
    If not set, telemetry export is disabled.
    """
    global _telemetry_initialized
    if _telemetry_initialized:
        return

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not otlp_endpoint:
        _telemetry_initialized = True
        return

    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: SERVICE_NAME,
    })

    # Configure tracing
    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    # Configure logging
    logger_provider = LoggerProvider(resource=resource)
    log_exporter = OTLPLogExporter(endpoint=otlp_endpoint, insecure=True)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    set_logger_provider(logger_provider)

    _telemetry_initialized = True


def get_otel_logging_handler() -> LoggingHandler | None:
    """
    Get an OpenTelemetry logging handler if telemetry is configured.

    Returns:
        LoggingHandler if OTLP endpoint is configured, None otherwise.
    """
    from opentelemetry._logs import get_logger_provider

    provider = get_logger_provider()
    if isinstance(provider, LoggerProvider):
        return LoggingHandler(logger_provider=provider)
    return None
