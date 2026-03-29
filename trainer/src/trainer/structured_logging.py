"""Structured logging configuration for cloud deployments.

This module provides JSON-formatted logging for Google Cloud Logging integration.
It can be toggled via the CARTRIDGE_LOGGING_FORMAT environment variable or
config.toml [logging] section.

Usage:
    from trainer.structured_logging import setup_logging, set_trace_context

    # In your main() function:
    setup_logging(level="INFO", component="trainer")

    # For distributed tracing:
    trace_id = generate_trace_id()
    set_trace_context(trace_id)
    # All subsequent logs will include trace_id
"""

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from typing import Any

from pythonjsonlogger import jsonlogger

# Thread-safe trace context using context variables
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id: ContextVar[str | None] = ContextVar("span_id", default=None)
_parent_span: ContextVar[str | None] = ContextVar("parent_span", default=None)


def generate_trace_id() -> str:
    """Generate a unique trace ID (32-character hex string).

    Returns:
        A 32-character hexadecimal trace ID compatible with Google Cloud Trace.
    """
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate a unique span ID (16-character hex string).

    Returns:
        A 16-character hexadecimal span ID compatible with Google Cloud Trace.
    """
    return uuid.uuid4().hex[:16]


def set_trace_context(
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span: str | None = None,
) -> None:
    """Set the current trace context for correlation.

    Args:
        trace_id: Trace ID to use (generates new one if None).
        span_id: Span ID to use (generates new one if None).
        parent_span: Parent span ID for nested operations.
    """
    _trace_id.set(trace_id or generate_trace_id())
    _span_id.set(span_id or generate_span_id())
    if parent_span:
        _parent_span.set(parent_span)


def get_trace_context() -> dict[str, str | None]:
    """Get current trace context for logging or propagation.

    Returns:
        Dictionary with trace_id, span_id, and parent_span (if set).
    """
    ctx = {
        "trace_id": _trace_id.get(),
        "span_id": _span_id.get(),
    }
    parent = _parent_span.get()
    if parent:
        ctx["parent_span"] = parent
    return ctx


def clear_trace_context() -> None:
    """Clear the current trace context."""
    _trace_id.set(None)
    _span_id.set(None)
    _parent_span.set(None)


def format_gcp_trace(project_id: str, trace_id: str) -> str:
    """Format trace ID for Google Cloud Logging.

    Args:
        project_id: Google Cloud project ID.
        trace_id: The trace ID.

    Returns:
        Formatted trace string for logging.googleapis.com/trace field.
    """
    return f"projects/{project_id}/traces/{trace_id}"


class CloudJsonFormatter(jsonlogger.JsonFormatter):
    """JSON formatter optimized for Google Cloud Logging.

    Outputs logs in a format that Google Cloud Logging can parse and index,
    with proper severity mapping and structured fields.
    """

    # Map Python log levels to Google Cloud Logging severity
    SEVERITY_MAP = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }

    def __init__(self, component: str = "trainer", include_timestamp: bool = True):
        """Initialize the formatter.

        Args:
            component: Component name to include in every log (trainer, orchestrator, etc.)
            include_timestamp: Whether to include timestamp (set False if cloud adds it)
        """
        self.component = component
        self.include_timestamp = include_timestamp
        # Define format with timestamp if needed
        fmt = "%(message)s"
        if include_timestamp:
            fmt = "%(asctime)s " + fmt
        super().__init__(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S%z")

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        """Add structured fields to the log record."""
        super().add_fields(log_record, record, message_dict)

        # Add severity for Google Cloud Logging
        log_record["severity"] = self.SEVERITY_MAP.get(record.levelname, "DEFAULT")

        # Add component identifier
        log_record["component"] = self.component

        # Add module/target for context
        log_record["target"] = record.name

        # Add trace context if set (for distributed tracing)
        trace_ctx = get_trace_context()
        if trace_ctx.get("trace_id"):
            log_record["trace_id"] = trace_ctx["trace_id"]
            # Add Google Cloud Logging trace format if project ID is available
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if project_id:
                log_record["logging.googleapis.com/trace"] = format_gcp_trace(
                    project_id, trace_ctx["trace_id"]
                )
        if trace_ctx.get("span_id"):
            log_record["span_id"] = trace_ctx["span_id"]
            # Also set the GCP-specific spanId field
            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
            if project_id:
                log_record["logging.googleapis.com/spanId"] = trace_ctx["span_id"]
        if trace_ctx.get("parent_span"):
            log_record["parent_span"] = trace_ctx["parent_span"]

        # Remove redundant fields
        if "levelname" in log_record:
            del log_record["levelname"]

        # Add timestamp in ISO format if enabled
        if self.include_timestamp and "asctime" in log_record:
            log_record["timestamp"] = log_record.pop("asctime")


def get_logging_format() -> str:
    """Determine logging format from environment or config.

    Returns:
        "json" or "text"
    """
    # Environment variable takes precedence
    env_format = os.environ.get("CARTRIDGE_LOGGING_FORMAT", "").lower()
    if env_format in ("json", "text"):
        return env_format

    # Try to load from config.toml
    try:
        from .central_config import get_config

        config = get_config()
        return config.logging.format.lower()
    except Exception:
        return "text"


def setup_logging(
    level: str = "INFO",
    component: str = "trainer",
    silence_noisy: bool = True,
) -> None:
    """Configure logging with optional JSON format for cloud deployments.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        component: Component name to include in logs
        silence_noisy: Whether to silence noisy third-party loggers
    """
    log_format = get_logging_format()
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level.upper())

    if log_format == "json":
        # JSON format for cloud logging
        include_timestamp = os.environ.get(
            "CARTRIDGE_LOGGING_INCLUDE_TIMESTAMPS", "true"
        ).lower() in ("true", "1", "yes")
        formatter = CloudJsonFormatter(
            component=component,
            include_timestamp=include_timestamp,
        )
    else:
        # Human-readable format for local development
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Silence noisy loggers
    if silence_noisy:
        from .logging_utils import silence_noisy_loggers

        silence_noisy_loggers()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds structured fields to log messages.

    This adapter allows adding extra fields to log messages that will be
    properly formatted as JSON fields when using the JSON formatter.

    Usage:
        logger = StructuredLoggerAdapter(
            logging.getLogger(__name__),
            {"component": "trainer", "env_id": "tictactoe"}
        )
        logger.info("Training started", extra={"step": 100, "loss": 0.5})
    """

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Process the logging message and keyword arguments."""
        extra = kwargs.get("extra", {})
        # Merge adapter extra with call extra
        merged_extra = {**self.extra, **extra}
        kwargs["extra"] = merged_extra
        return msg, kwargs
