"""Structured logging module with JSON output.

This module provides a structured logging system with:
- Pydantic models for log records
- JSON formatting for machine-readable output
- Settings-based configuration (via shared.settings)
"""

from __future__ import annotations

import logging
import traceback
from datetime import UTC, datetime
from logging import Logger, StreamHandler
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel


if TYPE_CHECKING:
    from shared.enums import LogLevel
    from shared.settings import Settings


class LogRecordModel(BaseModel):
    """Pydantic model for structured log records.

    Attributes:
        level: Log level name (DEBUG, INFO, etc.).
        message: The log message.
        timestamp: ISO format timestamp.
        module: Module name where log was created.
        function: Function name where log was created.
        exception: Exception traceback if present.
        extra: Additional context data.
    """

    level: str
    message: str
    timestamp: str
    module: str | None = None
    function: str | None = None
    exception: str | None = None
    extra: dict[str, Any] | None = None


class JsonFormatter(logging.Formatter):
    """Custom formatter that outputs JSON-structured log records."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string representation of the log record.
        """
        exception_str: str | None = None
        if record.exc_info:
            exception_str = "".join(traceback.format_exception(*record.exc_info))

        extra_data: dict[str, Any] | None = None
        extra_keys: set[str] = set(record.__dict__.keys()) - {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "taskName",
            "message",
        }
        if extra_keys:
            extra_data = {key: record.__dict__[key] for key in extra_keys}

        log_model: LogRecordModel = LogRecordModel(
            level=record.levelname,
            message=record.getMessage(),
            timestamp=datetime.now(UTC).isoformat(),
            module=record.module,
            function=record.funcName,
            exception=exception_str,
            extra=extra_data if extra_data else None,
        )

        return log_model.model_dump_json(exclude_none=True)


def _get_log_level_from_settings() -> LogLevel:
    """Get log level from settings with lazy import.

    Uses lazy import to avoid circular dependencies.

    Returns:
        Log level enum from settings.
    """
    from shared.settings import settings as app_settings  # noqa: PLC0415

    loaded_settings: Settings = app_settings
    return loaded_settings.log_level


def setup_logger(
    name: str = "langgraph",
    level: str | LogLevel | None = None,
    *,
    use_settings: bool = True,
) -> Logger:
    """Configure and return a logger with JSON formatting.

    Args:
        name: Name for the logger.
        level: Log level string or LogLevel enum (DEBUG, INFO, WARNING, ERROR,
               CRITICAL). If not provided and use_settings is True, uses
               settings.log_level. If not provided and use_settings is False,
               defaults to INFO.
        use_settings: Whether to load level from settings when not provided.
                     Set to False to avoid settings import (useful for testing).

    Returns:
        Configured logger instance.

    Example:
        >>> logger = setup_logger("my-module")
        >>> logger.info("Application started")

        >>> # With explicit level (ignores settings)
        >>> logger = setup_logger("debug-logger", level="DEBUG")

        >>> # Without settings (for testing)
        >>> logger = setup_logger("test", use_settings=False)
    """
    resolved_level: str | LogLevel
    if level is None:
        resolved_level = _get_log_level_from_settings() if use_settings else "INFO"
    else:
        resolved_level = level

    level_str: str = str(resolved_level).upper()
    log_level: int = getattr(logging, level_str, logging.INFO)

    configured_logger: Logger = logging.getLogger(name)
    configured_logger.setLevel(log_level)

    # Avoid adding duplicate handlers
    if not configured_logger.handlers:
        handler: StreamHandler[Any] = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        configured_logger.addHandler(handler)

    return configured_logger


# Pre-configured global logger instance
logger: Logger = setup_logger()
