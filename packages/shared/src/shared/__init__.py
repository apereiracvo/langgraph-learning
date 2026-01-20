"""Shared utilities for LangGraph learning patterns.

This package provides:
- enums: StrEnum classes for fixed string sets (Environment, LogLevel, LLMProvider)
- exceptions: Base exception classes for the project
- settings: Centralized configuration management
- logger: Structured JSON logging
- utils: Utility functions

Example:
    >>> from shared import settings
    >>> print(settings.environment)
    'development'

    >>> from shared.logger import logger
    >>> logger.info("Application started")

    >>> from shared.enums import Environment, LogLevel, LLMProvider
    >>> if settings.environment == Environment.PRODUCTION:
    ...     print("Production mode")

    >>> from shared.exceptions import LLMConfigurationError
    >>> raise LLMConfigurationError("openai", "API key missing")
"""

from shared.enums import Environment, LLMProvider, LogLevel
from shared.exceptions import (
    LangGraphLearningError,
    LLMConfigurationError,
    PromptLoadError,
)
from shared.logger import logger, setup_logger
from shared.settings import Settings, clear_settings_cache, get_settings, settings
from shared.utils import format_response


__all__: list[str] = [
    "Environment",
    "LLMConfigurationError",
    "LLMProvider",
    "LangGraphLearningError",
    "LogLevel",
    "PromptLoadError",
    "Settings",
    "clear_settings_cache",
    "format_response",
    "get_settings",
    "logger",
    "settings",
    "setup_logger",
]
