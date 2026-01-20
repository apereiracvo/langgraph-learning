"""Enumerations for fixed string sets in settings.

This module provides StrEnum classes for type-safe configuration values
that have a fixed set of valid options.

Example:
    >>> from shared.enums import Environment, LogLevel
    >>> env = Environment.PRODUCTION
    >>> print(env)  # "production"
    >>> print(env == "production")  # True
"""

from __future__ import annotations

from enum import StrEnum


class Environment(StrEnum):
    """Application environment enumeration.

    Defines the valid deployment environments for the application.

    Attributes:
        DEVELOPMENT: Local development environment.
        STAGING: Pre-production staging environment.
        PRODUCTION: Live production environment.

    Example:
        >>> from shared.enums import Environment
        >>> env = Environment.DEVELOPMENT
        >>> print(env)
        'development'
        >>> env == "development"
        True
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LLMProvider(StrEnum):
    """Supported LLM providers.

    Defines the valid LLM providers for the application.

    Attributes:
        OPENAI: OpenAI's GPT models.
        ANTHROPIC: Anthropic's Claude models.
        GOOGLE: Google's Gemini models.

    Example:
        >>> from shared.enums import LLMProvider
        >>> provider = LLMProvider.OPENAI
        >>> print(provider)
        'openai'
        >>> provider == "openai"
        True
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class LogLevel(StrEnum):
    """Logging level enumeration.

    Defines the valid log levels for the application, matching
    Python's standard logging levels.

    Attributes:
        DEBUG: Detailed information for debugging.
        INFO: General operational information.
        WARNING: Indication of potential issues.
        ERROR: Error events that might still allow operation.
        CRITICAL: Severe errors that may prevent operation.

    Example:
        >>> from shared.enums import LogLevel
        >>> level = LogLevel.INFO
        >>> print(level)
        'INFO'
        >>> level == "INFO"
        True
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
