"""Shared utilities for LangGraph learning patterns.

This package provides:
- enums: StrEnum classes for fixed string sets (Environment, LogLevel, LLMProvider)
- exceptions: Base exception classes for the project
- settings: Centralized configuration management
- logger: Structured JSON logging
- utils: Utility functions
- llm: Multi-provider LLM factory functions
- prompts: Prompt loading and template utilities

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

    >>> from shared.llm import create_llm, get_available_providers
    >>> llm = create_llm(LLMProvider.OPENAI, settings)

    >>> from shared.prompts import load_system_prompt
    >>> prompt = load_system_prompt("system.md", prompts_dir)
"""

from shared.enums import Environment, LLMProvider, LogLevel
from shared.exceptions import (
    LangGraphLearningError,
    LLMConfigurationError,
    PromptLoadError,
)
from shared.llm import DEFAULT_MODELS, create_llm, get_available_providers
from shared.logger import logger, setup_logger
from shared.prompts import clear_prompt_cache, load_system_prompt, render_prompt
from shared.settings import Settings, clear_settings_cache, get_settings, settings
from shared.utils import format_response


__all__: list[str] = [
    "DEFAULT_MODELS",
    "Environment",
    "LLMConfigurationError",
    "LLMProvider",
    "LangGraphLearningError",
    "LogLevel",
    "PromptLoadError",
    "Settings",
    "clear_prompt_cache",
    "clear_settings_cache",
    "create_llm",
    "format_response",
    "get_available_providers",
    "get_settings",
    "load_system_prompt",
    "logger",
    "render_prompt",
    "settings",
    "setup_logger",
]
