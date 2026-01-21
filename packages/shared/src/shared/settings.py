"""Centralized settings management using PydanticSettings.

This module provides type-safe configuration management for the LangGraph
Learning monorepo. Settings are loaded from environment variables and .env
files with validation at import time.

Example:
    >>> from shared import settings
    >>> print(settings.environment)
    'development'
    >>> if settings.llm.openai_api_key:
    ...     key = settings.llm.openai_api_key.get_secret_value()

Environment Variables:
    Core settings are loaded directly (e.g., ENVIRONMENT, DEBUG).
    Nested settings use double underscore delimiter (e.g., LLM__OPENAI_API_KEY).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Self

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.enums import Environment, LLMProvider, LogLevel


class LLMSettings(BaseModel):
    """LLM provider API key configuration.

    Supports OpenAI, Anthropic, and Google API keys. At least one key should be
    configured for patterns that use language models.

    Attributes:
        openai_api_key: OpenAI API key for GPT models.
        anthropic_api_key: Anthropic API key for Claude models.
        google_api_key: Google API key for Gemini models.
        default_model: Default model identifier to use.
        provider: Default LLM provider to use.
    """

    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key for GPT models",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key for Claude models",
    )
    google_api_key: SecretStr | None = Field(
        default=None,
        description="Google API key for Gemini models",
    )
    default_model: str = Field(
        default="gpt-4",
        description="Default model identifier",
    )
    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Default LLM provider (openai, anthropic, google)",
    )

    @field_validator(
        "openai_api_key", "anthropic_api_key", "google_api_key", mode="before"
    )
    @classmethod
    def reject_placeholder_keys(cls, v: str | None) -> str | None:
        """Reject placeholder API keys that are not real values.

        Args:
            v: The value to validate.

        Returns:
            The value if valid, None if it's a placeholder.
        """
        if v is None:
            return None

        # Strip whitespace
        v_stripped: str = v.strip()
        if not v_stripped:
            return None

        # Common placeholder patterns
        placeholders: list[str] = [
            "your-openai-key-here",
            "your-anthropic-key-here",
            "your-google-key-here",
            "sk-xxx",
            "your-key-here",
            "your-api-key-here",
            "placeholder",
            "changeme",
        ]

        v_lower: str = v_stripped.lower()
        if v_lower in placeholders:
            return None

        # Reject patterns starting with common placeholder prefixes
        if v_lower.startswith(("your-", "your_")):
            return None

        # Reject if it matches placeholder pattern: contains only x's after prefix
        placeholder_pattern: re.Pattern[str] = re.compile(r"^sk-x+$", re.IGNORECASE)
        if placeholder_pattern.match(v_stripped):
            return None

        return v_stripped


class LangSmithSettings(BaseModel):
    """LangSmith tracing and observability configuration.

    Controls LangSmith integration for debugging and monitoring LangGraph
    workflows. Supports both cloud LangSmith and self-hosted deployments.

    Attributes:
        tracing_enabled: Whether LangSmith tracing is enabled.
        api_key: LangSmith API key.
        project: LangSmith project name for organizing traces.
        endpoint: Custom endpoint URL for self-hosted LangSmith.
        workspace_id: Workspace ID for multi-workspace accounts.
        tracing_background: Whether to trace in background thread.
    """

    tracing_enabled: bool = Field(
        default=False,
        description="Enable LangSmith tracing (LANGSMITH__TRACING_ENABLED)",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="LangSmith API key (LANGSMITH__API_KEY)",
    )
    project: str = Field(
        default="langgraph-learning",
        description="LangSmith project name (LANGSMITH__PROJECT)",
    )
    endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith endpoint URL (for self-hosted)",
    )
    workspace_id: str | None = Field(
        default=None,
        description="Workspace ID for multi-workspace accounts",
    )
    tracing_background: bool = Field(
        default=True,
        description="Enable background tracing (disable for serverless)",
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def reject_placeholder_keys(cls, v: str | None) -> str | None:
        """Reject placeholder API keys.

        Args:
            v: The value to validate.

        Returns:
            The value if valid, None if it's a placeholder.
        """
        if v is None:
            return None

        v_stripped: str = v.strip()
        if not v_stripped:
            return None

        placeholders: list[str] = [
            "your-langsmith-key-here",
            "your-langchain-key-here",
            "your-key-here",
            "your-api-key-here",
            "placeholder",
            "changeme",
        ]

        v_lower: str = v_stripped.lower()
        if v_lower in placeholders:
            return None

        if v_lower.startswith(("your-", "your_")):
            return None

        return v_stripped


class Settings(BaseSettings):
    """Main application settings.

    Settings are loaded from environment variables with the following priority:
    1. Environment variables
    2. .env file in current working directory
    3. Default values

    Nested settings use double underscore delimiter (e.g., LLM__OPENAI_API_KEY).

    Both flat (backward compatible) and nested env vars are supported:
    - `OPENAI_API_KEY` -> `settings.llm.openai_api_key` (flat, backward compat)
    - `LLM__OPENAI_API_KEY` -> `settings.llm.openai_api_key` (nested, preferred)

    Attributes:
        environment: Application environment (development, staging, production).
        debug: Enable debug mode.
        verbose: Enable verbose output.
        log_level: Logging level.
        llm: LLM provider settings.
        langsmith: LangSmith tracing settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # Core settings
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    verbose: bool = Field(
        default=False,
        description="Enable verbose output",
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Log level for the application",
    )

    # Nested settings groups
    llm: LLMSettings = Field(default_factory=LLMSettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)

    # Backward compatibility: flat environment variable mappings
    # These allow both flat (OPENAI_API_KEY) and nested (LLM__OPENAI_API_KEY) access
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key (backward compat, prefer LLM__OPENAI_API_KEY)",
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Anthropic API key (backward compat, prefer LLM__ANTHROPIC_API_KEY)",
    )
    google_api_key: SecretStr | None = Field(
        default=None,
        description="Google API key (backward compat, prefer LLM__GOOGLE_API_KEY)",
    )
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM provider selection (backward compat, prefer LLM__PROVIDER)",
    )
    langchain_api_key: SecretStr | None = Field(
        default=None,
        description="LangChain API key (backward compat, prefer LANGSMITH__API_KEY)",
    )
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangChain tracing (backward compat)",
    )
    langchain_project: str = Field(
        default="langgraph-learning",
        description="LangChain project (backward compat)",
    )

    @field_validator(
        "openai_api_key",
        "anthropic_api_key",
        "google_api_key",
        "langchain_api_key",
        mode="before",
    )
    @classmethod
    def reject_placeholder_keys(cls, v: str | None) -> str | None:
        """Reject placeholder API keys for flat env vars.

        Args:
            v: The value to validate.

        Returns:
            The value if valid, None if it's a placeholder.
        """
        if v is None:
            return None

        v_stripped: str = v.strip()
        if not v_stripped:
            return None

        placeholders: list[str] = [
            "your-openai-key-here",
            "your-anthropic-key-here",
            "your-google-key-here",
            "your-langsmith-key-here",
            "your-langchain-key-here",
            "your-key-here",
            "your-api-key-here",
            "sk-xxx",
            "placeholder",
            "changeme",
        ]

        v_lower: str = v_stripped.lower()
        if v_lower in placeholders:
            return None

        if v_lower.startswith(("your-", "your_")):
            return None

        placeholder_pattern: re.Pattern[str] = re.compile(r"^sk-x+$", re.IGNORECASE)
        if placeholder_pattern.match(v_stripped):
            return None

        return v_stripped

    @model_validator(mode="after")
    def sync_flat_to_nested(self) -> Self:
        """Synchronize flat env vars with nested settings after initialization.

        This ensures backward compatibility by copying values from flat env vars
        (e.g., OPENAI_API_KEY) to nested settings (e.g., settings.llm.openai_api_key)
        when the nested value is not already set.

        Returns:
            Self with synchronized settings.
        """
        # Sync LLM settings from flat env vars if nested not set
        if self.openai_api_key and not self.llm.openai_api_key:
            object.__setattr__(self.llm, "openai_api_key", self.openai_api_key)
        if self.anthropic_api_key and not self.llm.anthropic_api_key:
            object.__setattr__(self.llm, "anthropic_api_key", self.anthropic_api_key)
        if self.google_api_key and not self.llm.google_api_key:
            object.__setattr__(self.llm, "google_api_key", self.google_api_key)

        # Sync LLM provider from flat env var
        if (
            self.llm_provider != LLMProvider.OPENAI
            and self.llm.provider == LLMProvider.OPENAI
        ):
            object.__setattr__(self.llm, "provider", self.llm_provider)

        # Sync LangSmith settings from flat env vars
        if self.langchain_api_key and not self.langsmith.api_key:
            object.__setattr__(self.langsmith, "api_key", self.langchain_api_key)
        if self.langchain_tracing_v2 and not self.langsmith.tracing_enabled:
            object.__setattr__(self.langsmith, "tracing_enabled", True)
        if (
            self.langchain_project != "langgraph-learning"
            and self.langsmith.project == "langgraph-learning"
        ):
            object.__setattr__(self.langsmith, "project", self.langchain_project)

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance.

    Uses lru_cache to ensure only one Settings instance is created,
    providing singleton-like behavior.

    Returns:
        Cached Settings instance.

    Example:
        >>> settings = get_settings()
        >>> settings.environment
        'development'
    """
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache.

    Useful for testing when you need to reload settings with different
    environment variables.

    Example:
        >>> from shared.settings import clear_settings_cache, get_settings
        >>> clear_settings_cache()
        >>> settings = get_settings()  # Creates new instance
    """
    get_settings.cache_clear()


# Module-level singleton for convenient import
settings: Settings = get_settings()
