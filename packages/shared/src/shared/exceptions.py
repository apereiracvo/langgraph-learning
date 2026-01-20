"""Shared exceptions for the LangGraph Learning project.

This module provides reusable exception classes for common error scenarios
across all patterns.

Example:
    >>> from shared.exceptions import LLMConfigurationError, PromptLoadError
    >>> raise LLMConfigurationError("openai", "API key missing")
    LLMConfigurationError: LLM configuration error for openai: API key missing
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from shared.enums import LLMProvider




class LangGraphLearningError(Exception):
    """Base exception for all project errors.

    All custom exceptions in this project should inherit from this class
    to allow for catching any project-specific error.

    Attributes:
        message: Detailed error message.

    Example:
        >>> try:
        ...     raise LangGraphLearningError("Something went wrong")
        ... except LangGraphLearningError as e:
        ...     print(e.message)
        'Something went wrong'
    """

    def __init__(self, message: str) -> None:
        """Initialize the base error.

        Args:
            message: Detailed error description.
        """
        self.message: str = message
        super().__init__(message)


class LLMConfigurationError(LangGraphLearningError):
    """Raised when LLM configuration is invalid or incomplete.

    This exception is raised when:
    - Required API key is missing for the selected provider
    - Invalid provider is specified
    - LLM initialization fails

    Attributes:
        provider: The LLM provider that failed configuration.
        message: Detailed error message.

    Example:
        >>> from shared.enums import LLMProvider
        >>> raise LLMConfigurationError(LLMProvider.OPENAI, "API key missing")
        LLMConfigurationError: LLM configuration error for openai: API key missing
    """

    def __init__(self, provider: LLMProvider | str, message: str) -> None:
        """Initialize the configuration error.

        Args:
            provider: The provider that failed.
            message: Detailed error description.
        """
        self.provider: LLMProvider | str = provider
        full_message: str = f"LLM configuration error for {provider}: {message}"
        super().__init__(full_message)
        # Override message to keep original detail
        self.message = message


class PromptLoadError(LangGraphLearningError):
    """Raised when a prompt file cannot be loaded.

    This exception is raised when:
    - The prompt file does not exist
    - The file cannot be read
    - The file path is not a regular file

    Attributes:
        path: The path that failed to load.
        message: Detailed error message.

    Example:
        >>> from pathlib import Path
        >>> raise PromptLoadError(Path("/missing.md"), "File not found")
        PromptLoadError: Failed to load prompt from /missing.md: File not found
    """

    def __init__(self, path: Path | str, message: str) -> None:
        """Initialize the prompt load error.

        Args:
            path: The path that failed.
            message: Detailed error description.
        """
        self.path: Path = Path(path)
        full_message: str = f"Failed to load prompt from {path}: {message}"
        super().__init__(full_message)
        # Override message to keep original detail
        self.message = message
