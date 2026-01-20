"""LLM factory and configuration for multi-provider support.

This module provides a factory pattern for creating LLM instances from
different providers (OpenAI, Anthropic, Google) based on configuration.

Supported Models (as of January 2026):
- OpenAI: gpt-5 (GPT-4.5 deprecated, GPT-5 is current flagship)
- Anthropic: claude-sonnet-4-5-20250929 (Sonnet 4.5)
- Google: gemini-2.5-flash (stable)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from shared.enums import LLMProvider
from shared.exceptions import LLMConfigurationError


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from shared.settings import Settings


__all__: list[str] = [
    "DEFAULT_MODELS",
    "create_llm",
    "get_available_providers",
]


# Default model identifiers for each provider
DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OPENAI: "gpt-5",  # GPT-5 as of Aug 2025
    LLMProvider.ANTHROPIC: "claude-sonnet-4-5-20250929",  # Claude Sonnet 4.5
    LLMProvider.GOOGLE: "gemini-2.5-flash",  # Gemini 2.5 Flash stable
}


def create_llm(
    provider: LLMProvider,
    settings: Settings,
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Create an LLM instance for the specified provider.

    Factory function that creates the appropriate LangChain chat model
    based on the provider selection. Uses API keys from the shared settings.

    Args:
        provider: The LLM provider to use.
        settings: Application settings containing API keys.
        model: Optional model override. If not provided, uses default for provider.
        temperature: Sampling temperature (0.0-2.0). Defaults to 0.7.
        max_tokens: Maximum tokens in response. None for provider default.

    Returns:
        A configured BaseChatModel instance ready for invocation.

    Raises:
        LLMConfigurationError: If the API key is missing or invalid.

    Example:
        >>> from shared.settings import settings
        >>> llm = create_llm(LLMProvider.OPENAI, settings)
        >>> response = llm.invoke("Hello!")
    """
    model_name: str = model or DEFAULT_MODELS[provider]

    if provider == LLMProvider.OPENAI:
        return _create_openai_llm(settings, model_name, temperature, max_tokens)
    if provider == LLMProvider.ANTHROPIC:
        return _create_anthropic_llm(settings, model_name, temperature, max_tokens)
    if provider == LLMProvider.GOOGLE:
        return _create_google_llm(settings, model_name, temperature, max_tokens)

    # This branch handles potential future providers or invalid input
    msg: str = f"Unsupported provider: {provider}"  # type: ignore[unreachable]
    raise LLMConfigurationError(provider, msg)


def _create_openai_llm(
    settings: Settings,
    model: str,
    temperature: float,
    max_tokens: int | None,
) -> ChatOpenAI:
    """Create an OpenAI chat model.

    Args:
        settings: Application settings with OpenAI API key.
        model: The model identifier (e.g., "gpt-5").
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.

    Returns:
        Configured ChatOpenAI instance.

    Raises:
        LLMConfigurationError: If OPENAI_API_KEY is not set.
    """
    api_key_secret = settings.llm.openai_api_key
    if not api_key_secret:
        raise LLMConfigurationError(
            LLMProvider.OPENAI,
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment.",
        )

    api_key: str = api_key_secret.get_secret_value()

    llm: ChatOpenAI = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return llm


def _create_anthropic_llm(
    settings: Settings,
    model: str,
    temperature: float,
    max_tokens: int | None,
) -> ChatAnthropic:
    """Create an Anthropic chat model.

    Args:
        settings: Application settings with Anthropic API key.
        model: The model identifier (e.g., "claude-sonnet-4-5-20250929").
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.

    Returns:
        Configured ChatAnthropic instance.

    Raises:
        LLMConfigurationError: If ANTHROPIC_API_KEY is not set.
    """
    api_key_secret = settings.llm.anthropic_api_key
    if not api_key_secret:
        raise LLMConfigurationError(
            LLMProvider.ANTHROPIC,
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment.",
        )

    api_key: str = api_key_secret.get_secret_value()

    # Anthropic requires max_tokens to be set
    resolved_max_tokens: int = max_tokens or 4096

    llm: ChatAnthropic = ChatAnthropic(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=resolved_max_tokens,
    )
    return llm


def _create_google_llm(
    settings: Settings,
    model: str,
    temperature: float,
    max_tokens: int | None,
) -> ChatGoogleGenerativeAI:
    """Create a Google Gemini chat model.

    Args:
        settings: Application settings with Google API key.
        model: The model identifier (e.g., "gemini-2.5-flash").
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.

    Returns:
        Configured ChatGoogleGenerativeAI instance.

    Raises:
        LLMConfigurationError: If GOOGLE_API_KEY is not set.
    """
    api_key_secret = settings.llm.google_api_key
    if not api_key_secret:
        raise LLMConfigurationError(
            LLMProvider.GOOGLE,
            "GOOGLE_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment.",
        )

    api_key: str = api_key_secret.get_secret_value()

    llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return llm


def get_available_providers(settings: Settings) -> list[LLMProvider]:
    """Get list of providers with configured API keys.

    Checks which providers have valid API keys configured and returns
    them as a list. Useful for fallback logic or provider selection UI.

    Args:
        settings: Application settings to check.

    Returns:
        List of providers with valid API key configuration.

    Example:
        >>> from shared.settings import settings
        >>> available = get_available_providers(settings)
        >>> print(available)
        [<LLMProvider.OPENAI: 'openai'>, <LLMProvider.ANTHROPIC: 'anthropic'>]
    """
    available: list[LLMProvider] = []

    if settings.llm.openai_api_key:
        available.append(LLMProvider.OPENAI)

    if settings.llm.anthropic_api_key:
        available.append(LLMProvider.ANTHROPIC)

    if settings.llm.google_api_key:
        available.append(LLMProvider.GOOGLE)

    return available
