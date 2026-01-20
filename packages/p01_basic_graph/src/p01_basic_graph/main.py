"""Entry point for the LLM Graph pattern.

This pattern demonstrates:
- LangGraph StateGraph with LLM nodes
- Multi-provider LLM support (OpenAI, Anthropic, Google)
- External system prompt loading from markdown files

Run with: task run -- p01-basic-graph

Configuration:
    LLM provider is configured via settings.llm.provider, which can be set
    via LLM__PROVIDER or LLM_PROVIDER environment variables.
    Defaults to 'openai' if not set.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from p01_basic_graph.graph import create_graph
from p01_basic_graph.llm import get_available_providers
from p01_basic_graph.prompts import load_system_prompt
from p01_basic_graph.state import GraphState, create_initial_state
from shared.exceptions import LLMConfigurationError, PromptLoadError
from shared.logger import logger
from shared.settings import settings
from shared.utils import format_response


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from shared.enums import LLMProvider


def get_provider_from_settings() -> LLMProvider:
    """Get LLM provider from settings.

    Reads the provider from settings.llm.provider and returns
    the corresponding LLMProvider enum value.

    Returns:
        The selected LLMProvider.
    """
    return settings.llm.provider


def _validate_provider(
    provider: LLMProvider,
    available: list[LLMProvider],
) -> bool:
    """Validate that the selected provider is available.

    Args:
        provider: The selected LLM provider.
        available: List of available providers.

    Returns:
        True if provider is available and valid, False otherwise.
    """
    if not available:
        logger.error(
            "No LLM providers configured. Please set at least one API key "
            "(OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY)."
        )
        return False

    if provider not in available:
        logger.error(
            f"Provider '{provider.value}' is not configured. "
            f"Available providers: {[p.value for p in available]}"
        )
        return False

    return True


def _display_result(user_input: str, result: dict[str, Any], provider: LLMProvider) -> None:
    """Display the graph execution result.

    Args:
        user_input: The original user input.
        result: The graph execution result.
        provider: The LLM provider used.
    """
    logger.info("Graph execution completed")
    logger.info(format_response(result))

    print("\n" + "=" * 60)
    print("USER INPUT:")
    print("-" * 60)
    print(user_input)
    print("\n" + "=" * 60)
    print(f"LLM RESPONSE ({provider.value.upper()}):")
    print("-" * 60)
    print(result.get("llm_response", "No response"))
    print("=" * 60 + "\n")


def main() -> None:
    """Run the LLM graph example.

    Main entry point that:
    1. Loads configuration from environment
    2. Loads the system prompt from file
    3. Creates and invokes the graph
    4. Displays the result
    """
    logger.info("Starting Pattern 01: LLM-Powered Graph")

    # Check available providers
    available: list[LLMProvider] = get_available_providers(settings)
    logger.info(
        "Available LLM providers",
        extra={"providers": [p.value for p in available]},
    )

    # Get provider from settings
    provider: LLMProvider = get_provider_from_settings()

    logger.info(f"Selected provider: {provider.value}")

    # Validate provider availability
    if not _validate_provider(provider, available):
        sys.exit(1)

    # Load system prompt
    try:
        system_prompt: str = load_system_prompt()
        logger.info("System prompt loaded successfully")
    except PromptLoadError as e:
        logger.error(f"Failed to load system prompt: {e}")
        sys.exit(1)

    # Create the graph
    try:
        graph: CompiledStateGraph = create_graph(settings)  # type: ignore[type-arg]
        logger.info("Graph created successfully")
    except LLMConfigurationError as e:
        logger.error(f"LLM configuration error: {e}")
        sys.exit(1)

    # Create initial state and invoke
    user_input: str = "What is LangGraph and how does it help with building AI agents?"
    initial_state: GraphState = create_initial_state(
        user_input=user_input,
        provider=provider,
        system_prompt=system_prompt,
    )

    logger.info("Invoking graph", extra={"user_input": user_input})

    try:
        result: dict[str, Any] = graph.invoke(initial_state)
    except LLMConfigurationError as e:
        logger.error(f"LLM error during invocation: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during graph invocation: {e}")
        sys.exit(1)

    _display_result(user_input, result, provider)


if __name__ == "__main__":
    main()
