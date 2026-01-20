"""Entry point for the LLM Graph pattern.

This pattern demonstrates:
- LangGraph StateGraph with LLM nodes
- Multi-provider LLM support (OpenAI, Anthropic, Google)
- External system prompt loading from markdown files
- Sequential execution across ALL available LLM providers

Run with: task run -- p01-basic-graph

Configuration:
    The pattern automatically detects and uses all configured providers.
    Configure API keys via environment variables:
    - OPENAI_API_KEY for OpenAI
    - ANTHROPIC_API_KEY for Anthropic
    - GOOGLE_API_KEY for Google
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from p01_basic_graph.graph import create_graph
from p01_basic_graph.prompts import load_system_prompt
from p01_basic_graph.state import GraphState, create_initial_state
from shared.exceptions import LLMConfigurationError, PromptLoadError
from shared.llm import get_available_providers
from shared.logger import logger
from shared.settings import settings
from shared.utils import format_response


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from shared.enums import LLMProvider


def _validate_providers(available: list[LLMProvider]) -> bool:
    """Validate that at least one provider is available.

    Args:
        available: List of available providers.

    Returns:
        True if at least one provider is available, False otherwise.
    """
    if not available:
        logger.error(
            "No LLM providers configured. Please set at least one API key "
            "(OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY)."
        )
        return False

    return True


def _display_result(
    user_input: str, result: dict[str, Any], provider: LLMProvider
) -> None:
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


def _run_with_provider(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
    provider: LLMProvider,
    user_input: str,
    system_prompt: str,
) -> bool:
    """Execute the graph with a specific provider.

    Args:
        graph: The compiled graph to invoke.
        provider: The LLM provider to use.
        user_input: The user's input message.
        system_prompt: The system prompt content.

    Returns:
        True if execution succeeded, False otherwise.
    """
    logger.info(f"Invoking graph with provider: {provider.value}")

    initial_state: GraphState = create_initial_state(
        user_input=user_input,
        provider=provider,
        system_prompt=system_prompt,
    )

    try:
        result: dict[str, Any] = graph.invoke(initial_state)
    except LLMConfigurationError as e:
        logger.error(f"LLM error with {provider.value}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error with {provider.value}: {e}")
        return False
    else:
        _display_result(user_input, result, provider)
        return True


def main() -> None:
    """Run the LLM graph example with all available providers.

    Main entry point that:
    1. Loads configuration from environment
    2. Detects all available LLM providers
    3. Loads the system prompt from file
    4. Creates the graph once
    5. Sequentially invokes the graph with each available provider
    6. Displays results for each provider
    """
    logger.info("Starting Pattern 01: LLM-Powered Graph (Multi-Provider)")

    # Check available providers
    available: list[LLMProvider] = get_available_providers(settings)
    logger.info(
        "Available LLM providers",
        extra={"providers": [p.value for p in available]},
    )

    # Validate at least one provider is available
    if not _validate_providers(available):
        sys.exit(1)

    print("\n" + "#" * 60)
    print(f"# Found {len(available)} configured provider(s):")
    for p in available:
        print(f"#   - {p.value.upper()}")
    print("#" * 60)

    # Load system prompt
    try:
        system_prompt: str = load_system_prompt()
        logger.info("System prompt loaded successfully")
    except PromptLoadError as e:
        logger.error(f"Failed to load system prompt: {e}")
        sys.exit(1)

    # Create the graph (once, reused for all providers)
    try:
        graph: CompiledStateGraph = create_graph(settings)  # type: ignore[type-arg]
        logger.info("Graph created successfully")
    except LLMConfigurationError as e:
        logger.error(f"LLM configuration error: {e}")
        sys.exit(1)

    # User input to test with all providers
    user_input: str = "What is LangGraph and how does it help with building AI agents?"

    # Execute with each available provider sequentially
    success_count: int = 0
    for provider in available:
        print("\n" + "#" * 60)
        print(f"# Executing with provider: {provider.value.upper()}")
        print("#" * 60)

        if _run_with_provider(graph, provider, user_input, system_prompt):
            success_count += 1

    # Summary
    print("\n" + "#" * 60)
    print("# EXECUTION SUMMARY")
    print("#" * 60)
    print(f"# Total providers: {len(available)}")
    print(f"# Successful: {success_count}")
    print(f"# Failed: {len(available) - success_count}")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
