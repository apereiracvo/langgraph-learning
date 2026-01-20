"""Entry point for the Async LLM Graph pattern.

This pattern demonstrates:
- Async LangGraph StateGraph with async LLM nodes
- Multi-provider LLM support (OpenAI, Anthropic, Google)
- Async graph invocation with `ainvoke()`
- Streaming support with `astream()`

Run with: task run -- p02-async-basic-graph

Configuration:
    LLM provider is configured via settings.llm.provider, which can be set
    via LLM__PROVIDER or LLM_PROVIDER environment variables.
    Defaults to 'openai' if not set.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

from p02_async_basic_graph.graph import create_graph
from p02_async_basic_graph.prompts import load_system_prompt
from p02_async_basic_graph.state import GraphState, create_initial_state
from shared.exceptions import LLMConfigurationError, PromptLoadError
from shared.llm import get_available_providers
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


async def run_with_ainvoke(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
    initial_state: GraphState,
    provider: LLMProvider,
    user_input: str,
) -> None:
    """Run the graph using async ainvoke.

    Demonstrates basic async invocation where the entire result
    is returned after completion.

    Args:
        graph: The compiled graph to invoke.
        initial_state: The initial state for the graph.
        provider: The LLM provider being used.
        user_input: The original user input for display.
    """
    logger.info("Invoking graph with ainvoke()", extra={"user_input": user_input})

    try:
        result: dict[str, Any] = await graph.ainvoke(initial_state)
    except LLMConfigurationError as e:
        logger.error(f"LLM error during invocation: {e}")
        sys.exit(1)
    except asyncio.CancelledError:
        logger.warning("Graph execution was cancelled")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during graph invocation: {e}")
        sys.exit(1)

    _display_result(user_input, result, provider)


async def run_with_astream(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
    initial_state: GraphState,
    provider: LLMProvider,
    user_input: str,
) -> None:
    """Run the graph using async streaming.

    Demonstrates streaming execution where state updates are
    yielded as they happen from each node.

    Args:
        graph: The compiled graph to invoke.
        initial_state: The initial state for the graph.
        provider: The LLM provider being used.
        user_input: The original user input for display.
    """
    logger.info("Invoking graph with astream()", extra={"user_input": user_input})

    print("\n" + "=" * 60)
    print("USER INPUT:")
    print("-" * 60)
    print(user_input)
    print("\n" + "=" * 60)
    print(f"STREAMING LLM RESPONSE ({provider.value.upper()}):")
    print("-" * 60)

    try:
        async for event in graph.astream(initial_state, stream_mode="updates"):
            for node_name, node_output in event.items():
                logger.debug(f"Received update from node: {node_name}")
                if "llm_response" in node_output:
                    print(node_output["llm_response"])
    except LLMConfigurationError as e:
        logger.error(f"LLM error during streaming: {e}")
        sys.exit(1)
    except asyncio.CancelledError:
        logger.warning("Graph streaming was cancelled")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during graph streaming: {e}")
        sys.exit(1)

    print("=" * 60 + "\n")


async def async_main() -> None:
    """Run the async LLM graph example.

    Async entry point that:
    1. Loads configuration from environment
    2. Loads the system prompt from file
    3. Creates and invokes the graph asynchronously
    4. Displays the result

    Demonstrates both `ainvoke()` and `astream()` patterns.
    """
    logger.info("Starting Pattern 02: Async LLM-Powered Graph")

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

    # Create initial state
    user_input: str = "What is LangGraph and how does it help with building AI agents?"
    initial_state: GraphState = create_initial_state(
        user_input=user_input,
        provider=provider,
        system_prompt=system_prompt,
    )

    # Demonstrate async invocation with ainvoke()
    print("\n" + "#" * 60)
    print("# DEMO 1: Using ainvoke() - Full response after completion")
    print("#" * 60)
    await run_with_ainvoke(graph, initial_state, provider, user_input)

    # Demonstrate streaming with astream()
    print("\n" + "#" * 60)
    print("# DEMO 2: Using astream() - Streaming state updates")
    print("#" * 60)
    # Create fresh state for second demo
    initial_state_streaming: GraphState = create_initial_state(
        user_input="Explain async/await in Python in 2-3 sentences.",
        provider=provider,
        system_prompt=system_prompt,
    )
    await run_with_astream(
        graph,
        initial_state_streaming,
        provider,
        "Explain async/await in Python in 2-3 sentences.",
    )


def main() -> None:
    """Entry point wrapper for async main.

    This function serves as the synchronous entry point required
    by the package script configuration. It wraps the async main
    function with `asyncio.run()`.
    """
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
