"""Entry point for the Async LLM Graph pattern.

This pattern demonstrates:
- Async LangGraph StateGraph with async LLM nodes
- Multi-provider LLM support (OpenAI, Anthropic, Google)
- Parallel execution across ALL available providers with asyncio.gather
- Async graph invocation with `ainvoke()`
- Streaming support with `astream()` (single provider demo)

Run with: task run -- p02-async-basic-graph

Configuration:
    The pattern automatically detects and uses all configured providers.
    Configure API keys via environment variables:
    - OPENAI_API_KEY for OpenAI
    - ANTHROPIC_API_KEY for Anthropic
    - GOOGLE_API_KEY for Google
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


async def run_with_ainvoke_for_provider(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
    provider: LLMProvider,
    user_input: str,
    system_prompt: str,
) -> tuple[LLMProvider, dict[str, Any] | None, str | None]:
    """Run the graph using async ainvoke for a specific provider.

    Demonstrates basic async invocation where the entire result
    is returned after completion. Returns the result for later display.

    Args:
        graph: The compiled graph to invoke.
        provider: The LLM provider to use.
        user_input: The user's input message.
        system_prompt: The system prompt content.

    Returns:
        A tuple of (provider, result_dict, error_message).
        If successful, error_message is None.
        If failed, result_dict is None and error_message contains the error.
    """
    logger.info(
        f"Invoking graph with ainvoke() using {provider.value}",
        extra={"user_input": user_input},
    )

    initial_state: GraphState = create_initial_state(
        user_input=user_input,
        provider=provider,
        system_prompt=system_prompt,
    )

    try:
        result: dict[str, Any] = await graph.ainvoke(initial_state)
    except LLMConfigurationError as e:
        error_msg: str = f"LLM error with {provider.value}: {e}"
        logger.error(error_msg)
        return (provider, None, error_msg)
    except asyncio.CancelledError:
        logger.warning(f"Graph execution was cancelled for {provider.value}")
        raise
    except Exception as e:
        error_msg = f"Unexpected error with {provider.value}: {e}"
        logger.error(error_msg)
        return (provider, None, error_msg)
    else:
        return (provider, result, None)


async def run_with_astream(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
    provider: LLMProvider,
    user_input: str,
    system_prompt: str,
) -> None:
    """Run the graph using async streaming.

    Demonstrates streaming execution where state updates are
    yielded as they happen from each node.

    Args:
        graph: The compiled graph to invoke.
        provider: The LLM provider to use.
        user_input: The user's input message.
        system_prompt: The system prompt content.
    """
    logger.info(
        f"Invoking graph with astream() using {provider.value}",
        extra={"user_input": user_input},
    )

    initial_state: GraphState = create_initial_state(
        user_input=user_input,
        provider=provider,
        system_prompt=system_prompt,
    )

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
    """Run the async LLM graph example with all available providers.

    Async entry point that:
    1. Loads configuration from environment
    2. Detects all available LLM providers
    3. Loads the system prompt from file
    4. Creates the graph once
    5. Executes all providers in PARALLEL using asyncio.gather
    6. Displays results for each provider
    7. Demonstrates streaming with a single provider

    Demonstrates both `ainvoke()` (parallel) and `astream()` patterns.
    """
    logger.info("Starting Pattern 02: Async LLM-Powered Graph (Multi-Provider)")

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

    # User input for parallel execution demo
    user_input: str = "What is LangGraph and how does it help with building AI agents?"

    # DEMO 1: Parallel execution with asyncio.gather
    print("\n" + "#" * 60)
    print("# DEMO 1: Using ainvoke() - Parallel execution across ALL providers")
    print("#" * 60)

    # Create tasks for all providers
    tasks = [
        run_with_ainvoke_for_provider(graph, provider, user_input, system_prompt)
        for provider in available
    ]

    # Execute all in parallel
    results: list[
        tuple[LLMProvider, dict[str, Any] | None, str | None]
    ] = await asyncio.gather(*tasks)

    # Display results in order
    success_count: int = 0
    for provider, result, error in results:
        print("\n" + "#" * 60)
        print(f"# Result from provider: {provider.value.upper()}")
        print("#" * 60)

        if result is not None:
            _display_result(user_input, result, provider)
            success_count += 1
        else:
            print(f"\n[ERROR] {error}\n")

    # Summary for DEMO 1
    print("\n" + "#" * 60)
    print("# DEMO 1 EXECUTION SUMMARY")
    print("#" * 60)
    print(f"# Total providers: {len(available)}")
    print(f"# Successful: {success_count}")
    print(f"# Failed: {len(available) - success_count}")
    print("#" * 60)

    # DEMO 2: Streaming with first available provider only
    print("\n" + "#" * 60)
    print("# DEMO 2: Using astream() - Streaming state updates (single provider)")
    print(f"# Using: {available[0].value.upper()}")
    print("#" * 60)

    stream_user_input: str = "Explain async/await in Python in 2-3 sentences."
    await run_with_astream(
        graph,
        available[0],
        stream_user_input,
        system_prompt,
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
