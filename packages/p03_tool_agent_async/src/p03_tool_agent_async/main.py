"""Entry point for the Async Tool Agent pattern.

This pattern demonstrates:
- Async LangGraph tool-calling agents
- Custom tool definitions with @tool decorator
- Basic tool-calling agent with StateGraph
- ReAct agent using LangGraph's prebuilt pattern
- Multi-provider LLM support (OpenAI, Anthropic, Google)
- Async graph invocation with `ainvoke()`

Run with: task run -- p03-tool-agent-async

Configuration:
    The pattern automatically detects and uses the first configured provider.
    Configure API keys via environment variables:
    - OPENAI_API_KEY for OpenAI
    - ANTHROPIC_API_KEY for Anthropic
    - GOOGLE_API_KEY for Google
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from p03_tool_agent_async.agents import create_basic_agent, create_react_agent
from p03_tool_agent_async.prompts import load_system_prompt
from p03_tool_agent_async.state import AgentState, create_initial_state
from shared.exceptions import LLMConfigurationError, PromptLoadError
from shared.llm import get_available_providers
from shared.logger import logger
from shared.settings import settings


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from p03_tool_agent_async.agents.react_agent import ReactAgentWrapper
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


def _display_messages(messages: list[BaseMessage], title: str) -> None:
    """Display conversation messages in a formatted way.

    Args:
        messages: List of messages to display.
        title: Title for the message section.
    """
    print(f"\n{title}")
    print("-" * 60)

    for msg in messages:
        if isinstance(msg, HumanMessage):
            print(f"[USER]: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print("[ASSISTANT]: (calling tools...)")
                for tc in msg.tool_calls:
                    print(f"  -> Tool: {tc['name']}")
                    print(f"     Args: {tc['args']}")
            elif msg.content:
                print(f"[ASSISTANT]: {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"[TOOL {msg.name}]: {msg.content}")


def _display_result(
    query: str,
    result: AgentState | dict[str, Any],
    provider: LLMProvider,
    agent_type: str,
) -> None:
    """Display the agent execution result.

    Args:
        query: The original user query.
        result: The agent execution result.
        provider: The LLM provider used.
        agent_type: Type of agent ("Basic" or "ReAct").
    """
    print("\n" + "=" * 60)
    print(f"{agent_type} AGENT RESULT ({provider.value.upper()})")
    print("=" * 60)

    print(f"\nQuery: {query}")

    # Extract messages
    messages: list[BaseMessage] = result.get("messages", [])
    _display_messages(messages, "Conversation:")

    # Show tools called
    tools_called: list[str] = result.get("tool_calls_made", [])
    if tools_called:
        print(f"\nTools called: {', '.join(tools_called)}")

    # Show final response
    final_response: str = result.get("final_response", "")
    if final_response:
        print(f"\nFinal Response: {final_response}")

    print("=" * 60)


async def run_basic_agent(
    provider: LLMProvider,
    query: str,
    system_prompt: str,
) -> AgentState | None:
    """Run the basic tool-calling agent.

    Args:
        provider: The LLM provider to use.
        query: The user's query.
        system_prompt: The system prompt content.

    Returns:
        The final agent state, or None if an error occurred.
    """
    logger.info(
        f"Running basic agent with {provider.value}",
        extra={"query": query},
    )

    try:
        agent: CompiledStateGraph = create_basic_agent(settings)  # type: ignore[type-arg]

        initial_state: AgentState = create_initial_state(
            user_input=query,
            provider=provider,
            system_prompt=system_prompt,
        )

        result: dict[str, Any] = await agent.ainvoke(initial_state)

        # Extract final response from last AI message
        messages: list[BaseMessage] = result.get("messages", [])
        final_response: str = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_response = str(msg.content)
                break

        # Build complete state
        complete_state: AgentState = {
            "messages": messages,
            "provider": provider,
            "system_prompt": system_prompt,
            "tool_calls_made": result.get("tool_calls_made", []),
            "final_response": final_response,
            "metadata": result.get("metadata", {}),
        }

    except LLMConfigurationError as e:
        logger.error(f"LLM configuration error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running basic agent: {e}")
        return None
    else:
        return complete_state


async def run_react_agent(
    provider: LLMProvider,
    query: str,
    system_prompt: str,
) -> AgentState | None:
    """Run the ReAct agent.

    Args:
        provider: The LLM provider to use.
        query: The user's query.
        system_prompt: The system prompt content.

    Returns:
        The final agent state, or None if an error occurred.
    """
    logger.info(
        f"Running ReAct agent with {provider.value}",
        extra={"query": query},
    )

    try:
        agent: ReactAgentWrapper = create_react_agent(settings)

        initial_state: AgentState = create_initial_state(
            user_input=query,
            provider=provider,
            system_prompt=system_prompt,
        )

        result: AgentState = await agent.ainvoke(initial_state)

    except LLMConfigurationError as e:
        logger.error(f"LLM configuration error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error running ReAct agent: {e}")
        return None
    else:
        return result


def _print_demo_header(demo_num: int, description: str) -> None:
    """Print a demo section header.

    Args:
        demo_num: The demo number.
        description: Description of the demo.
    """
    print("\n" + "#" * 60)
    print(f"# DEMO {demo_num}: {description}")
    print("#" * 60)


def _print_summary(
    provider: LLMProvider,
    results: dict[str, AgentState | None],
) -> None:
    """Print the execution summary.

    Args:
        provider: The LLM provider used.
        results: Dictionary of result names to their states.
    """
    print("\n" + "#" * 60)
    print("# EXECUTION SUMMARY")
    print("#" * 60)
    print(f"# Provider: {provider.value.upper()}")
    for name, result in results.items():
        status = "SUCCESS" if result else "FAILED"
        print(f"# {name}: {status}")
    print("#" * 60 + "\n")


async def _run_demos(
    provider: LLMProvider,
    system_prompt: str,
) -> dict[str, AgentState | None]:
    """Run all demo scenarios.

    Args:
        provider: The LLM provider to use.
        system_prompt: The system prompt content.

    Returns:
        Dictionary mapping demo names to their results.
    """
    calc_query: str = "What is 25 * 4 + 10?"
    weather_query: str = "What's the weather in Tokyo?"
    results: dict[str, AgentState | None] = {}

    # DEMO 1: Basic Agent with Calculator
    _print_demo_header(1, "Basic Tool-Calling Agent - Calculator")
    results["Basic Agent - Calculator"] = await run_basic_agent(
        provider, calc_query, system_prompt
    )
    if results["Basic Agent - Calculator"]:
        _display_result(calc_query, results["Basic Agent - Calculator"], provider, "Basic")

    # DEMO 2: Basic Agent with Weather
    _print_demo_header(2, "Basic Tool-Calling Agent - Weather")
    results["Basic Agent - Weather"] = await run_basic_agent(
        provider, weather_query, system_prompt
    )
    if results["Basic Agent - Weather"]:
        _display_result(weather_query, results["Basic Agent - Weather"], provider, "Basic")

    # DEMO 3: ReAct Agent with Calculator
    _print_demo_header(3, "ReAct Agent - Calculator")
    results["ReAct Agent - Calculator"] = await run_react_agent(
        provider, calc_query, system_prompt
    )
    if results["ReAct Agent - Calculator"]:
        _display_result(calc_query, results["ReAct Agent - Calculator"], provider, "ReAct")

    # DEMO 4: ReAct Agent with Weather
    _print_demo_header(4, "ReAct Agent - Weather")
    results["ReAct Agent - Weather"] = await run_react_agent(
        provider, weather_query, system_prompt
    )
    if results["ReAct Agent - Weather"]:
        _display_result(weather_query, results["ReAct Agent - Weather"], provider, "ReAct")

    return results


async def async_main() -> None:
    """Run the async tool agent examples.

    Async entry point that:
    1. Loads configuration from environment
    2. Detects the first available LLM provider
    3. Loads the system prompt from file
    4. Runs both agent types with example queries
    5. Displays results for each agent

    Demonstrates tool-calling with:
    - Calculator tool: "What is 25 * 4 + 10?"
    - Weather tool: "What's the weather in Tokyo?"
    """
    logger.info("Starting Pattern 03: Async Tool Agent")

    # Check available providers
    available: list[LLMProvider] = get_available_providers(settings)
    logger.info(
        "Available LLM providers",
        extra={"providers": [p.value for p in available]},
    )

    # Validate at least one provider is available
    if not _validate_providers(available):
        sys.exit(1)

    # Use first available provider
    provider: LLMProvider = available[0]

    print("\n" + "#" * 60)
    print("# Pattern 03: Async Tool Agent")
    print("#" * 60)
    print(f"# Using provider: {provider.value.upper()}")
    print("#" * 60)

    # Load system prompt
    try:
        system_prompt: str = load_system_prompt()
        logger.info("System prompt loaded successfully")
    except PromptLoadError as e:
        logger.error(f"Failed to load system prompt: {e}")
        sys.exit(1)

    # Run all demos and collect results
    results = await _run_demos(provider, system_prompt)

    # Print summary
    _print_summary(provider, results)


def main() -> None:
    """Entry point wrapper for async main.

    This function serves as the synchronous entry point required
    by the package script configuration. It wraps the async main
    function with `asyncio.run()`.
    """
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
