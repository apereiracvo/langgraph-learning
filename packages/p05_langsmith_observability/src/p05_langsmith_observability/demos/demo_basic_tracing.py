"""Demo 1: Basic automatic tracing with LangGraph.

This demo shows how LangChain/LangGraph automatically traces
all operations when LANGSMITH__TRACING_ENABLED=true.

Key concepts demonstrated:
- Zero-code tracing with environment variables
- Automatic capture of agent execution
- Tool call tracing
- Message flow visualization
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from p05_langsmith_observability.agent import create_traced_agent
from p05_langsmith_observability.prompts import load_system_prompt
from p05_langsmith_observability.state import AgentState, create_initial_state
from shared.exceptions import LLMConfigurationError, PromptLoadError
from shared.llm import get_available_providers
from shared.logger import logger
from shared.observability import is_tracing_enabled
from shared.settings import settings


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from shared.enums import LLMProvider


# region Private Functions


def _display_messages(messages: list[BaseMessage]) -> None:
    """Display conversation messages in a formatted way.

    Args:
        messages: List of messages to display.
    """
    print("\nConversation:")
    print("-" * 40)

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

    print("-" * 40)


# endregion


# region Public Functions


async def run_basic_tracing_demo() -> None:
    """Run the basic tracing demo.

    Demonstrates automatic trace capture of:
    - Agent invocation
    - Tool calls
    - LLM interactions
    - Message flow

    The demo executes a simple calculation query and shows how
    LangSmith automatically captures the entire execution flow.
    """
    print(f"\nTracing enabled: {is_tracing_enabled()}")

    if is_tracing_enabled():
        print(f"Project: {settings.langsmith.project}")
        print("Traces will be sent to LangSmith automatically")
    else:
        print("Tracing disabled - demo will run but no traces sent")
        print("Enable with: LANGSMITH__TRACING_ENABLED=true")

    # Check for available providers
    providers = get_available_providers(settings)
    if not providers:
        logger.error("No LLM providers configured")
        print("\nERROR: No LLM provider configured. Please set an API key.")
        return

    provider: LLMProvider = providers[0]
    print(f"\nUsing provider: {provider.value}")

    # Load system prompt
    try:
        system_prompt: str = load_system_prompt()
    except PromptLoadError as e:
        logger.error(f"Failed to load system prompt: {e}")
        print(f"\nERROR: {e}")
        return

    # Create agent
    try:
        agent: CompiledStateGraph = create_traced_agent(settings)  # type: ignore[type-arg]
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        print(f"\nERROR: Failed to create agent: {e}")
        return

    # Execute query
    query: str = "What is 25 * 4 + 10?"
    print(f"\nQuery: {query}")
    print("\n[Executing agent...]")

    initial_state: AgentState = create_initial_state(
        user_input=query,
        provider=provider,
        system_prompt=system_prompt,
    )

    try:
        result: dict[str, Any] = await agent.ainvoke(initial_state)
    except LLMConfigurationError as e:
        logger.error(f"LLM configuration error: {e}")
        print(f"\nERROR: {e}")
        return
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        print(f"\nERROR: Agent execution failed: {e}")
        return

    # Display results
    print("\nRun completed!")

    if is_tracing_enabled():
        print("\nView traces at: https://smith.langchain.com/")
        print(f"Project: {settings.langsmith.project}")

    # Display conversation
    messages: list[BaseMessage] = result.get("messages", [])
    _display_messages(messages)

    # Show what to look for in LangSmith
    print("\nWhat to look for in LangSmith UI:")
    print("  1. Agent Decision spans showing LLM calls")
    print("  2. Tool Execution spans with input/output")
    print("  3. Message flow between nodes")
    print("  4. Latency metrics for each operation")
    print("  5. Token usage (if available from provider)")


# endregion
