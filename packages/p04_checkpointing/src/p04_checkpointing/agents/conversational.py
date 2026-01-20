"""Conversational agent with automatic checkpointing.

Demonstrates multi-turn conversations where state persists across
invocations using the same thread_id. This is the simplest checkpointing
use case - the graph logic is identical to a non-checkpointed agent,
but state is automatically saved after each node.

Key Pattern:
- Same thread_id = same conversation (history preserved)
- Different thread_id = new conversation
- State is saved after each node execution
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from p04_checkpointing.state import CheckpointState
from p04_checkpointing.tools import get_all_tools
from shared.llm import create_llm


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from shared.enums import LLMProvider
    from shared.settings import Settings




def _should_continue(state: CheckpointState) -> str:
    """Determine whether to continue with tools or end.

    Args:
        state: Current agent state with messages.

    Returns:
        "tools" if there are pending tool calls, END otherwise.
    """
    messages: list[BaseMessage] = state["messages"]
    if not messages:
        return END

    last_message = messages[-1]

    # Check if the last message is an AIMessage with tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return END


def _create_agent_node(
    settings: Settings,
    tools: list[BaseTool],
) -> Callable[[CheckpointState, RunnableConfig], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the agent node function that calls the LLM.

    Args:
        settings: Application settings for LLM configuration.
        tools: List of tools to bind to the LLM.

    Returns:
        An async node function for the agent.
    """

    async def agent_node(
        state: CheckpointState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Process the current state and decide on action.

        Args:
            state: Current agent state.
            config: Runnable config for context propagation.

        Returns:
            State update with the LLM response message.
        """
        provider: LLMProvider = state["provider"]
        system_prompt: str = state["system_prompt"]
        messages: list[BaseMessage] = list(state["messages"])

        # Create LLM and bind tools
        llm: BaseChatModel = create_llm(provider, settings, temperature=0.7)
        llm_with_tools = llm.bind_tools(tools)

        # Prepend system message
        system_message: SystemMessage = SystemMessage(content=system_prompt)
        full_messages: list[BaseMessage] = [system_message, *messages]

        # Invoke LLM asynchronously
        response: AIMessage = await llm_with_tools.ainvoke(full_messages, config=config)

        return {"messages": [response]}

    return agent_node


def _create_tool_node(
    tools: list[BaseTool],
) -> Callable[[CheckpointState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the tool execution node.

    Args:
        tools: List of available tools.

    Returns:
        An async node function for tool execution.
    """
    tools_by_name: dict[str, BaseTool] = {tool.name: tool for tool in tools}

    async def tool_node(state: CheckpointState) -> dict[str, Any]:
        """Execute tool calls and return results.

        Args:
            state: Current agent state.

        Returns:
            State update with tool result messages and tracking.
        """
        messages: list[BaseMessage] = list(state["messages"])
        if not messages:
            return {"messages": [], "tool_calls_made": []}

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": [], "tool_calls_made": []}

        tool_messages: list[ToolMessage] = []
        tool_names_called: list[str] = []

        for tool_call in last_message.tool_calls:
            tool_name: str = tool_call["name"]
            tool_args: dict[str, Any] = tool_call["args"]
            tool_call_id: str = str(tool_call["id"])

            tool_names_called.append(tool_name)

            if tool_name not in tools_by_name:
                result: str = f"Error: Tool '{tool_name}' not found."
            else:
                try:
                    tool_func = tools_by_name[tool_name]
                    result = await tool_func.ainvoke(tool_args)
                except Exception as e:
                    result = f"Error executing {tool_name}: {e!s}"

            tool_messages.append(
                ToolMessage(
                    content=json.dumps(result) if not isinstance(result, str) else result,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            )

        return {
            "messages": tool_messages,
            "tool_calls_made": tool_names_called,
        }

    return tool_node


def create_conversational_agent(
    settings: Settings,
    checkpointer: BaseCheckpointSaver,  # type: ignore[type-arg]
) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Create a conversational agent with checkpointing.

    The agent maintains conversation history across invocations
    when called with the same thread_id. This demonstrates the
    simplest checkpointing pattern - automatic state persistence.

    Args:
        settings: Application settings for LLM configuration.
        checkpointer: The checkpointer instance for state persistence.

    Returns:
        A compiled StateGraph with checkpointing enabled.

    Example:
        >>> async with get_checkpointer(CheckpointerType.MEMORY) as cp:
        ...     agent = create_conversational_agent(settings, cp)
        ...     config = {"configurable": {"thread_id": "user-123"}}
        ...
        ...     # First turn
        ...     state1 = create_initial_state("Hi, I'm Alice", provider, prompt)
        ...     result1 = await agent.ainvoke(state1, config)
        ...
        ...     # Second turn - history is preserved
        ...     result2 = await agent.ainvoke(
        ...         {"messages": [HumanMessage(content="What's my name?")]},
        ...         config,
        ...     )
        ...     # Agent remembers "Alice" from first turn
    """
    tools: list[BaseTool] = get_all_tools()

    # Create graph builder
    builder: StateGraph[CheckpointState] = StateGraph(CheckpointState)

    # Create nodes
    agent_node_fn = _create_agent_node(settings, tools)
    tool_node_fn = _create_tool_node(tools)

    # Add nodes
    builder.add_node("agent", agent_node_fn)  # type: ignore[call-overload]
    builder.add_node("tools", tool_node_fn)  # type: ignore[call-overload]

    # Define edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        _should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    builder.add_edge("tools", "agent")

    # Compile WITH checkpointer - this enables state persistence
    compiled: CompiledStateGraph = builder.compile(  # type: ignore[type-arg]
        checkpointer=checkpointer
    )

    return compiled
