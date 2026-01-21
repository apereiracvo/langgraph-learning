"""Traced agent implementation with LangSmith observability.

This module implements a tool-calling agent with explicit tracing decorators
and metadata enrichment to demonstrate LangSmith observability features.

Key patterns demonstrated:
- @traceable decorator on node functions
- Dynamic metadata addition during execution
- Tag-based filtering support
- RunnableConfig metadata passing
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from p05_langsmith_observability.state import AgentState
from p05_langsmith_observability.tools import (
    get_all_tools,
    get_all_tools_with_failing,
)
from shared.llm import create_llm
from shared.observability import add_run_metadata, add_run_tags


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Sequence

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

    from shared.enums import LLMProvider
    from shared.settings import Settings


# region Constants


class NodeName(StrEnum):
    """Node names for the traced agent graph.

    Attributes:
        AGENT: The agent decision-making node.
        TOOLS: The tool execution node.
    """

    AGENT = "agent"
    TOOLS = "tools"


# endregion


# region Private Functions


def _should_continue(state: AgentState) -> str:
    """Determine whether to continue with tools or end.

    Implements the conditional routing logic for the ReAct-style loop:
    - If the last message has tool calls, route to "tools"
    - Otherwise, route to END

    Args:
        state: Current agent state with messages.

    Returns:
        NodeName.TOOLS if there are pending tool calls, END otherwise.
    """
    messages: Sequence[BaseMessage] = state["messages"]
    if not messages:
        return END

    last_message = messages[-1]

    # Check if the last message is an AIMessage with tool calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return NodeName.TOOLS

    return END


def _create_agent_node(
    settings: Settings,
    tools: list[BaseTool],
) -> Callable[[AgentState, RunnableConfig], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the traced agent node function.

    The agent node:
    1. Retrieves the LLM based on provider in state
    2. Binds tools to the LLM
    3. Invokes the LLM with the conversation history
    4. Adds tracing metadata about the decision

    Args:
        settings: Application settings for LLM configuration.
        tools: List of tools to bind to the LLM.

    Returns:
        An async node function for the agent with tracing.
    """

    @traceable(name="Agent Decision", run_type="llm")
    async def agent_node(
        state: AgentState,
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
        metadata: dict[str, Any] = state.get("metadata", {})

        # Add metadata about the agent decision
        add_run_metadata(
            {
                "provider": provider.value,
                "message_count": len(messages),
                "has_system_prompt": bool(system_prompt),
            }
        )

        # Add state metadata to trace
        if metadata:
            add_run_metadata({"state_metadata": metadata})

        # Add provider tag for filtering
        add_run_tags([f"provider:{provider.value}"])

        # Create LLM and bind tools
        llm: BaseChatModel = create_llm(provider, settings, temperature=0.0)
        llm_with_tools = llm.bind_tools(tools)

        # Prepend system message
        system_message: SystemMessage = SystemMessage(content=system_prompt)
        full_messages: list[BaseMessage] = [system_message, *messages]

        # Invoke LLM asynchronously
        response: AIMessage = await llm_with_tools.ainvoke(full_messages, config=config)

        # Add response metadata
        has_tool_calls = bool(response.tool_calls)
        add_run_metadata(
            {
                "has_tool_calls": has_tool_calls,
                "tool_call_count": len(response.tool_calls) if has_tool_calls else 0,
            }
        )

        if has_tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            add_run_tags([f"calling:{name}" for name in tool_names])

        return {"messages": [response]}

    return agent_node


def _create_tool_node(
    tools: list[BaseTool],
) -> Callable[[AgentState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the traced tool execution node.

    This node executes any tool calls from the last AI message and
    returns ToolMessages with the results.

    Args:
        tools: List of available tools.

    Returns:
        An async node function for tool execution with tracing.
    """
    tools_by_name: dict[str, BaseTool] = {tool.name: tool for tool in tools}

    @traceable(name="Tool Execution", run_type="tool")
    async def tool_node(state: AgentState) -> dict[str, Any]:
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

        # Add metadata about tools being executed
        add_run_metadata(
            {
                "tool_call_count": len(last_message.tool_calls),
                "available_tools": list(tools_by_name.keys()),
            }
        )

        for tool_call in last_message.tool_calls:
            tool_name: str = tool_call["name"]
            tool_args: dict[str, Any] = tool_call["args"]
            tool_call_id: str = str(tool_call["id"])

            tool_names_called.append(tool_name)

            # Add per-tool metadata
            add_run_metadata(
                {
                    f"tool_{tool_name}_args": tool_args,
                }
            )

            if tool_name not in tools_by_name:
                result: str = f"Error: Tool '{tool_name}' not found."
                add_run_metadata({f"tool_{tool_name}_error": "not_found"})
            else:
                try:
                    tool_func = tools_by_name[tool_name]
                    # Invoke tool asynchronously
                    result = await tool_func.ainvoke(tool_args)
                    add_run_metadata({f"tool_{tool_name}_success": True})
                except Exception as e:
                    result = f"Error executing {tool_name}: {e!s}"
                    add_run_metadata(
                        {
                            f"tool_{tool_name}_error": str(e),
                            f"tool_{tool_name}_error_type": type(e).__name__,
                        }
                    )

            tool_messages.append(
                ToolMessage(
                    content=json.dumps(result)
                    if not isinstance(result, str)
                    else result,
                    name=tool_name,
                    tool_call_id=tool_call_id,
                )
            )

        # Add summary tags
        add_run_tags([f"tool:{name}" for name in tool_names_called])

        return {
            "messages": tool_messages,
            "tool_calls_made": tool_names_called,
        }

    return tool_node


# endregion


# region Public Functions


def create_traced_agent(
    settings: Settings,
    *,
    include_failing_tool: bool = False,
) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Create and compile a traced tool-calling agent.

    Builds a StateGraph with the standard tool-calling pattern,
    enhanced with LangSmith tracing decorators on each node.

    Args:
        settings: Application settings for LLM configuration.
        include_failing_tool: If True, includes the failing tool for
            error tracing demonstrations.

    Returns:
        A compiled StateGraph ready for async invocation.

    Example:
        >>> from shared.settings import settings
        >>> agent = create_traced_agent(settings)
        >>> result = await agent.ainvoke(initial_state)
    """
    # Get tools based on configuration
    tools: list[BaseTool] = (
        get_all_tools_with_failing() if include_failing_tool else get_all_tools()
    )

    # Create graph builder
    builder: StateGraph[AgentState] = StateGraph(AgentState)

    # Create traced nodes
    agent_node_fn = _create_agent_node(settings, tools)
    tool_node_fn = _create_tool_node(tools)

    # Add nodes
    builder.add_node(NodeName.AGENT, agent_node_fn)  # type: ignore[call-overload]
    builder.add_node(NodeName.TOOLS, tool_node_fn)  # type: ignore[call-overload]

    # Define edges
    builder.add_edge(START, NodeName.AGENT)
    builder.add_conditional_edges(
        NodeName.AGENT,
        _should_continue,
        {
            NodeName.TOOLS: NodeName.TOOLS,
            END: END,
        },
    )
    builder.add_edge(NodeName.TOOLS, NodeName.AGENT)

    # Compile and return
    compiled: CompiledStateGraph = builder.compile()  # type: ignore[type-arg]

    return compiled


# endregion
