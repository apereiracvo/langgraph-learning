"""Human-in-the-loop approval agent.

Demonstrates pausing execution for human approval before executing
sensitive actions like sending emails or writing files.

Key Pattern:
- Uses interrupt() function to pause execution
- Resumes with Command(resume=...) containing approval decision
- Sensitive tools (email, file write) require approval
- Safe tools (calculator) execute immediately

Graph Flow:
    START -> agent -> route_action -> (request_approval | execute_tools | respond)
                                             |
                                             v
                                        [INTERRUPT]
                                             |
                                        (on resume)
                                             v
                                       execute_tools -> agent (loop)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from p04_checkpointing.state import CheckpointState
from p04_checkpointing.tools import get_all_tools, get_sensitive_tools
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




def _create_agent_node(
    settings: Settings,
    tools: list[BaseTool],
    sensitive_tool_names: set[str],
) -> Callable[[CheckpointState, RunnableConfig], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the agent node that generates responses and detects sensitive actions.

    Args:
        settings: Application settings for LLM configuration.
        tools: List of tools to bind to the LLM.
        sensitive_tool_names: Set of tool names that require approval.

    Returns:
        An async node function for the agent.
    """

    async def agent_node(
        state: CheckpointState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        """Generate response, potentially requesting tool calls.

        Args:
            state: Current agent state.
            config: Runnable config for context propagation.

        Returns:
            State update with response and pending action if sensitive.
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

        # Check if response contains sensitive tool calls
        pending_action: str | None = None
        if response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"] in sensitive_tool_names:
                    pending_action = f"Tool: {tc['name']}\nArgs: {tc['args']}"
                    break

        return {
            "messages": [response],
            "pending_action": pending_action,
        }

    return agent_node


def _route_action(state: CheckpointState) -> str:
    """Route based on whether approval is needed.

    Args:
        state: Current agent state.

    Returns:
        Next node name: "request_approval", "execute_tools", or "respond".
    """
    # If there's a pending action needing approval
    if state.get("pending_action"):
        return "request_approval"

    # If there are tool calls (but not sensitive ones)
    messages: list[BaseMessage] = state["messages"]
    if messages:
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "execute_tools"

    return "respond"


def request_approval_node(state: CheckpointState) -> dict[str, Any]:
    """Interrupt and wait for human approval.

    IMPORTANT: Code before interrupt() re-executes on resume.
    Keep pre-interrupt code idempotent.

    Args:
        state: Current agent state.

    Returns:
        State update with approval decision after resume.
    """
    pending = state.get("pending_action", "Unknown action")

    # Interrupt execution and wait for human input
    # The interrupt() call pauses here until resumed with Command(resume=...)
    approval: str = interrupt({
        "message": "Approval required for sensitive action:",
        "action": pending,
        "options": ["approve", "reject"],
    })

    # This code runs AFTER resume with the approval value
    is_approved = approval == "approve"

    return {
        "action_approved": is_approved,
        "pending_action": None,  # Clear pending action
    }


def _create_execute_tools_node(
    tools: list[BaseTool],
) -> Callable[[CheckpointState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create the tool execution node.

    Args:
        tools: List of available tools.

    Returns:
        An async node function for tool execution.
    """
    tools_by_name: dict[str, BaseTool] = {tool.name: tool for tool in tools}

    async def execute_tools_node(state: CheckpointState) -> dict[str, Any]:
        """Execute tool calls, respecting approval status.

        Args:
            state: Current agent state.

        Returns:
            State update with tool results.
        """
        # Check if action was rejected
        if state.get("action_approved") is False:
            return {
                "messages": [],
                "final_response": "Action was rejected by user. The requested action was not executed.",
                "action_approved": None,
            }

        messages: list[BaseMessage] = list(state["messages"])
        if not messages:
            return {"messages": [], "tool_calls_made": [], "action_approved": None}

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": [], "tool_calls_made": [], "action_approved": None}

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
            "action_approved": None,  # Reset approval state
        }

    return execute_tools_node


def respond_node(state: CheckpointState) -> dict[str, Any]:
    """Extract final response from messages.

    Args:
        state: Current agent state.

    Returns:
        State update with final response.
    """
    # If final_response already set (e.g., from rejection), keep it
    if state.get("final_response"):
        return {}

    messages: list[BaseMessage] = state["messages"]
    final: str = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            final = str(msg.content)
            break

    return {"final_response": final}


def _route_after_tools(state: CheckpointState) -> str:
    """Route after tool execution.

    If tools were executed (tool messages present), loop back to agent.
    If rejection happened (final_response set, no tool messages), go to respond.

    Args:
        state: Current agent state.

    Returns:
        Next node: "agent" or "respond".
    """
    # If final_response is set (rejection case), go to respond/END
    if state.get("final_response"):
        return "respond"

    # Otherwise loop back to agent for follow-up
    return "agent"


def create_approval_agent(
    settings: Settings,
    checkpointer: BaseCheckpointSaver,  # type: ignore[type-arg]
) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Create an agent that requires approval for sensitive actions.

    The agent pauses before executing sensitive tools (email, file write)
    and waits for human approval before proceeding.

    Args:
        settings: Application settings for LLM configuration.
        checkpointer: The checkpointer instance for state persistence.

    Returns:
        A compiled StateGraph with human-in-the-loop capability.

    Example:
        >>> async with get_checkpointer(CheckpointerType.MEMORY) as cp:
        ...     agent = create_approval_agent(settings, cp)
        ...     config = {"configurable": {"thread_id": "approval-flow"}}
        ...
        ...     # First call - will pause at approval
        ...     state = create_initial_state(
        ...         "Send email to bob@example.com",
        ...         provider,
        ...         prompt,
        ...     )
        ...     result1 = await agent.ainvoke(state, config)
        ...
        ...     # Check if paused
        ...     current = await agent.aget_state(config)
        ...     if current.next:
        ...         # Resume with approval
        ...         from langgraph.types import Command
        ...         result2 = await agent.ainvoke(Command(resume="approve"), config)
    """
    tools: list[BaseTool] = get_all_tools()
    sensitive_tool_names: set[str] = {t.name for t in get_sensitive_tools()}

    # Create nodes
    agent_node_fn = _create_agent_node(settings, tools, sensitive_tool_names)
    execute_tools_fn = _create_execute_tools_node(tools)

    # Build graph
    builder: StateGraph[CheckpointState] = StateGraph(CheckpointState)

    builder.add_node("agent", agent_node_fn)  # type: ignore[call-overload]
    builder.add_node("request_approval", request_approval_node)
    builder.add_node("execute_tools", execute_tools_fn)  # type: ignore[arg-type]
    builder.add_node("respond", respond_node)

    # Define edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        _route_action,
        {
            "request_approval": "request_approval",
            "execute_tools": "execute_tools",
            "respond": "respond",
        },
    )
    builder.add_edge("request_approval", "execute_tools")
    builder.add_conditional_edges(
        "execute_tools",
        _route_after_tools,
        {
            "agent": "agent",  # Loop back for follow-up after successful tool execution
            "respond": "respond",  # Go to END after rejection
        },
    )
    builder.add_edge("respond", END)

    # Compile WITH checkpointer - required for interrupt() to work
    compiled: CompiledStateGraph = builder.compile(  # type: ignore[type-arg]
        checkpointer=checkpointer
    )

    return compiled
