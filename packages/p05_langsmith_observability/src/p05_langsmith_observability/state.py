"""State definitions for the LangSmith Observability pattern.

This module defines the agent state schema using TypedDict with full type hints.
The state extends the basic agent state with additional fields for tracing
metadata and demonstration-specific context.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from shared.enums import LLMProvider  # noqa: TC001


# region Types


class AgentState(TypedDict):
    """Agent state for traced tool-calling conversation.

    This state schema tracks all information needed for tool-calling agents
    with additional fields for observability demonstration.

    Attributes:
        messages: Conversation history using LangGraph's message accumulator.
            Uses the add_messages reducer to properly append new messages.
        provider: The LLM provider being used (OpenAI, Anthropic, or Google).
        system_prompt: The loaded system prompt content.
        tool_calls_made: List of tool names that were invoked during execution.
        final_response: The final text response after tool execution.
        metadata: Additional context or configuration data for tracing.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    provider: LLMProvider
    system_prompt: str
    tool_calls_made: list[str]
    final_response: str
    metadata: dict[str, Any]


# endregion


# region Public Functions


def create_initial_state(
    user_input: str,
    provider: LLMProvider,
    system_prompt: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> AgentState:
    """Create an initial agent state with the user's input message.

    Args:
        user_input: The user's input message to process.
        provider: The LLM provider to use.
        system_prompt: The system prompt content.
        metadata: Optional metadata to attach to the state.

    Returns:
        A fully initialized AgentState ready for graph invocation.

    Example:
        >>> state = create_initial_state(
        ...     user_input="What is 25 * 4?",
        ...     provider=LLMProvider.OPENAI,
        ...     system_prompt="You are a helpful assistant with tools.",
        ...     metadata={"user_id": "123", "session_id": "abc"},
        ... )
    """
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_input)],
        "provider": provider,
        "system_prompt": system_prompt,
        "tool_calls_made": [],
        "final_response": "",
        "metadata": metadata or {},
    }
    return initial_state


# endregion
