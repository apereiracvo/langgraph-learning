"""State definitions for the Tool Agent pattern.

This module defines the agent state schema using TypedDict with full type hints.
The state tracks messages, tool calls, and execution metadata for tool-calling agents.

The state schema supports both basic tool-calling agents and ReAct agents by
storing the full message history using LangGraph's message accumulator pattern.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from shared.enums import LLMProvider  # noqa: TC001


class AgentState(TypedDict):
    """Agent state for tool-calling conversation.

    This state schema tracks all information needed for tool-calling agents,
    including the full message history, tool execution results, and configuration.

    Attributes:
        messages: Conversation history using LangGraph's message accumulator.
            Uses the add_messages reducer to properly append new messages.
            This includes HumanMessages, AIMessages (with tool_calls), and ToolMessages.
        provider: The LLM provider being used (OpenAI, Anthropic, or Google).
        system_prompt: The loaded system prompt content.
        tool_calls_made: List of tool names that were invoked during execution.
        final_response: The final text response after tool execution.
        metadata: Additional context or configuration data.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    provider: LLMProvider
    system_prompt: str
    tool_calls_made: list[str]
    final_response: str
    metadata: dict[str, Any]


def create_initial_state(
    user_input: str,
    provider: LLMProvider,
    system_prompt: str,
) -> AgentState:
    """Create an initial agent state with the user's input message.

    Args:
        user_input: The user's input message to process.
        provider: The LLM provider to use.
        system_prompt: The system prompt content.

    Returns:
        A fully initialized AgentState ready for graph invocation.

    Example:
        >>> state = create_initial_state(
        ...     user_input="What is 25 * 4?",
        ...     provider=LLMProvider.OPENAI,
        ...     system_prompt="You are a helpful assistant with tools.",
        ... )
    """
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_input)],
        "provider": provider,
        "system_prompt": system_prompt,
        "tool_calls_made": [],
        "final_response": "",
        "metadata": {},
    }
    return initial_state


