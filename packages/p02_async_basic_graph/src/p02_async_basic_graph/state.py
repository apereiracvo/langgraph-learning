"""State definitions for the Async LLM Graph pattern.

This module defines the graph state schema using TypedDict with full type hints.
The state tracks the conversation, LLM responses, and configuration.

Note:
    State definitions are identical to the synchronous version since
    async execution does not change the state schema.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage  # noqa: TC002
from langgraph.graph.message import add_messages

from shared.enums import LLMProvider  # noqa: TC001


class GraphState(TypedDict):
    """Graph state for async LLM-powered conversation.

    This state schema tracks all information needed for LLM interactions.

    Attributes:
        messages: Conversation history using LangGraph's message accumulator.
            Uses the add_messages reducer to properly append new messages.
        user_input: The current user input/query to process.
        llm_response: The LLM's response to the user input.
        provider: The LLM provider being used (OpenAI, Anthropic, or Google).
        system_prompt: The loaded system prompt content.
        metadata: Additional context or configuration data.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    user_input: str
    llm_response: str
    provider: LLMProvider
    system_prompt: str
    metadata: dict[str, Any]


def create_initial_state(
    user_input: str,
    provider: LLMProvider,
    system_prompt: str,
) -> GraphState:
    """Create an initial graph state with default values.

    Args:
        user_input: The user's input message to process.
        provider: The LLM provider to use.
        system_prompt: The system prompt content.

    Returns:
        A fully initialized GraphState ready for async graph invocation.

    Example:
        >>> state = create_initial_state(
        ...     user_input="Hello!",
        ...     provider=LLMProvider.OPENAI,
        ...     system_prompt="You are helpful.",
        ... )
    """
    initial_state: GraphState = {
        "messages": [],
        "user_input": user_input,
        "llm_response": "",
        "provider": provider,
        "system_prompt": system_prompt,
        "metadata": {},
    }
    return initial_state
