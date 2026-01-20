"""State definitions for the Checkpointing pattern.

Defines the state schema that will be persisted across checkpoints.
Includes conversation history, tool tracking, and approval state for
human-in-the-loop workflows.

The state uses LangGraph's add_messages reducer to automatically
accumulate messages across invocations, enabling multi-turn conversations.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from shared.enums import LLMProvider  # noqa: TC001


class CheckpointState(TypedDict):
    """State schema for checkpointed conversations.

    This state is persisted at each checkpoint, enabling:
    - Conversation memory across invocations
    - Human-in-the-loop approval workflows
    - Fault recovery from any checkpoint

    Attributes:
        messages: Conversation history using add_messages reducer.
            Automatically accumulates messages across invocations.
        provider: The LLM provider being used.
        system_prompt: The loaded system prompt content.
        pending_action: Description of action awaiting approval (if any).
        action_approved: Whether the pending action was approved.
        tool_calls_made: List of tool names invoked during execution.
        final_response: The final text response.
        metadata: Additional context (turn count, timestamps, etc.).
    """

    messages: Annotated[list[BaseMessage], add_messages]
    provider: LLMProvider
    system_prompt: str
    pending_action: str | None
    action_approved: bool | None
    tool_calls_made: list[str]
    final_response: str
    metadata: dict[str, Any]


def create_initial_state(
    user_input: str,
    provider: LLMProvider,
    system_prompt: str,
) -> CheckpointState:
    """Create an initial state for a new conversation.

    Args:
        user_input: The user's first message.
        provider: The LLM provider to use.
        system_prompt: The system prompt content.

    Returns:
        A fully initialized CheckpointState.

    Example:
        >>> state = create_initial_state(
        ...     user_input="Hello!",
        ...     provider=LLMProvider.OPENAI,
        ...     system_prompt="You are helpful.",
        ... )
    """
    return CheckpointState(
        messages=[HumanMessage(content=user_input)],
        provider=provider,
        system_prompt=system_prompt,
        pending_action=None,
        action_approved=None,
        tool_calls_made=[],
        final_response="",
        metadata={"turn_count": 1},
    )
