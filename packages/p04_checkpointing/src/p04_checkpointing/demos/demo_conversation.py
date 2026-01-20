"""Multi-turn conversation demonstration.

Shows how checkpointing maintains conversation history across multiple
invocations with the same thread_id. This is the most common use case
for checkpointing - enabling stateful conversations.

Key Concepts Demonstrated:
- Conversation memory persists across ainvoke() calls
- Using the same thread_id continues the conversation
- Using a different thread_id starts a fresh conversation
- State history can be retrieved with aget_state_history()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from p04_checkpointing.agents.conversational import create_conversational_agent
from p04_checkpointing.checkpointer.factory import CheckpointerType, get_checkpointer
from p04_checkpointing.prompts import load_system_prompt
from p04_checkpointing.state import create_initial_state
from shared.llm import get_available_providers
from shared.settings import settings


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


# region Constants


# Display constants
MAX_CONTENT_LENGTH = 200
MAX_CHECKPOINTS_DISPLAY = 5


# endregion


# region Private Functions


def _display_messages(messages: list[BaseMessage], title: str) -> None:
    """Display conversation messages in a formatted way.

    Args:
        messages: List of messages to display.
        title: Title for the message section.
    """
    print(f"\n{title}")
    print("-" * 50)

    for msg in messages:
        if isinstance(msg, HumanMessage):
            print(f"[USER]: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print("[ASSISTANT]: (calling tools...)")
                for tc in msg.tool_calls:
                    print(f"  -> {tc['name']}: {tc['args']}")
            elif msg.content:
                # Truncate long responses for display
                content = str(msg.content)
                if len(content) > MAX_CONTENT_LENGTH:
                    content = content[:MAX_CONTENT_LENGTH] + "..."
                print(f"[ASSISTANT]: {content}")


# endregion


# region Public Functions


async def run_conversation_demo() -> None:
    """Demonstrate multi-turn conversation with memory.

    Shows:
    1. First message establishes context (user's name)
    2. Second message references previous context
    3. Agent correctly recalls information from first turn
    4. State history shows all checkpoints
    """
    # Get first available provider
    available = get_available_providers(settings)
    if not available:
        print("ERROR: No LLM providers configured. Please set an API key.")
        return

    provider = available[0]
    print(f"Using provider: {provider.value.upper()}")

    # Load system prompt
    system_prompt = load_system_prompt()

    async with get_checkpointer(CheckpointerType.MEMORY) as checkpointer:
        agent: CompiledStateGraph = create_conversational_agent(  # type: ignore[type-arg]
            settings, checkpointer
        )

        thread_id = "conversation-demo-thread"
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

        # === TURN 1: Establish context ===
        print("\n" + "=" * 50)
        print("TURN 1: Establishing context")
        print("=" * 50)

        state1 = create_initial_state(
            user_input="Hi! My name is Alice and I love Python programming.",
            provider=provider,
            system_prompt=system_prompt,
        )

        result1 = await agent.ainvoke(state1, config)  # type: ignore[arg-type]
        messages1: list[BaseMessage] = result1.get("messages", [])
        _display_messages(messages1, "Conversation after Turn 1:")

        # === TURN 2: Reference previous context ===
        print("\n" + "=" * 50)
        print("TURN 2: Testing memory")
        print("=" * 50)

        # NOTE: We only send the new message - history is loaded from checkpoint
        result2 = await agent.ainvoke(
            {"messages": [HumanMessage(content="What's my name and what do I like?")]},
            config,  # type: ignore[arg-type]
        )
        messages2: list[BaseMessage] = result2.get("messages", [])
        _display_messages(messages2, "Conversation after Turn 2:")

        # Verify the agent remembered the context
        last_response = messages2[-1] if messages2 else None
        if isinstance(last_response, AIMessage) and last_response.content:
            content_lower = str(last_response.content).lower()
            remembered_name = "alice" in content_lower
            remembered_interest = "python" in content_lower

            print("\n" + "-" * 50)
            print("Memory Check:")
            print(f"  Remembered name (Alice): {'Yes' if remembered_name else 'No'}")
            print(
                f"  Remembered interest (Python): "
                f"{'Yes' if remembered_interest else 'No'}"
            )

        # === Show checkpoint history ===
        print("\n" + "=" * 50)
        print("CHECKPOINT HISTORY")
        print("=" * 50)

        checkpoint_count = 0
        async for state_snapshot in agent.aget_state_history(config):  # type: ignore[arg-type]
            checkpoint_count += 1
            step = (
                state_snapshot.metadata.get("step", "?")
                if state_snapshot.metadata
                else "?"
            )
            msg_count = len(state_snapshot.values.get("messages", []))
            print(f"  Checkpoint {checkpoint_count}: Step {step}, {msg_count} messages")

            # Limit display to avoid overwhelming output
            if checkpoint_count >= MAX_CHECKPOINTS_DISPLAY:
                print(f"  ... (showing first {MAX_CHECKPOINTS_DISPLAY} checkpoints)")
                break

        print(f"\nTotal checkpoints found: {checkpoint_count}+")
        print("-" * 50)


# endregion
