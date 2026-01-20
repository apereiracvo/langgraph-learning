"""Human-in-the-loop approval demonstration.

Shows how to pause execution for human approval and resume after
receiving the approval decision. This pattern is essential for
workflows where certain actions require human oversight.

Key Concepts Demonstrated:
- interrupt() pauses execution and saves state
- Command(resume=...) provides the human's decision
- Agent can proceed or cancel based on approval
- Checkpointing enables the pause/resume pattern
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command

from p04_checkpointing.agents.approval import create_approval_agent
from p04_checkpointing.checkpointer.factory import CheckpointerType, get_checkpointer
from p04_checkpointing.prompts import load_system_prompt
from p04_checkpointing.state import create_initial_state
from shared.llm import get_available_providers
from shared.settings import settings


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from shared.enums import LLMProvider



# Display constants
_MAX_CONTENT_LENGTH = 300
_RECENT_MESSAGES_COUNT = 3


def _display_messages(messages: list[BaseMessage]) -> None:
    """Display conversation messages in a formatted way.

    Args:
        messages: List of messages to display.
    """
    for msg in messages:
        if isinstance(msg, HumanMessage):
            print(f"[USER]: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print("[ASSISTANT]: (requesting tool calls...)")
                for tc in msg.tool_calls:
                    print(f"  -> Tool: {tc['name']}")
                    print(f"     Args: {tc['args']}")
            elif msg.content:
                content = str(msg.content)
                if len(content) > _MAX_CONTENT_LENGTH:
                    content = content[:_MAX_CONTENT_LENGTH] + "..."
                print(f"[ASSISTANT]: {content}")


def _get_recent_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Get the most recent messages for display.

    Args:
        messages: Full message list.

    Returns:
        Last N messages.
    """
    if len(messages) > _RECENT_MESSAGES_COUNT:
        return messages[-_RECENT_MESSAGES_COUNT:]
    return messages


async def _run_approval_scenario(
    agent: CompiledStateGraph,  # type: ignore[type-arg]
    provider: LLMProvider,
    system_prompt: str,
    thread_id: str,
    user_input: str,
    approval_decision: str,
    scenario_name: str,
) -> None:
    """Run a single approval scenario.

    Args:
        agent: The compiled approval agent.
        provider: LLM provider to use.
        system_prompt: System prompt content.
        thread_id: Thread ID for this scenario.
        user_input: User's request.
        approval_decision: "approve" or "reject".
        scenario_name: Display name for the scenario.
    """
    print("\n" + "=" * 60)
    print(f"SCENARIO: {scenario_name}")
    print("=" * 60)

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

    # Initial request
    print("\n--- Initial Request ---")
    state = create_initial_state(
        user_input=user_input,
        provider=provider,
        system_prompt=system_prompt,
    )

    # First invocation - should pause for approval
    print("Invoking agent (expecting pause for approval)...")
    result1: dict[str, Any] = await agent.ainvoke(state, config)  # type: ignore[arg-type]

    # Check if agent is paused
    current_state = await agent.aget_state(config)  # type: ignore[arg-type]
    if current_state.next:
        print(f"\nAgent PAUSED at node(s): {current_state.next}")
        pending = current_state.values.get("pending_action")
        if pending:
            print(f"Pending action:\n{pending}")

        # Apply human decision
        decision_text = "APPROVES" if approval_decision == "approve" else "REJECTS"
        print(f"\n--- Human {decision_text} the action ---")
        result2: dict[str, Any] = await agent.ainvoke(
            Command(resume=approval_decision),
            config,  # type: ignore[arg-type]
        )

        messages: list[BaseMessage] = result2.get("messages", [])
        print(f"\nConversation after {approval_decision}:")
        _display_messages(_get_recent_messages(messages))

        final_response = result2.get("final_response", "")
        if final_response:
            print(f"\nFinal Response: {final_response}")

        tools_called = result2.get("tool_calls_made", [])
        if tools_called:
            print(f"Tools executed: {', '.join(tools_called)}")
        elif approval_decision == "reject":
            print("Tools executed: None (action was rejected)")
    else:
        print("Agent completed without pausing (no sensitive action detected)")
        messages = result1.get("messages", [])
        _display_messages(messages)


async def run_approval_demo() -> None:
    """Demonstrate human-in-the-loop approval workflow.

    Shows:
    1. User requests sensitive action (send email)
    2. Agent recognizes sensitive tool and pauses for approval
    3. Human provides approval (simulated)
    4. Agent executes the action and responds

    Also demonstrates rejection flow where the action is cancelled.
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
        agent: CompiledStateGraph = create_approval_agent(  # type: ignore[type-arg]
            settings, checkpointer
        )

        # Scenario 1: Approved Action
        await _run_approval_scenario(
            agent=agent,
            provider=provider,
            system_prompt=system_prompt,
            thread_id="approval-demo-approved",
            user_input="Please send an email to bob@example.com saying 'Hello Bob!'",
            approval_decision="approve",
            scenario_name="Sensitive Action with APPROVAL",
        )

        # Scenario 2: Rejected Action
        await _run_approval_scenario(
            agent=agent,
            provider=provider,
            system_prompt=system_prompt,
            thread_id="approval-demo-rejected",
            user_input="Write a file to /tmp/secret.txt with content 'secret data'",
            approval_decision="reject",
            scenario_name="Sensitive Action with REJECTION",
        )

        print("\n" + "-" * 60)
        print("Approval demo completed!")
        print("-" * 60)
