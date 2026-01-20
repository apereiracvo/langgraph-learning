"""Fault recovery demonstration.

Shows how to resume execution from a checkpoint after a simulated failure.
This pattern is critical for long-running workflows where failures can
occur at any point.

Key Concepts Demonstrated:
- State is saved after each successful node
- Failures don't lose progress from completed nodes
- aupdate_state() can modify checkpointed state before resume
- ainvoke(None, config) resumes from the last checkpoint
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from p04_checkpointing.checkpointer.factory import CheckpointerType, get_checkpointer
from p04_checkpointing.state import CheckpointState
from shared.enums import LLMProvider


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


# region Constants


class NodeName(StrEnum):
    """Node names for the recovery demo graph.

    Attributes:
        STEP_1: First processing step.
        STEP_2: Second processing step (simulates failure).
        STEP_3: Third processing step.
    """

    STEP_1 = "step_1"
    STEP_2 = "step_2"
    STEP_3 = "step_3"


# Type alias for step functions
StepFunction = Callable[[CheckpointState], dict[str, Any]]

# Error message constant
NETWORK_ERROR_MSG = "Simulated network error in step_2"


# endregion


# region Exceptions


class SimulatedNetworkError(Exception):
    """Simulated network failure for demonstration purposes."""


# endregion


# region Private Functions


def _create_step_functions(
    execution_log: list[str],
) -> tuple[StepFunction, StepFunction, StepFunction]:
    """Create step functions for the demo graph.

    Args:
        execution_log: List to track execution order.

    Returns:
        Tuple of (step_1, step_2, step_3) functions.
    """

    def step_1(state: CheckpointState) -> dict[str, Any]:
        """First step - always succeeds."""
        execution_log.append("step_1")
        metadata = dict(state.get("metadata", {}))
        metadata["step_1_complete"] = True
        metadata["step_1_result"] = "Data processed successfully"
        return {"metadata": metadata}

    def step_2(state: CheckpointState) -> dict[str, Any]:
        """Second step - fails on first attempt, succeeds on retry."""
        execution_log.append("step_2")

        # Simulate failure on first attempt
        metadata = state.get("metadata", {})
        if not metadata.get("step_2_retried"):
            raise SimulatedNetworkError(NETWORK_ERROR_MSG)

        # Success on retry
        metadata = dict(metadata)
        metadata["step_2_complete"] = True
        metadata["step_2_result"] = "External API call succeeded"
        return {"metadata": metadata}

    def step_3(state: CheckpointState) -> dict[str, Any]:
        """Third step - completes workflow."""
        execution_log.append("step_3")
        metadata = dict(state.get("metadata", {}))
        metadata["step_3_complete"] = True
        return {
            "metadata": metadata,
            "final_response": "Workflow completed successfully!",
        }

    return step_1, step_2, step_3


def _build_recovery_graph(
    step_1: StepFunction,
    step_2: StepFunction,
    step_3: StepFunction,
) -> StateGraph[CheckpointState]:
    """Build the recovery demo graph.

    Args:
        step_1: First step function.
        step_2: Second step function.
        step_3: Third step function.

    Returns:
        Configured StateGraph.
    """
    builder: StateGraph[CheckpointState] = StateGraph(CheckpointState)
    builder.add_node(NodeName.STEP_1, step_1)  # type: ignore[call-overload]
    builder.add_node(NodeName.STEP_2, step_2)  # type: ignore[call-overload]
    builder.add_node(NodeName.STEP_3, step_3)  # type: ignore[call-overload]

    builder.add_edge(START, NodeName.STEP_1)
    builder.add_edge(NodeName.STEP_1, NodeName.STEP_2)
    builder.add_edge(NodeName.STEP_2, NodeName.STEP_3)
    builder.add_edge(NodeName.STEP_3, END)

    return builder


async def _run_phase_1(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
    initial_state: CheckpointState,
    config: dict[str, Any],
    execution_log: list[str],
) -> None:
    """Run Phase 1: Initial execution that will fail.

    Args:
        graph: Compiled graph.
        initial_state: Starting state.
        config: Run configuration.
        execution_log: Execution tracking list.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Initial Execution (will fail at step_2)")
    print("=" * 60)

    try:
        await graph.ainvoke(initial_state, config)  # type: ignore[arg-type]
        print("ERROR: Expected failure did not occur!")
    except SimulatedNetworkError as e:
        print(f"\nCaught expected failure: {e}")
        print(f"Execution log: {execution_log}")
        print("\nNote: step_1 completed, step_2 failed")


async def _run_phase_2(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run Phase 2: Inspect checkpoint state.

    Args:
        graph: Compiled graph.
        config: Run configuration.

    Returns:
        Saved metadata from checkpoint.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Inspect Checkpoint State")
    print("=" * 60)

    state_snapshot = await graph.aget_state(config)  # type: ignore[arg-type]
    saved_metadata: dict[str, Any] = state_snapshot.values.get("metadata", {})
    next_nodes = state_snapshot.next

    print(f"\nSaved state metadata: {saved_metadata}")
    print(f"Next nodes to execute: {next_nodes}")
    print(f"step_1_complete: {saved_metadata.get('step_1_complete', False)}")
    print(f"step_2_complete: {saved_metadata.get('step_2_complete', False)}")

    return saved_metadata


async def _run_phase_3(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
    config: dict[str, Any],
    saved_metadata: dict[str, Any],
    execution_log: list[str],
) -> dict[str, Any]:
    """Run Phase 3: Update state and resume execution.

    Args:
        graph: Compiled graph.
        config: Run configuration.
        saved_metadata: Metadata from checkpoint.
        execution_log: Execution tracking list.

    Returns:
        Final result from resumed execution.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Update State and Resume Execution")
    print("=" * 60)

    # Clear execution log for resume
    execution_log.clear()

    # Update state to mark retry flag
    print("\nUpdating state with retry flag...")
    updated_metadata = {
        **saved_metadata,
        "step_2_retried": True,
    }
    await graph.aupdate_state(
        config,  # type: ignore[arg-type]
        {"metadata": updated_metadata},
    )

    # Verify update
    updated_snapshot = await graph.aget_state(config)  # type: ignore[arg-type]
    print(f"Updated metadata: {updated_snapshot.values.get('metadata', {})}")

    # Resume from checkpoint (None = continue from last checkpoint)
    print("\nResuming execution...")
    result = await graph.ainvoke(None, config)  # type: ignore[arg-type]

    print(f"\nExecution log after resume: {execution_log}")
    print(f"Final response: {result.get('final_response', 'No response')}")

    return result


def _print_summary(result: dict[str, Any]) -> None:
    """Print the final summary and takeaways.

    Args:
        result: Final result from execution.
    """
    final_metadata = result.get("metadata", {})
    print("\n" + "-" * 60)
    print("Final State Summary:")
    print(f"  step_1_complete: {final_metadata.get('step_1_complete', False)}")
    print(f"  step_2_complete: {final_metadata.get('step_2_complete', False)}")
    print(f"  step_3_complete: {final_metadata.get('step_3_complete', False)}")
    print("-" * 60)

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print(
        """
1. Checkpointing saved state after step_1 completed
2. When step_2 failed, we didn't lose step_1's work
3. We updated the checkpoint to add a retry flag
4. Resuming with ainvoke(None, config) continued from step_2
5. The workflow completed successfully on retry
        """
    )


# endregion


# region Public Functions


async def run_fault_recovery_demo() -> None:
    """Demonstrate fault recovery using checkpoints.

    Steps:
    1. Create a multi-step graph (step_1 -> step_2 -> step_3)
    2. Run until step_2, which fails on first attempt
    3. Show the saved checkpoint state (step_1 completed)
    4. Update state to mark retry flag
    5. Resume execution from checkpoint
    6. Complete successfully

    This demonstrates how checkpointing enables recovery from failures
    without losing progress from completed steps.
    """
    # Track execution for demonstration
    execution_log: list[str] = []

    # Create step functions and build graph
    step_1, step_2, step_3 = _create_step_functions(execution_log)
    builder = _build_recovery_graph(step_1, step_2, step_3)

    async with get_checkpointer(CheckpointerType.MEMORY) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)

        thread_id = "recovery-demo-thread"
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}

        # Initial state
        initial_state: CheckpointState = {
            "messages": [],
            "provider": LLMProvider.OPENAI,  # Not used in this demo
            "system_prompt": "",
            "pending_action": None,
            "action_approved": None,
            "tool_calls_made": [],
            "final_response": "",
            "metadata": {"workflow_id": "demo-123"},
        }

        # Run the three phases
        await _run_phase_1(graph, initial_state, config, execution_log)
        saved_metadata = await _run_phase_2(graph, config)
        result = await _run_phase_3(graph, config, saved_metadata, execution_log)

        # Print summary
        _print_summary(result)


# endregion
