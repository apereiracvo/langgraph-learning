"""Basic graph definition.

This module demonstrates the simplest possible LangGraph:
- A single node that processes input
- Linear flow from START -> process -> END
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, START, StateGraph


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


class State(TypedDict):
    """Graph state definition.

    Attributes:
        input: The input message to process.
        output: The processed output message.
    """

    input: str
    output: str


def process_node(state: State) -> dict[str, str]:
    """Process the input and generate output.

    Args:
        state: The current graph state.

    Returns:
        A dictionary with the output key to update state.
    """
    input_text: str = state["input"]
    return {"output": f"Processed: {input_text}"}


def create_graph() -> CompiledStateGraph[State]:
    """Create and compile the basic graph.

    Returns:
        A compiled StateGraph ready for invocation.
    """
    # Initialize the graph builder with our state schema
    builder: StateGraph[State] = StateGraph(State)

    # Add the processing node
    builder.add_node("process", process_node)

    # Define the edges (flow)
    builder.add_edge(START, "process")
    builder.add_edge("process", END)

    # Compile and return the graph
    return builder.compile()  # type: ignore[return-value]
