"""Entry point for the Basic Graph pattern.

This pattern demonstrates the fundamental concepts of LangGraph:
- Creating a StateGraph
- Defining state with TypedDict
- Adding nodes and edges
- Compiling and invoking the graph

Run with: task run -- p01-basic-graph
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from p01_basic_graph.graph import State, create_graph
from shared.logger import logger
from shared.utils import format_response


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def main() -> None:
    """Run the basic graph example."""
    logger.info("Starting Pattern 01: Basic Graph")

    # Create and compile the graph
    graph: CompiledStateGraph[State] = create_graph()

    # Invoke the graph with sample input
    result: dict[str, Any] = graph.invoke({"input": "Hello, LangGraph!"})  # type: ignore[arg-type]

    logger.info("Graph execution completed", extra={"result": result})
    logger.info(format_response(result))


if __name__ == "__main__":
    main()
