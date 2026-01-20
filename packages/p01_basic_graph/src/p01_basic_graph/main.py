"""Entry point for the Basic Graph pattern.

This pattern demonstrates the fundamental concepts of LangGraph:
- Creating a StateGraph
- Defining state with TypedDict
- Adding nodes and edges
- Compiling and invoking the graph

Run with: task run -- p01-basic-graph
"""

from __future__ import annotations

from p01_basic_graph.graph import create_graph
from shared.utils import format_response, setup_environment


def main() -> None:
    """Run the basic graph example."""
    setup_environment()

    print("=" * 50)
    print("Pattern 01: Basic Graph")
    print("=" * 50)

    # Create and compile the graph
    graph = create_graph()

    # Invoke the graph with sample input
    result = graph.invoke({"input": "Hello, LangGraph!"})  # type: ignore[arg-type]

    print(format_response(result))
    print("=" * 50)


if __name__ == "__main__":
    main()
