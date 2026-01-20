"""Unit tests for the Basic Graph pattern."""

from __future__ import annotations

from p01_basic_graph.graph import State, create_graph, process_node


class TestProcessNode:
    """Tests for the process_node function."""

    def test_process_node_returns_processed_output(self) -> None:
        """Test that process_node correctly processes input."""
        state: State = {"input": "Hello", "output": ""}
        result = process_node(state)

        assert "output" in result
        assert result["output"] == "Processed: Hello"

    def test_process_node_handles_empty_input(self) -> None:
        """Test that process_node handles empty input."""
        state: State = {"input": "", "output": ""}
        result = process_node(state)

        assert result["output"] == "Processed: "


class TestCreateGraph:
    """Tests for the create_graph function."""

    def test_create_graph_returns_compiled_graph(self) -> None:
        """Test that create_graph returns a compiled graph."""
        graph = create_graph()

        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_graph_invoke_produces_output(self) -> None:
        """Test that invoking the graph produces expected output."""
        graph = create_graph()
        result = graph.invoke({"input": "Test message"})

        assert "output" in result
        assert result["output"] == "Processed: Test message"
