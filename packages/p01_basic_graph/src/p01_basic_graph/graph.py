"""LLM-powered graph definition.

This module defines the main LangGraph StateGraph with LLM integration.
The graph processes user input through an LLM node using the configured
provider and system prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from p01_basic_graph.state import GraphState
from shared.llm import create_llm


if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

    from shared.enums import LLMProvider
    from shared.settings import Settings


def create_llm_node(
    settings: Settings,
) -> Callable[[GraphState], dict[str, Any]]:
    """Create an LLM node function with the given settings.

    This factory function creates a node that:
    1. Retrieves the LLM based on the provider in state
    2. Builds messages from system prompt and user input
    3. Invokes the LLM and returns the response

    Args:
        settings: Application settings for LLM configuration.

    Returns:
        A node function that processes state through an LLM.

    Example:
        >>> from shared.settings import settings
        >>> node_fn = create_llm_node(settings)
        >>> # node_fn is now ready to be added to a StateGraph
    """

    def llm_node(state: GraphState) -> dict[str, Any]:
        """Process user input through the LLM.

        This node function:
        1. Creates an LLM instance for the configured provider
        2. Builds the message list with system prompt and user input
        3. Invokes the LLM and captures the response
        4. Returns state updates with the response

        Args:
            state: Current graph state containing user input and config.

        Returns:
            Dictionary with state updates including llm_response and messages.
        """
        provider: LLMProvider = state["provider"]
        system_prompt: str = state["system_prompt"]
        user_input: str = state["user_input"]

        # Create LLM for the specified provider
        llm: BaseChatModel = create_llm(provider, settings)

        # Build message list
        system_message: SystemMessage = SystemMessage(content=system_prompt)
        human_message: HumanMessage = HumanMessage(content=user_input)

        messages: list[SystemMessage | HumanMessage] = [system_message, human_message]

        # Invoke LLM
        response: AIMessage = llm.invoke(messages)

        # Extract response content
        response_content: str = str(response.content)

        # Return state updates
        state_updates: dict[str, Any] = {
            "llm_response": response_content,
            "messages": [human_message, response],
        }

        return state_updates

    return llm_node


def create_graph(settings: Settings) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Create and compile the LLM-powered graph.

    Builds a StateGraph with a single LLM node that processes
    user input and returns an AI response.

    Args:
        settings: Application settings for LLM configuration.

    Returns:
        A compiled StateGraph ready for invocation.

    Example:
        >>> from shared.settings import settings
        >>> graph = create_graph(settings)
        >>> result = graph.invoke(
        ...     {
        ...         "user_input": "Hello!",
        ...         "provider": LLMProvider.OPENAI,
        ...         "system_prompt": "You are helpful.",
        ...         "messages": [],
        ...         "llm_response": "",
        ...         "metadata": {},
        ...     }
        ... )
    """
    # Create the graph builder
    builder: StateGraph[GraphState] = StateGraph(GraphState)

    # Create the LLM node
    llm_node_fn: Callable[[GraphState], dict[str, Any]] = create_llm_node(settings)

    # Add nodes
    builder.add_node("llm", llm_node_fn)  # type: ignore[arg-type]

    # Define edges
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)

    # Compile and return
    compiled_graph: CompiledStateGraph = builder.compile()  # type: ignore[type-arg]

    return compiled_graph


