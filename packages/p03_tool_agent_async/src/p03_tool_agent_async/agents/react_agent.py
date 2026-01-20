"""ReAct agent implementation using LangChain's create_agent.

This module implements a ReAct (Reasoning + Acting) agent using LangChain's
`create_agent` function. The ReAct pattern alternates between:
1. Reasoning: The LLM analyzes the current state and decides what to do
2. Acting: The agent executes a tool based on the reasoning
3. Observation: The tool result is fed back to the LLM

LangChain provides `create_agent` as a prebuilt solution that handles
the ReAct loop automatically. This module wraps it to integrate with our
state management and multi-provider support.

Key patterns demonstrated:
- Using LangChain's `create_agent` for tool-calling loops
- Custom state adapter for our AgentState schema
- Async invocation with state translation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent as lc_create_agent
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from p03_tool_agent_async.tools import get_all_tools
from shared.llm import create_llm


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.graph.state import CompiledStateGraph

    from p03_tool_agent_async.state import AgentState
    from shared.enums import LLMProvider
    from shared.settings import Settings


class ReactAgentWrapper:
    """Wrapper around LangChain's create_agent.

    This wrapper adapts the prebuilt agent to work with our custom
    AgentState schema and provides async invocation with state translation.

    Attributes:
        settings: Application settings for LLM configuration.
        tools: List of tools available to the agent.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the ReAct agent wrapper.

        Args:
            settings: Application settings for LLM configuration.
        """
        self.settings: Settings = settings
        self.tools: list[BaseTool] = get_all_tools()

    def _create_agent_for_provider(
        self,
        provider: LLMProvider,
        system_prompt: str,
    ) -> CompiledStateGraph:  # type: ignore[type-arg]
        """Create a ReAct agent for a specific provider.

        Args:
            provider: The LLM provider to use.
            system_prompt: The system prompt for the agent.

        Returns:
            A compiled agent graph.
        """
        llm: BaseChatModel = create_llm(
            provider,
            self.settings,
            temperature=0.0,
        )

        # Create the agent with system prompt
        agent: CompiledStateGraph = lc_create_agent(  # type: ignore[type-arg]
            model=llm,
            tools=self.tools,
            system_prompt=system_prompt,
        )

        return agent

    async def ainvoke(
        self,
        state: AgentState,
        config: RunnableConfig | None = None,
    ) -> AgentState:
        """Invoke the ReAct agent asynchronously.

        Translates our AgentState to the prebuilt agent's expected format,
        invokes the agent, and translates the result back.

        Args:
            state: The initial agent state.
            config: Optional runnable config.

        Returns:
            Updated AgentState with the agent's response.
        """
        provider: LLMProvider = state["provider"]
        system_prompt: str = state["system_prompt"]
        messages: list[BaseMessage] = list(state["messages"])

        # Create agent for this provider with system prompt
        agent: CompiledStateGraph = self._create_agent_for_provider(  # type: ignore[type-arg]
            provider,
            system_prompt,
        )

        # Invoke the agent (system prompt is already configured)
        result: dict[str, Any] = await agent.ainvoke(
            {"messages": messages},
            config=config,
        )

        # Extract results
        result_messages: list[BaseMessage] = result.get("messages", [])

        # Track tool calls made
        tool_calls_made: list[str] = []
        for msg in result_messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_calls_made.extend(tc["name"] for tc in msg.tool_calls)

        # Get final response (last AI message content)
        final_response: str = ""
        for msg in reversed(result_messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_response = str(msg.content)
                break

        # Filter out system message from result for cleaner state
        filtered_messages: list[BaseMessage] = [
            m for m in result_messages if not isinstance(m, SystemMessage)
        ]

        # Build updated state
        updated_state: AgentState = {
            "messages": filtered_messages,
            "provider": provider,
            "system_prompt": system_prompt,
            "tool_calls_made": tool_calls_made,
            "final_response": final_response,
            "metadata": state.get("metadata", {}),
        }

        return updated_state


def create_react_agent(settings: Settings) -> ReactAgentWrapper:
    """Create a ReAct agent wrapper.

    This function creates a wrapper around LangChain's create_agent
    that integrates with our state management and multi-provider support.

    Args:
        settings: Application settings for LLM configuration.

    Returns:
        A ReactAgentWrapper ready for async invocation.

    Example:
        >>> from shared.settings import settings
        >>> agent = create_react_agent(settings)
        >>> result = await agent.ainvoke(initial_state)
    """
    return ReactAgentWrapper(settings)


