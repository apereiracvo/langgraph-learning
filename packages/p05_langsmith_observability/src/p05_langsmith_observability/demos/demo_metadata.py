"""Demo 3: Metadata and tags for filtering and organization.

This demo shows how to attach metadata and tags to traces for
filtering, grouping, and organizing traces in LangSmith.

Key concepts demonstrated:
- Adding user_id, session_id, request_id
- Tagging by environment and version
- Dynamic metadata addition
- RunnableConfig metadata passing
- Filtering traces by metadata in LangSmith UI
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from p05_langsmith_observability.agent import create_traced_agent
from p05_langsmith_observability.prompts import load_system_prompt
from p05_langsmith_observability.state import AgentState, create_initial_state
from shared.exceptions import LLMConfigurationError, PromptLoadError
from shared.llm import get_available_providers
from shared.logger import logger
from shared.observability import is_tracing_enabled
from shared.settings import settings


if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

    from shared.enums import LLMProvider


# region Types


class RequestMetadata:
    """Metadata schema for request tracing.

    This class provides a structured way to define request metadata
    that will be attached to traces for filtering.

    Attributes:
        user_id: Unique identifier for the user.
        session_id: Session identifier for grouping conversations.
        request_id: Unique identifier for this specific request.
        environment: Deployment environment (development, staging, production).
        version: Application version string.
        feature_flags: Dictionary of enabled feature flags.
    """

    def __init__(
        self,
        *,
        user_id: str,
        session_id: str | None = None,
        request_id: str | None = None,
        environment: str = "development",
        version: str = "1.0.0",
        feature_flags: dict[str, bool] | None = None,
    ) -> None:
        """Initialize request metadata.

        Args:
            user_id: Unique identifier for the user.
            session_id: Session ID (auto-generated if not provided).
            request_id: Request ID (auto-generated if not provided).
            environment: Deployment environment.
            version: Application version.
            feature_flags: Dictionary of enabled feature flags.
        """
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())
        self.request_id = request_id or str(uuid.uuid4())
        self.environment = environment
        self.version = version
        self.feature_flags = feature_flags or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for tracing.

        Returns:
            Dictionary representation of metadata.
        """
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "environment": self.environment,
            "version": self.version,
            "feature_flags": self.feature_flags,
        }

    def get_tags(self) -> list[str]:
        """Get tags for filtering.

        Returns:
            List of tags derived from metadata.
        """
        tags = [
            f"env:{self.environment}",
            f"version:{self.version}",
            f"user:{self.user_id}",
        ]

        # Add feature flag tags
        for flag, enabled in self.feature_flags.items():
            if enabled:
                tags.append(f"feature:{flag}")

        return tags


# endregion


# region Private Functions


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
                print("[ASSISTANT]: (calling tools...)")
                for tc in msg.tool_calls:
                    print(f"  -> Tool: {tc['name']}")
            elif msg.content:
                print(f"[ASSISTANT]: {msg.content}")
        elif isinstance(msg, ToolMessage):
            print(f"[TOOL {msg.name}]: {msg.content}")


async def _run_traced_query(
    agent: CompiledStateGraph,  # type: ignore[type-arg]
    query: str,
    provider: LLMProvider,
    system_prompt: str,
    metadata: RequestMetadata,
) -> dict[str, Any] | None:
    """Run a query with metadata attached.

    Args:
        agent: Compiled agent graph.
        query: User query.
        provider: LLM provider.
        system_prompt: System prompt content.
        metadata: Request metadata for tracing.

    Returns:
        Agent result or None if error.
    """
    # Create initial state with metadata
    initial_state: AgentState = create_initial_state(
        user_input=query,
        provider=provider,
        system_prompt=system_prompt,
        metadata=metadata.to_dict(),
    )

    # Create RunnableConfig with metadata and tags
    # This is how you pass metadata through the LangChain/LangGraph runtime
    config: RunnableConfig = {
        "metadata": metadata.to_dict(),
        "tags": metadata.get_tags(),
        "run_name": f"Query: {query[:30]}...",
    }

    try:
        return await agent.ainvoke(initial_state, config=config)
    except LLMConfigurationError as e:
        logger.error(f"LLM configuration error: {e}")
        return None
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        return None


async def _run_query_with_metadata(
    agent: CompiledStateGraph,  # type: ignore[type-arg]
    provider: LLMProvider,
    system_prompt: str,
    query: str,
    metadata: RequestMetadata,
    title: str,
) -> None:
    """Run a single query with metadata and display results.

    Args:
        agent: Compiled agent graph.
        provider: LLM provider.
        system_prompt: System prompt content.
        query: User query.
        metadata: Request metadata for tracing.
        title: Title for this query section.
    """
    print(f"\n--- {title} ---")
    print(f"Metadata: {metadata.to_dict()}")
    print(f"Tags: {metadata.get_tags()}")
    print(f"\nQuery: {query}")

    result = await _run_traced_query(agent, query, provider, system_prompt, metadata)

    if result:
        print("\nResponse:")
        _display_messages(result.get("messages", []))


def _print_filtering_instructions(session_id: str) -> None:
    """Print LangSmith filtering instructions.

    Args:
        session_id: The session ID used in the demo.
    """
    print("\n--- Filtering Traces in LangSmith ---")
    print("\nCommon filter queries you can use:")
    print("  1. By user: metadata.user_id = 'user_alice_123'")
    print("  2. By environment: has(tags, 'env:production')")
    print("  3. By session: metadata.session_id = '<session-id>'")
    print("  4. By feature flag: has(tags, 'feature:beta_tools')")
    print("  5. By version: has(tags, 'version:2.0.0')")

    if is_tracing_enabled():
        print(f"\nSession ID for both queries: {session_id}")
        print("Use this to find related traces in the same session")
        print("\nView traces at: https://smith.langchain.com/")


# endregion


# region Public Functions


async def run_metadata_demo() -> None:
    """Run the metadata and tags demo.

    Demonstrates:
    - Adding user_id, session_id, request_id
    - Tagging by environment and version
    - Multiple queries with different metadata
    - How to filter traces in LangSmith UI
    """
    print(f"\nTracing enabled: {is_tracing_enabled()}")

    if is_tracing_enabled():
        print("Metadata will be attached to traces")
    else:
        print("Tracing disabled - metadata demo will run but no traces sent")

    # Check for available providers
    providers = get_available_providers(settings)
    if not providers:
        logger.error("No LLM providers configured")
        print("\nERROR: No LLM provider configured. Please set an API key.")
        return

    provider: LLMProvider = providers[0]
    print(f"\nUsing provider: {provider.value}")

    # Load system prompt
    try:
        system_prompt: str = load_system_prompt()
    except PromptLoadError as e:
        logger.error(f"Failed to load system prompt: {e}")
        print(f"\nERROR: {e}")
        return

    # Create agent
    try:
        agent: CompiledStateGraph = create_traced_agent(settings)  # type: ignore[type-arg]
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        print(f"\nERROR: Failed to create agent: {e}")
        return

    # Query 1: User A in production
    metadata_1 = RequestMetadata(
        user_id="user_alice_123",
        environment="production",
        version="2.0.0",
        feature_flags={"new_ui": True, "beta_tools": False},
    )
    await _run_query_with_metadata(
        agent,
        provider,
        system_prompt,
        "What is 100 / 4?",
        metadata_1,
        "Query 1: User A (Production)",
    )

    # Query 2: User B in development with same session
    metadata_2 = RequestMetadata(
        user_id="user_bob_456",
        session_id=metadata_1.session_id,  # Same session for grouping demo
        environment="development",
        version="2.1.0-beta",
        feature_flags={"new_ui": True, "beta_tools": True},
    )
    await _run_query_with_metadata(
        agent,
        provider,
        system_prompt,
        "What's the weather in Paris?",
        metadata_2,
        "Query 2: User B (Development)",
    )

    # Print filtering instructions
    _print_filtering_instructions(metadata_1.session_id)


# endregion
