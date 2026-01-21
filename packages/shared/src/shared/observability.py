"""Centralized observability utilities for LangSmith tracing.

This module provides a unified interface for tracing configuration,
context management, and graceful degradation when LangSmith is unavailable.

Key Features:
- Configure tracing from application settings
- Context manager for creating traced spans
- Helper functions for adding metadata and tags
- Graceful degradation when LangSmith is not available

Example:
    >>> from shared.observability import configure_tracing, trace_context
    >>> configure_tracing()
    >>> with trace_context("my_operation", metadata={"key": "value"}) as ctx:
    ...     # Your code here
    ...     ctx.metadata["additional"] = "data"
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from shared.logger import logger


if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Literal

    from langsmith import Client

    RunType = Literal[
        "tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"
    ]


# region Constants


# Try to import LangSmith; gracefully handle if not installed
try:
    import langsmith as ls
    from langsmith import Client
    from langsmith.run_helpers import get_current_run_tree

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    ls = None  # type: ignore[assignment]
    Client = None  # type: ignore[assignment, misc]
    get_current_run_tree = None  # type: ignore[assignment]


# Lazy-loaded client instance
_client: Client | None = None


# endregion


# region Types


@dataclass
class RunContext:
    """Context for a traced run.

    Provides a mutable context object that can be used within a trace_context
    block to add metadata and tags dynamically.

    Attributes:
        run_id: Unique identifier for the run.
        name: Name of the run.
        run_type: Type of run (chain, tool, llm, etc.).
        metadata: Mutable metadata dict that can be modified during execution.
        tags: Mutable tags list that can be modified during execution.
        active: Whether tracing is active for this context.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    run_type: str = "chain"
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    active: bool = False


# endregion


# region Public Functions


def configure_tracing() -> bool:
    """Configure LangSmith tracing from application settings.

    Reads the LangSmith settings and sets the appropriate environment
    variables for the LangSmith SDK to pick up. This function should be
    called at application startup.

    Returns:
        True if tracing is configured and enabled, False otherwise.

    Example:
        >>> from shared.observability import configure_tracing
        >>> if configure_tracing():
        ...     print("Tracing enabled!")
        ... else:
        ...     print("Tracing disabled or not configured")
    """
    from shared.settings import settings  # noqa: PLC0415

    if not LANGSMITH_AVAILABLE:
        logger.warning("LangSmith SDK not installed. Tracing disabled.")
        return False

    langsmith_settings = settings.langsmith

    if not langsmith_settings.tracing_enabled:
        logger.debug("LangSmith tracing disabled in settings.")
        return False

    if not langsmith_settings.api_key:
        logger.warning(
            "LangSmith tracing enabled but no API key configured. "
            "Set LANGSMITH__API_KEY."
        )
        return False

    # Set environment variables for LangSmith SDK
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_settings.api_key.get_secret_value()
    os.environ["LANGCHAIN_PROJECT"] = langsmith_settings.project

    # Set custom endpoint if provided (for self-hosted)
    if langsmith_settings.endpoint != "https://api.smith.langchain.com":
        os.environ["LANGCHAIN_ENDPOINT"] = langsmith_settings.endpoint

    # Set workspace ID if provided
    if langsmith_settings.workspace_id:
        os.environ["LANGSMITH_WORKSPACE_ID"] = langsmith_settings.workspace_id

    # Configure background tracing
    background = "true" if langsmith_settings.tracing_background else "false"
    os.environ["LANGSMITH_TRACING_BACKGROUND"] = background

    logger.info(
        "LangSmith tracing configured",
        extra={
            "project": langsmith_settings.project,
            "background": langsmith_settings.tracing_background,
        },
    )

    return True


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is currently enabled.

    Checks both that the LangSmith SDK is available and that tracing
    has been configured via environment variables.

    Returns:
        True if tracing is enabled and configured, False otherwise.

    Example:
        >>> from shared.observability import is_tracing_enabled
        >>> if is_tracing_enabled():
        ...     print("Traces will be sent to LangSmith")
    """
    return (
        LANGSMITH_AVAILABLE
        and os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    )


def get_langsmith_client() -> Client | None:
    """Get a LangSmith client instance (lazy initialization).

    Returns a cached client instance if available, or creates a new one.
    Returns None if LangSmith is not available or not configured.

    Returns:
        LangSmith Client if available and configured, None otherwise.

    Example:
        >>> client = get_langsmith_client()
        >>> if client:
        ...     runs = client.list_runs(project_name="my-project")
    """
    global _client  # noqa: PLW0603

    if not LANGSMITH_AVAILABLE or not is_tracing_enabled():
        return None

    if _client is None:
        try:
            _client = Client()
        except Exception as e:
            logger.warning(f"Failed to create LangSmith client: {e}")
            return None

    return _client


@contextmanager
def trace_context(
    name: str,
    *,
    run_type: RunType = "chain",
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Generator[RunContext, None, None]:
    """Context manager for creating traced spans.

    Creates a traced span that wraps the code block. Provides graceful
    degradation when LangSmith is unavailable - the code will still execute
    but no traces will be sent.

    Args:
        name: Name for the trace span.
        run_type: Type of run (chain, tool, llm, prompt, retriever, embedding).
        metadata: Optional metadata dict to attach to the span.
        tags: Optional tags list to attach to the span.

    Yields:
        RunContext with mutable metadata and tags that can be modified
        during execution.

    Example:
        >>> with trace_context("my_operation", metadata={"user_id": "123"}) as ctx:
        ...     ctx.metadata["additional"] = "data"
        ...     ctx.tags.append("processed")
        ...     # Your code here
    """
    ctx = RunContext(
        name=name,
        run_type=run_type,
        metadata=metadata.copy() if metadata else {},
        tags=list(tags) if tags else [],
        active=is_tracing_enabled(),
    )

    if not ctx.active or not LANGSMITH_AVAILABLE or ls is None:
        # Graceful degradation - just yield context without tracing
        yield ctx
        return

    # Use LangSmith trace context
    with ls.trace(
        name=name,
        run_type=run_type,
        metadata=ctx.metadata,
        tags=ctx.tags,
    ) as run_tree:
        ctx.run_id = str(run_tree.id)
        yield ctx


def add_run_metadata(metadata: dict[str, Any]) -> None:
    """Add metadata to the current run if tracing is active.

    Dynamically adds metadata to the currently active run tree.
    Does nothing if tracing is not enabled.

    Args:
        metadata: Key-value pairs to add to current run.

    Example:
        >>> add_run_metadata({"user_id": "123", "request_id": "abc"})
    """
    if (
        not is_tracing_enabled()
        or not LANGSMITH_AVAILABLE
        or get_current_run_tree is None
    ):
        return

    try:
        run_tree = get_current_run_tree()
        if run_tree and hasattr(run_tree, "metadata"):
            run_tree.metadata.update(metadata)
    except Exception as e:
        logger.debug(f"Failed to add run metadata: {e}")


def add_run_tags(tags: list[str]) -> None:
    """Add tags to the current run if tracing is active.

    Dynamically adds tags to the currently active run tree.
    Does nothing if tracing is not enabled.

    Args:
        tags: Tags to add to current run.

    Example:
        >>> add_run_tags(["production", "high-priority"])
    """
    if (
        not is_tracing_enabled()
        or not LANGSMITH_AVAILABLE
        or get_current_run_tree is None
    ):
        return

    try:
        run_tree = get_current_run_tree()
        if run_tree and hasattr(run_tree, "tags") and run_tree.tags is not None:
            run_tree.tags.extend(tags)
    except Exception as e:
        logger.debug(f"Failed to add run tags: {e}")


def flush_traces() -> None:
    """Ensure all pending traces are submitted (synchronous version).

    Call this before application exit to ensure all traces are sent.
    This is the synchronous version - use in sync contexts.
    """
    if not LANGSMITH_AVAILABLE:
        return

    try:
        from langchain_core.tracers.langchain import (  # noqa: PLC0415
            wait_for_all_tracers,
        )

        wait_for_all_tracers()
    except ImportError:
        # Older LangChain version or not installed
        logger.debug("wait_for_all_tracers not available")
    except Exception as e:
        logger.warning(f"Failed to flush traces: {e}")


async def aflush_traces() -> None:
    """Ensure all pending traces are submitted (async version).

    Call this before application exit to ensure all traces are sent.
    This is the async version - use in async contexts.

    Note: Currently uses the synchronous flush under the hood as
    LangSmith doesn't have a native async flush. This may block
    briefly but ensures traces are submitted.
    """
    # Currently just call the sync version
    # LangSmith tracing is background by default so this should be quick
    flush_traces()


# endregion
