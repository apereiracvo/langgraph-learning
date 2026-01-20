"""Checkpointer factory for plug-n-play persistence backends.

Provides a factory pattern for creating checkpointers based on configuration.
Supports InMemorySaver for testing and AsyncSqliteSaver for persistence.

The key design goal is to make checkpointers transparent to graph logic:
the same graph code works with any checkpointer backend.

Example:
    >>> async with get_checkpointer(CheckpointerType.MEMORY) as cp:
    ...     graph = builder.compile(checkpointer=cp)
    ...     await graph.ainvoke(state, config)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from enum import StrEnum
from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from langgraph.checkpoint.base import BaseCheckpointSaver


class CheckpointerType(StrEnum):
    """Supported checkpointer backend types.

    Attributes:
        MEMORY: In-memory checkpointer for testing (data lost on exit).
        SQLITE: SQLite-based async checkpointer for persistent storage.
    """

    MEMORY = "memory"
    SQLITE = "sqlite"


async def create_checkpointer(
    checkpointer_type: CheckpointerType,
    *,
    db_path: Path | str | None = None,
) -> BaseCheckpointSaver:  # type: ignore[type-arg]
    """Create a checkpointer instance based on the specified type.

    Factory function that creates the appropriate checkpointer backend.
    For SQLITE type, returns an AsyncSqliteSaver that requires context management.

    Note: For SQLITE checkpointers, prefer using get_checkpointer() context manager
    to ensure proper lifecycle management.

    Args:
        checkpointer_type: The type of checkpointer to create.
        db_path: Path for SQLite database file. Required for SQLITE type.
            Ignored for MEMORY type.

    Returns:
        A configured checkpointer instance ready for use.

    Raises:
        ValueError: If SQLITE type is requested without db_path.

    Example:
        >>> # In-memory for testing
        >>> checkpointer = await create_checkpointer(CheckpointerType.MEMORY)
        >>>
        >>> # SQLite for persistence (prefer get_checkpointer context manager)
        >>> checkpointer = await create_checkpointer(
        ...     CheckpointerType.SQLITE,
        ...     db_path="checkpoints.db",
        ... )
    """
    if checkpointer_type == CheckpointerType.MEMORY:
        return MemorySaver()

    if checkpointer_type == CheckpointerType.SQLITE:
        if db_path is None:
            msg = "db_path is required for SQLITE checkpointer"
            raise ValueError(msg)

        # Lazy import for optional SQLite support
        from langgraph.checkpoint.sqlite.aio import (  # noqa: PLC0415
            AsyncSqliteSaver,
        )

        # AsyncSqliteSaver requires context manager for proper lifecycle
        # We return it directly; caller must use async context manager
        return AsyncSqliteSaver.from_conn_string(str(db_path))  # type: ignore[return-value]

    # All CheckpointerType values are handled above, this is defensive
    msg = f"Unsupported checkpointer type: {checkpointer_type}"  # type: ignore[unreachable]
    raise ValueError(msg)


@asynccontextmanager
async def get_checkpointer(
    checkpointer_type: CheckpointerType,
    *,
    db_path: Path | str | None = None,
) -> AsyncIterator[BaseCheckpointSaver]:  # type: ignore[type-arg]
    """Context manager for checkpointer lifecycle management.

    Handles proper setup and cleanup for checkpointers that require it.
    MemorySaver is returned directly; AsyncSqliteSaver is properly
    initialized and cleaned up.

    This is the recommended way to create checkpointers as it ensures
    proper resource cleanup, especially for database-backed checkpointers.

    Args:
        checkpointer_type: The type of checkpointer to create.
        db_path: Path for SQLite database file (required for SQLITE).

    Yields:
        A configured checkpointer instance.

    Raises:
        ValueError: If SQLITE type is requested without db_path.

    Example:
        >>> async with get_checkpointer(CheckpointerType.MEMORY) as cp:
        ...     graph = builder.compile(checkpointer=cp)
        ...     result = await graph.ainvoke(state, config)

        >>> async with get_checkpointer(
        ...     CheckpointerType.SQLITE, db_path="checkpoints.db"
        ... ) as cp:
        ...     graph = builder.compile(checkpointer=cp)
        ...     result = await graph.ainvoke(state, config)
    """
    if checkpointer_type == CheckpointerType.MEMORY:
        yield MemorySaver()
        return

    if checkpointer_type == CheckpointerType.SQLITE:
        if db_path is None:
            msg = "db_path is required for SQLITE checkpointer"
            raise ValueError(msg)

        # Lazy import for optional SQLite support
        from langgraph.checkpoint.sqlite.aio import (  # noqa: PLC0415
            AsyncSqliteSaver,
        )

        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            yield checkpointer
            return

    # All CheckpointerType values are handled above, this is defensive
    msg = f"Unsupported checkpointer type: {checkpointer_type}"  # type: ignore[unreachable]
    raise ValueError(msg)
