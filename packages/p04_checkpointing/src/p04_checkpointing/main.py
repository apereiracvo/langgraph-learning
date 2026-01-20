"""Entry point for the Checkpointing pattern.

This pattern demonstrates:
- Plug-n-play checkpointers (MemorySaver, AsyncSqliteSaver)
- Multi-turn conversations with state persistence
- Human-in-the-loop approval workflows with interrupt()
- Fault recovery from checkpoints

Run with: task run -- p04-checkpointing

Configuration:
    The pattern automatically detects and uses the first configured provider.
    Configure API keys via environment variables:
    - OPENAI_API_KEY for OpenAI
    - ANTHROPIC_API_KEY for Anthropic
    - GOOGLE_API_KEY for Google
"""

from __future__ import annotations

import asyncio
import sys
from enum import StrEnum
from typing import TYPE_CHECKING

from p04_checkpointing.demos.demo_approval import run_approval_demo
from p04_checkpointing.demos.demo_conversation import run_conversation_demo
from p04_checkpointing.demos.demo_recovery import run_fault_recovery_demo
from shared.llm import get_available_providers
from shared.logger import logger
from shared.settings import settings


if TYPE_CHECKING:
    from shared.enums import LLMProvider


class DemoType(StrEnum):
    """Available demo types.

    Attributes:
        CONVERSATION: Multi-turn conversation demo.
        APPROVAL: Human-in-the-loop approval demo.
        RECOVERY: Fault recovery demo.
        ALL: Run all demos.
    """

    CONVERSATION = "conversation"
    APPROVAL = "approval"
    RECOVERY = "recovery"
    ALL = "all"


def _print_header() -> None:
    """Print the pattern header."""
    print("\n" + "=" * 60)
    print("Pattern 04: LangGraph Checkpointing")
    print("=" * 60)


def _print_demo_header(demo_num: int, title: str) -> None:
    """Print a demo section header.

    Args:
        demo_num: The demo number.
        title: The demo title.
    """
    print("\n")
    print("#" * 60)
    print(f"# DEMO {demo_num}: {title}")
    print("#" * 60)


async def async_main(demo_type: DemoType = DemoType.ALL) -> None:
    """Run the checkpointing pattern demos.

    Args:
        demo_type: Which demo to run (default: all).
    """
    logger.info("Starting Pattern 04: Checkpointing")

    # Validate providers
    available: list[LLMProvider] = get_available_providers(settings)
    if not available:
        logger.error(
            "No LLM providers configured. Please set at least one API key "
            "(OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY)."
        )
        sys.exit(1)

    _print_header()
    print(f"Available providers: {[p.value for p in available]}")
    print("=" * 60)

    # Run selected demos
    if demo_type in (DemoType.ALL, DemoType.CONVERSATION):
        _print_demo_header(1, "Multi-Turn Conversation with Memory")
        await run_conversation_demo()

    if demo_type in (DemoType.ALL, DemoType.APPROVAL):
        _print_demo_header(2, "Human-in-the-Loop Approval")
        await run_approval_demo()

    if demo_type in (DemoType.ALL, DemoType.RECOVERY):
        _print_demo_header(3, "Fault Recovery")
        await run_fault_recovery_demo()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60 + "\n")


def main() -> None:
    """Entry point wrapper for async main.

    This function serves as the synchronous entry point required
    by the package script configuration. It wraps the async main
    function with `asyncio.run()`.
    """
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
