"""Entry point for the LangSmith Observability pattern.

This pattern demonstrates LangSmith integration for tracing, monitoring,
and debugging LangGraph agents. The demos cover:

1. Basic Automatic Tracing: Zero-code tracing with LangGraph
2. Custom Spans: @traceable decorator for fine-grained control
3. Metadata & Tags: Filtering and organizing traces
4. Error Tracing: Debugging workflows and error capture

Run with: task run -- p05-langsmith-observability

Configuration:
    Enable tracing by setting:
    - LANGSMITH__TRACING_ENABLED=true
    - LANGSMITH__API_KEY=your-key

    Get your API key from: https://smith.langchain.com/
"""

from __future__ import annotations

import asyncio
import sys

from p05_langsmith_observability.demos.demo_basic_tracing import run_basic_tracing_demo
from p05_langsmith_observability.demos.demo_custom_spans import run_custom_spans_demo
from p05_langsmith_observability.demos.demo_error_tracing import run_error_tracing_demo
from p05_langsmith_observability.demos.demo_metadata import run_metadata_demo
from shared.logger import logger
from shared.observability import aflush_traces, configure_tracing, is_tracing_enabled
from shared.settings import settings


# region Private Functions


def _print_header() -> None:
    """Print pattern header with tracing status."""
    print("\n" + "#" * 60)
    print("# Pattern 05: LangSmith Observability")
    print("#" * 60)

    tracing_status = "ENABLED" if is_tracing_enabled() else "DISABLED"
    print(f"# Tracing: {tracing_status}")
    print(f"# Project: {settings.langsmith.project}")

    if is_tracing_enabled():
        print("# View traces: https://smith.langchain.com/")
    else:
        print("# Enable with: LANGSMITH__TRACING_ENABLED=true")

    print("#" * 60)


def _print_demo_header(demo_num: int, title: str) -> None:
    """Print a demo section header.

    Args:
        demo_num: The demo number (1-4).
        title: Title of the demo.
    """
    print("\n" + "=" * 60)
    print(f"DEMO {demo_num}: {title}")
    print("=" * 60)


def _print_summary() -> None:
    """Print execution summary."""
    print("\n" + "#" * 60)
    print("# EXECUTION SUMMARY")
    print("#" * 60)
    print(f"# Tracing was: {'ENABLED' if is_tracing_enabled() else 'DISABLED'}")
    print(f"# Project: {settings.langsmith.project}")

    if is_tracing_enabled():
        print("#")
        print("# View your traces at: https://smith.langchain.com/")
        print(f"# Look for project: {settings.langsmith.project}")
        print("#")
        print("# Useful filters:")
        print("#   - By demo: run_name contains 'Demo'")
        print("#   - Errors only: is_error = true")
        print("#   - By tag: has(tags, 'env:development')")

    print("#" * 60 + "\n")


# endregion


# region Public Functions


async def async_main() -> None:
    """Run all LangSmith observability demos.

    This is the async entry point that:
    1. Configures LangSmith tracing from settings
    2. Displays tracing status
    3. Runs all four demo modules
    4. Flushes traces at the end
    """
    logger.info("Starting Pattern 05: LangSmith Observability")

    # Configure tracing from settings
    tracing_configured = configure_tracing()

    if not tracing_configured:
        logger.warning(
            "LangSmith tracing is not configured. Demos will run but traces "
            "will not be sent. Set LANGSMITH__TRACING_ENABLED=true and provide "
            "LANGSMITH__API_KEY to enable tracing."
        )

    # Print header
    _print_header()

    try:
        # Demo 1: Basic Automatic Tracing
        _print_demo_header(1, "Basic Automatic Tracing")
        await run_basic_tracing_demo()

        # Demo 2: Custom Spans
        _print_demo_header(2, "Custom Spans with @traceable")
        await run_custom_spans_demo()

        # Demo 3: Metadata & Tags
        _print_demo_header(3, "Metadata and Tags")
        await run_metadata_demo()

        # Demo 4: Error Tracing
        _print_demo_header(4, "Error Tracing")
        await run_error_tracing_demo()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in demos: {e}")
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
    finally:
        # Always flush traces before exit
        print("\n[Flushing traces...]")
        await aflush_traces()
        logger.info("All demos completed, traces flushed")

    # Print summary
    _print_summary()


def main() -> None:
    """Entry point wrapper for async main.

    This function serves as the synchronous entry point required
    by the package script configuration. It wraps the async main
    function with `asyncio.run()`.
    """
    asyncio.run(async_main())


# endregion


if __name__ == "__main__":
    main()
