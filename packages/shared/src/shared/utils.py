"""Shared utilities for all patterns."""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv


def setup_environment() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


def format_response(response: dict[str, Any]) -> str:
    """Format a LangGraph response for display.

    Args:
        response: The response dictionary from a LangGraph invocation.

    Returns:
        A formatted string representation of the response.
    """
    return f"Result: {response}"
