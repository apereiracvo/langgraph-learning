"""Shared utilities for all patterns.

This module provides utility functions used across LangGraph patterns.
"""

from __future__ import annotations

from typing import Any


def format_response(response: dict[str, Any]) -> str:
    """Format a LangGraph response for display.

    Args:
        response: The response dictionary from a LangGraph invocation.

    Returns:
        A formatted string representation of the response.

    Example:
        >>> result = {"output": "Hello, world!"}
        >>> print(format_response(result))
        Result: {'output': 'Hello, world!'}
    """
    return f"Result: {response}"
