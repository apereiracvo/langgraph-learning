"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_input() -> dict[str, str]:
    """Provide sample input for graph tests."""
    return {"input": "Test message"}
