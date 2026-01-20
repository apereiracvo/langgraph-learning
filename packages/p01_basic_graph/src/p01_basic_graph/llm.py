"""LLM utilities re-exported from shared package.

This module provides backward-compatible re-exports of the LLM factory
functions from the shared package. New code should import directly from
shared.llm and shared.enums for consistency.
"""

from shared.enums import LLMProvider
from shared.llm import DEFAULT_MODELS, create_llm, get_available_providers


__all__: list[str] = [
    "DEFAULT_MODELS",
    "LLMProvider",
    "create_llm",
    "get_available_providers",
]
