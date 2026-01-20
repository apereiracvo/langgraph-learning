"""Prompt loading and management utilities for p03_tool_agent_async.

This module provides pattern-specific wrappers around the shared prompt
utilities. The main value is providing the prompts directory automatically
so pattern code doesn't need to construct paths.
"""

from __future__ import annotations

from pathlib import Path

from shared.prompts import (
    clear_prompt_cache as _clear_prompt_cache,
    load_system_prompt as _load_system_prompt,
    render_prompt,
)


__all__: list[str] = [
    "clear_prompt_cache",
    "get_prompts_directory",
    "load_system_prompt",
    "render_prompt",
]


def get_prompts_directory() -> Path:
    """Get the prompts directory path for this pattern.

    Returns the path to the prompts directory relative to this package.

    Returns:
        Path to the prompts directory.
    """
    package_dir: Path = Path(__file__).parent.parent.parent
    prompts_dir: Path = package_dir / "prompts"
    return prompts_dir


def load_system_prompt(
    filename: str = "system.md",
    *,
    prompts_dir: Path | None = None,
) -> str:
    """Load a system prompt from a markdown file.

    Wrapper around shared.prompts.load_system_prompt that provides
    the default prompts directory for this pattern.

    Args:
        filename: Name of the prompt file (default: "system.md").
        prompts_dir: Optional custom prompts directory. If not provided,
            uses the default prompts directory for this pattern.

    Returns:
        The prompt content as a string.

    Raises:
        PromptLoadError: If the file cannot be found or read.

    Example:
        >>> prompt = load_system_prompt()
        >>> print(prompt[:50])
        '# Tool-Calling Assistant...'

        >>> # Custom prompt file
        >>> prompt = load_system_prompt("custom.md")
    """
    directory: Path = prompts_dir or get_prompts_directory()
    return _load_system_prompt(filename, directory)


def clear_prompt_cache() -> None:
    """Clear the prompt loading cache.

    Useful for testing or when prompts may have changed on disk.

    Example:
        >>> clear_prompt_cache()
        >>> prompt = load_system_prompt()  # Reloads from disk
    """
    _clear_prompt_cache()
