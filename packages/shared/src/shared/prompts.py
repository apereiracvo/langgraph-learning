"""Prompt loading and management utilities.

This module handles loading system prompts from external markdown files,
with caching and template variable substitution support.

Each pattern should provide its own prompts directory when calling
load_system_prompt(), making this module reusable across all patterns.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from shared.exceptions import PromptLoadError


if TYPE_CHECKING:
    from pathlib import Path




@lru_cache(maxsize=32)
def load_system_prompt(
    filename: str,
    prompts_dir: Path,
) -> str:
    """Load a system prompt from a markdown file.

    Loads the prompt content from the specified file with caching
    to avoid repeated disk reads.

    Args:
        filename: Name of the prompt file (e.g., "system.md").
        prompts_dir: Path to the prompts directory containing the file.

    Returns:
        The prompt content as a string.

    Raises:
        PromptLoadError: If the file cannot be found or read.

    Example:
        >>> from pathlib import Path
        >>> prompts_dir = Path("/path/to/package/prompts")
        >>> prompt = load_system_prompt("system.md", prompts_dir)
        >>> print(prompt[:50])
        '# Assistant System Prompt...'

        >>> # Custom prompt file
        >>> prompt = load_system_prompt("custom.md", prompts_dir)
    """
    prompt_path: Path = prompts_dir / filename

    if not prompt_path.exists():
        raise PromptLoadError(
            prompt_path,
            f"Prompt file not found. Expected at: {prompt_path}",
        )

    if not prompt_path.is_file():
        raise PromptLoadError(
            prompt_path,
            "Path exists but is not a file.",
        )

    try:
        content: str = prompt_path.read_text(encoding="utf-8")
    except OSError as e:
        raise PromptLoadError(prompt_path, f"Failed to read file: {e}") from e

    return content.strip()


def render_prompt(
    template: str,
    variables: dict[str, Any],
    *,
    strict: bool = False,
) -> str:
    """Render a prompt template with variable substitution.

    Performs f-string style variable substitution on the template.
    Variables are substituted using {variable_name} syntax.

    Args:
        template: The prompt template string with {placeholders}.
        variables: Dictionary of variable names to values.
        strict: If True, raises KeyError for missing variables.
            If False (default), leaves missing placeholders unchanged.

    Returns:
        The rendered prompt with variables substituted.

    Raises:
        KeyError: If strict=True and a required variable is missing.

    Example:
        >>> template = "Hello, {name}! You are a {role}."
        >>> rendered = render_prompt(template, {"name": "Claude", "role": "assistant"})
        >>> print(rendered)
        'Hello, Claude! You are a assistant.'
    """
    if strict:
        return template.format(**variables)

    # Non-strict mode: substitute what we can, leave rest unchanged
    result: str = template
    for key, value in variables.items():
        placeholder: str = "{" + key + "}"
        result = result.replace(placeholder, str(value))

    return result


def clear_prompt_cache() -> None:
    """Clear the prompt loading cache.

    Useful for testing or when prompts may have changed on disk.

    Example:
        >>> clear_prompt_cache()
        >>> prompt = load_system_prompt("system.md", prompts_dir)  # Reloads from disk
    """
    load_system_prompt.cache_clear()
