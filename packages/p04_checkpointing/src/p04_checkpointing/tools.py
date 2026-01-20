"""Tool definitions for the Checkpointing pattern.

Includes both safe tools (calculator) and sensitive tools (email sender,
file writer) that require human approval in the approval workflow.

Tool Categories:
- Safe tools: Can execute without approval (calculator)
- Sensitive tools: Require human approval (send_email, write_file)

The distinction between safe and sensitive tools is used by the approval
agent to determine when to pause for human input.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class EmailInput(BaseModel):
    """Input schema for email sender tool.

    Attributes:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body content.
    """

    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")


class FileWriteInput(BaseModel):
    """Input schema for file writer tool.

    Attributes:
        path: File path to write to.
        content: Content to write.
    """

    path: str = Field(description="File path to write to")
    content: str = Field(description="Content to write")


class CalculatorInput(BaseModel):
    """Input schema for calculator tool.

    Attributes:
        expression: Math expression to evaluate.
    """

    expression: str = Field(description="Math expression to evaluate (e.g., '2 + 2')")


@tool(args_schema=EmailInput)
async def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient.

    SENSITIVE: This action requires human approval before execution.

    Use this tool when the user asks to send an email. The tool will
    simulate sending the email and return a confirmation.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body content.

    Returns:
        Confirmation message with email details.

    Example:
        >>> result = await send_email(
        ...     to="bob@example.com", subject="Hello", body="How are you?"
        ... )
        >>> print(result)
        'Email sent to bob@example.com with subject: Hello'
    """
    # Simulated email sending (body included in simulation)
    _ = body  # Used in real implementation
    return f"Email sent to {to} with subject: {subject}"


@tool(args_schema=FileWriteInput)
async def write_file(path: str, content: str) -> str:
    """Write content to a file.

    SENSITIVE: This action requires human approval before execution.

    Use this tool when the user asks to write or save content to a file.
    The tool will simulate writing the file and return a confirmation.

    Args:
        path: File path to write to.
        content: Content to write.

    Returns:
        Confirmation message with file details.

    Example:
        >>> result = await write_file(path="/tmp/notes.txt", content="Important notes here")
        >>> print(result)
        'File written to /tmp/notes.txt (20 chars)'
    """
    # Simulated file write
    return f"File written to {path} ({len(content)} chars)"


@tool(args_schema=CalculatorInput)
async def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Safe operation - does not require approval.

    Use this tool when the user asks to calculate or evaluate a math
    expression. Supports basic arithmetic operations.

    Args:
        expression: Math expression to evaluate. Supports +, -, *, /, (, ), and numbers.

    Returns:
        Result of the calculation or an error message.

    Example:
        >>> result = await calculator("2 + 2 * 3")
        >>> print(result)
        '2 + 2 * 3 = 8'
    """
    # Safe eval for basic math - restrict to safe characters
    allowed_chars = set("0123456789+-*/(). ")
    if not all(c in allowed_chars for c in expression):
        return (
            "Error: Invalid characters in expression. Only numbers and +-*/() allowed."
        )

    try:
        result = eval(expression)  # noqa: S307
    except SyntaxError:
        return f"Error: Invalid expression syntax: {expression}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error evaluating expression: {e}"
    else:
        return f"{expression} = {result}"


def get_all_tools() -> list[Any]:
    """Get all available tools.

    Returns:
        List of all tool functions that can be bound to an LLM.

    Example:
        >>> tools = get_all_tools()
        >>> llm_with_tools = llm.bind_tools(tools)
    """
    return [send_email, write_file, calculator]


def get_sensitive_tools() -> list[Any]:
    """Get tools that require human approval.

    These tools perform actions that could have external side effects
    (sending emails, writing files) and should be approved by a human
    before execution.

    Returns:
        List of sensitive tool functions.

    Example:
        >>> sensitive = get_sensitive_tools()
        >>> sensitive_names = {t.name for t in sensitive}
        >>> # {'send_email', 'write_file'}
    """
    return [send_email, write_file]


def get_safe_tools() -> list[Any]:
    """Get tools that don't require approval.

    These tools perform read-only or computational operations
    with no external side effects.

    Returns:
        List of safe tool functions.

    Example:
        >>> safe = get_safe_tools()
        >>> safe_names = {t.name for t in safe}
        >>> # {'calculator'}
    """
    return [calculator]
