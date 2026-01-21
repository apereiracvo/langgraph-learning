"""Traced tool definitions for the LangSmith Observability pattern.

This module provides async tools with additional tracing decorators to
demonstrate custom span creation and metadata enrichment in LangSmith.

Tools included:
- calculator: Perform basic math operations with tracing
- weather_lookup: Simulated weather lookup with tracing
- failing_tool: Tool that intentionally fails for error tracing demo

Best Practices Demonstrated:
- @traceable decorator for custom span creation
- Metadata enrichment within traced functions
- Error handling with trace capture
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from langchain_core.tools import tool
from langsmith import traceable
from pydantic import BaseModel, Field

from shared.observability import add_run_metadata


# region Constants


class MathOperation(StrEnum):
    """Supported mathematical operations for the calculator tool."""

    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


# Mock weather data for demonstration purposes
MOCK_WEATHER_DATA: dict[str, dict[str, str | int]] = {
    "tokyo": {
        "condition": "Partly Cloudy",
        "temperature": 18,
        "humidity": 65,
        "wind": "Light breeze from the east",
    },
    "new york": {
        "condition": "Sunny",
        "temperature": 22,
        "humidity": 45,
        "wind": "Calm",
    },
    "london": {
        "condition": "Rainy",
        "temperature": 12,
        "humidity": 85,
        "wind": "Moderate wind from the west",
    },
    "paris": {
        "condition": "Overcast",
        "temperature": 15,
        "humidity": 70,
        "wind": "Light wind from the north",
    },
    "san francisco": {
        "condition": "Foggy",
        "temperature": 16,
        "humidity": 80,
        "wind": "Gentle ocean breeze",
    },
    "sydney": {
        "condition": "Clear and sunny",
        "temperature": 28,
        "humidity": 55,
        "wind": "Warm northerly wind",
    },
}


# endregion


# region Types


class CalculatorInput(BaseModel):
    """Input schema for the calculator tool.

    Attributes:
        a: The first number in the calculation.
        b: The second number in the calculation.
        operation: The mathematical operation to perform.
    """

    a: float = Field(description="The first number in the calculation")
    b: float = Field(description="The second number in the calculation")
    operation: MathOperation = Field(
        description="The operation to perform: add, subtract, multiply, or divide"
    )


class WeatherInput(BaseModel):
    """Input schema for the weather lookup tool.

    Attributes:
        city: The name of the city to look up weather for.
    """

    city: str = Field(description="The name of the city to look up weather for")


class FailingToolInput(BaseModel):
    """Input schema for the failing tool (error demo).

    Attributes:
        error_type: Type of error to simulate.
    """

    error_type: str = Field(
        default="generic",
        description="Type of error to simulate: generic, validation, timeout",
    )


# endregion


# region Private Functions


@tool(args_schema=CalculatorInput)
@traceable(name="Calculator Tool", run_type="tool")
async def calculator(a: float, b: float, operation: MathOperation) -> str:
    """Perform basic math operations on two numbers.

    Use this tool when you need to calculate the result of a mathematical
    operation. Supports addition, subtraction, multiplication, and division.

    Args:
        a: The first number in the calculation.
        b: The second number in the calculation.
        operation: The operation to perform (add, subtract, multiply, divide).

    Returns:
        A string describing the calculation and its result.

    Examples:
        - calculator(5, 3, "add") -> "5.0 + 3.0 = 8.0"
        - calculator(10, 2, "divide") -> "10.0 / 2.0 = 5.0"
    """
    # Add metadata about the calculation
    add_run_metadata(
        {
            "operand_a": a,
            "operand_b": b,
            "operation_type": operation,
        }
    )

    result: float
    operation_symbol: str

    if operation == MathOperation.ADD:
        result = a + b
        operation_symbol = "+"
    elif operation == MathOperation.SUBTRACT:
        result = a - b
        operation_symbol = "-"
    elif operation == MathOperation.MULTIPLY:
        result = a * b
        operation_symbol = "*"
    elif operation == MathOperation.DIVIDE:
        if b == 0:
            add_run_metadata({"error": "division_by_zero"})
            return f"Error: Cannot divide {a} by zero."
        result = a / b
        operation_symbol = "/"

    add_run_metadata({"result": result})
    return f"{a} {operation_symbol} {b} = {result}"


@tool(args_schema=WeatherInput)
@traceable(name="Weather Lookup Tool", run_type="tool")
async def weather_lookup(city: str) -> str:
    """Look up the current weather for a specified city.

    Use this tool when you need to find out the current weather conditions
    in a particular city. Returns temperature, conditions, humidity, and wind.

    Note: This is a simulated weather service for demonstration purposes.
    In a production environment, this would call a real weather API.

    Args:
        city: The name of the city to look up weather for.

    Returns:
        A formatted string with weather information for the city,
        or an error message if the city is not found.
    """
    city_lower: str = city.lower().strip()

    # Add metadata about the lookup
    add_run_metadata(
        {
            "requested_city": city,
            "normalized_city": city_lower,
        }
    )

    if city_lower in MOCK_WEATHER_DATA:
        weather = MOCK_WEATHER_DATA[city_lower]
        add_run_metadata(
            {
                "city_found": True,
                "temperature": weather["temperature"],
                "condition": weather["condition"],
            }
        )
        return (
            f"Weather in {city.title()}:\n"
            f"  Condition: {weather['condition']}\n"
            f"  Temperature: {weather['temperature']}C\n"
            f"  Humidity: {weather['humidity']}%\n"
            f"  Wind: {weather['wind']}"
        )

    # City not found - provide helpful response
    add_run_metadata({"city_found": False})
    available_cities: str = ", ".join(
        c.title() for c in sorted(MOCK_WEATHER_DATA.keys())
    )
    return (
        f"Weather data not available for '{city}'. "
        f"Available cities: {available_cities}."
    )


@tool(args_schema=FailingToolInput)
@traceable(name="Failing Tool (Demo)", run_type="tool")
async def failing_tool(error_type: str = "generic") -> str:
    """A tool that intentionally fails for error tracing demonstration.

    This tool is used to demonstrate how LangSmith captures errors
    in traces for debugging purposes.

    Args:
        error_type: Type of error to simulate (generic, validation, timeout).

    Returns:
        Never returns successfully - always raises an exception.

    Raises:
        ValueError: For validation errors.
        TimeoutError: For timeout errors.
        RuntimeError: For generic errors.
    """
    add_run_metadata(
        {
            "error_type_requested": error_type,
            "intentional_failure": True,
        }
    )

    if error_type == "validation":
        msg = "Validation failed: Invalid input data provided"
        raise ValueError(msg)
    if error_type == "timeout":
        msg = "Operation timed out after 30 seconds"
        raise TimeoutError(msg)

    msg = "Simulated generic error for demonstration"
    raise RuntimeError(msg)


# endregion


# region Public Functions


def get_all_tools() -> list[Any]:
    """Get all available tools for the agent (excluding failing_tool).

    Returns:
        A list of all tool functions that can be bound to an LLM.

    Example:
        >>> tools = get_all_tools()
        >>> llm_with_tools = llm.bind_tools(tools)
    """
    return [calculator, weather_lookup]


def get_all_tools_with_failing() -> list[Any]:
    """Get all tools including the failing tool for error demos.

    Returns:
        A list of all tool functions including the failing tool.

    Example:
        >>> tools = get_all_tools_with_failing()
        >>> # Use for error tracing demonstration
    """
    return [calculator, weather_lookup, failing_tool]


# endregion
