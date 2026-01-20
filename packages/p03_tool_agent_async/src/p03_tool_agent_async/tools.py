"""Async tool definitions for the Tool Agent pattern.

This module provides async custom tools that demonstrate LangGraph's tool-calling
capabilities. Tools are defined using the @tool decorator on async functions
with comprehensive docstrings and type hints to guide the LLM's tool selection.

Tools included:
- calculator: Perform basic math operations (add, subtract, multiply, divide)
- weather_lookup: Simulated weather lookup for a city (returns mock data)

Best Practices Demonstrated:
- Async tools for non-blocking execution in async agents
- Clear, descriptive docstrings that help the LLM understand when to use tools
- Type hints for all parameters and return values
- Robust error handling with informative error messages
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field


# region Constants


class MathOperation(StrEnum):
    """Supported mathematical operations for the calculator tool."""

    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


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


# region Tool Functions


@tool(args_schema=CalculatorInput)
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
            return f"Error: Cannot divide {a} by zero."
        result = a / b
        operation_symbol = "/"
    return f"{a} {operation_symbol} {b} = {result}"


@tool(args_schema=WeatherInput)
async def weather_lookup(city: str) -> str:
    """Look up the current weather for a specified city.

    Use this tool when you need to find out the current weather conditions
    in a particular city. Returns temperature, conditions, humidity, and wind.

    Note: This is a simulated weather service for demonstration purposes.
    In a production environment, this would call a real weather API using
    an async HTTP client like aiohttp or httpx.

    Args:
        city: The name of the city to look up weather for.

    Returns:
        A formatted string with weather information for the city,
        or an error message if the city is not found.

    Examples:
        - weather_lookup("Tokyo") -> Weather details for Tokyo
        - weather_lookup("Unknown City") -> Error message
    """
    city_lower: str = city.lower().strip()

    if city_lower in MOCK_WEATHER_DATA:
        weather = MOCK_WEATHER_DATA[city_lower]
        return (
            f"Weather in {city.title()}:\n"
            f"  Condition: {weather['condition']}\n"
            f"  Temperature: {weather['temperature']}C\n"
            f"  Humidity: {weather['humidity']}%\n"
            f"  Wind: {weather['wind']}"
        )

    # City not found - provide helpful response
    available_cities: str = ", ".join(
        c.title() for c in sorted(MOCK_WEATHER_DATA.keys())
    )
    return (
        f"Weather data not available for '{city}'. "
        f"Available cities: {available_cities}."
    )


# endregion


# region Public Functions


def get_all_tools() -> list[Any]:
    """Get all available tools for the agent.

    Returns:
        A list of all tool functions that can be bound to an LLM.

    Example:
        >>> tools = get_all_tools()
        >>> llm_with_tools = llm.bind_tools(tools)
    """
    return [calculator, weather_lookup]


# endregion
