"""Pattern 03: Async Tool-Calling Agent with ReAct Pattern.

This pattern demonstrates:
- Async LangGraph tool-calling agents
- Custom tool definitions with @tool decorator and Pydantic schemas
- Basic tool-calling agent with StateGraph
- ReAct agent using LangGraph's prebuilt pattern
- Multi-provider LLM support (OpenAI, Anthropic, Google)
- Async graph invocation with `ainvoke()`

Tools included:
- calculator: Perform basic math operations
- weather_lookup: Simulated weather lookup for cities

Agent types:
- Basic Agent: Simple tool-calling agent with manual StateGraph
- ReAct Agent: Uses LangGraph's prebuilt create_react_agent
"""

from __future__ import annotations

from p03_tool_agent_async.agents import create_basic_agent, create_react_agent
from p03_tool_agent_async.prompts import load_system_prompt
from p03_tool_agent_async.state import AgentState, create_initial_state
from p03_tool_agent_async.tools import (
    CalculatorInput,
    MathOperation,
    WeatherInput,
    calculator,
    get_all_tools,
    weather_lookup,
)
from shared.enums import LLMProvider
from shared.exceptions import LLMConfigurationError, PromptLoadError
from shared.llm import create_llm


__all__: list[str] = [
    "AgentState",
    "CalculatorInput",
    "LLMConfigurationError",
    "LLMProvider",
    "MathOperation",
    "PromptLoadError",
    "WeatherInput",
    "calculator",
    "create_basic_agent",
    "create_initial_state",
    "create_llm",
    "create_react_agent",
    "get_all_tools",
    "load_system_prompt",
    "weather_lookup",
]
