"""Pattern 05: LangSmith Observability and Tracing.

This pattern demonstrates LangSmith integration for tracing, monitoring,
and debugging LangGraph agents. It covers:

- Automatic tracing with environment variables
- Custom spans with @traceable decorator
- Metadata and tags for filtering
- Error tracing and debugging workflows

Key components:
- agent: Traced tool-calling agent implementation
- tools: Tools with custom tracing decorators
- demos: Four demonstration modules showcasing features

Usage:
    Run via task runner: task run -- p05-langsmith-observability

Configuration:
    Enable tracing by setting environment variables:
    - LANGSMITH__TRACING_ENABLED=true
    - LANGSMITH__API_KEY=your-key

Example:
    >>> from p05_langsmith_observability.agent import create_traced_agent
    >>> from shared.settings import settings
    >>> agent = create_traced_agent(settings)
    >>> result = await agent.ainvoke(initial_state)
"""

from __future__ import annotations
