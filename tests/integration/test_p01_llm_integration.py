"""Integration tests for the LLM Graph pattern.

These tests make real LLM API calls and require valid API keys.
Run with: task test-integration

Environment Variables Required:
    OPENAI_API_KEY: For OpenAI tests
    ANTHROPIC_API_KEY: For Anthropic tests
    GOOGLE_API_KEY: For Google Gemini tests
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

from p01_basic_graph.graph import create_graph
from p01_basic_graph.llm import (
    DEFAULT_MODELS,
    LLMProvider,
    create_llm,
    get_available_providers,
)
from p01_basic_graph.prompts import load_system_prompt
from p01_basic_graph.state import GraphState, create_initial_state
from shared.settings import settings


if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langgraph.graph.state import CompiledStateGraph


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def has_openai_key() -> bool:
    """Check if OpenAI API key is configured."""
    return settings.llm.openai_api_key is not None


def has_anthropic_key() -> bool:
    """Check if Anthropic API key is configured."""
    return settings.llm.anthropic_api_key is not None


def has_google_key() -> bool:
    """Check if Google API key is configured."""
    return os.environ.get("GOOGLE_API_KEY") is not None


class TestLLMCreation:
    """Integration tests for LLM creation with real credentials."""

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_create_openai_llm(self) -> None:
        """Test creating OpenAI LLM with real credentials."""
        llm: BaseChatModel = create_llm(LLMProvider.OPENAI, settings)

        assert llm is not None

    @pytest.mark.skipif(not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
    def test_create_anthropic_llm(self) -> None:
        """Test creating Anthropic LLM with real credentials."""
        llm: BaseChatModel = create_llm(LLMProvider.ANTHROPIC, settings)

        assert llm is not None

    @pytest.mark.skipif(not has_google_key(), reason="GOOGLE_API_KEY not set")
    def test_create_google_llm(self) -> None:
        """Test creating Google LLM with real credentials."""
        llm: BaseChatModel = create_llm(LLMProvider.GOOGLE, settings)

        assert llm is not None


class TestLLMInvocation:
    """Integration tests for direct LLM invocation."""

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_openai_simple_invoke(self) -> None:
        """Test simple invocation with OpenAI."""
        llm: BaseChatModel = create_llm(
            LLMProvider.OPENAI,
            settings,
            temperature=0.0,  # Deterministic for testing
        )

        response: AIMessage = llm.invoke("Say 'Hello' and nothing else.")

        assert response.content is not None
        assert len(str(response.content)) > 0

    @pytest.mark.skipif(not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
    def test_anthropic_simple_invoke(self) -> None:
        """Test simple invocation with Anthropic."""
        llm: BaseChatModel = create_llm(
            LLMProvider.ANTHROPIC,
            settings,
            temperature=0.0,
        )

        response: AIMessage = llm.invoke("Say 'Hello' and nothing else.")

        assert response.content is not None
        assert len(str(response.content)) > 0

    @pytest.mark.skipif(not has_google_key(), reason="GOOGLE_API_KEY not set")
    def test_google_simple_invoke(self) -> None:
        """Test simple invocation with Google Gemini."""
        llm: BaseChatModel = create_llm(
            LLMProvider.GOOGLE,
            settings,
            temperature=0.0,
        )

        response: AIMessage = llm.invoke("Say 'Hello' and nothing else.")

        assert response.content is not None
        assert len(str(response.content)) > 0


class TestGraphInvocation:
    """Integration tests for full graph invocation with real LLMs."""

    @pytest.fixture
    def system_prompt(self) -> str:
        """Load the system prompt."""
        return load_system_prompt()

    @pytest.mark.skipif(not has_openai_key(), reason="OPENAI_API_KEY not set")
    def test_graph_with_openai(self, system_prompt: str) -> None:
        """Test full graph invocation with OpenAI."""
        graph: CompiledStateGraph = create_graph(settings)

        initial_state: GraphState = create_initial_state(
            user_input="What is 2 + 2? Answer with just the number.",
            provider=LLMProvider.OPENAI,
            system_prompt=system_prompt,
        )

        result: dict[str, Any] = graph.invoke(initial_state)

        assert "llm_response" in result
        assert len(result["llm_response"]) > 0
        # Should contain "4" somewhere in response
        assert "4" in result["llm_response"]

    @pytest.mark.skipif(not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set")
    def test_graph_with_anthropic(self, system_prompt: str) -> None:
        """Test full graph invocation with Anthropic."""
        graph: CompiledStateGraph = create_graph(settings)

        initial_state: GraphState = create_initial_state(
            user_input="What is 2 + 2? Answer with just the number.",
            provider=LLMProvider.ANTHROPIC,
            system_prompt=system_prompt,
        )

        result: dict[str, Any] = graph.invoke(initial_state)

        assert "llm_response" in result
        assert len(result["llm_response"]) > 0
        assert "4" in result["llm_response"]

    @pytest.mark.skipif(not has_google_key(), reason="GOOGLE_API_KEY not set")
    def test_graph_with_google(self, system_prompt: str) -> None:
        """Test full graph invocation with Google Gemini."""
        graph: CompiledStateGraph = create_graph(settings)

        initial_state: GraphState = create_initial_state(
            user_input="What is 2 + 2? Answer with just the number.",
            provider=LLMProvider.GOOGLE,
            system_prompt=system_prompt,
        )

        result: dict[str, Any] = graph.invoke(initial_state)

        assert "llm_response" in result
        assert len(result["llm_response"]) > 0
        assert "4" in result["llm_response"]


class TestParameterizedProviders:
    """Parameterized tests that run across all available providers."""

    @pytest.fixture
    def available_providers(self) -> list[LLMProvider]:
        """Get list of providers with configured keys."""
        return get_available_providers(settings)

    @pytest.mark.parametrize(
        "provider",
        [
            pytest.param(
                LLMProvider.OPENAI,
                id="openai",
                marks=pytest.mark.skipif(
                    not has_openai_key(), reason="OPENAI_API_KEY not set"
                ),
            ),
            pytest.param(
                LLMProvider.ANTHROPIC,
                id="anthropic",
                marks=pytest.mark.skipif(
                    not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set"
                ),
            ),
            pytest.param(
                LLMProvider.GOOGLE,
                id="google",
                marks=pytest.mark.skipif(
                    not has_google_key(), reason="GOOGLE_API_KEY not set"
                ),
            ),
        ],
    )
    def test_llm_responds_coherently(self, provider: LLMProvider) -> None:
        """Test that each provider responds coherently to a simple question."""
        llm: BaseChatModel = create_llm(provider, settings, temperature=0.0)

        response: AIMessage = llm.invoke(
            "Complete this sequence: 1, 2, 3, __. Respond with only the number."
        )

        response_text: str = str(response.content).strip()
        # Should contain "4" - allowing for different formatting
        assert "4" in response_text, f"Expected '4' in response, got: {response_text}"

    @pytest.mark.parametrize(
        "provider",
        [
            pytest.param(
                LLMProvider.OPENAI,
                id="openai",
                marks=pytest.mark.skipif(
                    not has_openai_key(), reason="OPENAI_API_KEY not set"
                ),
            ),
            pytest.param(
                LLMProvider.ANTHROPIC,
                id="anthropic",
                marks=pytest.mark.skipif(
                    not has_anthropic_key(), reason="ANTHROPIC_API_KEY not set"
                ),
            ),
            pytest.param(
                LLMProvider.GOOGLE,
                id="google",
                marks=pytest.mark.skipif(
                    not has_google_key(), reason="GOOGLE_API_KEY not set"
                ),
            ),
        ],
    )
    def test_full_graph_flow(self, provider: LLMProvider) -> None:
        """Test full graph flow with each provider."""
        graph: CompiledStateGraph = create_graph(settings)
        system_prompt: str = load_system_prompt()

        initial_state: GraphState = create_initial_state(
            user_input="Respond with exactly the word 'success'.",
            provider=provider,
            system_prompt=system_prompt,
        )

        result: dict[str, Any] = graph.invoke(initial_state)

        assert "llm_response" in result
        response_lower: str = result["llm_response"].lower()
        assert "success" in response_lower


class TestModelVersions:
    """Tests to verify correct model versions are being used."""

    def test_default_models_are_current(self) -> None:
        """Test that default model identifiers are current."""
        # These are the expected current model versions as of Jan 2026
        expected_models: dict[LLMProvider, str] = {
            LLMProvider.OPENAI: "gpt-5",
            LLMProvider.ANTHROPIC: "claude-sonnet-4-5-20250929",
            LLMProvider.GOOGLE: "gemini-2.5-flash",
        }

        for provider, expected_model in expected_models.items():
            assert DEFAULT_MODELS[provider] == expected_model, (
                f"Model mismatch for {provider}: "
                f"expected {expected_model}, got {DEFAULT_MODELS[provider]}"
            )
