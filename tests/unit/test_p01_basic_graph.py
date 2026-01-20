"""Unit tests for the LLM Graph pattern.

These tests mock LLM responses and test graph structure without
making actual API calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from p01_basic_graph.graph import create_graph, create_llm_node
from p01_basic_graph.state import GraphState, create_initial_state
from shared.enums import LLMProvider
from shared.exceptions import LLMConfigurationError, PromptLoadError
from shared.llm import DEFAULT_MODELS, create_llm, get_available_providers
from shared.prompts import clear_prompt_cache, load_system_prompt, render_prompt


if TYPE_CHECKING:
    from pathlib import Path

    from langgraph.graph.state import CompiledStateGraph

    from shared.settings import Settings


class TestLLMProvider:
    """Tests for LLMProvider enum."""

    def test_provider_values(self) -> None:
        """Test that provider enum has correct values."""
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.GOOGLE == "google"

    def test_provider_is_str_enum(self) -> None:
        """Test that providers can be used as strings."""
        provider: LLMProvider = LLMProvider.OPENAI
        assert f"Provider: {provider}" == "Provider: openai"


class TestCreateLLM:
    """Tests for the create_llm factory function."""

    def test_create_llm_raises_for_missing_openai_key(
        self,
        mock_settings_no_keys: Settings,
    ) -> None:
        """Test that missing OpenAI key raises error."""
        with pytest.raises(LLMConfigurationError) as exc_info:
            create_llm(LLMProvider.OPENAI, mock_settings_no_keys)

        assert "OPENAI_API_KEY" in str(exc_info.value)
        assert exc_info.value.provider == LLMProvider.OPENAI

    def test_create_llm_raises_for_missing_anthropic_key(
        self,
        mock_settings_no_keys: Settings,
    ) -> None:
        """Test that missing Anthropic key raises error."""
        with pytest.raises(LLMConfigurationError) as exc_info:
            create_llm(LLMProvider.ANTHROPIC, mock_settings_no_keys)

        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_create_llm_raises_for_missing_google_key(
        self,
        mock_settings_no_keys: Settings,
    ) -> None:
        """Test that missing Google key raises error."""
        with pytest.raises(LLMConfigurationError) as exc_info:
            create_llm(LLMProvider.GOOGLE, mock_settings_no_keys)

        assert "GOOGLE_API_KEY" in str(exc_info.value)

    def test_default_models_exist_for_all_providers(self) -> None:
        """Test that default models are defined for all providers."""
        for provider in LLMProvider:
            assert provider in DEFAULT_MODELS
            assert DEFAULT_MODELS[provider] is not None


class TestGetAvailableProviders:
    """Tests for get_available_providers function."""

    def test_no_providers_when_no_keys(
        self,
        mock_settings_no_keys: Settings,
    ) -> None:
        """Test that no providers are available without keys."""
        available: list[LLMProvider] = get_available_providers(mock_settings_no_keys)
        assert available == []

    def test_openai_available_when_key_set(
        self,
        mock_settings_openai: Settings,
    ) -> None:
        """Test that OpenAI is available when key is set."""
        available: list[LLMProvider] = get_available_providers(mock_settings_openai)
        assert LLMProvider.OPENAI in available


class TestPromptLoading:
    """Tests for prompt loading utilities."""

    def setup_method(self) -> None:
        """Clear cache before each test."""
        clear_prompt_cache()

    def test_load_system_prompt_returns_string(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that load_system_prompt returns prompt content."""
        prompt_file: Path = tmp_path / "system.md"
        prompt_file.write_text("You are helpful.")

        content: str = load_system_prompt("system.md", prompts_dir=tmp_path)

        assert content == "You are helpful."

    def test_load_system_prompt_strips_whitespace(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that prompt content is stripped."""
        prompt_file: Path = tmp_path / "system.md"
        prompt_file.write_text("  Content with whitespace  \n\n")

        content: str = load_system_prompt("system.md", prompts_dir=tmp_path)

        assert content == "Content with whitespace"

    def test_load_system_prompt_raises_for_missing_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that missing file raises PromptLoadError."""
        with pytest.raises(PromptLoadError) as exc_info:
            load_system_prompt("nonexistent.md", prompts_dir=tmp_path)

        assert "not found" in str(exc_info.value)

    def test_render_prompt_substitutes_variables(self) -> None:
        """Test that render_prompt performs substitution."""
        template: str = "Hello, {name}!"
        result: str = render_prompt(template, {"name": "World"})

        assert result == "Hello, World!"

    def test_render_prompt_nonstrict_preserves_missing(self) -> None:
        """Test non-strict mode preserves missing placeholders."""
        template: str = "Hello, {name}! Your {role} is ready."
        result: str = render_prompt(template, {"name": "Alice"})

        assert result == "Hello, Alice! Your {role} is ready."

    def test_render_prompt_strict_raises_for_missing(self) -> None:
        """Test strict mode raises for missing variables."""
        template: str = "Hello, {name}!"

        with pytest.raises(KeyError):
            render_prompt(template, {}, strict=True)


class TestGraphState:
    """Tests for GraphState and state creation."""

    def test_create_initial_state_sets_values(self) -> None:
        """Test that create_initial_state sets all values."""
        state: GraphState = create_initial_state(
            user_input="Hello",
            provider=LLMProvider.OPENAI,
            system_prompt="Be helpful",
        )

        assert state["user_input"] == "Hello"
        assert state["provider"] == LLMProvider.OPENAI
        assert state["system_prompt"] == "Be helpful"
        assert state["messages"] == []
        assert state["llm_response"] == ""
        assert state["metadata"] == {}


class TestCreateGraph:
    """Tests for graph creation and structure."""

    def test_create_graph_returns_compiled_graph(
        self,
        mock_settings_openai: Settings,
    ) -> None:
        """Test that create_graph returns a compiled graph."""
        with patch("p01_basic_graph.graph.create_llm") as mock_create_llm:
            mock_llm: MagicMock = MagicMock()
            mock_llm.invoke.return_value = AIMessage(content="Hello!")
            mock_create_llm.return_value = mock_llm

            graph: CompiledStateGraph = create_graph(mock_settings_openai)

            assert graph is not None
            assert hasattr(graph, "invoke")

    def test_graph_invoke_with_mocked_llm(
        self,
        mock_settings_openai: Settings,
    ) -> None:
        """Test graph invocation with mocked LLM."""
        with patch("p01_basic_graph.graph.create_llm") as mock_create_llm:
            mock_llm: MagicMock = MagicMock()
            mock_response: AIMessage = AIMessage(content="I am an AI assistant.")
            mock_llm.invoke.return_value = mock_response
            mock_create_llm.return_value = mock_llm

            graph: CompiledStateGraph = create_graph(mock_settings_openai)

            initial_state: GraphState = create_initial_state(
                user_input="What are you?",
                provider=LLMProvider.OPENAI,
                system_prompt="You are helpful.",
            )

            result: dict[str, Any] = graph.invoke(initial_state)

            assert result["llm_response"] == "I am an AI assistant."
            assert mock_llm.invoke.called


class TestCreateLLMNode:
    """Tests for the LLM node factory function."""

    def test_create_llm_node_returns_callable(
        self,
        mock_settings_openai: Settings,
    ) -> None:
        """Test that create_llm_node returns a callable."""
        node_fn = create_llm_node(mock_settings_openai)

        assert callable(node_fn)

    def test_llm_node_calls_llm_with_messages(
        self,
        mock_settings_openai: Settings,
    ) -> None:
        """Test that LLM node builds and sends messages correctly."""
        with patch("p01_basic_graph.graph.create_llm") as mock_create_llm:
            mock_llm: MagicMock = MagicMock()
            mock_response: AIMessage = AIMessage(content="Response")
            mock_llm.invoke.return_value = mock_response
            mock_create_llm.return_value = mock_llm

            node_fn = create_llm_node(mock_settings_openai)

            state: GraphState = create_initial_state(
                user_input="Test input",
                provider=LLMProvider.OPENAI,
                system_prompt="System prompt",
            )

            result: dict[str, Any] = node_fn(state)

            # Verify LLM was called with correct messages
            call_args = mock_llm.invoke.call_args[0][0]
            assert len(call_args) == 2
            assert call_args[0].content == "System prompt"
            assert call_args[1].content == "Test input"

            # Verify response
            assert result["llm_response"] == "Response"


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_settings_no_keys() -> Settings:
    """Create mock settings with no API keys."""
    mock: MagicMock = MagicMock()
    mock.llm.openai_api_key = None
    mock.llm.anthropic_api_key = None
    mock.llm.google_api_key = None
    return mock


@pytest.fixture
def mock_settings_openai() -> Settings:
    """Create mock settings with OpenAI key."""
    mock: MagicMock = MagicMock()
    mock.llm.openai_api_key.get_secret_value.return_value = "sk-test-key"
    mock.llm.anthropic_api_key = None
    mock.llm.google_api_key = None
    return mock


@pytest.fixture
def mock_settings_anthropic() -> Settings:
    """Create mock settings with Anthropic key."""
    mock: MagicMock = MagicMock()
    mock.llm.openai_api_key = None
    mock.llm.anthropic_api_key.get_secret_value.return_value = "sk-ant-test-key"
    mock.llm.google_api_key = None
    return mock
