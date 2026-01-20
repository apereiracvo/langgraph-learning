"""Unit tests for shared settings module.

Tests cover:
- Default values
- Environment variable loading
- Nested delimiter parsing
- Placeholder rejection
- SecretStr masking
- Backward compatibility
- Enum validation
"""

from __future__ import annotations

import pytest

from shared.enums import Environment, LLMProvider, LogLevel
from shared.settings import (
    LangSmithSettings,
    LLMSettings,
    Settings,
    clear_settings_cache,
    get_settings,
    settings,
)


class TestLLMSettings:
    """Tests for LLMSettings validation."""

    def test_default_values(self) -> None:
        """LLMSettings should have sensible defaults."""
        llm_settings: LLMSettings = LLMSettings()

        assert llm_settings.openai_api_key is None
        assert llm_settings.anthropic_api_key is None
        assert llm_settings.google_api_key is None
        assert llm_settings.default_model == "gpt-4"
        assert llm_settings.provider == LLMProvider.OPENAI

    def test_placeholder_rejection_your_key_here(self) -> None:
        """Placeholder 'your-key-here' patterns should be treated as None."""
        llm_settings: LLMSettings = LLMSettings(
            openai_api_key="your-openai-key-here",  # type: ignore[arg-type]
            anthropic_api_key="your-anthropic-key-here",  # type: ignore[arg-type]
            google_api_key="your-google-key-here",  # type: ignore[arg-type]
        )

        assert llm_settings.openai_api_key is None
        assert llm_settings.anthropic_api_key is None
        assert llm_settings.google_api_key is None

    def test_placeholder_rejection_sk_xxx(self) -> None:
        """Placeholder 'sk-xxx' patterns should be treated as None."""
        llm_settings: LLMSettings = LLMSettings(
            openai_api_key="sk-xxx",  # type: ignore[arg-type]
        )

        assert llm_settings.openai_api_key is None

    def test_placeholder_rejection_sk_xxxx_multiple(self) -> None:
        """Placeholder 'sk-xxxx' with multiple x's should be treated as None."""
        llm_settings: LLMSettings = LLMSettings(
            openai_api_key="sk-xxxxxxxx",  # type: ignore[arg-type]
        )

        assert llm_settings.openai_api_key is None

    def test_placeholder_rejection_your_prefix(self) -> None:
        """Keys starting with 'your-' should be treated as None."""
        llm_settings: LLMSettings = LLMSettings(
            openai_api_key="your-custom-placeholder",  # type: ignore[arg-type]
        )

        assert llm_settings.openai_api_key is None

    def test_placeholder_rejection_empty_string(self) -> None:
        """Empty strings should be treated as None."""
        llm_settings: LLMSettings = LLMSettings(
            openai_api_key="",  # type: ignore[arg-type]
        )

        assert llm_settings.openai_api_key is None

    def test_placeholder_rejection_whitespace(self) -> None:
        """Whitespace-only strings should be treated as None."""
        llm_settings: LLMSettings = LLMSettings(
            openai_api_key="   ",  # type: ignore[arg-type]
        )

        assert llm_settings.openai_api_key is None

    def test_valid_api_key_accepted(self) -> None:
        """Valid API keys should be stored as SecretStr."""
        llm_settings: LLMSettings = LLMSettings(
            openai_api_key="sk-real-key-abc123def456",  # type: ignore[arg-type]
        )

        assert llm_settings.openai_api_key is not None
        assert (
            llm_settings.openai_api_key.get_secret_value() == "sk-real-key-abc123def456"
        )

    def test_secret_str_masking(self) -> None:
        """SecretStr should mask value in repr and str."""
        llm_settings: LLMSettings = LLMSettings(
            openai_api_key="sk-secret-key-12345",  # type: ignore[arg-type]
        )

        # repr should not contain the actual key
        repr_str: str = repr(llm_settings)
        assert "sk-secret-key-12345" not in repr_str
        assert "**********" in repr_str


class TestLangSmithSettings:
    """Tests for LangSmithSettings validation."""

    def test_default_values(self) -> None:
        """LangSmithSettings should have sensible defaults."""
        langsmith_settings: LangSmithSettings = LangSmithSettings()

        assert langsmith_settings.tracing_enabled is False
        assert langsmith_settings.api_key is None
        assert langsmith_settings.project == "langgraph-learning"

    def test_placeholder_rejection(self) -> None:
        """Placeholder API keys should be treated as None."""
        langsmith_settings: LangSmithSettings = LangSmithSettings(
            api_key="your-langsmith-key-here",  # type: ignore[arg-type]
        )

        assert langsmith_settings.api_key is None

    def test_valid_api_key_accepted(self) -> None:
        """Valid API keys should be stored."""
        langsmith_settings: LangSmithSettings = LangSmithSettings(
            api_key="ls-real-api-key-12345",  # type: ignore[arg-type]
        )

        assert langsmith_settings.api_key is not None
        assert langsmith_settings.api_key.get_secret_value() == "ls-real-api-key-12345"


class TestSettings:
    """Tests for main Settings class."""

    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Settings should have sensible defaults."""
        # Clear environment variables that might affect defaults
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        monkeypatch.delenv("DEBUG", raising=False)
        monkeypatch.delenv("VERBOSE", raising=False)
        monkeypatch.delenv("LOG_LEVEL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("LLM__OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LLM__ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGSMITH__TRACING_ENABLED", raising=False)

        clear_settings_cache()

        test_settings: Settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert test_settings.environment == Environment.DEVELOPMENT
        assert test_settings.debug is False
        assert test_settings.verbose is False
        assert test_settings.log_level == LogLevel.INFO

    def test_environment_literal_validation(self) -> None:
        """Environment must be a valid literal value."""
        from pydantic import ValidationError  # noqa: PLC0415

        with pytest.raises(ValidationError) as exc_info:
            Settings(environment="invalid", _env_file=None)  # type: ignore[arg-type, call-arg]

        errors: list[dict[str, object]] = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("environment",)

    def test_log_level_literal_validation(self) -> None:
        """Log level must be a valid literal value."""
        from pydantic import ValidationError  # noqa: PLC0415

        with pytest.raises(ValidationError) as exc_info:
            Settings(log_level="TRACE", _env_file=None)  # type: ignore[arg-type, call-arg]

        errors: list[dict[str, object]] = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("log_level",)

    def test_environment_variable_loading(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Settings should load from environment variables."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        clear_settings_cache()

        test_settings: Settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert test_settings.environment == Environment.PRODUCTION
        assert test_settings.debug is True
        assert test_settings.log_level == LogLevel.DEBUG

    def test_nested_env_vars_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nested env vars with __ delimiter should work for LLM."""
        monkeypatch.setenv("LLM__OPENAI_API_KEY", "sk-nested-key-12345")
        monkeypatch.setenv("LLM__DEFAULT_MODEL", "gpt-4-turbo")
        # Clear flat vars
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        clear_settings_cache()

        test_settings: Settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert test_settings.llm.openai_api_key is not None
        assert (
            test_settings.llm.openai_api_key.get_secret_value() == "sk-nested-key-12345"
        )
        assert test_settings.llm.default_model == "gpt-4-turbo"

    def test_nested_env_vars_langsmith(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Nested env vars with __ delimiter should work for LangSmith."""
        monkeypatch.setenv("LANGSMITH__TRACING_ENABLED", "true")
        monkeypatch.setenv("LANGSMITH__API_KEY", "ls-nested-key-12345")
        monkeypatch.setenv("LANGSMITH__PROJECT", "my-custom-project")
        # Clear flat vars
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

        clear_settings_cache()

        test_settings: Settings = Settings(_env_file=None)  # type: ignore[call-arg]

        assert test_settings.langsmith.tracing_enabled is True
        assert test_settings.langsmith.api_key is not None
        assert (
            test_settings.langsmith.api_key.get_secret_value() == "ls-nested-key-12345"
        )
        assert test_settings.langsmith.project == "my-custom-project"

    def test_backward_compat_flat_env_vars(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Flat env vars should sync to nested settings for backward compat."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-flat-key-12345")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "ls-flat-key-12345")
        monkeypatch.setenv("LANGCHAIN_PROJECT", "flat-project")
        # Clear nested vars
        monkeypatch.delenv("LLM__OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH__TRACING_ENABLED", raising=False)
        monkeypatch.delenv("LANGSMITH__API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH__PROJECT", raising=False)

        clear_settings_cache()

        test_settings: Settings = Settings(_env_file=None)  # type: ignore[call-arg]

        # Flat vars should be synced to nested
        assert test_settings.llm.openai_api_key is not None
        assert (
            test_settings.llm.openai_api_key.get_secret_value() == "sk-flat-key-12345"
        )
        assert test_settings.langsmith.tracing_enabled is True
        assert test_settings.langsmith.api_key is not None
        assert test_settings.langsmith.api_key.get_secret_value() == "ls-flat-key-12345"
        assert test_settings.langsmith.project == "flat-project"

    def test_nested_takes_precedence_over_flat(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When both flat and nested env vars set, nested should take precedence."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-flat-key")
        monkeypatch.setenv("LLM__OPENAI_API_KEY", "sk-nested-key")

        clear_settings_cache()

        test_settings: Settings = Settings(_env_file=None)  # type: ignore[call-arg]

        # Nested should be used (not overwritten by flat)
        assert test_settings.llm.openai_api_key is not None
        assert test_settings.llm.openai_api_key.get_secret_value() == "sk-nested-key"

    def test_backward_compat_google_api_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Flat GOOGLE_API_KEY should sync to nested settings."""
        monkeypatch.setenv("GOOGLE_API_KEY", "google-flat-key-12345")
        monkeypatch.delenv("LLM__GOOGLE_API_KEY", raising=False)

        clear_settings_cache()

        test_settings: Settings = Settings(_env_file=None)  # type: ignore[call-arg]

        # Flat var should be synced to nested
        assert test_settings.llm.google_api_key is not None
        assert (
            test_settings.llm.google_api_key.get_secret_value()
            == "google-flat-key-12345"
        )

    def test_backward_compat_llm_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Flat LLM_PROVIDER should sync to nested settings."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.delenv("LLM__PROVIDER", raising=False)

        clear_settings_cache()

        test_settings: Settings = Settings(_env_file=None)  # type: ignore[call-arg]

        # Flat var should be synced to nested
        assert test_settings.llm.provider == LLMProvider.ANTHROPIC

    def test_nested_google_api_key_takes_precedence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When both flat and nested Google API key set, nested should take precedence."""
        monkeypatch.setenv("GOOGLE_API_KEY", "google-flat-key")
        monkeypatch.setenv("LLM__GOOGLE_API_KEY", "google-nested-key")

        clear_settings_cache()

        test_settings: Settings = Settings(_env_file=None)  # type: ignore[call-arg]

        # Nested should be used (not overwritten by flat)
        assert test_settings.llm.google_api_key is not None
        assert (
            test_settings.llm.google_api_key.get_secret_value() == "google-nested-key"
        )

    def test_nested_llm_provider_takes_precedence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When both flat and nested LLM provider set, nested should take precedence."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM__PROVIDER", "google")

        clear_settings_cache()

        test_settings: Settings = Settings(_env_file=None)  # type: ignore[call-arg]

        # Nested should be used (not overwritten by flat)
        assert test_settings.llm.provider == LLMProvider.GOOGLE


class TestSettingsCaching:
    """Tests for settings caching behavior."""

    def test_get_settings_returns_cached_instance(self) -> None:
        """get_settings should return the same instance (cached)."""
        clear_settings_cache()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_clear_settings_cache_creates_new_instance(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """clear_settings_cache should allow creating a new instance."""
        monkeypatch.setenv("DEBUG", "false")
        clear_settings_cache()
        settings1 = get_settings()

        monkeypatch.setenv("DEBUG", "true")
        clear_settings_cache()
        settings2 = get_settings()

        assert settings1 is not settings2

    def test_module_level_settings_singleton(self) -> None:
        """Module-level settings should be a singleton."""
        # Import settings twice to test singleton behavior
        # We already imported 'settings' at module level
        settings1 = settings
        settings2 = settings

        assert settings1 is settings2


class TestModelDump:
    """Tests for settings serialization."""

    def test_model_dump_masks_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """model_dump should mask SecretStr values."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-key-12345")
        monkeypatch.delenv("LLM__OPENAI_API_KEY", raising=False)

        clear_settings_cache()

        test_settings: Settings = Settings(_env_file=None)  # type: ignore[call-arg]
        dumped: dict[str, object] = test_settings.model_dump()

        # The secret should be masked
        llm_dump: dict[str, object] = dumped["llm"]  # type: ignore[assignment]
        assert "sk-secret-key-12345" not in str(llm_dump)

    def test_model_dump_json_masks_secrets(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """model_dump_json should mask SecretStr values."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-key-12345")
        monkeypatch.delenv("LLM__OPENAI_API_KEY", raising=False)

        clear_settings_cache()

        test_settings: Settings = Settings(_env_file=None)  # type: ignore[call-arg]
        json_str: str = test_settings.model_dump_json()

        # The actual secret should not appear in JSON
        assert "sk-secret-key-12345" not in json_str


class TestEnumValidation:
    """Tests for enum-based field validation."""

    def test_environment_enum_values(self) -> None:
        """Environment enum should have expected values."""
        assert Environment.DEVELOPMENT == "development"
        assert Environment.STAGING == "staging"
        assert Environment.PRODUCTION == "production"

    def test_log_level_enum_values(self) -> None:
        """LogLevel enum should have expected values."""
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"

    def test_environment_str_enum_behavior(self) -> None:
        """Environment enum should behave as str for comparisons."""
        env: Environment = Environment.PRODUCTION
        assert env == "production"
        assert str(env) == "production"
        assert isinstance(env, str)

    def test_log_level_str_enum_behavior(self) -> None:
        """LogLevel enum should behave as str for comparisons."""
        level: LogLevel = LogLevel.WARNING
        assert level == "WARNING"
        assert str(level) == "WARNING"
        assert isinstance(level, str)

    def test_settings_environment_accepts_enum(self) -> None:
        """Settings should accept Environment enum directly."""
        test_settings: Settings = Settings(
            environment=Environment.STAGING,
            _env_file=None,  # type: ignore[call-arg]
        )
        assert test_settings.environment == Environment.STAGING
        assert test_settings.environment == "staging"

    def test_settings_log_level_accepts_enum(self) -> None:
        """Settings should accept LogLevel enum directly."""
        test_settings: Settings = Settings(
            log_level=LogLevel.ERROR,
            _env_file=None,  # type: ignore[call-arg]
        )
        assert test_settings.log_level == LogLevel.ERROR
        assert test_settings.log_level == "ERROR"

    def test_settings_environment_accepts_string(self) -> None:
        """Settings should coerce valid string to Environment enum."""
        test_settings: Settings = Settings(
            environment="production",  # type: ignore[arg-type]
            _env_file=None,  # type: ignore[call-arg]
        )
        assert test_settings.environment == Environment.PRODUCTION
        assert isinstance(test_settings.environment, Environment)

    def test_settings_log_level_accepts_string(self) -> None:
        """Settings should coerce valid string to LogLevel enum."""
        test_settings: Settings = Settings(
            log_level="WARNING",  # type: ignore[arg-type]
            _env_file=None,  # type: ignore[call-arg]
        )
        assert test_settings.log_level == LogLevel.WARNING
        assert isinstance(test_settings.log_level, LogLevel)

    def test_settings_environment_rejects_invalid(self) -> None:
        """Settings should reject invalid environment values."""
        from pydantic import ValidationError  # noqa: PLC0415

        with pytest.raises(ValidationError) as exc_info:
            Settings(
                environment="invalid",  # type: ignore[arg-type]
                _env_file=None,  # type: ignore[call-arg]
            )

        errors: list[dict[str, object]] = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("environment",)

    def test_settings_log_level_rejects_invalid(self) -> None:
        """Settings should reject invalid log level values."""
        from pydantic import ValidationError  # noqa: PLC0415

        with pytest.raises(ValidationError) as exc_info:
            Settings(
                log_level="TRACE",  # type: ignore[arg-type]
                _env_file=None,  # type: ignore[call-arg]
            )

        errors: list[dict[str, object]] = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("log_level",)

    def test_llm_provider_enum_values(self) -> None:
        """LLMProvider enum should have expected values."""
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC == "anthropic"
        assert LLMProvider.GOOGLE == "google"

    def test_llm_provider_str_enum_behavior(self) -> None:
        """LLMProvider enum should behave as str for comparisons."""
        provider: LLMProvider = LLMProvider.OPENAI
        assert provider == "openai"
        assert str(provider) == "openai"
        assert isinstance(provider, str)

    def test_llm_settings_provider_accepts_enum(self) -> None:
        """LLMSettings should accept LLMProvider enum directly."""
        llm_settings: LLMSettings = LLMSettings(provider=LLMProvider.GOOGLE)
        assert llm_settings.provider == LLMProvider.GOOGLE
        assert llm_settings.provider == "google"

    def test_llm_settings_provider_accepts_string(self) -> None:
        """LLMSettings should coerce valid string to LLMProvider enum."""
        llm_settings: LLMSettings = LLMSettings(
            provider="anthropic",  # type: ignore[arg-type]
        )
        assert llm_settings.provider == LLMProvider.ANTHROPIC
        assert isinstance(llm_settings.provider, LLMProvider)

    def test_llm_settings_provider_rejects_invalid(self) -> None:
        """LLMSettings should reject invalid provider values."""
        from pydantic import ValidationError  # noqa: PLC0415

        with pytest.raises(ValidationError) as exc_info:
            LLMSettings(
                provider="invalid_provider",  # type: ignore[arg-type]
            )

        errors: list[dict[str, object]] = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("provider",)
