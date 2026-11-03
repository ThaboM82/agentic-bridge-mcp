"""
Comprehensive tests for application configuration.

These tests cover default settings, environment parsing, URL validation,
CORS origins, upload extensions, production safeguards, secret redaction,
LLM provider credentials, checkpoint backends, and safe startup diagnostics.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.core.config import Settings, get_settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_settings_kwargs() -> dict[str, Any]:
    """Return baseline settings suitable for isolated configuration tests."""
    return {
        "_env_file": None,
        "APP_ENV": "test",
        "DEBUG": True,
        "SECRET_KEY": "a" * 32,
        "DATABASE_URL": (
            "postgresql+asyncpg://user:password@localhost:5432/agenticbridge"
        ),
        "REDIS_URL": "redis://localhost:6379/0",
        "TRUSTED_HOSTS": ["localhost", "testserver"],
        "WORKFLOW_CHECKPOINT_BACKEND": "memory",
        "DEFAULT_LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-openai-key",
    }


@pytest.fixture
def settings(
    base_settings_kwargs: dict[str, Any],
) -> Settings:
    """Return a valid isolated Settings instance."""
    return Settings(
        **base_settings_kwargs
    )


# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------


def test_default_settings_load():
    settings = get_settings()

    assert (
        settings.APP_NAME == "AgenticBridge"
    )
    assert settings.APP_ENV in {
        "local",
        "development",
        "test",
        "staging",
        "production",
    }
    assert settings.DATABASE_URL.startswith(
        "postgresql"
    )
    assert settings.REDIS_URL.startswith(
        "redis"
    )


def test_get_settings_returns_configured_values(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "APP_NAME", "Custom Bridge"
    )
    monkeypatch.setenv(
        "APP_ENV", "test"
    )

    settings = get_settings()

    assert (
        settings.APP_NAME
        == "Custom Bridge"
    )
    assert settings.APP_ENV == "test"


def test_settings_support_all_supported_environments():
    supported_environments = {
        "local",
        "development",
        "test",
        "staging",
        "production",
    }

    for environment in (
        supported_environments
    ):
        settings = Settings(
            _env_file=None,
            APP_ENV=environment,
        )

        assert (
            settings.APP_ENV
            == environment
        )


def test_settings_reject_unknown_environment():
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            APP_ENV="unknown",
        )


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------


def test_comma_separated_origins_are_parsed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000, https://app.example.com",
    )

    settings = Settings(
        _env_file=None,
        ALLOWED_ORIGINS=(
            "http://localhost:3000, https://app.example.com"
        ),
    )

    assert settings.ALLOWED_ORIGINS == [
        "http://localhost:3000",
        "https://app.example.com",
    ]


def test_allowed_origins_strip_whitespace():
    settings = Settings(
        _env_file=None,
        ALLOWED_ORIGINS=(
            "  http://localhost:3000  ,  https://app.example.com  "
        ),
    )

    assert settings.ALLOWED_ORIGINS == [
        "http://localhost:3000",
        "https://app.example.com",
    ]


def test_allowed_origins_remove_empty_entries():
    settings = Settings(
        _env_file=None,
        ALLOWED_ORIGINS=(
            "http://localhost:3000,,https://app.example.com,"
        ),
    )

    assert settings.ALLOWED_ORIGINS == [
        "http://localhost:3000",
        "https://app.example.com",
    ]


def test_allowed_origins_default_to_empty_list():
    settings = Settings(
        _env_file=None,
        ALLOWED_ORIGINS="",
    )

    assert (
        settings.ALLOWED_ORIGINS
        == []
    )


# ---------------------------------------------------------------------------
# Upload configuration
# ---------------------------------------------------------------------------


def test_upload_extensions_are_normalized():
    settings = Settings(
        _env_file=None,
        ALLOWED_UPLOAD_EXTENSIONS=(
            "PDF, docx, .txt"
        ),
    )

    assert (
        settings.ALLOWED_UPLOAD_EXTENSIONS
        == [".pdf", ".docx", ".txt"]
    )


def test_upload_extensions_remove_duplicates():
    settings = Settings(
        _env_file=None,
        ALLOWED_UPLOAD_EXTENSIONS=(
            ".pdf, PDF, .pdf, docx"
        ),
    )

    assert (
        settings.ALLOWED_UPLOAD_EXTENSIONS
        == [".pdf", ".docx"]
    )


def test_upload_extensions_remove_empty_values():
    settings = Settings(
        _env_file=None,
        ALLOWED_UPLOAD_EXTENSIONS=(
            ".pdf,,docx,"
        ),
    )

    assert (
        settings.ALLOWED_UPLOAD_EXTENSIONS
        == [".pdf", ".docx"]
    )


def test_upload_extensions_default_to_empty_list():
    settings = Settings(
        _env_file=None,
        ALLOWED_UPLOAD_EXTENSIONS="",
    )

    assert (
        settings.ALLOWED_UPLOAD_EXTENSIONS
        == []
    )


# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------


def test_invalid_database_url_is_rejected():
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            DATABASE_URL="mysql://user:password@localhost/db",
        )


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql://user:password@localhost/db",
        "postgresql+asyncpg://user:password@localhost/db",
        "postgresql+psycopg://user:password@localhost/db",
    ],
)
def test_supported_database_urls_are_accepted(
    database_url: str,
):
    settings = Settings(
        _env_file=None,
        DATABASE_URL=database_url,
    )

    assert (
        settings.DATABASE_URL
        == database_url
    )


@pytest.mark.parametrize(
    "invalid_database_url",
    [
        "",
        "mysql://user:password@localhost/db",
        "sqlite:///./local.db",
        "not-a-database-url",
        "http://localhost/db",
    ],
)
def test_unsupported_database_urls_are_rejected(
    invalid_database_url: str,
):
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            DATABASE_URL=invalid_database_url,
        )


def test_database_url_safe_redacts_credentials():
    settings = Settings(
        _env_file=None,
        DATABASE_URL="postgresql+asyncpg://user:password@localhost/db",
    )

    assert (
        settings.database_url_safe
        == "[REDACTED_DATABASE_URL]"
    )


# ---------------------------------------------------------------------------
# Redis configuration
# ---------------------------------------------------------------------------


def test_invalid_redis_url_is_rejected():
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            REDIS_URL="memcached://localhost:11211",
        )


@pytest.mark.parametrize(
    "redis_url",
    [
        "redis://localhost:6379/0",
        "rediss://localhost:6379/0",
        "redis://user:password@localhost:6379/0",
    ],
)
def test_supported_redis_urls_are_accepted(
    redis_url: str,
):
    settings = Settings(
        _env_file=None,
        REDIS_URL=redis_url,
    )

    assert (
        settings.REDIS_URL
        == redis_url
    )


@pytest.mark.parametrize(
    "invalid_redis_url",
    [
        "",
        "memcached://localhost:11211",
        "postgresql://localhost:5432/db",
        "not-a-redis-url",
        "http://localhost:6379",
    ],
)
def test_unsupported_redis_urls_are_rejected(
    invalid_redis_url: str,
):
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            REDIS_URL=invalid_redis_url,
        )


def test_redis_url_safe_redacts_connection_details():
    settings = Settings(
        _env_file=None,
        REDIS_URL="redis://user:password@localhost:6379/0",
    )

    assert (
        settings.redis_url_safe
        == "[REDACTED_REDIS_URL]"
    )


# ---------------------------------------------------------------------------
# Production safeguards
# ---------------------------------------------------------------------------


def test_production_configuration_requires_secret():
    settings = Settings(
        _env_file=None,
        APP_ENV="production",
        DEBUG=False,
        SECRET_KEY="",
        TRUSTED_HOSTS=["app.example.com"],
        WORKFLOW_CHECKPOINT_BACKEND="memory",
        DEFAULT_LLM_PROVIDER="openai",
        OPENAI_API_KEY="test-key",
    )

    with pytest.raises(ValueError):
        settings.validate_production_configuration()


def test_production_configuration_valid():
    settings = Settings(
        _env_file=None,
        APP_ENV="production",
        DEBUG=False,
        SECRET_KEY="a" * 32,
        TRUSTED_HOSTS=["app.example.com"],
        WORKFLOW_CHECKPOINT_BACKEND="memory",
        DEFAULT_LLM_PROVIDER="openai",
        OPENAI_API_KEY="test-key",
    )

    settings.validate_production_configuration()


def test_production_configuration_rejects_debug_mode():
    settings = Settings(
        _env_file=None,
        APP_ENV="production",
        DEBUG=True,
        SECRET_KEY="a" * 32,
        TRUSTED_HOSTS=["app.example.com"],
        WORKFLOW_CHECKPOINT_BACKEND="memory",
        DEFAULT_LLM_PROVIDER="openai",
        OPENAI_API_KEY="test-key",
    )

    with pytest.raises(ValueError):
        settings.validate_production_configuration()


def test_production_configuration_rejects_placeholder_secret():
    settings = Settings(
        _env_file=None,
        APP_ENV="production",
        DEBUG=False,
        SECRET_KEY="changeme",
        TRUSTED_HOSTS=["app.example.com"],
        WORKFLOW_CHECKPOINT_BACKEND="memory",
        DEFAULT_LLM_PROVIDER="openai",
        OPENAI_API_KEY="test-key",
    )

    with pytest.raises(ValueError):
        settings.validate_production_configuration()


def test_production_configuration_rejects_short_secret():
    settings = Settings(
        _env_file=None,
        APP_ENV="production",
        DEBUG=False,
        SECRET_KEY="short",
        TRUSTED_HOSTS=["app.example.com"],
        WORKFLOW_CHECKPOINT_BACKEND="memory",
        DEFAULT_LLM_PROVIDER="openai",
        OPENAI_API_KEY="test-key",
    )

    with pytest.raises(ValueError):
        settings.validate_production_configuration()


def test_production_configuration_rejects_wildcard_trusted_host():
    settings = Settings(
        _env_file=None,
        APP_ENV="production",
        DEBUG=False,
        SECRET_KEY="a" * 32,
        TRUSTED_HOSTS=["*"],
        WORKFLOW_CHECKPOINT_BACKEND="memory",
        DEFAULT_LLM_PROVIDER="openai",
        OPENAI_API_KEY="test-key",
    )

    with pytest.raises(ValueError):
        settings.validate_production_configuration()


def test_production_configuration_rejects_memory_checkpoint_backend():
    settings = Settings(
        _env_file=None,
        APP_ENV="production",
        DEBUG=False,
        SECRET_KEY="a" * 32,
        TRUSTED_HOSTS=["app.example.com"],
        WORKFLOW_CHECKPOINT_BACKEND="memory",
        DEFAULT_LLM_PROVIDER="openai",
        OPENAI_API_KEY="test-key",
    )

    with pytest.raises(ValueError):
        settings.validate_production_configuration()


# ---------------------------------------------------------------------------
# Safe startup diagnostics
# ---------------------------------------------------------------------------


def test_safe_startup_diagnostics_do_not_leak_secrets():
    settings = Settings(
        _env_file=None,
        SECRET_KEY="a" * 32,
        DATABASE_URL=(
            "postgresql+asyncpg://user:password@localhost/db"
        ),
        REDIS_URL="redis://localhost:6379/0",
    )

    diagnostics = (
        settings.get_safe_startup_diagnostics()
    )

    serialized = str(diagnostics)

    assert (
        "password" not in serialized
    )
    assert (
        "postgresql+asyncpg://"
        not in serialized
    )
    assert (
        settings.database_url_safe
        == "[REDACTED_DATABASE_URL]"
    )
    assert (
        settings.redis_url_safe
        == "[REDACTED_REDIS_URL]"
    )


def test_safe_startup_diagnostics_include_app_metadata():
    settings = Settings(
        _env_file=None,
        APP_NAME="AgenticBridge",
        APP_ENV="test",
    )

    diagnostics = (
        settings.get_safe_startup_diagnostics()
    )

    assert (
        diagnostics["app_name"]
        == "AgenticBridge"
    )
    assert (
        diagnostics["app_env"] == "test"
    )


def test_safe_startup_diagnostics_do_not_leak_secret_key():
    settings = Settings(
        _env_file=None,
        SECRET_KEY="a" * 32,
    )

    diagnostics = (
        settings.get_safe_startup_diagnostics()
    )

    serialized = str(diagnostics)

    assert "a" * 32 not in serialized
    assert "secret_key" not in serialized


def test_safe_startup_diagnostics_do_not_leak_llm_api_keys():
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="sk-should-not-leak",
        ANTHROPIC_API_KEY="sk-ant-should-not-leak",
    )

    diagnostics = (
        settings.get_safe_startup_diagnostics()
    )

    serialized = str(diagnostics)

    assert (
        "sk-should-not-leak"
        not in serialized
    )
    assert (
        "sk-ant-should-not-leak"
        not in serialized
    )


# ---------------------------------------------------------------------------
# LLM provider configuration
# ---------------------------------------------------------------------------


def test_get_llm_api_key():
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="openai-key",
    )

    assert (
        settings.get_llm_api_key(
            "openai"
        )
        == "openai-key"
    )

    with pytest.raises(ValueError):
        settings.get_llm_api_key(
            "unknown"
        )


@pytest.mark.parametrize(
    "provider,api_key",
    [
        ("openai", "openai-key"),
        ("anthropic", "anthropic-key"),
    ],
)
def test_get_llm_api_key_returns_provider_specific_key(
    provider: str,
    api_key: str,
):
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY=(
            "openai-key"
            if provider == "openai"
            else ""
        ),
        ANTHROPIC_API_KEY=(
            "anthropic-key"
            if provider == "anthropic"
            else ""
        ),
    )

    assert (
        settings.get_llm_api_key(
            provider
        )
        == api_key
    )


def test_get_llm_api_key_rejects_empty_provider():
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="openai-key",
    )

    with pytest.raises(ValueError):
        settings.get_llm_api_key("")


def test_get_llm_api_key_rejects_none_provider():
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="openai-key",
    )

    with pytest.raises(ValueError):
        settings.get_llm_api_key(None)


def test_get_llm_api_key_rejects_missing_provider_key():
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="",
    )

    with pytest.raises(ValueError):
        settings.get_llm_api_key(
            "openai"
        )


# ---------------------------------------------------------------------------
# Checkpoint backend configuration
# ---------------------------------------------------------------------------


def test_memory_checkpoint_backend_is_supported():
    settings = Settings(
        _env_file=None,
        WORKFLOW_CHECKPOINT_BACKEND="memory",
    )

    assert (
        settings.WORKFLOW_CHECKPOINT_BACKEND
        == "memory"
    )


def test_postgres_checkpoint_backend_is_supported():
    settings = Settings(
        _env_file=None,
        WORKFLOW_CHECKPOINT_BACKEND="postgres",
    )

    assert (
        settings.WORKFLOW_CHECKPOINT_BACKEND
        == "postgres"
    )


def test_unknown_checkpoint_backend_is_rejected():
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            WORKFLOW_CHECKPOINT_BACKEND="unknown",
        )


def test_empty_checkpoint_backend_is_rejected():
    with pytest.raises(ValueError):
        Settings(
            _env_file=None,
            WORKFLOW_CHECKPOINT_BACKEND="",
        )


# ---------------------------------------------------------------------------
# Security-oriented configuration
# ---------------------------------------------------------------------------


def test_settings_do_not_expose_secret_in_repr():
    settings = Settings(
        _env_file=None,
        SECRET_KEY="a" * 32,
    )

    serialized_settings = str(
        settings
    )

    assert "a" * 32 not in (
        serialized_settings
    )


def test_settings_do_not_expose_database_credentials_in_repr():
    settings = Settings(
        _env_file=None,
        DATABASE_URL="postgresql+asyncpg://user:password@localhost/db",
    )

    serialized_settings = str(
        settings
    )

    assert (
        "password@localhost"
        not in serialized_settings
    )


def test_settings_do_not_expose_redis_credentials_in_repr():
    settings = Settings(
        _env_file=None,
        REDIS_URL="redis://user:password@localhost:6379/0",
    )

    serialized_settings = str(
        settings
    )

    assert (
        "password@localhost"
        not in serialized_settings
    )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(
        pytest.main(
            [
                __file__,
                "-v",
            ]
        )
    )
