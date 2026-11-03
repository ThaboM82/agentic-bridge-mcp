"""
Unit tests for database configuration and connection utilities.

These tests verify async database URL normalization, SSL configuration,
pool settings, safe diagnostics, and invalid-configuration handling. They do
not connect to a live database.
"""

from __future__ import annotations

import pytest

from app.core.database import (
    DatabaseNotConfiguredError,
    _build_engine_arguments,
    _get_ssl_mode,
    _is_production_environment,
    _mask_database_url,
    _normalize_async_database_url,
    _validate_database_url,
)


# ---------------------------------------------------------------------------
# Async database URL normalization
# ---------------------------------------------------------------------------


def test_normalizes_postgres_url():
    assert (
        _normalize_async_database_url(
            "postgres://user:password@localhost:5432/db"
        )
        == "postgresql+asyncpg://user:password@localhost:5432/db"
    )


def test_normalizes_postgresql_url():
    assert (
        _normalize_async_database_url(
            "postgresql://user:password@localhost:5432/db"
        )
        == "postgresql+asyncpg://user:password@localhost:5432/db"
    )


def test_preserves_already_async_url():
    url = "postgresql+asyncpg://user:password@localhost:5432/db"

    assert _normalize_async_database_url(url) == url


def test_preserves_query_parameters():
    normalized_url = _normalize_async_database_url(
        "postgresql://user:password@localhost:5432/db"
        "?sslmode=require&connect_timeout=10"
    )

    assert normalized_url == (
        "postgresql+asyncpg://user:password@localhost:5432/db"
        "?sslmode=require&connect_timeout=10"
    )


def test_normalizes_url_with_whitespace():
    assert (
        _normalize_async_database_url(
            "  postgresql://user:password@localhost:5432/db  "
        )
        == "postgresql+asyncpg://user:password@localhost:5432/db"
    )


def test_rejects_non_postgres_url():
    with pytest.raises(DatabaseNotConfiguredError):
        _normalize_async_database_url(
            "mysql://user:password@localhost/db"
        )


def test_rejects_sqlite_url():
    with pytest.raises(DatabaseNotConfiguredError):
        _normalize_async_database_url(
            "sqlite:///./local.db"
        )


def test_rejects_empty_url():
    with pytest.raises(DatabaseNotConfiguredError):
        _normalize_async_database_url("")

    with pytest.raises(DatabaseNotConfiguredError):
        _normalize_async_database_url("   ")


def test_rejects_none_url():
    with pytest.raises(DatabaseNotConfiguredError):
        _normalize_async_database_url(None)


def test_rejects_non_string_url():
    with pytest.raises(DatabaseNotConfiguredError):
        _normalize_async_database_url(12345)


def test_rejects_url_without_credentials():
    with pytest.raises(DatabaseNotConfiguredError):
        _normalize_async_database_url(
            "postgresql://localhost:5432/db"
        )


# ---------------------------------------------------------------------------
# Database URL validation
# ---------------------------------------------------------------------------


def test_validate_database_url_accepts_async_url():
    url = "postgresql+asyncpg://user:password@localhost:5432/db"

    assert _validate_database_url(url) == url


def test_validate_database_url_rejects_sync_postgres_url():
    with pytest.raises(DatabaseNotConfiguredError):
        _validate_database_url(
            "postgresql://user:password@localhost:5432/db"
        )


def test_validate_database_url_rejects_invalid_scheme():
    with pytest.raises(DatabaseNotConfiguredError):
        _validate_database_url(
            "mysql://user:password@localhost:3306/db"
        )


def test_validate_database_url_rejects_empty_url():
    with pytest.raises(DatabaseNotConfiguredError):
        _validate_database_url("")


# ---------------------------------------------------------------------------
# SSL configuration
# ---------------------------------------------------------------------------


def test_ssl_mode_defaults_to_require_in_production():
    assert (
        _get_ssl_mode(
            environment="production",
            configured_ssl_mode=None,
        )
        == "require"
    )


def test_ssl_mode_defaults_to_prefer_outside_production():
    assert (
        _get_ssl_mode(
            environment="local",
            configured_ssl_mode=None,
        )
        == "prefer"
    )

    assert (
        _get_ssl_mode(
            environment="test",
            configured_ssl_mode=None,
        )
        == "prefer"
    )


def test_ssl_mode_honors_explicit_configuration():
    assert (
        _get_ssl_mode(
            environment="production",
            configured_ssl_mode="verify-full",
        )
        == "verify-full"
    )


def test_ssl_mode_rejects_invalid_value():
    with pytest.raises(DatabaseNotConfiguredError):
        _get_ssl_mode(
            environment="production",
            configured_ssl_mode="invalid-mode",
        )


# ---------------------------------------------------------------------------
# Production detection
# ---------------------------------------------------------------------------


def test_production_environment_detection():
    assert _is_production_environment("production") is True
    assert _is_production_environment("PRODUCTION") is True
    assert _is_production_environment("staging") is False
    assert _is_production_environment("local") is False
    assert _is_production_environment(None) is False


# ---------------------------------------------------------------------------
# Connection arguments
# ---------------------------------------------------------------------------


def test_build_engine_arguments_defaults():
    engine_arguments = _build_engine_arguments(
        environment="local",
        pool_size=10,
        max_overflow=20,
        pool_timeout_seconds=30,
        pool_recycle_seconds=1_800,
        database_echo=False,
    )

    assert engine_arguments["pool_size"] == 10
    assert engine_arguments["max_overflow"] == 20
    assert engine_arguments["pool_timeout"] == 30
    assert engine_arguments["pool_recycle"] == 1_800
    assert engine_arguments["echo"] is False
    assert engine_arguments["pool_pre_ping"] is True


def test_build_engine_arguments_enables_ssl_in_production():
    engine_arguments = _build_engine_arguments(
        environment="production",
        pool_size=10,
        max_overflow=20,
        pool_timeout_seconds=30,
        pool_recycle_seconds=1_800,
        database_echo=False,
    )

    connect_args = engine_arguments["connect_args"]

    assert connect_args["ssl"] == "require"


def test_build_engine_arguments_rejects_invalid_pool_size():
    with pytest.raises(DatabaseNotConfiguredError):
        _build_engine_arguments(
            environment="local",
            pool_size=0,
            max_overflow=20,
            pool_timeout_seconds=30,
            pool_recycle_seconds=1_800,
            database_echo=False,
        )


def test_build_engine_arguments_rejects_invalid_overflow():
    with pytest.raises(DatabaseNotConfiguredError):
        _build_engine_arguments(
            environment="local",
            pool_size=10,
            max_overflow=-1,
            pool_timeout_seconds=30,
            pool_recycle_seconds=1_800,
            database_echo=False,
        )


def test_build_engine_arguments_rejects_invalid_timeout():
    with pytest.raises(DatabaseNotConfiguredError):
        _build_engine_arguments(
            environment="local",
            pool_size=10,
            max_overflow=20,
            pool_timeout_seconds=0,
            pool_recycle_seconds=1_800,
            database_echo=False,
        )


# ---------------------------------------------------------------------------
# Safe diagnostics
# ---------------------------------------------------------------------------


def test_mask_database_url_hides_credentials():
    masked_url = _mask_database_url(
        "postgresql+asyncpg://user:password@localhost:5432/db"
    )

    assert masked_url == "postgresql+asyncpg://user:[REDACTED]@localhost:5432/db"
    assert "password" not in masked_url


def test_mask_database_url_handles_invalid_url():
    assert (
        _mask_database_url("not-a-database-url")
        == "[REDACTED_DATABASE_URL]"
    )

    assert (
        _mask_database_url("")
        == "[REDACTED_DATABASE_URL]"
    )

    assert (
        _mask_database_url(None)
        == "[REDACTED_DATABASE_URL]"
    )


def test_mask_database_url_does_not_leak_query_credentials():
    masked_url = _mask_database_url(
        "postgresql+asyncpg://user:password@localhost:5432/db"
        "?sslmode=require&password=another-secret"
    )

    assert "password" not in masked_url
    assert "another-secret" not in masked_url
