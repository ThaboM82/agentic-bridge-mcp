"""
Async SQLAlchemy database engine, session management, and transaction helpers.

Architecture:
- `engine` owns connection pooling.
- `AsyncSessionLocal` creates request-scoped sessions.
- `get_db()` yields a session without committing automatically.
- Services explicitly commit when completing a successful unit of work.
- `get_transactional_session()` provides an all-or-nothing transaction for
  background jobs and batch operations.

Security rules:
- Never log or expose the full database URL, password, query string, or
  connection string.
- Redact credentials in diagnostics and error messages.
- Require PostgreSQL with the asyncpg driver.
- Use TLS in production unless an operator explicitly configures another SSL
  mode.
- Keep pool sizes bounded to prevent connection exhaustion.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


logger = logging.getLogger("agentic_bridge.database")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_DATABASE_SCHEMES: Final[tuple[str, ...]] = (
    "postgres",
    "postgresql",
    "postgresql+asyncpg",
)

ASYNC_POSTGRES_SCHEME: Final[str] = "postgresql+asyncpg"

SUPPORTED_SSL_MODES: Final[tuple[str, ...]] = (
    "disable",
    "allow",
    "prefer",
    "require",
    "verify-ca",
    "verify-full",
)

DEFAULT_SSL_MODE_PRODUCTION: Final[str] = "require"
DEFAULT_SSL_MODE_NON_PRODUCTION: Final[str] = "prefer"

MIN_POOL_SIZE: Final[int] = 1
MAX_POOL_SIZE: Final[int] = 100

MIN_MAX_OVERFLOW: Final[int] = 0
MAX_MAX_OVERFLOW: Final[int] = 200

MIN_POOL_TIMEOUT_SECONDS: Final[int] = 1
MAX_POOL_TIMEOUT_SECONDS: Final[int] = 300

MIN_POOL_RECYCLE_SECONDS: Final[int] = 60
MAX_POOL_RECYCLE_SECONDS: Final[int] = 86_400

SENSITIVE_QUERY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "sslcert",
        "sslkey",
        "sslrootcert",
    }
)

REDACTED_CREDENTIAL = "[REDACTED]"
REDACTED_DATABASE_URL = "[REDACTED_DATABASE_URL]"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DatabaseNotConfiguredError(RuntimeError):
    """Raised when a valid asynchronous database URL cannot be resolved."""


# ---------------------------------------------------------------------------
# URL normalization and validation
# ---------------------------------------------------------------------------


def _normalize_async_database_url(
    raw_url: str | None,
) -> str:
    """
    Normalize common PostgreSQL URL formats for SQLAlchemy asyncpg.

    Supported inputs:
    - postgresql+asyncpg://user:password@host:port/database
    - postgresql://user:password@host:port/database
    - postgres://user:password@host:port/database

    Render, Heroku, Supabase, and many hosted PostgreSQL providers may return
    `postgres://` or `postgresql://`; SQLAlchemy asyncpg requires the
    `postgresql+asyncpg://` dialect.

    Args:
        raw_url: Database URL supplied by configuration.

    Returns:
        A normalized async PostgreSQL database URL.

    Raises:
        DatabaseNotConfiguredError: If the URL is missing, malformed, or uses
            an unsupported database scheme.
    """
    if raw_url is None or not isinstance(raw_url, str):
        raise DatabaseNotConfiguredError(
            "DATABASE_URL must be a PostgreSQL URL compatible with asyncpg."
        )

    url = raw_url.strip()

    if not url:
        raise DatabaseNotConfiguredError(
            "DATABASE_URL must not be empty."
        )

    try:
        parsed_url = urlsplit(url)
    except ValueError as exc:
        raise DatabaseNotConfiguredError(
            "DATABASE_URL is malformed."
        ) from exc

    scheme = parsed_url.scheme.strip().lower()

    if scheme not in SUPPORTED_DATABASE_SCHEMES:
        raise DatabaseNotConfiguredError(
            "DATABASE_URL must use a PostgreSQL URL compatible with asyncpg."
        )

    if not parsed_url.netloc:
        raise DatabaseNotConfiguredError(
            "DATABASE_URL must include a database host."
        )

    if not parsed_url.username or not parsed_url.password:
        raise DatabaseNotConfiguredError(
            "DATABASE_URL must include database credentials."
        )

    if scheme == ASYNC_POSTGRES_SCHEME:
        return url

    # Preserve the path and query string while replacing only the scheme.
    normalized_url = urlunsplit(
        (
            ASYNC_POSTGRES_SCHEME,
            parsed_url.netloc,
            parsed_url.path,
            parsed_url.query,
            parsed_url.fragment,
        )
    )

    return normalized_url


def _validate_database_url(
    database_url: str | None,
) -> str:
    """
    Validate that a database URL uses the async PostgreSQL dialect.

    Args:
        database_url: Database URL to validate.

    Returns:
        The validated database URL.

    Raises:
        DatabaseNotConfiguredError: If the URL does not use async PostgreSQL.
    """
    normalized_url = _normalize_async_database_url(database_url)

    if not normalized_url.startswith(
        f"{ASYNC_POSTGRES_SCHEME}://"
    ):
        raise DatabaseNotConfiguredError(
            "DATABASE_URL must use the postgresql+asyncpg dialect."
        )

    return normalized_url


def _mask_database_url(
    database_url: str | None,
) -> str:
    """
    Return a database URL safe for diagnostics.

    Credentials and sensitive query parameters are redacted. The full URL is
    never returned or logged.

    Args:
        database_url: Database URL to mask.

    Returns:
        A redacted database URL.
    """
    if not database_url or not isinstance(database_url, str):
        return REDACTED_DATABASE_URL

    try:
        parsed_url = urlsplit(database_url.strip())
    except ValueError:
        return REDACTED_DATABASE_URL

    if not parsed_url.netloc:
        return REDACTED_DATABASE_URL

    try:
        safe_query_pairs = [
            (key, REDACTED_CREDENTIAL)
            if key.lower() in SENSITIVE_QUERY_KEYS
            else (key, value)
            for key, value in parse_qsl(
                parsed_url.query,
                keep_blank_values=True,
            )
        ]
    except ValueError:
        safe_query_pairs = []

    safe_query = urlencode(safe_query_pairs)

    if parsed_url.username and parsed_url.password:
        netloc = (
            f"{parsed_url.username}:{REDACTED_CREDENTIAL}"
            f"@{parsed_url.hostname or ''}"
        )

        if parsed_url.port:
            netloc = f"{netloc}:{parsed_url.port}"
    else:
        netloc = parsed_url.netloc

    return urlunsplit(
        (
            parsed_url.scheme,
            netloc,
            parsed_url.path,
            safe_query,
            "",
        )
    )


# ---------------------------------------------------------------------------
# SSL and engine configuration
# ---------------------------------------------------------------------------


def _is_production_environment(
    environment: str | None,
) -> bool:
    """
    Return whether an environment name represents production.

    Args:
        environment: Environment name.

    Returns:
        `True` when the environment is production.
    """
    if not environment or not isinstance(environment, str):
        return False

    return environment.strip().lower() == "production"


def _get_ssl_mode(
    *,
    environment: str | None,
    configured_ssl_mode: str | None,
) -> str:
    """
    Resolve the PostgreSQL SSL mode.

    Args:
        environment: Deployment environment.
        configured_ssl_mode: Explicit SSL mode from configuration.

    Returns:
        A supported PostgreSQL SSL mode.

    Raises:
        DatabaseNotConfiguredError: If the configured SSL mode is unsupported.
    """
    if configured_ssl_mode:
        normalized_ssl_mode = configured_ssl_mode.strip().lower()

        if normalized_ssl_mode not in SUPPORTED_SSL_MODES:
            raise DatabaseNotConfiguredError(
                "DATABASE_SSL_MODE must be one of: "
                + ", ".join(SUPPORTED_SSL_MODES)
            )

        return normalized_ssl_mode

    if _is_production_environment(environment):
        return DEFAULT_SSL_MODE_PRODUCTION

    return DEFAULT_SSL_MODE_NON_PRODUCTION


def _build_engine_arguments(
    *,
    environment: str | None,
    pool_size: int,
    max_overflow: int,
    pool_timeout_seconds: int,
    pool_recycle_seconds: int,
    database_echo: bool,
    database_ssl_mode: str | None = None,
) -> dict[str, Any]:
    """
    Build bounded SQLAlchemy async-engine arguments.

    Args:
        environment: Deployment environment.
        pool_size: Base connection-pool size.
        max_overflow: Maximum temporary connections above the pool size.
        pool_timeout_seconds: Seconds to wait for a pooled connection.
        pool_recycle_seconds: Seconds before a connection is recycled.
        database_echo: Whether to emit SQL debugging output.
        database_ssl_mode: Optional explicit PostgreSQL SSL mode.

    Returns:
        Keyword arguments for `create_async_engine`.

    Raises:
        DatabaseNotConfiguredError: If pool settings are invalid.
    """
    if not isinstance(pool_size, int) or isinstance(
        pool_size,
        bool,
    ):
        raise DatabaseNotConfiguredError(
            "Database pool size must be an integer."
        )

    if not MIN_POOL_SIZE <= pool_size <= MAX_POOL_SIZE:
        raise DatabaseNotConfiguredError(
            f"Database pool size must be between {MIN_POOL_SIZE} "
            f"and {MAX_POOL_SIZE}."
        )

    if not isinstance(max_overflow, int) or isinstance(
        max_overflow,
        bool,
    ):
        raise DatabaseNotConfiguredError(
            "Database max overflow must be an integer."
        )

    if not MIN_MAX_OVERFLOW <= max_overflow <= MAX_MAX_OVERFLOW:
        raise DatabaseNotConfiguredError(
            f"Database max overflow must be between {MIN_MAX_OVERFLOW} "
            f"and {MAX_MAX_OVERFLOW}."
        )

    if not isinstance(
        pool_timeout_seconds,
        int,
    ) or isinstance(pool_timeout_seconds, bool):
        raise DatabaseNotConfiguredError(
            "Database pool timeout must be an integer."
        )

    if not (
        MIN_POOL_TIMEOUT_SECONDS
        <= pool_timeout_seconds
        <= MAX_POOL_TIMEOUT_SECONDS
    ):
        raise DatabaseNotConfiguredError(
            f"Database pool timeout must be between "
            f"{MIN_POOL_TIMEOUT_SECONDS} and "
            f"{MAX_POOL_TIMEOUT_SECONDS} seconds."
        )

    if not isinstance(
        pool_recycle_seconds,
        int,
    ) or isinstance(pool_recycle_seconds, bool):
        raise DatabaseNotConfiguredError(
            "Database pool recycle interval must be an integer."
        )

    if not (
        MIN_POOL_RECYCLE_SECONDS
        <= pool_recycle_seconds
        <= MAX_POOL_RECYCLE_SECONDS
    ):
        raise DatabaseNotConfiguredError(
            f"Database pool recycle interval must be between "
            f"{MIN_POOL_RECYCLE_SECONDS} and "
            f"{MAX_POOL_RECYCLE_SECONDS} seconds."
        )

    ssl_mode = _get_ssl_mode(
        environment=environment,
        configured_ssl_mode=database_ssl_mode,
    )

    return {
        "echo": bool(database_echo),
        "future": True,
        "pool_pre_ping": True,
        "pool_size": pool_size,
        "max_overflow": max_overflow,
        "pool_timeout": pool_timeout_seconds,
        "pool_recycle": pool_recycle_seconds,
        "connect_args": {
            # Prevent asyncpg from holding server-side prepared statements in
            # ways that can conflict with transaction-pooling proxies such as
            # PgBouncer.
            "statement_cache_size": 0,
            "ssl": ssl_mode,
        },
    }


def _resolve_database_url() -> str:
    """
    Resolve the database URL from explicit `DATABASE_URL` or `POSTGRES_*`
    settings.

    Returns:
        A normalized async PostgreSQL database URL.

    Raises:
        DatabaseNotConfiguredError: If no valid database URL can be resolved.
    """
    if getattr(settings, "DATABASE_URL", None):
        return _normalize_async_database_url(
            str(settings.DATABASE_URL)
        )

    postgres_user = getattr(settings, "POSTGRES_USER", None)
    postgres_password = getattr(
        settings,
        "POSTGRES_PASSWORD",
        None,
    )
    postgres_host = getattr(settings, "POSTGRES_HOST", None)
    postgres_port = getattr(settings, "POSTGRES_PORT", None)
    postgres_db = getattr(settings, "POSTGRES_DB", None)

    if all(
        (
            postgres_user,
            postgres_password,
            postgres_host,
            postgres_port,
            postgres_db,
        )
    ):
        assembled_url = (
            f"postgresql://{postgres_user}:{postgres_password}"
            f"@{postgres_host}:{postgres_port}/{postgres_db}"
        )

        return _normalize_async_database_url(assembled_url)

    fallback_url = getattr(
        settings,
        "SQLALCHEMY_DATABASE_URI",
        None,
    )

    if fallback_url:
        return _normalize_async_database_url(
            str(fallback_url)
        )

    raise DatabaseNotConfiguredError(
        "DATABASE_URL is not configured."
    )


# ---------------------------------------------------------------------------
# Engine and session factory
# ---------------------------------------------------------------------------


DATABASE_URL: Final[str] = _resolve_database_url()

_engine_arguments = _build_engine_arguments(
    environment=getattr(settings, "ENVIRONMENT", None),
    pool_size=getattr(settings, "DB_POOL_SIZE", 10),
    max_overflow=getattr(settings, "DB_MAX_OVERFLOW", 20),
    pool_timeout_seconds=getattr(
        settings,
        "DB_POOL_TIMEOUT_SECONDS",
        30,
    ),
    pool_recycle_seconds=getattr(
        settings,
        "DB_POOL_RECYCLE_SECONDS",
        1_800,
    ),
    database_echo=getattr(settings, "DB_ECHO", False),
    database_ssl_mode=getattr(
        settings,
        "DATABASE_SSL_MODE",
        None,
    ),
)

engine = create_async_engine(
    DATABASE_URL,
    **_engine_arguments,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    """Declarative base for all SQLAlchemy ORM models."""


# ---------------------------------------------------------------------------
# Session dependencies
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield a request-scoped database session.

    This dependency does not commit automatically. Services should call
    `await session.commit()` after a successful unit of work.

    If an exception occurs, the session is rolled back before being closed.

    Yields:
        An async SQLAlchemy session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_transactional_session() -> AsyncGenerator[
    AsyncSession,
    None,
]:
    """
    Provide a session with an explicit all-or-nothing transaction.

    Use this for background jobs, batch ingestion, and operations that must
    either fully succeed or fully roll back.

    Yields:
        An async SQLAlchemy session.

    Raises:
        SQLAlchemyError: If the transaction cannot be committed.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()

            logger.exception(
                "database_transaction_rolled_back",
            )

            raise


# ---------------------------------------------------------------------------
# Health checks and lifecycle
# ---------------------------------------------------------------------------


async def check_database_connection() -> bool:
    """
    Execute a lightweight connectivity check.

    Used by the `/ready` endpoint and container health checks.

    Returns:
        `True` when the database accepts a simple query; otherwise `False`.
    """
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

        return True

    except SQLAlchemyError:
        logger.exception(
            "database_connectivity_check_failed",
        )

        return False


async def dispose_engine() -> None:
    """Dispose all pooled database connections during shutdown."""
    await engine.dispose()

    logger.info(
        "database_engine_disposed",
    )


# ---------------------------------------------------------------------------
# Safe diagnostics
# ---------------------------------------------------------------------------


def get_database_metadata() -> dict[str, Any]:
    """
    Return safe, non-secret database metadata for diagnostics.

    Never return or log the full database URL, password, or connection string.

    Returns:
        A dictionary of non-sensitive database diagnostics.
    """
    pool_status = getattr(engine.pool, "status", None)

    return {
        "database_scheme": ASYNC_POSTGRES_SCHEME,
        "database_url": _mask_database_url(DATABASE_URL),
        "environment": getattr(
            settings,
            "ENVIRONMENT",
            None,
        ),
        "pool_size": _engine_arguments["pool_size"],
        "max_overflow": _engine_arguments["max_overflow"],
        "pool_timeout_seconds": _engine_arguments[
            "pool_timeout"
        ],
        "pool_recycle_seconds": _engine_arguments[
            "pool_recycle"
        ],
        "pool_pre_ping": _engine_arguments[
            "pool_pre_ping"
        ],
        "echo": _engine_arguments["echo"],
        "ssl_mode": _engine_arguments["connect_args"][
            "ssl"
        ],
        "pool_status": (
            pool_status() if callable(pool_status) else None
        ),
    }


def get_database_url_masked() -> str:
    """
    Return a redacted database URL for logs and health diagnostics.

    Returns:
        A database URL with credentials and sensitive query values redacted.
    """
    return _mask_database_url(DATABASE_URL)