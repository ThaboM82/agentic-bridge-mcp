"""
LangGraph checkpoint persistence for workflow execution.

This module provides a narrow factory for workflow checkpointers. It selects
the configured backend, validates configuration, and returns a LangGraph
checkpointer without exposing credentials or connection details.

Supported backends:
- `memory`: In-memory checkpointer for local development and tests.
- `postgres`: PostgreSQL-backed checkpointer for durable production execution.

Security and operational rules:
- Never log database URLs, credentials, connection strings, or checkpoint
  payloads.
- Treat checkpoint state as sensitive because it can contain workflow context.
- Use PostgreSQL persistence for production and multi-worker deployments.
- Use the memory backend only for local development, tests, or ephemeral jobs.
- Keep checkpoint TTL and cleanup policy in infrastructure or a dedicated
  maintenance job, not inside workflow nodes.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import settings
from app.core.exceptions import ConfigurationError


logger = logging.getLogger("agentic_bridge.graph.checkpoints")

CheckpointBackend = Literal["memory", "postgres"]

SUPPORTED_CHECKPOINT_BACKENDS: tuple[str, ...] = (
    "memory",
    "postgres",
)

DEFAULT_CHECKPOINT_BACKEND: CheckpointBackend = "memory"


def _normalize_backend(
    backend: str | None,
) -> CheckpointBackend:
    """
    Validate and normalize a checkpoint backend name.

    Args:
        backend: Backend name supplied by configuration or caller.

    Returns:
        A supported backend identifier.

    Raises:
        ConfigurationError: If the backend is unsupported.
    """
    if not backend or not backend.strip():
        return DEFAULT_CHECKPOINT_BACKEND

    normalized_backend = backend.strip().lower()

    if normalized_backend not in SUPPORTED_CHECKPOINT_BACKENDS:
        raise ConfigurationError(
            f"Unsupported workflow checkpoint backend '{normalized_backend}'. "
            f"Supported backends: {', '.join(SUPPORTED_CHECKPOINT_BACKENDS)}."
        )

    return normalized_backend  # type: ignore[return-value]


def _get_configured_backend() -> CheckpointBackend:
    """
    Read the configured checkpoint backend.

    Returns:
        The configured backend identifier.

    Raises:
        ConfigurationError: If configured backend is unsupported.
    """
    configured_backend = getattr(
        settings,
        "WORKFLOW_CHECKPOINT_BACKEND",
        None,
    )

    return _normalize_backend(configured_backend)


def _get_postgres_connection_kwargs() -> dict[str, Any]:
    """
    Build connection arguments for the PostgreSQL checkpointer.

    Returns:
        Connection keyword arguments. The connection string is intentionally
        excluded from logs.

    Raises:
        ConfigurationError: If PostgreSQL checkpoint configuration is missing.
    """
    database_url = getattr(
        settings,
        "WORKFLOW_CHECKPOINT_DATABASE_URL",
        None,
    )

    if not database_url or not str(database_url).strip():
        raise ConfigurationError(
            "PostgreSQL workflow checkpointing requires "
            "WORKFLOW_CHECKPOINT_DATABASE_URL."
        )

    return {
        "dsn": str(database_url).strip(),
    }


def create_checkpointer(
    *,
    backend: str | None = None,
) -> BaseCheckpointSaver:
    """
    Create a LangGraph checkpointer for workflow execution.

    Args:
        backend: Optional explicit backend override. When omitted, the backend
            is read from application settings.

    Returns:
        A configured LangGraph checkpointer.

    Raises:
        ConfigurationError: If the requested backend is unsupported or its
            configuration is incomplete.
    """
    selected_backend = _normalize_backend(
        backend or _get_configured_backend()
    )

    if selected_backend == "memory":
        logger.info(
            "workflow_checkpointer_created",
            backend="memory",
        )

        return InMemorySaver()

    if selected_backend == "postgres":
        try:
            from langgraph.checkpoint.postgres.aio import (
                AsyncPostgresSaver,
            )
        except ImportError as exc:
            raise ConfigurationError(
                "PostgreSQL workflow checkpointing requires the "
                "langgraph-checkpoint-postgres package."
            ) from exc

        connection_kwargs = _get_postgres_connection_kwargs()

        logger.info(
            "workflow_checkpointer_created",
            backend="postgres",
        )

        return AsyncPostgresSaver.from_conn_string(
            connection_kwargs["dsn"]
        )

    raise ConfigurationError(
        f"Unsupported workflow checkpoint backend '{selected_backend}'."
    )


def create_memory_checkpointer() -> InMemorySaver:
    """
    Create an in-memory checkpointer explicitly.

    Use this only for local development, unit tests, or ephemeral execution.
    In-memory checkpoints are lost on process restart and are not shared across
    workers.
    """
    logger.info(
        "workflow_memory_checkpointer_created",
        backend="memory",
    )

    return InMemorySaver()


def create_postgres_checkpointer() -> Any:
    """
    Create a PostgreSQL-backed async checkpointer explicitly.

    Returns:
        An async PostgreSQL checkpointer.

    Raises:
        ConfigurationError: If PostgreSQL checkpoint configuration is missing
            or the required package is not installed.
    """
    return create_checkpointer(backend="postgres")


def get_checkpoint_backend_name(
    *,
    backend: str | None = None,
) -> str:
    """
    Return the checkpoint backend name for logging and health checks.

    Args:
        backend: Optional explicit backend override.

    Returns:
        A supported backend name.

    Raises:
        ConfigurationError: If the backend is unsupported.
    """
    return _normalize_backend(
        backend or _get_configured_backend()
    )