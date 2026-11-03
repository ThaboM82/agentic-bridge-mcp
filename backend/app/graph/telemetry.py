"""
Workflow telemetry for AgenticBridge LangGraph executions.

This module provides safe, bounded telemetry helpers for workflow lifecycle,
node execution, provider usage, recovery, human review, and publication events.
It is designed to work with structured logging and can be extended to OpenTelemetry
or Prometheus without changing workflow nodes.

Security rules:
- Never log prompts, completions, generated code, document contents,
  embeddings, credentials, database URLs, API keys, or customer PII.
- Treat telemetry payloads and workflow state as untrusted input.
- Redact sensitive-looking keys and values.
- Bound all strings, lists, and metadata maps.
- Telemetry failures must never break workflow execution.
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, AsyncIterator, Literal
from uuid import UUID, uuid4

from app.core.utils import truncate_text


logger = logging.getLogger("agentic_bridge.graph.telemetry")

MAX_EVENT_NAME_CHARS = 100
MAX_FIELD_CHARS = 500
MAX_METADATA_KEYS = 50
MAX_METADATA_VALUE_CHARS = 1_000
MAX_TAG_COUNT = 20
MAX_TAG_LENGTH = 100
MAX_LIST_ITEMS = 20

SENSITIVE_KEY_FRAGMENTS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "cookie",
    "session",
    "private_key",
    "access_key",
    "refresh_token",
    "signed_url",
    "storage_url",
    "file_url",
    "download_url",
    "prompt",
    "completion",
    "messages",
    "input",
    "output",
    "embedding",
    "vector",
    "generated_code",
    "database_url",
    "connection_string",
)

SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(?:sk|pk)_-[A-Za-z0-9_-]{20,}"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?token|"
        r"refresh[_-]?token|password)\s*[=:]\s*\S+"
    ),
    re.compile(
        r"(?i)postgresql(?:\+asyncpg)?://[^\s\"']+:[^\s\"']+@"
    ),
    re.compile(
        r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/-]+=*"
    ),
)

REDACTED_VALUE = "[REDACTED]"
UNSERIALIZABLE_VALUE = "[UNSERIALIZABLE]"

TelemetryLevel = Literal[
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
]


# ---------------------------------------------------------------------------
# Time and identity helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _coerce_id(
    value: str | UUID | None,
) -> str | None:
    """Validate and normalize an identifier-like telemetry value."""
    if value is None:
        return None

    if isinstance(value, UUID):
        return str(value)

    if not isinstance(value, str):
        return None

    normalized_value = value.strip()

    return normalized_value or None


def generate_telemetry_event_id() -> str:
    """Generate a non-sensitive telemetry correlation ID."""
    return f"telemetry_{uuid4()}"


# ---------------------------------------------------------------------------
# Payload sanitization
# ---------------------------------------------------------------------------


def _redact_string(value: str) -> str:
    """Redact likely secrets from an untrusted string."""
    redacted_value = value

    for pattern in SECRET_VALUE_PATTERNS:
        redacted_value = pattern.sub(
            REDACTED_VALUE,
            redacted_value,
        )

    return redacted_value


def sanitize_telemetry_value(
    value: Any,
    *,
    max_length: int = MAX_METADATA_VALUE_CHARS,
) -> Any:
    """
    Sanitize and bound a telemetry value.

    Args:
        value: Arbitrary telemetry payload value.
        max_length: Maximum string length.

    Returns:
        A safe, bounded representation of the value.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return truncate_text(
            _redact_string(value),
            max_length,
        )

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, dict):
        return sanitize_telemetry_metadata(value)

    if isinstance(value, (list, tuple, set)):
        return [
            sanitize_telemetry_value(
                item,
                max_length=min(
                    max_length,
                    MAX_FIELD_CHARS,
                ),
            )
            for item in list(value)[:MAX_LIST_ITEMS]
        ]

    return UNSERIALIZABLE_VALUE


def sanitize_telemetry_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Sanitize and bound telemetry metadata.

    Args:
        metadata: Arbitrary metadata supplied by workflow code.

    Returns:
        A bounded metadata dictionary with sensitive values redacted.
    """
    if not metadata or not isinstance(metadata, dict):
        return {}

    safe_metadata: dict[str, Any] = {}

    for key, value in list(metadata.items())[:MAX_METADATA_KEYS]:
        if not isinstance(key, str):
            continue

        normalized_key = key.strip()

        if not normalized_key:
            continue

        key_lower = normalized_key.lower()

        if any(
            fragment in key_lower
            for fragment in SENSITIVE_KEY_FRAGMENTS
        ):
            safe_metadata[normalized_key] = REDACTED_VALUE
            continue

        safe_metadata[normalized_key] = sanitize_telemetry_value(
            value
        )

    return safe_metadata


def sanitize_event_name(
    event_name: str | None,
) -> str:
    """
    Validate and bound a telemetry event name.

    Args:
        event_name: Event name supplied by workflow code.

    Returns:
        A bounded event name.

    Raises:
        ValueError: If the event name is missing or malformed.
    """
    if not event_name or not isinstance(event_name, str):
        raise ValueError("Telemetry event name is required.")

    normalized_event_name = event_name.strip().lower()

    if not normalized_event_name:
        raise ValueError("Telemetry event name is required.")

    if len(normalized_event_name) > MAX_EVENT_NAME_CHARS:
        raise ValueError(
            "Telemetry event name exceeds the allowed length."
        )

    return normalized_event_name


def build_telemetry_tags(
    *,
    execution_id: str | UUID | None = None,
    tenant_id: str | UUID | None = None,
    workflow_step: str | None = None,
    workflow_name: str | None = None,
) -> list[str]:
    """
    Build bounded, non-sensitive telemetry tags.

    Args:
        execution_id: Workflow execution identifier.
        tenant_id: Tenant identifier.
        workflow_step: Current workflow step.
        workflow_name: Workflow name.

    Returns:
        A bounded list of telemetry tags.
    """
    tags: list[str] = []

    tag_values = {
        "execution": execution_id,
        "tenant": tenant_id,
        "step": workflow_step,
        "workflow": workflow_name,
    }

    for prefix, value in tag_values.items():
        normalized_value = _coerce_id(value)

        if normalized_value:
            tags.append(
                f"{prefix}:{truncate_text(normalized_value, MAX_TAG_LENGTH)}"
            )

    return tags[:MAX_TAG_COUNT]


# ---------------------------------------------------------------------------
# Core event emission
# ---------------------------------------------------------------------------


def emit_event(
    event_name: str,
    *,
    level: TelemetryLevel = "INFO",
    execution_id: str | UUID | None = None,
    tenant_id: str | UUID | None = None,
    user_id: str | UUID | None = None,
    workflow_step: str | None = None,
    workflow_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Emit a structured, sanitized workflow telemetry event.

    Args:
        event_name: Stable event name, such as `workflow_started`.
        level: Log level.
        execution_id: Workflow execution identifier.
        tenant_id: Tenant identifier.
        user_id: User identifier.
        workflow_step: Current workflow step.
        workflow_name: Workflow name.
        metadata: Optional non-sensitive event metadata.

    Returns:
        The emitted telemetry event ID.

    Raises:
        ValueError: If the event name is invalid.
    """
    normalized_event_name = sanitize_event_name(event_name)

    event_id = generate_telemetry_event_id()

    safe_metadata = sanitize_telemetry_metadata(
        {
            **(metadata or {}),
            "event_id": event_id,
            "execution_id": _coerce_id(execution_id),
            "tenant_id": _coerce_id(tenant_id),
            "user_id": _coerce_id(user_id),
            "workflow_step": workflow_step,
            "workflow_name": workflow_name,
            "timestamp": _utc_now_iso(),
        }
    )

    log_method = {
        "DEBUG": logger.debug,
        "INFO": logger.info,
        "WARNING": logger.warning,
        "ERROR": logger.error,
    }[level]

    try:
        log_method(
            normalized_event_name,
            **safe_metadata,
        )
    except Exception:
        # Observability must never break workflow execution.
        logger.exception(
            "workflow_telemetry_event_failed",
            event_id=event_id,
            event_name=normalized_event_name,
        )

    return event_id


# ---------------------------------------------------------------------------
# Workflow lifecycle events
# ---------------------------------------------------------------------------


def emit_workflow_started(
    *,
    execution_id: str | UUID,
    brief_id: str | UUID,
    tenant_id: str | UUID | None = None,
    user_id: str | UUID | None = None,
    workflow_name: str | None = None,
    workflow_version: str | None = None,
) -> str:
    """Emit a workflow-started event."""
    return emit_event(
        "workflow_started",
        execution_id=execution_id,
        tenant_id=tenant_id,
        user_id=user_id,
        workflow_name=workflow_name,
        metadata={
            "brief_id": _coerce_id(brief_id),
            "workflow_version": workflow_version,
        },
    )


def emit_workflow_completed(
    *,
    execution_id: str | UUID,
    brief_id: str | UUID,
    tenant_id: str | UUID | None = None,
    user_id: str | UUID | None = None,
    workflow_name: str | None = None,
    duration_ms: float | None = None,
    token_usage: dict[str, int] | None = None,
    estimated_cost_usd: float | None = None,
) -> str:
    """Emit a workflow-completed event."""
    return emit_event(
        "workflow_completed",
        execution_id=execution_id,
        tenant_id=tenant_id,
        user_id=user_id,
        workflow_name=workflow_name,
        metadata={
            "brief_id": _coerce_id(brief_id),
            "duration_ms": duration_ms,
            "token_usage": token_usage or {},
            "estimated_cost_usd": estimated_cost_usd,
        },
    )


def emit_workflow_failed(
    *,
    execution_id: str | UUID,
    brief_id: str | UUID,
    tenant_id: str | UUID | None = None,
    user_id: str | UUID | None = None,
    workflow_name: str | None = None,
    error_code: str | None = None,
    error_step: str | None = None,
    recoverable: bool = False,
) -> str:
    """Emit a workflow-failed event without exposing error details."""
    return emit_event(
        "workflow_failed",
        level="ERROR",
        execution_id=execution_id,
        tenant_id=tenant_id,
        user_id=user_id,
        workflow_name=workflow_name,
        metadata={
            "brief_id": _coerce_id(brief_id),
            "error_code": error_code,
            "error_step": error_step,
            "recoverable": recoverable,
        },
    )


def emit_workflow_cancelled(
    *,
    execution_id: str | UUID,
    brief_id: str | UUID,
    tenant_id: str | UUID | None = None,
    user_id: str | UUID | None = None,
    workflow_name: str | None = None,
    cancelled_by: str | UUID | None = None,
) -> str:
    """Emit a workflow-cancelled event."""
    return emit_event(
        "workflow_cancelled",
        level="WARNING",
        execution_id=execution_id,
        tenant_id=tenant_id,
        user_id=user_id,
        workflow_name=workflow_name,
        metadata={
            "brief_id": _coerce_id(brief_id),
            "cancelled_by": _coerce_id(cancelled_by),
        },
    )


# ---------------------------------------------------------------------------
# Node and provider events
# ---------------------------------------------------------------------------


def emit_node_started(
    *,
    execution_id: str | UUID,
    node_name: str,
    tenant_id: str | UUID | None = None,
    workflow_step: str | None = None,
) -> str:
    """Emit a node-started event."""
    return emit_event(
        "workflow_node_started",
        execution_id=execution_id,
        tenant_id=tenant_id,
        workflow_step=workflow_step,
        metadata={
            "node_name": node_name,
        },
    )


def emit_node_completed(
    *,
    execution_id: str | UUID,
    node_name: str,
    duration_ms: float,
    tenant_id: str | UUID | None = None,
    workflow_step: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Emit a node-completed event."""
    return emit_event(
        "workflow_node_completed",
        execution_id=execution_id,
        tenant_id=tenant_id,
        workflow_step=workflow_step,
        metadata={
            "node_name": node_name,
            "duration_ms": duration_ms,
            **(metadata or {}),
        },
    )


def emit_node_failed(
    *,
    execution_id: str | UUID,
    node_name: str,
    duration_ms: float,
    error_code: str | None = None,
    recoverable: bool = False,
    tenant_id: str | UUID | None = None,
    workflow_step: str | None = None,
) -> str:
    """Emit a node-failed event without exposing exception details."""
    return emit_event(
        "workflow_node_failed",
        level="ERROR",
        execution_id=execution_id,
        tenant_id=tenant_id,
        workflow_step=workflow_step,
        metadata={
            "node_name": node_name,
            "duration_ms": duration_ms,
            "error_code": error_code,
            "recoverable": recoverable,
        },
    )


def emit_provider_usage(
    *,
    execution_id: str | UUID,
    provider_name: str,
    provider_model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
    latency_ms: float | None = None,
    estimated_cost_usd: float | None = None,
    tenant_id: str | UUID | None = None,
    workflow_step: str | None = None,
) -> str:
    """
    Emit non-sensitive language-model usage telemetry.

    Args:
        execution_id: Workflow execution identifier.
        provider_name: Provider name.
        provider_model: Provider model identifier.
        prompt_tokens: Prompt tokens used.
        completion_tokens: Completion tokens used.
        total_tokens: Total tokens used, if supplied by the provider.
        latency_ms: Provider latency in milliseconds.
        estimated_cost_usd: Estimated cost in USD.
        tenant_id: Tenant identifier.
        workflow_step: Workflow step that invoked the provider.

    Returns:
        The emitted telemetry event ID.
    """
    calculated_total = prompt_tokens + completion_tokens

    return emit_event(
        "workflow_provider_usage",
        execution_id=execution_id,
        tenant_id=tenant_id,
        workflow_step=workflow_step,
        metadata={
            "provider_name": provider_name,
            "provider_model": provider_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                total_tokens
                if total_tokens is not None
                else calculated_total
            ),
            "latency_ms": latency_ms,
            "estimated_cost_usd": estimated_cost_usd,
        },
    )


# ---------------------------------------------------------------------------
# Recovery, review, and publication events
# ---------------------------------------------------------------------------


def emit_retry_scheduled(
    *,
    execution_id: str | UUID,
    node_name: str,
    retry_count: int,
    backoff_seconds: float,
    error_code: str | None = None,
    tenant_id: str | UUID | None = None,
) -> str:
    """Emit a retry-scheduled event."""
    return emit_event(
        "workflow_retry_scheduled",
        level="WARNING",
        execution_id=execution_id,
        tenant_id=tenant_id,
        metadata={
            "node_name": node_name,
            "retry_count": retry_count,
            "backoff_seconds": backoff_seconds,
            "error_code": error_code,
        },
    )


def emit_human_review_requested(
    *,
    execution_id: str | UUID,
    reason: str,
    workflow_step: str | None = None,
    tenant_id: str | UUID | None = None,
    interrupt_id: str | UUID | None = None,
) -> str:
    """Emit a human-review-requested event."""
    return emit_event(
        "workflow_human_review_requested",
        level="WARNING",
        execution_id=execution_id,
        tenant_id=tenant_id,
        workflow_step=workflow_step,
        metadata={
            "reason": reason,
            "interrupt_id": _coerce_id(interrupt_id),
        },
    )


def emit_human_review_resolved(
    *,
    execution_id: str | UUID,
    decision: str,
    reviewer_id: str | UUID | None = None,
    workflow_step: str | None = None,
    tenant_id: str | UUID | None = None,
    interrupt_id: str | UUID | None = None,
) -> str:
    """Emit a human-review-resolved event."""
    return emit_event(
        "workflow_human_review_resolved",
        execution_id=execution_id,
        tenant_id=tenant_id,
        user_id=reviewer_id,
        workflow_step=workflow_step,
        metadata={
            "decision": decision,
            "interrupt_id": _coerce_id(interrupt_id),
        },
    )


def emit_publication_completed(
    *,
    execution_id: str | UUID,
    artifact_count: int,
    publication_target: str | None = None,
    duration_ms: float | None = None,
    tenant_id: str | UUID | None = None,
) -> str:
    """Emit a publication-completed event."""
    return emit_event(
        "workflow_publication_completed",
        execution_id=execution_id,
        tenant_id=tenant_id,
        metadata={
            "artifact_count": artifact_count,
            "publication_target": publication_target,
            "duration_ms": duration_ms,
        },
    )


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def track_node_telemetry(
    *,
    execution_id: str | UUID,
    node_name: str,
    tenant_id: str | UUID | None = None,
    workflow_step: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Track node execution duration and emit lifecycle telemetry.

    Usage:
        ```python
        async with track_node_telemetry(
            execution_id=state["execution_id"],
            node_name="requirements",
            tenant_id=state.get("tenant_id"),
            workflow_step="requirements",
        ) as telemetry:
            result = await run_requirements_node(state)
            telemetry["requirement_count"] = len(result["extracted_requirements"])
        ```

    Args:
        execution_id: Workflow execution identifier.
        node_name: Node name.
        tenant_id: Tenant identifier.
        workflow_step: Current workflow step.
        metadata: Optional non-sensitive node metadata.

    Yields:
        A mutable dictionary that nodes can use to add non-sensitive result
        metadata before completion telemetry is emitted.
    """
    started_at = time.perf_counter()

    emit_node_started(
        execution_id=execution_id,
        node_name=node_name,
        tenant_id=tenant_id,
        workflow_step=workflow_step,
    )

    telemetry_data: dict[str, Any] = {
        **(metadata or {}),
    }

    try:
        yield telemetry_data

        duration_ms = (time.perf_counter() - started_at) * 1_000

        emit_node_completed(
            execution_id=execution_id,
            node_name=node_name,
            duration_ms=round(duration_ms, 3),
            tenant_id=tenant_id,
            workflow_step=workflow_step,
            metadata=telemetry_data,
        )

    except Exception as exc:
        duration_ms = (time.perf_counter() - started_at) * 1_000

        emit_node_failed(
            execution_id=execution_id,
            node_name=node_name,
            duration_ms=round(duration_ms, 3),
            error_code=type(exc).__name__,
            recoverable=False,
            tenant_id=tenant_id,
            workflow_step=workflow_step,
        )

        raise