"""
LangChain callback handlers for workflow observability.

This module provides safe, bounded callback handlers for LLM and tool events.
It records structured telemetry without logging prompts, completions, document
content, credentials, customer PII, or raw model output.

Security rules:
- Never log prompt text, completion text, tool arguments, tool results,
  document contents, embeddings, or credentials.
- Redact unknown or sensitive-looking values.
- Bound all recorded strings and payload sizes.
- Treat callback payloads as untrusted input.
- Callback failures must never break workflow execution.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from uuid import UUID, uuid4

from langchain_core.callbacks import (
    AsyncCallbackHandler,
    BaseCallbackHandler,
)

from app.core.utils import truncate_text


logger = logging.getLogger("agentic_bridge.graph.callbacks")

MAX_EVENT_FIELD_CHARS = 500
MAX_EVENT_METADATA_KEYS = 50
MAX_EVENT_METADATA_VALUE_CHARS = 1_000
MAX_SERIALIZED_PAYLOAD_CHARS = 2_000

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


def _is_sensitive_key(key: str) -> bool:
    """Return whether a metadata key appears sensitive."""
    normalized_key = key.strip().lower()

    return any(
        fragment in normalized_key
        for fragment in SENSITIVE_KEY_FRAGMENTS
    )


def _redact_string(value: str) -> str:
    """Redact likely secrets from an untrusted string."""
    redacted_value = value

    for pattern in SECRET_VALUE_PATTERNS:
        redacted_value = pattern.sub(
            REDACTED_VALUE,
            redacted_value,
        )

    return redacted_value


def sanitize_callback_value(
    value: Any,
    *,
    max_length: int = MAX_EVENT_METADATA_VALUE_CHARS,
) -> Any:
    """
    Sanitize and bound a callback metadata value.

    Args:
        value: Arbitrary callback payload value.
        max_length: Maximum length for string values.

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
        return sanitize_callback_metadata(value)

    if isinstance(value, (list, tuple, set)):
        normalized_items: list[Any] = []

        for item in list(value)[:20]:
            normalized_items.append(
                sanitize_callback_value(
                    item,
                    max_length=min(
                        max_length,
                        MAX_EVENT_FIELD_CHARS,
                    ),
                )
            )

        return normalized_items

    # Do not serialize arbitrary objects, exceptions, models, buffers, or
    # provider payloads into logs.
    return "[UNSERIALIZABLE]"


def sanitize_callback_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Sanitize and bound callback metadata.

    Args:
        metadata: Arbitrary metadata supplied by LangChain or a workflow node.

    Returns:
        A bounded metadata dictionary with sensitive values redacted.
    """
    if not metadata or not isinstance(metadata, dict):
        return {}

    safe_metadata: dict[str, Any] = {}

    for key, value in list(metadata.items())[:MAX_EVENT_METADATA_KEYS]:
        if not isinstance(key, str):
            continue

        normalized_key = key.strip()

        if not normalized_key:
            continue

        if _is_sensitive_key(normalized_key):
            safe_metadata[normalized_key] = REDACTED_VALUE
            continue

        safe_metadata[normalized_key] = sanitize_callback_value(value)

    return safe_metadata


def build_callback_tags(
    *,
    execution_id: str | UUID | None = None,
    workflow_step: str | None = None,
    tenant_id: str | UUID | None = None,
) -> list[str]:
    """
    Build bounded, non-sensitive callback tags.

    Args:
        execution_id: Workflow execution identifier.
        workflow_step: Current workflow step name.
        tenant_id: Tenant identifier.

    Returns:
        A list of bounded callback tags.
    """
    tags: list[str] = []

    if execution_id:
        tags.append(
            f"execution:{truncate_text(str(execution_id), 100)}"
        )

    if workflow_step:
        tags.append(
            f"step:{truncate_text(str(workflow_step), 100)}"
        )

    if tenant_id:
        tags.append(
            f"tenant:{truncate_text(str(tenant_id), 100)}"
        )

    return tags


class WorkflowCallbackHandler(BaseCallbackHandler):
    """
    Synchronous callback handler for workflow observability.

    This handler records only event names, model/tool identifiers, token usage,
    status, duration metadata supplied by LangChain, and sanitized metadata. It
    never records prompt or completion content.
    """

    raise_error: bool = False

    def __init__(
        self,
        *,
        execution_id: str | UUID | None = None,
        workflow_step: str | None = None,
        tenant_id: str | UUID | None = None,
    ) -> None:
        self.execution_id = (
            str(execution_id) if execution_id else None
        )
        self.workflow_step = workflow_step
        self.tenant_id = (
            str(tenant_id) if tenant_id else None
        )
        self.tags = build_callback_tags(
            execution_id=self.execution_id,
            workflow_step=self.workflow_step,
            tenant_id=self.tenant_id,
        )

    def _log_event(
        self,
        event_name: str,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        """Emit a structured, sanitized callback event."""
        try:
            safe_metadata = sanitize_callback_metadata(metadata)
            safe_extra = sanitize_callback_metadata(extra)

            logger.info(
                event_name,
                execution_id=self.execution_id,
                workflow_step=self.workflow_step,
                tenant_id=self.tenant_id,
                run_id=str(run_id) if run_id else None,
                parent_run_id=(
                    str(parent_run_id) if parent_run_id else None
                ),
                **safe_metadata,
                **safe_extra,
            )
        except Exception:
            # Observability must never break workflow execution.
            logger.exception(
                "workflow_callback_event_failed",
                execution_id=self.execution_id,
                workflow_step=self.workflow_step,
            )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the start of an LLM call without recording prompts."""
        model_name = None

        if isinstance(serialized, dict):
            model_name = serialized.get("name") or serialized.get(
                "model_name"
            )

        self._log_event(
            "workflow_llm_started",
            run_id=run_id,
            parent_run_id=parent_run_id,
            model_name=model_name,
            prompt_count=len(prompts) if isinstance(prompts, list) else None,
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record successful LLM completion metadata."""
        token_usage = getattr(response, "llm_output", None) or {}

        self._log_event(
            "workflow_llm_completed",
            run_id=run_id,
            parent_run_id=parent_run_id,
            token_usage=sanitize_callback_value(token_usage),
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record an LLM failure without recording provider payloads."""
        self._log_event(
            "workflow_llm_failed",
            run_id=run_id,
            parent_run_id=parent_run_id,
            error_type=type(error).__name__,
        )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record tool invocation without recording tool input."""
        tool_name = None

        if isinstance(serialized, dict):
            tool_name = serialized.get("name")

        self._log_event(
            "workflow_tool_started",
            run_id=run_id,
            parent_run_id=parent_run_id,
            tool_name=tool_name,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record successful tool completion without recording tool output."""
        self._log_event(
            "workflow_tool_completed",
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record a tool failure without recording tool output."""
        self._log_event(
            "workflow_tool_failed",
            run_id=run_id,
            parent_run_id=parent_run_id,
            error_type=type(error).__name__,
        )


class AsyncWorkflowCallbackHandler(AsyncCallbackHandler):
    """
    Asynchronous callback handler for async workflow execution.

    This handler mirrors `WorkflowCallbackHandler` but is suitable for async
    LangChain and LangGraph execution paths.
    """

    raise_error: bool = False

    def __init__(
        self,
        *,
        execution_id: str | UUID | None = None,
        workflow_step: str | None = None,
        tenant_id: str | UUID | None = None,
    ) -> None:
        self.execution_id = (
            str(execution_id) if execution_id else None
        )
        self.workflow_step = workflow_step
        self.tenant_id = (
            str(tenant_id) if tenant_id else None
        )
        self.tags = build_callback_tags(
            execution_id=self.execution_id,
            workflow_step=self.workflow_step,
            tenant_id=self.tenant_id,
        )

    async def _log_event(
        self,
        event_name: str,
        *,
        run_id: UUID | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        """Emit a structured, sanitized async callback event."""
        try:
            safe_metadata = sanitize_callback_metadata(metadata)
            safe_extra = sanitize_callback_metadata(extra)

            logger.info(
                event_name,
                execution_id=self.execution_id,
                workflow_step=self.workflow_step,
                tenant_id=self.tenant_id,
                run_id=str(run_id) if run_id else None,
                parent_run_id=(
                    str(parent_run_id) if parent_run_id else None
                ),
                **safe_metadata,
                **safe_extra,
            )
        except Exception:
            logger.exception(
                "workflow_async_callback_event_failed",
                execution_id=self.execution_id,
                workflow_step=self.workflow_step,
            )

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the start of an async LLM call without recording prompts."""
        model_name = None

        if isinstance(serialized, dict):
            model_name = serialized.get("name") or serialized.get(
                "model_name"
            )

        await self._log_event(
            "workflow_llm_started",
            run_id=run_id,
            parent_run_id=parent_run_id,
            model_name=model_name,
            prompt_count=len(prompts) if isinstance(prompts, list) else None,
        )

    async def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record successful async LLM completion metadata."""
        token_usage = getattr(response, "llm_output", None) or {}

        await self._log_event(
            "workflow_llm_completed",
            run_id=run_id,
            parent_run_id=parent_run_id,
            token_usage=sanitize_callback_value(token_usage),
        )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record an async LLM failure without recording provider payloads."""
        await self._log_event(
            "workflow_llm_failed",
            run_id=run_id,
            parent_run_id=parent_run_id,
            error_type=type(error).__name__,
        )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record async tool invocation without recording tool input."""
        tool_name = None

        if isinstance(serialized, dict):
            tool_name = serialized.get("name")

        await self._log_event(
            "workflow_tool_started",
            run_id=run_id,
            parent_run_id=parent_run_id,
            tool_name=tool_name,
        )

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record successful async tool completion."""
        await self._log_event(
            "workflow_tool_completed",
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record an async tool failure without recording tool output."""
        await self._log_event(
            "workflow_tool_failed",
            run_id=run_id,
            parent_run_id=parent_run_id,
            error_type=type(error).__name__,
        )


def create_workflow_callbacks(
    *,
    execution_id: str | UUID | None = None,
    workflow_step: str | None = None,
    tenant_id: str | UUID | None = None,
) -> list[BaseCallbackHandler]:
    """
    Create callback handlers for a workflow invocation.

    Args:
        execution_id: Workflow execution identifier.
        workflow_step: Current workflow step name.
        tenant_id: Tenant identifier.

    Returns:
        A list containing a synchronous workflow callback handler.
    """
    return [
        WorkflowCallbackHandler(
            execution_id=execution_id,
            workflow_step=workflow_step,
            tenant_id=tenant_id,
        )
    ]


def create_async_workflow_callbacks(
    *,
    execution_id: str | UUID | None = None,
    workflow_step: str | None = None,
    tenant_id: str | UUID | None = None,
) -> list[AsyncCallbackHandler]:
    """
    Create async callback handlers for an async workflow invocation.

    Args:
        execution_id: Workflow execution identifier.
        workflow_step: Current workflow step name.
        tenant_id: Tenant identifier.

    Returns:
        A list containing an asynchronous workflow callback handler.
    """
    return [
        AsyncWorkflowCallbackHandler(
            execution_id=execution_id,
            workflow_step=workflow_step,
            tenant_id=tenant_id,
        )
    ]


def generate_callback_run_id() -> str:
    """Generate a non-sensitive callback correlation ID."""
    return f"callback_{uuid4()}"