"""
Request, user, tenant, workflow, and job logging context.

Context variables are safe for asyncio: each request, task, and workflow
execution receives its own isolated context, preventing IDs from leaking
between concurrent requests.

Use these helpers in middleware, LangGraph nodes, background jobs, and service
layers so every log line can be correlated back to a request, tenant, user,
workflow execution, ingestion job, or external provider call.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar
from typing import Any

import structlog


# ---------------------------------------------------------------------------
# Context variables
# ---------------------------------------------------------------------------

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")
tenant_id_var: ContextVar[str] = ContextVar("tenant_id", default="")
workflow_id_var: ContextVar[str] = ContextVar("workflow_id", default="")
workflow_step_var: ContextVar[str] = ContextVar("workflow_step", default="")
job_id_var: ContextVar[str] = ContextVar("job_id", default="")
document_id_var: ContextVar[str] = ContextVar("document_id", default="")
provider_var: ContextVar[str] = ContextVar("provider", default="")
model_var: ContextVar[str] = ContextVar("model", default="")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _safe_context_value(value: str | None, field_name: str) -> str:
    """
    Validate a value before placing it into logging context.

    Prevents malformed, oversized, or control-character-laden values from
    polluting structured logs or downstream log processors.
    """
    if not value:
        return ""

    normalized = str(value).strip()

    if not normalized:
        return ""

    if len(normalized) > 128:
        raise ValueError(f"{field_name} exceeds the maximum allowed length of 128.")

    if not _SAFE_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            f"{field_name} may contain only letters, numbers, underscores, "
            "hyphens, colons, and periods."
        )

    return normalized


def _bind_contextvars(**values: str) -> None:
    """Bind only non-empty values to structlog's context-local storage."""
    non_empty_values = {key: value for key, value in values.items() if value}

    if non_empty_values:
        structlog.contextvars.bind_contextvars(**non_empty_values)


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def generate_request_id() -> str:
    """Generate a new opaque request identifier."""
    return f"req_{uuid.uuid4().hex[:24]}"


def generate_correlation_id() -> str:
    """Generate a correlation ID for a logical business operation."""
    return f"corr_{uuid.uuid4().hex[:24]}"


def generate_workflow_id() -> str:
    """Generate a workflow execution identifier."""
    return f"wf_{uuid.uuid4().hex[:24]}"


def generate_job_id() -> str:
    """Generate an asynchronous job identifier."""
    return f"job_{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Request context
# ---------------------------------------------------------------------------


def bind_request_context(
    request_id: str,
    *,
    correlation_id: str | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
    workflow_id: str | None = None,
) -> None:
    """
    Initialize request-scoped logging context.

    Call this once in request middleware before downstream handlers execute.
    """
    safe_request_id = _safe_context_value(request_id, "request_id")

    if not safe_request_id:
        raise ValueError("request_id is required.")

    structlog.contextvars.clear_contextvars()

    request_id_var.set(safe_request_id)

    safe_correlation_id = _safe_context_value(correlation_id, "correlation_id")
    safe_user_id = _safe_context_value(user_id, "user_id")
    safe_tenant_id = _safe_context_value(tenant_id, "tenant_id")
    safe_workflow_id = _safe_context_value(workflow_id, "workflow_id")

    correlation_id_var.set(safe_correlation_id)
    user_id_var.set(safe_user_id)
    tenant_id_var.set(safe_tenant_id)
    workflow_id_var.set(safe_workflow_id)

    _bind_contextvars(
        request_id=safe_request_id,
        correlation_id=safe_correlation_id,
        user_id=safe_user_id,
        tenant_id=safe_tenant_id,
        workflow_id=safe_workflow_id,
    )


def get_request_id() -> str:
    """Return the current request ID."""
    return request_id_var.get()


def get_correlation_id() -> str:
    """Return the current correlation ID."""
    return correlation_id_var.get()


def get_user_id() -> str:
    """Return the authenticated user ID, if available."""
    return user_id_var.get()


def get_tenant_id() -> str:
    """Return the current tenant ID, if available."""
    return tenant_id_var.get()


def get_logging_context() -> dict[str, str]:
    """Return all currently bound context values."""
    return {
        "request_id": request_id_var.get(),
        "correlation_id": correlation_id_var.get(),
        "user_id": user_id_var.get(),
        "tenant_id": tenant_id_var.get(),
        "workflow_id": workflow_id_var.get(),
        "workflow_step": workflow_step_var.get(),
        "job_id": job_id_var.get(),
        "document_id": document_id_var.get(),
        "provider": provider_var.get(),
        "model": model_var.get(),
    }


# ---------------------------------------------------------------------------
# Workflow context
# ---------------------------------------------------------------------------


def bind_workflow_context(
    workflow_id: str,
    *,
    workflow_step: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
) -> None:
    """
    Attach workflow execution context to subsequent logs.

    Useful in LangGraph nodes, workflow services, and background workers.
    """
    safe_workflow_id = _safe_context_value(workflow_id, "workflow_id")

    if not safe_workflow_id:
        raise ValueError("workflow_id is required.")

    workflow_id_var.set(safe_workflow_id)

    safe_step = _safe_context_value(workflow_step, "workflow_step")
    safe_tenant_id = _safe_context_value(tenant_id, "tenant_id")
    safe_user_id = _safe_context_value(user_id, "user_id")

    if safe_step:
        workflow_step_var.set(safe_step)

    if safe_tenant_id:
        tenant_id_var.set(safe_tenant_id)

    if safe_user_id:
        user_id_var.set(safe_user_id)

    _bind_contextvars(
        workflow_id=safe_workflow_id,
        workflow_step=safe_step,
        tenant_id=safe_tenant_id,
        user_id=safe_user_id,
    )


def get_workflow_id() -> str:
    """Return the current workflow execution ID."""
    return workflow_id_var.get()


def get_workflow_step() -> str:
    """Return the current workflow step."""
    return workflow_step_var.get()


def bind_workflow_step(workflow_step: str) -> None:
    """Update the current workflow step in logging context."""
    safe_step = _safe_context_value(workflow_step, "workflow_step")

    if not safe_step:
        raise ValueError("workflow_step is required.")

    workflow_step_var.set(safe_step)
    structlog.contextvars.bind_contextvars(workflow_step=safe_step)


# ---------------------------------------------------------------------------
# Job and document context
# ---------------------------------------------------------------------------


def bind_job_context(
    job_id: str,
    *,
    workflow_id: str | None = None,
    tenant_id: str | None = None,
) -> None:
    """Attach an asynchronous job to the current logging context."""
    safe_job_id = _safe_context_value(job_id, "job_id")

    if not safe_job_id:
        raise ValueError("job_id is required.")

    job_id_var.set(safe_job_id)

    safe_workflow_id = _safe_context_value(workflow_id, "workflow_id")
    safe_tenant_id = _safe_context_value(tenant_id, "tenant_id")

    if safe_workflow_id:
        workflow_id_var.set(safe_workflow_id)

    if safe_tenant_id:
        tenant_id_var.set(safe_tenant_id)

    _bind_contextvars(
        job_id=safe_job_id,
        workflow_id=safe_workflow_id,
        tenant_id=safe_tenant_id,
    )


def get_job_id() -> str:
    """Return the current asynchronous job ID."""
    return job_id_var.get()


def bind_document_context(document_id: str) -> None:
    """Attach a document ID to the current logging context."""
    safe_document_id = _safe_context_value(document_id, "document_id")

    if not safe_document_id:
        raise ValueError("document_id is required.")

    document_id_var.set(safe_document_id)
    structlog.contextvars.bind_contextvars(document_id=safe_document_id)


def get_document_id() -> str:
    """Return the current document ID."""
    return document_id_var.get()


# ---------------------------------------------------------------------------
# LLM provider context
# ---------------------------------------------------------------------------


def bind_llm_context(
    provider: str,
    *,
    model: str | None = None,
) -> None:
    """
    Attach language-model provider and model metadata to logging context.

    Never bind API keys, prompts, raw completions, or customer content here.
    """
    safe_provider = _safe_context_value(provider, "provider")

    if not safe_provider:
        raise ValueError("provider is required.")

    provider_var.set(safe_provider)

    safe_model = _safe_context_value(model, "model")

    if safe_model:
        model_var.set(safe_model)

    _bind_contextvars(
        provider=safe_provider,
        model=safe_model,
    )


def get_llm_context() -> dict[str, str]:
    """Return current language-model context values."""
    return {
        "provider": provider_var.get(),
        "model": model_var.get(),
    }


# ---------------------------------------------------------------------------
# Temporary context
# ---------------------------------------------------------------------------


def bind_temporary_context(**values: Any) -> None:
    """
    Bind temporary structured values for the current execution context.

    Values are converted to strings and bounded to 256 characters. Do not use
    this for secrets, raw prompts, tokens, passwords, or large payloads.
    """
    safe_values: dict[str, str] = {}

    for key, value in values.items():
        normalized_key = _safe_context_value(str(key), "context key")

        if not normalized_key:
            continue

        normalized_value = str(value).strip()[:256]
        safe_values[normalized_key] = normalized_value

    if safe_values:
        structlog.contextvars.bind_contextvars(**safe_values)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def clear_logging_context() -> None:
    """Clear all request-scoped logging context."""
    structlog.contextvars.clear_contextvars()

    request_id_var.set("")
    correlation_id_var.set("")
    user_id_var.set("")
    tenant_id_var.set("")
    workflow_id_var.set("")
    workflow_step_var.set("")
    job_id_var.set("")
    document_id_var.set("")
    provider_var.set("")
    model_var.set("")