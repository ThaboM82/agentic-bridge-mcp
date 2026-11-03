"""
LangGraph workflow recovery and retry orchestration.

This module provides deterministic helpers for recording recoverable failures,
incrementing retry attempts, preparing retry state, and deciding whether a
workflow should retry, pause for human review, or terminate.

Security and reliability rules:
- Never store raw exception messages, stack traces, provider payloads,
  credentials, document contents, prompts, completions, or embeddings in
  workflow state.
- Treat workflow state as untrusted input.
- Bound all persisted strings and metadata.
- Recovery helpers must be pure and side-effect free.
- Never retry non-recoverable failures automatically.
- Human review is required when retries are exhausted or the failure is
  blocking.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Literal

from app.core.enums import (
    JobStatus,
    WorkflowStep,
)
from app.core.exceptions import (
    AuthenticationError,
    AuthorizationError,
    ExternalProviderError,
    ProviderTimeoutError,
    ValidationError,
    WorkflowStateError,
)
from app.core.utils import truncate_text


logger = logging.getLogger(
    "agentic_bridge.graph.recovery",
)


MAX_ERROR_MESSAGE_CHARS = 1_000
MAX_ERROR_METADATA_KEYS = 50
MAX_ERROR_METADATA_VALUE_CHARS = 1_000
MAX_RECOVERY_HISTORY_ENTRIES = 50


DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
MAX_RETRY_BACKOFF_SECONDS = 60.0


SENSITIVE_KEY_FRAGMENTS: tuple[
    str,
    ...,
] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "cookie",
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
    "stack_trace",
    "traceback",
)


REDACTED_VALUE = "[REDACTED]"


RecoveryAction = Literal[
    "retry",
    "human_review",
    "fail",
    "cancel",
    "resume",
]


ERROR_CODE_PROVIDER_TIMEOUT = (
    "provider_timeout"
)
ERROR_CODE_PROVIDER_UNAVAILABLE = (
    "provider_unavailable"
)
ERROR_CODE_VALIDATION_FAILED = (
    "validation_failed"
)
ERROR_CODE_AUTHORIZATION = (
    "authorization_error"
)
ERROR_CODE_AUTHENTICATION = (
    "authentication_error"
)
ERROR_CODE_REVIEW_CHANGES_REQUESTED = (
    "review_changes_requested"
)
ERROR_CODE_WORKFLOW_REJECTED = (
    "workflow_rejected"
)
ERROR_CODE_WORKFLOW_CANCELLED = (
    "workflow_cancelled"
)
ERROR_CODE_INVALID_STATE = "invalid_state"
ERROR_CODE_UNEXPECTED = (
    "unexpected_error"
)


RECOVERABLE_ERROR_CODES: frozenset[
    str
] = frozenset(
    {
        ERROR_CODE_PROVIDER_TIMEOUT,
        ERROR_CODE_PROVIDER_UNAVAILABLE,
        ERROR_CODE_VALIDATION_FAILED,
        ERROR_CODE_REVIEW_CHANGES_REQUESTED,
    }
)


# ---------------------------------------------------------------------------
# Time and sanitization helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _safe_get(
    state: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Safely read a workflow-state value."""
    if not isinstance(state, dict):
        return default

    return state.get(
        key,
        default,
    )


def _is_truthy(
    value: Any,
) -> bool:
    """Return whether a state value should be treated as true."""
    if value is None:
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "1",
            "yes",
            "y",
        }

    if isinstance(
        value,
        (int, float),
    ):
        return value > 0

    if isinstance(
        value,
        (list, tuple, set, dict),
    ):
        return len(value) > 0

    return bool(value)


def sanitize_recovery_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Sanitize and bound recovery metadata.
    """
    if (
        not metadata
        or not isinstance(metadata, dict)
    ):
        return {}

    safe_metadata: dict[str, Any] = {}

    for key, value in list(
        metadata.items(),
    )[:MAX_ERROR_METADATA_KEYS]:
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
            safe_metadata[
                normalized_key
            ] = REDACTED_VALUE

            continue

        if value is None or isinstance(
            value,
            (bool, int, float),
        ):
            safe_metadata[
                normalized_key
            ] = value

        elif isinstance(value, str):
            safe_metadata[
                normalized_key
            ] = truncate_text(
                value,
                MAX_ERROR_METADATA_VALUE_CHARS,
            )

        else:
            safe_metadata[
                normalized_key
            ] = "[UNSERIALIZABLE]"

    return safe_metadata


def sanitize_error_message(
    message: str | None,
) -> str:
    """
    Sanitize and bound an error message.
    """
    if not message or not isinstance(
        message,
        str,
    ):
        return (
            "An unexpected workflow error occurred."
        )

    normalized_message = message.strip()

    if not normalized_message:
        return (
            "An unexpected workflow error occurred."
        )

    return truncate_text(
        normalized_message,
        MAX_ERROR_MESSAGE_CHARS,
    )


def normalize_error_code(
    error_code: str | None,
) -> str:
    """
    Normalize and validate an error code.
    """
    if not error_code or not isinstance(
        error_code,
        str,
    ):
        raise ValidationError(
            "Error code is required.",
        )

    normalized_code = (
        error_code.strip().lower()
    )

    if not normalized_code:
        raise ValidationError(
            "Error code is required.",
        )

    if len(normalized_code) > 100:
        raise ValidationError(
            "Error code exceeds the allowed length.",
        )

    return normalized_code


# ---------------------------------------------------------------------------
# Recovery history
# ---------------------------------------------------------------------------


def _append_recovery_history(
    state: dict[str, Any],
    recovery_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Append a recovery record to bounded recovery history.
    """
    existing_history = state.get(
        "recovery_history",
    )

    if not isinstance(
        existing_history,
        list,
    ):
        existing_history = []

    updated_history = [
        *existing_history,
        recovery_record,
    ]

    if (
        len(updated_history)
        > MAX_RECOVERY_HISTORY_ENTRIES
    ):
        updated_history = updated_history[
            -MAX_RECOVERY_HISTORY_ENTRIES:
        ]

    return updated_history


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def classify_exception(
    exc: BaseException,
) -> tuple[str, bool, str]:
    """
    Classify an exception into a safe workflow error contract.
    """
    if isinstance(
        exc,
        ProviderTimeoutError,
    ):
        return (
            ERROR_CODE_PROVIDER_TIMEOUT,
            True,
            "The language-model provider timed out.",
        )

    if isinstance(
        exc,
        ExternalProviderError,
    ):
        return (
            ERROR_CODE_PROVIDER_UNAVAILABLE,
            True,
            "An external provider is temporarily unavailable.",
        )

    if isinstance(
        exc,
        ValidationError,
    ):
        return (
            ERROR_CODE_VALIDATION_FAILED,
            True,
            "Workflow output failed validation.",
        )

    if isinstance(
        exc,
        AuthorizationError,
    ):
        return (
            ERROR_CODE_AUTHORIZATION,
            False,
            "The caller is not authorized to perform this action.",
        )

    if isinstance(
        exc,
        AuthenticationError,
    ):
        return (
            ERROR_CODE_AUTHENTICATION,
            False,
            "Authentication is required.",
        )

    if isinstance(
        exc,
        WorkflowStateError,
    ):
        return (
            ERROR_CODE_INVALID_STATE,
            False,
            "Workflow state is invalid or incomplete.",
        )

    return (
        ERROR_CODE_UNEXPECTED,
        False,
        "An unexpected workflow error occurred.",
    )


def is_recoverable_error_code(
    error_code: str | None,
) -> bool:
    """
    Return whether an error code is eligible for automatic retry.
    """
    if not error_code:
        return False

    return (
        error_code.strip().lower()
        in RECOVERABLE_ERROR_CODES
    )


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


def get_retry_policy(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Return the effective retry policy for a workflow execution.
    """
    retry_count = _safe_get(
        state,
        "retry_count",
        0,
    )

    if not isinstance(
        retry_count,
        int,
    ) or isinstance(
        retry_count,
        bool,
    ):
        retry_count = 0

    max_retries = _safe_get(
        state,
        "max_retries",
        DEFAULT_MAX_RETRIES,
    )

    if not isinstance(
        max_retries,
        int,
    ) or isinstance(
        max_retries,
        bool,
    ):
        max_retries = DEFAULT_MAX_RETRIES

    if max_retries < 0:
        max_retries = 0

    retry_backoff_seconds = _safe_get(
        state,
        "retry_backoff_seconds",
        DEFAULT_RETRY_BACKOFF_SECONDS,
    )

    if not isinstance(
        retry_backoff_seconds,
        (int, float),
    ) or isinstance(
        retry_backoff_seconds,
        bool,
    ):
        retry_backoff_seconds = (
            DEFAULT_RETRY_BACKOFF_SECONDS
        )

    bounded_backoff = min(
        max(
            0.0,
            float(retry_backoff_seconds),
        ),
        MAX_RETRY_BACKOFF_SECONDS,
    )

    remaining_attempts = max(
        0,
        max_retries - retry_count,
    )

    return {
        "retry_count": retry_count,
        "max_retries": max_retries,
        "retry_backoff_seconds": bounded_backoff,
        "remaining_attempts": remaining_attempts,
        "retries_exhausted": remaining_attempts
        <= 0,
    }


def should_retry(
    state: dict[str, Any],
    *,
    error_code: str | None = None,
) -> bool:
    """
    Return whether a workflow failure is eligible for another retry.
    """
    effective_error_code = (
        error_code
        or _safe_get(
            state,
            "error_code",
        )
    )

    recoverable_error = _is_truthy(
        _safe_get(
            state,
            "recoverable_error",
        ),
    ) or is_recoverable_error_code(
        effective_error_code,
    )

    retry_policy = get_retry_policy(
        state,
    )

    return (
        recoverable_error
        and not retry_policy[
            "retries_exhausted"
        ]
        and not _is_truthy(
            _safe_get(
                state,
                "cancel_requested",
            ),
        )
    )


def calculate_retry_backoff(
    *,
    retry_count: int,
    base_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> float:
    """
    Calculate bounded exponential backoff for a retry attempt.
    """
    if retry_count < 0:
        retry_count = 0

    if not isinstance(
        base_backoff_seconds,
        (int, float),
    ) or isinstance(
        base_backoff_seconds,
        bool,
    ):
        base_backoff_seconds = (
            DEFAULT_RETRY_BACKOFF_SECONDS
        )

    bounded_base = min(
        max(
            0.0,
            float(base_backoff_seconds),
        ),
        MAX_RETRY_BACKOFF_SECONDS,
    )

    backoff_seconds = bounded_base * (
        2 ** min(retry_count, 5)
    )

    return min(
        backoff_seconds,
        MAX_RETRY_BACKOFF_SECONDS,
    )


# ---------------------------------------------------------------------------
# Recovery state updates
# ---------------------------------------------------------------------------


def record_recoverable_failure(
    state: dict[str, Any],
    *,
    message: str,
    error_step: WorkflowStep | None = None,
    error_code: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Record a recoverable workflow failure without incrementing retries.
    """
    normalized_error_code = (
        normalize_error_code(error_code)
    )

    safe_message = sanitize_error_message(
        message,
    )

    safe_metadata = (
        sanitize_recovery_metadata(
            metadata,
        )
    )

    recovery_record = {
        "recorded_at": _utc_now_iso(),
        "error_step": (
            error_step.value
            if error_step
            else None
        ),
        "error_code": normalized_error_code,
        "message": safe_message,
        "recoverable": True,
        "metadata": safe_metadata,
    }

    recovery_history = (
        _append_recovery_history(
            state,
            recovery_record,
        )
    )

    return {
        "current_error": safe_message,
        "error_step": error_step,
        "error_code": normalized_error_code,
        "error_occurred_at": recovery_record[
            "recorded_at"
        ],
        "recoverable_error": True,
        "workflow_status": (
            JobStatus.PROCESSING
        ),
        "recovery_history": recovery_history,
        "execution_logs": [
            f"WARNING: Recoverable workflow failure: {safe_message}",
        ],
    }


def record_blocking_failure(
    state: dict[str, Any],
    *,
    message: str,
    error_step: WorkflowStep | None = None,
    error_code: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Record a non-recoverable workflow failure.
    """
    normalized_error_code = (
        normalize_error_code(error_code)
    )

    safe_message = sanitize_error_message(
        message,
    )

    safe_metadata = (
        sanitize_recovery_metadata(
            metadata,
        )
    )

    recovery_record = {
        "recorded_at": _utc_now_iso(),
        "error_step": (
            error_step.value
            if error_step
            else None
        ),
        "error_code": normalized_error_code,
        "message": safe_message,
        "recoverable": False,
        "metadata": safe_metadata,
    }

    recovery_history = (
        _append_recovery_history(
            state,
            recovery_record,
        )
    )

    return {
        "current_error": safe_message,
        "error_step": error_step,
        "error_code": normalized_error_code,
        "error_occurred_at": recovery_record[
            "recorded_at"
        ],
        "recoverable_error": False,
        "workflow_status": JobStatus.FAILED,
        "recovery_history": recovery_history,
        "execution_logs": [
            f"ERROR: Blocking workflow failure: {safe_message}",
        ],
    }


def prepare_retry(
    state: dict[str, Any],
    *,
    error_step: WorkflowStep | None = None,
    error_code: str | None = None,
    message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Prepare state updates for another retry attempt.
    """
    effective_error_code = (
        normalize_error_code(
            error_code
            or _safe_get(
                state,
                "error_code",
            )
            or ERROR_CODE_UNEXPECTED,
        )
    )

    if not is_recoverable_error_code(
        effective_error_code,
    ):
        raise ValidationError(
            "Only recoverable workflow failures "
            "can be retried.",
        )

    retry_policy = get_retry_policy(
        state,
    )

    if retry_policy["retries_exhausted"]:
        raise ValidationError(
            "Workflow retry allowance has been "
            "exhausted.",
        )

    retry_count = (
        retry_policy["retry_count"] + 1
    )

    backoff_seconds = (
        calculate_retry_backoff(
            retry_count=retry_policy[
                "retry_count"
            ],
            base_backoff_seconds=retry_policy[
                "retry_backoff_seconds"
            ],
        )
    )

    recovery_record = {
        "recorded_at": _utc_now_iso(),
        "action": "retry",
        "error_step": (
            error_step.value
            if error_step
            else None
        ),
        "error_code": effective_error_code,
        "message": sanitize_error_message(
            message,
        ),
        "retry_count": retry_count,
        "backoff_seconds": backoff_seconds,
        "metadata": (
            sanitize_recovery_metadata(
                metadata,
            )
        ),
    }

    recovery_history = (
        _append_recovery_history(
            state,
            recovery_record,
        )
    )

    return {
        "retry_count": retry_count,
        "last_retry_at": recovery_record[
            "recorded_at"
        ],
        "retry_backoff_seconds": backoff_seconds,
        "current_error": None,
        "error_step": None,
        "error_code": None,
        "error_occurred_at": None,
        "recoverable_error": False,
        "workflow_status": (
            JobStatus.PROCESSING
        ),
        "recovery_history": recovery_history,
        "execution_logs": [
            (
                f"INFO: Preparing retry attempt {retry_count} "
                f"after {backoff_seconds:.2f} seconds."
            ),
        ],
    }


def prepare_human_review(
    state: dict[str, Any],
    *,
    reason: str,
    error_step: WorkflowStep | None = None,
    error_code: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Prepare state updates that pause a workflow for human review.
    """
    safe_reason = sanitize_error_message(
        reason,
    )

    safe_metadata = (
        sanitize_recovery_metadata(
            metadata,
        )
    )

    normalized_error_code = (
        error_code.strip().lower()
        if error_code
        else None
    )

    recovery_record = {
        "recorded_at": _utc_now_iso(),
        "action": "human_review",
        "error_step": (
            error_step.value
            if error_step
            else None
        ),
        "error_code": normalized_error_code,
        "message": safe_reason,
        "metadata": safe_metadata,
    }

    recovery_history = (
        _append_recovery_history(
            state,
            recovery_record,
        )
    )

    return {
        "approval_required": True,
        "approval_status": "pending",
        "review_status": "pending_approval",
        "current_error": safe_reason,
        "error_step": error_step,
        "error_code": normalized_error_code,
        "error_occurred_at": recovery_record[
            "recorded_at"
        ],
        "recoverable_error": True,
        "workflow_status": (
            JobStatus.PROCESSING
        ),
        "recovery_history": recovery_history,
        "execution_logs": [
            f"WARNING: Workflow paused for human review: {safe_reason}",
        ],
    }


def prepare_resume_after_review(
    state: dict[str, Any],
    *,
    reviewer_id: str | None = None,
    comment: str | None = None,
) -> dict[str, Any]:
    """
    Prepare state updates after a human reviewer approves a paused workflow.
    """
    safe_comment = truncate_text(
        (comment or "").strip(),
        MAX_ERROR_MESSAGE_CHARS,
    )

    recovery_record = {
        "recorded_at": _utc_now_iso(),
        "action": "resume",
        "reviewer_id": reviewer_id,
        "message": (
            safe_comment
            or "Workflow resumed after human approval."
        ),
    }

    recovery_history = (
        _append_recovery_history(
            state,
            recovery_record,
        )
    )

    return {
        "approval_required": False,
        "approval_status": "approved",
        "review_status": "approved",
        "approved_by": reviewer_id,
        "approved_at": recovery_record[
            "recorded_at"
        ],
        "approval_comment": safe_comment
        or None,
        "current_error": None,
        "error_step": None,
        "error_code": None,
        "error_occurred_at": None,
        "recoverable_error": False,
        "workflow_status": (
            JobStatus.PROCESSING
        ),
        "recovery_history": recovery_history,
        "execution_logs": [
            "INFO: Workflow resumed after human approval.",
        ],
    }


def prepare_cancellation(
    state: dict[str, Any],
    *,
    cancelled_by: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """
    Prepare state updates for workflow cancellation.
    """
    safe_reason = sanitize_error_message(
        reason
        or "Workflow cancellation requested.",
    )

    recovery_record = {
        "recorded_at": _utc_now_iso(),
        "action": "cancel",
        "cancelled_by": cancelled_by,
        "message": safe_reason,
    }

    recovery_history = (
        _append_recovery_history(
            state,
            recovery_record,
        )
    )

    return {
        "cancel_requested": True,
        "cancelled_by": cancelled_by,
        "cancellation_reason": safe_reason,
        "cancelled_at": recovery_record[
            "recorded_at"
        ],
        "workflow_status": (
            JobStatus.CANCELLED
        ),
        "current_error": safe_reason,
        "error_code": ERROR_CODE_WORKFLOW_CANCELLED,
        "recoverable_error": False,
        "recovery_history": recovery_history,
        "execution_logs": [
            f"WARNING: Workflow cancelled: {safe_reason}",
        ],
    }


# ---------------------------------------------------------------------------
# Recovery decision
# ---------------------------------------------------------------------------


def decide_recovery_action(
    state: dict[str, Any],
    *,
    error_code: str | None = None,
) -> RecoveryAction:
    """
    Decide the next recovery action for a failed workflow.
    """
    execution_id = _safe_get(
        state,
        "execution_id",
    )

    try:
        effective_error_code = (
            error_code
            or _safe_get(
                state,
                "error_code",
            )
        )

        if effective_error_code:
            effective_error_code = (
                normalize_error_code(
                    effective_error_code,
                )
            )

        if _is_truthy(
            _safe_get(
                state,
                "cancel_requested",
            ),
        ):
            action: RecoveryAction = "cancel"
            reason = (
                "Workflow cancellation was requested."
            )

        elif not effective_error_code:
            action = "resume"
            reason = (
                "No active workflow error."
            )

        elif not is_recoverable_error_code(
            effective_error_code,
        ):
            action = "fail"
            reason = (
                "Workflow failure is not recoverable."
            )

        elif should_retry(
            state,
            error_code=effective_error_code,
        ):
            action = "retry"
            reason = (
                "Recoverable failure has retry "
                "attempts remaining."
            )

        else:
            action = "human_review"
            reason = (
                "Recoverable failure exhausted retry "
                "attempts; human review is required."
            )

        logger.info(
            "workflow_recovery_action_selected",
            extra={
                "execution_id": execution_id,
                "action": action,
                "error_code": effective_error_code,
                "reason": reason,
            },
        )

        return action

    except ValidationError:
        logger.warning(
            "workflow_recovery_action_invalid_error_code",
            extra={
                "execution_id": execution_id,
            },
        )

        return "human_review"

    except Exception:
        logger.exception(
            "workflow_recovery_action_failed",
            extra={
                "execution_id": execution_id,
            },
        )

        return "fail"


__all__ = [
    "ERROR_CODE_AUTHENTICATION",
    "ERROR_CODE_AUTHORIZATION",
    "ERROR_CODE_INVALID_STATE",
    "ERROR_CODE_PROVIDER_TIMEOUT",
    "ERROR_CODE_PROVIDER_UNAVAILABLE",
    "ERROR_CODE_REVIEW_CHANGES_REQUESTED",
    "ERROR_CODE_UNEXPECTED",
    "ERROR_CODE_VALIDATION_FAILED",
    "ERROR_CODE_WORKFLOW_CANCELLED",
    "ERROR_CODE_WORKFLOW_REJECTED",
    "RecoveryAction",
    "calculate_retry_backoff",
    "classify_exception",
    "decide_recovery_action",
    "get_retry_policy",
    "is_recoverable_error_code",
    "prepare_cancellation",
    "prepare_human_review",
    "prepare_resume_after_review",
    "prepare_retry",
    "record_blocking_failure",
    "record_recoverable_failure",
    "sanitize_error_message",
    "sanitize_recovery_metadata",
    "should_retry",
]