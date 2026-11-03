"""
LangGraph human-in-the-loop interrupts.

This module provides safe helpers for pausing workflow execution for human
review, resuming after approval, and recording an immutable interrupt audit
trail. It does not make approval decisions or mutate external systems.

Security rules:
- Never include generated code, document contents, prompts, completions,
  embeddings, credentials, or customer PII in interrupt payloads.
- Treat interrupt metadata and reviewer input as untrusted.
- Bound all persisted strings and payload sizes.
- Use append-only interrupt history; never overwrite prior decisions.
- Human approval is required before publication or deployment-related actions.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from app.core.exceptions import ValidationError
from app.core.utils import truncate_text


logger = logging.getLogger("agentic_bridge.graph.interrupts")

MAX_INTERRUPT_REASON_CHARS = 1_000
MAX_INTERRUPT_COMMENT_CHARS = 2_000
MAX_INTERRUPT_METADATA_KEYS = 50
MAX_INTERRUPT_METADATA_VALUE_CHARS = 1_000
MAX_INTERRUPT_HISTORY_ENTRIES = 100

VALID_INTERRUPT_TYPES: tuple[str, ...] = (
    "human_review",
    "clarification_required",
    "approval_required",
    "validation_failed",
    "policy_review",
)

VALID_REVIEW_DECISIONS: tuple[str, ...] = (
    "approved",
    "changes_requested",
    "rejected",
)

SENSITIVE_KEY_FRAGMENTS: tuple[str, ...] = (
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "credential",
    "authorization",
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
)

REDACTED_VALUE = "[REDACTED]"

InterruptType = Literal[
    "human_review",
    "clarification_required",
    "approval_required",
    "validation_failed",
    "policy_review",
]

ReviewDecision = Literal[
    "approved",
    "changes_requested",
    "rejected",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _coerce_execution_id(
    execution_id: str | UUID | None,
) -> str | None:
    """Validate and normalize a workflow execution identifier."""
    if execution_id is None:
        return None

    if isinstance(execution_id, UUID):
        return str(execution_id)

    if not isinstance(execution_id, str):
        raise ValidationError(
            "Execution ID must be a UUID, UUID string, or None."
        )

    normalized_execution_id = execution_id.strip()

    if not normalized_execution_id:
        return None

    try:
        return str(UUID(normalized_execution_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValidationError(
            "Execution ID must be a valid UUID."
        ) from exc


def _validate_interrupt_type(
    interrupt_type: str | None,
) -> InterruptType:
    """
    Validate an interrupt type.

    Args:
        interrupt_type: Interrupt type supplied by a workflow node.

    Returns:
        A supported interrupt type.

    Raises:
        ValidationError: If the interrupt type is unsupported.
    """
    if not interrupt_type or not isinstance(interrupt_type, str):
        raise ValidationError("Interrupt type is required.")

    normalized_type = interrupt_type.strip().lower()

    if normalized_type not in VALID_INTERRUPT_TYPES:
        raise ValidationError(
            f"Unsupported interrupt type '{normalized_type}'."
        )

    return normalized_type  # type: ignore[return-value]


def _validate_review_decision(
    decision: str | None,
) -> ReviewDecision:
    """
    Validate a human review decision.

    Args:
        decision: Decision supplied by a human reviewer.

    Returns:
        A supported review decision.

    Raises:
        ValidationError: If the decision is unsupported.
    """
    if not decision or not isinstance(decision, str):
        raise ValidationError("Review decision is required.")

    normalized_decision = decision.strip().lower()

    if normalized_decision not in VALID_REVIEW_DECISIONS:
        raise ValidationError(
            f"Unsupported review decision '{normalized_decision}'."
        )

    return normalized_decision  # type: ignore[return-value]


def _sanitize_interrupt_metadata(
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Sanitize and bound interrupt metadata.

    Args:
        metadata: Arbitrary metadata supplied by a workflow node or reviewer.

    Returns:
        A bounded metadata dictionary with sensitive values redacted.
    """
    if not metadata or not isinstance(metadata, dict):
        return {}

    safe_metadata: dict[str, Any] = {}

    for key, value in list(metadata.items())[
        :MAX_INTERRUPT_METADATA_KEYS
    ]:
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

        if value is None or isinstance(
            value,
            (bool, int, float),
        ):
            safe_metadata[normalized_key] = value
        elif isinstance(value, str):
            safe_metadata[normalized_key] = truncate_text(
                value,
                MAX_INTERRUPT_METADATA_VALUE_CHARS,
            )
        elif isinstance(value, UUID):
            safe_metadata[normalized_key] = str(value)
        else:
            safe_metadata[normalized_key] = "[UNSERIALIZABLE]"

    return safe_metadata


def _append_interrupt_history(
    state: dict[str, Any],
    interrupt_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Append an interrupt record to immutable interrupt history.

    Args:
        state: Current workflow state.
        interrupt_record: Sanitized interrupt record.

    Returns:
        A bounded interrupt-history list.
    """
    existing_history = state.get("interrupt_history")

    if not isinstance(existing_history, list):
        existing_history = []

    updated_history = [*existing_history, interrupt_record]

    if len(updated_history) > MAX_INTERRUPT_HISTORY_ENTRIES:
        updated_history = updated_history[
            -MAX_INTERRUPT_HISTORY_ENTRIES:
        ]

    return updated_history


# ---------------------------------------------------------------------------
# Interrupt creation
# ---------------------------------------------------------------------------


def create_interrupt_payload(
    *,
    interrupt_type: str,
    reason: str,
    execution_id: str | UUID | None = None,
    workflow_step: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a safe interrupt payload for LangGraph.

    Args:
        interrupt_type: Type of interrupt.
        reason: Human-readable reason for the interrupt.
        execution_id: Workflow execution identifier.
        workflow_step: Workflow step requesting the interrupt.
        metadata: Optional non-sensitive interrupt metadata.

    Returns:
        A bounded interrupt payload.

    Raises:
        ValidationError: If interrupt inputs are invalid.
    """
    normalized_execution_id = _coerce_execution_id(execution_id)
    normalized_interrupt_type = _validate_interrupt_type(
        interrupt_type
    )

    if not reason or not isinstance(reason, str):
        raise ValidationError("Interrupt reason is required.")

    safe_reason = truncate_text(
        reason.strip(),
        MAX_INTERRUPT_REASON_CHARS,
    )

    safe_metadata = _sanitize_interrupt_metadata(metadata)

    payload: dict[str, Any] = {
        "interrupt_id": str(uuid4()),
        "interrupt_type": normalized_interrupt_type,
        "reason": safe_reason,
        "execution_id": normalized_execution_id,
        "workflow_step": workflow_step,
        "created_at": _utc_now_iso(),
        "status": "pending_review",
        "metadata": safe_metadata,
    }

    logger.info(
        "workflow_interrupt_created",
        interrupt_id=payload["interrupt_id"],
        interrupt_type=normalized_interrupt_type,
        execution_id=normalized_execution_id,
        workflow_step=workflow_step,
    )

    return payload


def request_human_review(
    state: dict[str, Any],
    *,
    reason: str,
    interrupt_type: str = "human_review",
    workflow_step: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create state updates that pause a workflow for human review.

    Args:
        state: Current workflow state.
        reason: Human-readable reason for requesting review.
        interrupt_type: Interrupt type.
        workflow_step: Workflow step requesting review.
        metadata: Optional non-sensitive interrupt metadata.

    Returns:
        State updates that mark the workflow as awaiting approval.

    Raises:
        ValidationError: If interrupt inputs are invalid.
    """
    execution_id = state.get("execution_id")

    interrupt_payload = create_interrupt_payload(
        interrupt_type=interrupt_type,
        reason=reason,
        execution_id=execution_id,
        workflow_step=workflow_step,
        metadata=metadata,
    )

    interrupt_history = _append_interrupt_history(
        state,
        interrupt_payload,
    )

    return {
        "approval_required": True,
        "approval_status": "pending",
        "review_status": "pending_approval",
        "current_interrupt": interrupt_payload,
        "interrupt_history": interrupt_history,
        "execution_logs": [
            (
                f"INFO: Workflow paused for {interrupt_type}: "
                f"{interrupt_payload['reason']}"
            )
        ],
    }


# ---------------------------------------------------------------------------
# Resume and decision handling
# ---------------------------------------------------------------------------


def resolve_human_review(
    state: dict[str, Any],
    *,
    decision: str,
    reviewer_id: str | UUID | None = None,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Apply a human review decision and create resume state updates.

    Args:
        state: Current workflow state.
        decision: Reviewer decision.
        reviewer_id: Optional reviewer identifier.
        comment: Optional reviewer comment.
        metadata: Optional non-sensitive decision metadata.

    Returns:
        State updates that resolve the interrupt and allow the workflow to
        continue, retry, or fail based on the decision.

    Raises:
        ValidationError: If the decision or inputs are invalid.
    """
    normalized_decision = _validate_review_decision(decision)

    current_interrupt = state.get("current_interrupt")

    if not isinstance(current_interrupt, dict):
        raise ValidationError(
            "Workflow does not have an active human-review interrupt."
        )

    interrupt_id = current_interrupt.get("interrupt_id")

    if not interrupt_id:
        raise ValidationError(
            "Active interrupt is missing an interrupt ID."
        )

    safe_comment = truncate_text(
        (comment or "").strip(),
        MAX_INTERRUPT_COMMENT_CHARS,
    )

    safe_metadata = _sanitize_interrupt_metadata(metadata)

    resolved_interrupt: dict[str, Any] = {
        **current_interrupt,
        "status": "resolved",
        "resolved_at": _utc_now_iso(),
        "decision": normalized_decision,
        "reviewer_id": (
            str(reviewer_id) if reviewer_id else None
        ),
        "comment": safe_comment,
        "decision_metadata": safe_metadata,
    }

    interrupt_history = _append_interrupt_history(
        state,
        resolved_interrupt,
    )

    state_updates: dict[str, Any] = {
        "approval_required": False,
        "approval_status": normalized_decision,
        "review_status": normalized_decision,
        "current_interrupt": None,
        "interrupt_history": interrupt_history,
        "approval_comment": safe_comment or None,
        "approved_by": (
            str(reviewer_id) if reviewer_id else None
        ),
        "approval_decided_at": _utc_now_iso(),
        "execution_logs": [
            (
                f"INFO: Human review resolved with decision "
                f"'{normalized_decision}'."
            )
        ],
    }

    if normalized_decision == "approved":
        state_updates["current_error"] = None
        state_updates["error_step"] = None
        state_updates["error_code"] = None
    elif normalized_decision == "changes_requested":
        state_updates["current_error"] = (
            "Human reviewer requested changes."
        )
        state_updates["error_step"] = current_interrupt.get(
            "workflow_step"
        )
        state_updates["error_code"] = "review_changes_requested"
        state_updates["recoverable_error"] = True
    else:
        state_updates["current_error"] = (
            "Workflow rejected by human reviewer."
        )
        state_updates["error_step"] = current_interrupt.get(
            "workflow_step"
        )
        state_updates["error_code"] = "workflow_rejected"
        state_updates["recoverable_error"] = False

    logger.info(
        "workflow_interrupt_resolved",
        interrupt_id=interrupt_id,
        decision=normalized_decision,
        execution_id=state.get("execution_id"),
    )

    return state_updates


def is_interrupt_active(
    state: dict[str, Any],
) -> bool:
    """
    Return whether a workflow interrupt is active.

    Args:
        state: Current workflow state.

    Returns:
        `True` when the workflow is awaiting human review.
    """
    current_interrupt = state.get("current_interrupt")

    if not isinstance(current_interrupt, dict):
        return False

    interrupt_status = current_interrupt.get("status")

    return interrupt_status == "pending_review"


def get_active_interrupt(
    state: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Return the active interrupt payload, if present.

    Args:
        state: Current workflow state.

    Returns:
        The active interrupt payload or `None`.
    """
    if not is_interrupt_active(state):
        return None

    current_interrupt = state.get("current_interrupt")

    if not isinstance(current_interrupt, dict):
        return None

    return current_interrupt


def get_interrupt_history(
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Return the bounded interrupt history.

    Args:
        state: Current workflow state.

    Returns:
        A copy of the interrupt-history list.
    """
    interrupt_history = state.get("interrupt_history")

    if not isinstance(interrupt_history, list):
        return []

    return list(interrupt_history)