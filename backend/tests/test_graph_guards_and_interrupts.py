"""
Comprehensive tests for workflow guards and human-review interrupts.

These tests cover retry decisions, blocking-error detection, approval
requirements, requirements and validation routing, interrupt creation,
human-review requests and resolutions, interrupt history, metadata
sanitization, and security boundaries.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.core.enums import (
    ApprovalStatus,
    JobStatus,
    WorkflowStep,
)
from app.core.exceptions import ValidationError
from app.graph.guards import (
    ROUTE_CONTINUE,
    ROUTE_FAILED,
    ROUTE_HUMAN_REVIEW,
    ROUTE_RETRY,
    has_blocking_error,
    requires_human_approval,
    route_after_requirements,
    route_after_validation,
    should_retry,
)
from app.graph.interrupts import (
    create_interrupt_payload,
    get_active_interrupt,
    get_interrupt_history,
    is_interrupt_active,
    request_human_review,
    resolve_human_review,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def execution_id() -> str:
    """Return a stable workflow execution ID."""
    return f"wf_{uuid4()}"


@pytest.fixture
def reviewer_id() -> str:
    """Return a stable reviewer ID."""
    return str(uuid4())


@pytest.fixture
def base_state(
    execution_id: str,
) -> dict:
    """Return a valid base workflow state."""
    return {
        "execution_id": execution_id,
        "brief_id": f"brief_{uuid4()}",
        "tenant_id": str(uuid4()),
        "user_id": str(uuid4()),
        "workflow_status": JobStatus.PROCESSING,
        "current_step": WorkflowStep.REQUIREMENTS,
        "retry_count": 0,
        "max_retries": 3,
        "approval_required": False,
        "review_status": None,
        "current_error": None,
        "recoverable_error": False,
        "interrupt_history": [],
        "current_interrupt": None,
    }


# ---------------------------------------------------------------------------
# Retry guards
# ---------------------------------------------------------------------------


def test_should_retry():
    assert should_retry(
        {
            "recoverable_error": True,
            "retry_count": 0,
            "max_retries": 3,
        }
    )


    assert not should_retry(
        {
            "recoverable_error": True,
            "retry_count": 3,
            "max_retries": 3,
        }
    )


    assert not should_retry(
        {
            "recoverable_error": False,
            "retry_count": 0,
            "max_retries": 3,
        }
    )


def test_should_retry_before_limit(base_state):
    base_state.update(
        {
            "recoverable_error": True,
            "retry_count": 2,
        }
    )

    assert should_retry(base_state) is True


def test_should_retry_at_limit(base_state):
    base_state.update(
        {
            "recoverable_error": True,
            "retry_count": 3,
        }
    )

    assert should_retry(base_state) is False


def test_should_retry_beyond_limit(base_state):
    base_state.update(
        {
            "recoverable_error": True,
            "retry_count": 5,
        }
    )

    assert should_retry(base_state) is False


def test_should_retry_rejects_non_recoverable_error(base_state):
    base_state.update(
        {
            "recoverable_error": False,
            "retry_count": 0,
        }
    )

    assert should_retry(base_state) is False


def test_should_retry_rejects_cancelled_workflow(base_state):
    base_state.update(
        {
            "recoverable_error": True,
            "cancel_requested": True,
        }
    )

    assert should_retry(base_state) is False


def test_should_retry_rejects_missing_error(base_state):
    base_state.update(
        {
            "recoverable_error": True,
            "current_error": None,
        }
    )

    assert should_retry(base_state) is False


# ---------------------------------------------------------------------------
# Blocking-error guards
# ---------------------------------------------------------------------------


def test_has_blocking_error():
    assert has_blocking_error(
        {
            "current_error": "Invalid input",
            "recoverable_error": False,
        }
    )


    assert not has_blocking_error(
        {
            "current_error": "Provider timeout",
            "recoverable_error": True,
        }
    )


def test_has_blocking_error_requires_error(base_state):
    assert has_blocking_error(base_state) is False


def test_has_blocking_error_with_recoverable_error(base_state):
    base_state.update(
        {
            "current_error": "Provider timeout",
            "recoverable_error": True,
        }
    )

    assert has_blocking_error(base_state) is False


def test_has_blocking_error_with_blocking_error(base_state):
    base_state.update(
        {
            "current_error": "Invalid workflow state",
            "recoverable_error": False,
        }
    )

    assert has_blocking_error(base_state) is True


# ---------------------------------------------------------------------------
# Approval guards
# ---------------------------------------------------------------------------


def test_requires_human_approval():
    assert requires_human_approval(
        {
            "approval_required": True,
        }
    )


    assert requires_human_approval(
        {
            "review_status": "pending_approval",
        }
    )


    assert not requires_human_approval(
        {
            "approval_required": False,
            "review_status": "approved",
        }
    )


def test_requires_human_approval_when_configured(base_state):
    base_state["approval_required"] = True

    assert (
        requires_human_approval(base_state)
        is True
    )


def test_requires_human_approval_when_pending(base_state):
    base_state["review_status"] = "pending_approval"

    assert (
        requires_human_approval(base_state)
        is True
    )


def test_does_not_require_approval_when_approved(base_state):
    base_state.update(
        {
            "approval_required": False,
            "review_status": "approved",
        }
    )

    assert (
        requires_human_approval(base_state)
        is False
    )


def test_does_not_require_approval_when_rejected(base_state):
    base_state.update(
        {
            "approval_required": False,
            "review_status": "rejected",
        }
    )

    assert (
        requires_human_approval(base_state)
        is False
    )


# ---------------------------------------------------------------------------
# Requirements routing
# ---------------------------------------------------------------------------


def test_route_after_requirements_success():
    route = route_after_requirements(
        {
            "execution_id": str(uuid4()),
            "extracted_requirements": [
                "Build a dashboard",
            ],
        }
    )

    assert route == ROUTE_CONTINUE


def test_route_after_requirements_missing_requirements():
    route = route_after_requirements(
        {
            "execution_id": str(uuid4()),
            "extracted_requirements": [],
        }
    )

    assert route == ROUTE_HUMAN_REVIEW


def test_route_after_requirements_none_requirements(base_state):
    base_state["extracted_requirements"] = None

    assert (
        route_after_requirements(base_state)
        == ROUTE_HUMAN_REVIEW
    )


def test_route_after_requirements_with_multiple_requirements(base_state):
    base_state["extracted_requirements"] = [
        "Build a dashboard",
        "Show key metrics",
    ]

    assert (
        route_after_requirements(base_state)
        == ROUTE_CONTINUE
    )


def test_route_after_requirements_cancelled(base_state):
    base_state.update(
        {
            "extracted_requirements": [
                "Build a dashboard"
            ],
            "cancel_requested": True,
        }
    )

    assert (
        route_after_requirements(base_state)
        == ROUTE_CANCELLED
        if hasattr(
            route_after_requirements,
            "__call__",
        )
        else ROUTE_HUMAN_REVIEW
    )


# ---------------------------------------------------------------------------
# Validation routing
# ---------------------------------------------------------------------------


def test_route_after_validation_success():
    route = route_after_validation(
        {
            "execution_id": str(uuid4()),
            "validation_outcome": "passed",
        }
    )

    assert route == ROUTE_HUMAN_REVIEW


def test_route_after_validation_retry():
    route = route_after_validation(
        {
            "execution_id": str(uuid4()),
            "validation_outcome": "failed",
            "recoverable_error": True,
            "retry_count": 0,
            "max_retries": 3,
        }
    )

    assert route == ROUTE_RETRY


def test_route_after_validation_failure():
    route = route_after_validation(
        {
            "execution_id": str(uuid4()),
            "validation_outcome": "failed",
            "recoverable_error": False,
        }
    )

    assert route == ROUTE_FAILED


def test_route_after_validation_retry_exhausted(base_state):
    base_state.update(
        {
            "validation_outcome": "failed",
            "recoverable_error": True,
            "retry_count": 3,
        }
    )

    assert (
        route_after_validation(base_state)
        == ROUTE_HUMAN_REVIEW
    )


def test_route_after_validation_warning_routes_to_review(base_state):
    base_state["validation_outcome"] = "warning"

    assert (
        route_after_validation(base_state)
        == ROUTE_HUMAN_REVIEW
    )


def test_route_after_validation_cancelled(base_state):
    base_state.update(
        {
            "validation_outcome": "passed",
            "cancel_requested": True,
        }
    )

    assert (
        route_after_validation(base_state)
        == ROUTE_CANCELLED
        if hasattr(
            route_after_validation,
            "__call__",
        )
        else ROUTE_HUMAN_REVIEW
    )


# ---------------------------------------------------------------------------
# Interrupt payload creation
# ---------------------------------------------------------------------------


def test_create_interrupt_payload():
    payload = create_interrupt_payload(
        interrupt_type="approval_required",
        reason=(
            "Human approval is required before publication."
        ),
        execution_id=str(uuid4()),
        workflow_step="review",
    )

    assert (
        payload["interrupt_type"]
        == "approval_required"
    )
    assert (
        payload["status"]
        == "pending_review"
    )
    assert payload["reason"].startswith(
        "Human approval"
    )


def test_create_interrupt_payload_rejects_invalid_type():
    with pytest.raises(ValidationError):
        create_interrupt_payload(
            interrupt_type="unknown",
            reason="Invalid interrupt.",
        )


def test_create_interrupt_payload_rejects_empty_reason():
    with pytest.raises(ValidationError):
        create_interrupt_payload(
            interrupt_type="approval_required",
            reason="",
        )


def test_create_interrupt_payload_rejects_whitespace_reason():
    with pytest.raises(ValidationError):
        create_interrupt_payload(
            interrupt_type="approval_required",
            reason="   ",
        )


def test_create_interrupt_payload_rejects_none_reason():
    with pytest.raises(ValidationError):
        create_interrupt_payload(
            interrupt_type="approval_required",
            reason=None,
        )


def test_create_interrupt_payload_generates_interrupt_id():
    payload = create_interrupt_payload(
        interrupt_type="approval_required",
        reason="Approval required.",
    )

    assert payload["interrupt_id"]
    assert payload["created_at"]


def test_create_interrupt_payload_preserves_workflow_context():
    execution_id = str(uuid4())

    payload = create_interrupt_payload(
        interrupt_type="approval_required",
        reason="Approval required.",
        execution_id=execution_id,
        workflow_step="review",
    )

    assert (
        payload["execution_id"]
        == execution_id
    )
    assert payload["workflow_step"] == "review"


# ---------------------------------------------------------------------------
# Human-review requests
# ---------------------------------------------------------------------------


def test_request_human_review(base_state):
    updates = request_human_review(
        base_state,
        reason="Review generated artifacts.",
        workflow_step="review",
    )

    assert (
        updates["approval_required"]
        is True
    )
    assert (
        updates["review_status"]
        == "pending_approval"
    )
    assert updates["current_interrupt"][
        "status"
    ] == "pending_review"
    assert (
        len(updates["interrupt_history"])
        == 1
    )


def test_request_human_review_sets_approval_status(base_state):
    updates = request_human_review(
        base_state,
        reason="Review required.",
    )

    assert (
        updates["approval_status"]
        == ApprovalStatus.PENDING
    )


def test_request_human_review_rejects_empty_reason(base_state):
    with pytest.raises(ValidationError):
        request_human_review(
            base_state,
            reason="",
        )


def test_request_human_review_rejects_whitespace_reason(base_state):
    with pytest.raises(ValidationError):
        request_human_review(
            base_state,
            reason="   ",
        )


def test_request_human_review_rejects_none_reason(base_state):
    with pytest.raises(ValidationError):
        request_human_review(
            base_state,
            reason=None,
        )


def test_request_human_review_sanitizes_reason(base_state):
    updates = request_human_review(
        base_state,
        reason=(
            "Review required. api_key=sk-should-not-leak"
        ),
    )

    serialized_updates = str(updates)

    assert (
        "sk-should-not-leak"
        not in serialized_updates
    )


# ---------------------------------------------------------------------------
# Active interrupt inspection
# ---------------------------------------------------------------------------


def test_is_interrupt_active(base_state):
    updates = request_human_review(
        base_state,
        reason="Review required.",
    )

    active_state = {
        **base_state,
        **updates,
    }

    assert (
        is_interrupt_active(active_state)
        is True
    )
    assert (
        get_active_interrupt(active_state)
        is not None
    )


def test_interrupt_is_not_active_without_interrupt(base_state):
    assert (
        is_interrupt_active(base_state)
        is False
    )
    assert (
        get_active_interrupt(base_state)
        is None
    )


def test_interrupt_is_not_active_after_resolution(base_state, reviewer_id):
    request_updates = request_human_review(
        base_state,
        reason="Review required.",
    )

    active_state = {
        **base_state,
        **request_updates,
    }

    resolution_updates = resolve_human_review(
        active_state,
        decision="approved",
        reviewer_id=reviewer_id,
    )

    resolved_state = {
        **active_state,
        **resolution_updates,
    }

    assert (
        is_interrupt_active(resolved_state)
        is False
    )
    assert (
        get_active_interrupt(resolved_state)
        is None
    )


# ---------------------------------------------------------------------------
# Human-review resolution
# ---------------------------------------------------------------------------


def test_resolve_human_review_approval(
    base_state,
    reviewer_id,
):
    request_updates = request_human_review(
        base_state,
        reason="Review required.",
    )

    active_state = {
        **base_state,
        **request_updates,
    }

    resolution = resolve_human_review(
        active_state,
        decision="approved",
        reviewer_id=reviewer_id,
        comment="Approved.",
    )

    assert (
        resolution["approval_required"]
        is False
    )
    assert (
        resolution["review_status"]
        == "approved"
    )
    assert (
        resolution["approval_status"]
        == ApprovalStatus.APPROVED
    )
    assert (
        resolution["current_interrupt"]
        is None
    )
    assert (
        resolution["current_error"]
        is None
    )
    assert (
        len(resolution["interrupt_history"])
        == 2
    )


def test_resolve_human_review_rejection(
    base_state,
    reviewer_id,
):
    request_updates = request_human_review(
        base_state,
        reason="Review required.",
    )

    active_state = {
        **base_state,
        **request_updates,
    }

    resolution = resolve_human_review(
        active_state,
        decision="rejected",
        reviewer_id=reviewer_id,
    )

    assert (
        resolution["review_status"]
        == "rejected"
    )
    assert (
        resolution["approval_status"]
        == ApprovalStatus.REJECTED
    )
    assert (
        resolution["recoverable_error"]
        is False
    )
    assert resolution["current_error"]
    assert (
        len(resolution["interrupt_history"])
        == 2
    )


def test_resolve_human_review_request_changes(
    base_state,
    reviewer_id,
):
    request_updates = request_human_review(
        base_state,
        reason="Review required.",
    )

    active_state = {
        **base_state,
        **request_updates,
    }

    resolution = resolve_human_review(
        active_state,
        decision="request_changes",
        reviewer_id=reviewer_id,
        comment="Clarify metrics.",
    )

    assert (
        resolution["review_status"]
        == "changes_requested"
    )
    assert (
        resolution["recoverable_error"]
        is True
    )
    assert (
        len(resolution["interrupt_history"])
        == 2
    )


def test_resolve_human_review_requires_active_interrupt():
    with pytest.raises(ValidationError):
        resolve_human_review(
            {},
            decision="approved",
        )


def test_resolve_human_review_rejects_invalid_decision(
    base_state,
    reviewer_id,
):
    request_updates = request_human_review(
        base_state,
        reason="Review required.",
    )

    active_state = {
        **base_state,
        **request_updates,
    }

    with pytest.raises(ValidationError):
        resolve_human_review(
            active_state,
            decision="maybe",
            reviewer_id=reviewer_id,
        )


def test_resolve_human_review_rejects_missing_reviewer(
    base_state,
):
    request_updates = request_human_review(
        base_state,
        reason="Review required.",
    )

    active_state = {
        **base_state,
        **request_updates,
    }

    with pytest.raises(ValidationError):
        resolve_human_review(
            active_state,
            decision="approved",
            reviewer_id="",
        )


# ---------------------------------------------------------------------------
# Interrupt history
# ---------------------------------------------------------------------------


def test_get_interrupt_history_returns_copy(base_state):
    updates = request_human_review(
        base_state,
        reason="Review required.",
    )

    active_state = {
        **base_state,
        **updates,
    }

    history = get_interrupt_history(
        active_state
    )

    assert len(history) == 1

    history.clear()

    assert (
        len(
            get_interrupt_history(
                active_state
            )
        )
        == 1
    )


def test_interrupt_history_records_request_and_resolution(
    base_state,
    reviewer_id,
):
    request_updates = request_human_review(
        base_state,
        reason="Review required.",
    )

    active_state = {
        **base_state,
        **request_updates,
    }

    resolution_updates = resolve_human_review(
        active_state,
        decision="approved",
        reviewer_id=reviewer_id,
    )

    resolved_state = {
        **active_state,
        **resolution_updates,
    }

    history = get_interrupt_history(
        resolved_state
    )

    assert len(history) == 2
    assert history[0]["event"] == "requested"
    assert history[1]["event"] == "resolved"


# ---------------------------------------------------------------------------
# Interrupt metadata sanitization
# ---------------------------------------------------------------------------


def test_interrupt_metadata_is_sanitized():
    payload = create_interrupt_payload(
        interrupt_type="policy_review",
        reason="Review policy.",
        metadata={
            "source": "confluence",
            "api_key": "should-not-leak",
            "generated_code": "print('hello')",
        },
    )

    assert (
        payload["metadata"]["source"]
        == "confluence"
    )
    assert (
        payload["metadata"]["api_key"]
        == "[REDACTED]"
    )
    assert (
        payload["metadata"][
            "generated_code"
        ]
        == "[REDACTED]"
    )


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "api_key",
        "apikey",
        "password",
        "secret",
        "token",
        "authorization",
        "credential",
        "prompt",
        "completion",
        "messages",
        "embedding",
        "vector",
        "generated_code",
        "storage_url",
        "signed_url",
        "database_url",
        "connection_string",
    ],
)
def test_interrupt_metadata_redacts_sensitive_keys(
    sensitive_key: str,
):
    payload = create_interrupt_payload(
        interrupt_type="policy_review",
        reason="Review policy.",
        metadata={
            "safe": "value",
            sensitive_key: "should-not-leak",
        },
    )

    assert (
        payload["metadata"]["safe"]
        == "value"
    )
    assert (
        payload["metadata"][sensitive_key]
        == "[REDACTED]"
    )


def test_interrupt_reason_does_not_leak_secrets(base_state):
    updates = request_human_review(
        base_state,
        reason=(
            "Review required. "
            "password=should-not-leak"
        ),
    )

    serialized_updates = str(updates)

    assert (
        "should-not-leak"
        not in serialized_updates
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
