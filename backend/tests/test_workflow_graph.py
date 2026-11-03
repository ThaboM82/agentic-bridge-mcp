"""
Comprehensive tests for workflow graph routing.

These tests cover ingestion, validation, approval, and error routing,
including success paths, warnings, failures, recoverable and non-recoverable
errors, retry limits, cancellation, approval outcomes, and terminal routing.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.core.enums import ApprovalStatus, JobStatus, WorkflowStep
from app.core.exceptions import ValidationError
from app.graph.state import (
    create_initial_state,
    set_error,
    set_workflow_status,
)
from app.graph.workflow import (
    END,
    route_after_approval,
    route_after_error,
    route_after_ingestion,
    route_after_validation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_state():
    return create_initial_state(
        execution_id=f"wf_{uuid4()}",
        brief_id=f"brief_{uuid4()}",
        raw_brief_text="Build a dashboard.",
        tenant_id=str(uuid4()),
        user_id=str(uuid4()),
        max_retries=3,
    )


@pytest.fixture
def state_factory():
    def _factory(**overrides):
        state = create_initial_state(
            execution_id=f"wf_{uuid4()}",
            brief_id=f"brief_{uuid4()}",
            raw_brief_text="Build a dashboard.",
            tenant_id=str(uuid4()),
            user_id=str(uuid4()),
            max_retries=3,
        )

        return {
            **state,
            **overrides,
        }

    return _factory


# ---------------------------------------------------------------------------
# Ingestion routing
# ---------------------------------------------------------------------------


def test_ingestion_routes_to_requirements(
    state_factory=None,
):
    state = create_initial_state(
        execution_id="wf_1",
        brief_id="brief_1",
        raw_brief_text="Build a dashboard.",
    )

    state_with_requirements = {
        **state,
        "extracted_requirements": [
            "Create a dashboard"
        ],
    }

    assert (
        route_after_ingestion(
            state_with_requirements
        )
        == "extract_requirements"
    )


def test_ingestion_routes_to_error_without_requirements():
    state = create_initial_state(
        execution_id="wf_1",
        brief_id="brief_1",
        raw_brief_text="Build a dashboard.",
    )

    assert (
        route_after_ingestion(state)
        == "handle_error"
    )


def test_ingestion_routes_to_error_on_failure():
    state = create_initial_state(
        execution_id="wf_1",
        brief_id="brief_1",
        raw_brief_text="Build a dashboard.",
    )

    failed_state = set_error(
        state,
        message="Ingestion failed.",
    )

    assert (
        route_after_ingestion(
            failed_state
        )
        == "handle_error"
    )


def test_ingestion_routes_to_requirements_with_multiple_requirements(
    base_state,
):
    state = {
        **base_state,
        "extracted_requirements": [
            "Create a dashboard",
            "Show key metrics",
        ],
    }

    assert (
        route_after_ingestion(state)
        == "extract_requirements"
    )


def test_ingestion_routes_to_error_with_empty_requirements(
    base_state,
):
    state = {
        **base_state,
        "extracted_requirements": [],
    }

    assert (
        route_after_ingestion(state)
        == "handle_error"
    )


def test_ingestion_routes_to_error_with_none_requirements(
    base_state,
):
    state = {
        **base_state,
        "extracted_requirements": None,
    }

    assert (
        route_after_ingestion(state)
        == "handle_error"
    )


def test_ingestion_routes_to_error_for_recoverable_failure(
    base_state,
):
    failed_state = set_error(
        base_state,
        message="Document ingestion failed.",
        error_step=WorkflowStep.INGESTION,
        error_code="ingestion_failed",
        recoverable=True,
    )

    assert (
        route_after_ingestion(
            failed_state
        )
        == "handle_error"
    )


def test_ingestion_routes_to_error_for_non_recoverable_failure(
    base_state,
):
    failed_state = set_error(
        base_state,
        message="Unsupported document format.",
        error_step=WorkflowStep.INGESTION,
        error_code="unsupported_document",
        recoverable=False,
    )

    assert (
        route_after_ingestion(
            failed_state
        )
        == "handle_error"
    )


# ---------------------------------------------------------------------------
# Validation routing
# ---------------------------------------------------------------------------


def test_validation_routes_to_review_on_warning(
    base_state,
):
    warning_state = {
        **base_state,
        "validation_outcome": "warning",
    }

    assert (
        route_after_validation(
            warning_state
        )
        == "review"
    )


def test_validation_routes_to_error_on_failure(
    base_state,
):
    failed_state = {
        **base_state,
        "validation_outcome": "failed",
    }

    assert (
        route_after_validation(
            failed_state
        )
        == "handle_error"
    )


def test_validation_routes_to_review_on_success(
    base_state,
):
    passed_state = {
        **base_state,
        "validation_outcome": "passed",
    }

    assert (
        route_after_validation(
            passed_state
        )
        == "review"
    )


def test_validation_routes_to_error_for_recoverable_failure(
    base_state,
):
    failed_state = {
        **base_state,
        "validation_outcome": "failed",
        "recoverable_error": True,
        "retry_count": 0,
        "current_error": "Validation failed.",
    }

    assert (
        route_after_validation(
            failed_state
        )
        == "handle_error"
    )


def test_validation_routes_to_error_for_non_recoverable_failure(
    base_state,
):
    failed_state = {
        **base_state,
        "validation_outcome": "failed",
        "recoverable_error": False,
        "current_error": "Invalid generated output.",
    }

    assert (
        route_after_validation(
            failed_state
        )
        == "handle_error"
    )


def test_validation_routes_to_error_for_unknown_outcome(
    base_state,
):
    unknown_state = {
        **base_state,
        "validation_outcome": "unknown",
    }

    assert (
        route_after_validation(
            unknown_state
        )
        == "handle_error"
    )


def test_validation_routes_to_error_for_missing_outcome(
    base_state,
):
    assert (
        route_after_validation(
            base_state
        )
        == "handle_error"
    )


# ---------------------------------------------------------------------------
# Approval routing
# ---------------------------------------------------------------------------


def test_approval_routes_to_publish_when_approved(
    base_state,
):
    approved_state = {
        **base_state,
        "approval_status": ApprovalStatus.APPROVED,
        "review_status": "approved",
    }

    assert (
        route_after_approval(
            approved_state
        )
        == "publish"
    )


def test_approval_routes_to_error_when_rejected(
    base_state,
):
    rejected_state = {
        **base_state,
        "approval_status": ApprovalStatus.REJECTED,
        "review_status": "rejected",
    }

    assert (
        route_after_approval(
            rejected_state
        )
        == "handle_error"
    )


def test_approval_routes_to_error_when_pending(
    base_state,
):
    pending_state = {
        **base_state,
        "approval_status": ApprovalStatus.PENDING,
        "review_status": "pending_approval",
    }

    assert (
        route_after_approval(
            pending_state
        )
        == "handle_error"
    )


def test_approval_routes_to_error_when_changes_requested(
    base_state,
):
    changes_state = {
        **base_state,
        "approval_status": ApprovalStatus.CHANGES_REQUESTED,
        "review_status": "changes_requested",
    }

    assert (
        route_after_approval(
            changes_state
        )
        == "handle_error"
    )


def test_approval_routes_to_error_without_review_status(
    base_state,
):
    assert (
        route_after_approval(base_state)
        == "handle_error"
    )


def test_approval_routes_to_error_for_unknown_review_status(
    base_state,
):
    unknown_state = {
        **base_state,
        "review_status": "unknown",
    }

    assert (
        route_after_approval(
            unknown_state
        )
        == "handle_error"
    )


# ---------------------------------------------------------------------------
# Error routing
# ---------------------------------------------------------------------------


def test_non_recoverable_error_ends_workflow(
    base_state,
):
    failed_state = set_error(
        base_state,
        message="Invalid workflow state.",
        recoverable=False,
    )

    assert (
        route_after_error(failed_state)
        == END
    )


def test_recoverable_error_can_retry(
    base_state,
):
    recoverable_state = set_error(
        base_state,
        message="Provider timeout.",
        error_step=WorkflowStep.DESIGN,
        recoverable=True,
    )

    assert (
        route_after_error(
            recoverable_state
        )
        == "retry"
    )


def test_recoverable_error_ends_workflow_after_retry_limit(
    base_state,
):
    exhausted_state = {
        **base_state,
        "recoverable_error": True,
        "retry_count": 3,
        "max_retries": 3,
        "current_error": "Provider timeout.",
    }

    assert (
        route_after_error(
            exhausted_state
        )
        == END
    )


def test_recoverable_error_ends_workflow_beyond_retry_limit(
    base_state,
):
    exhausted_state = {
        **base_state,
        "recoverable_error": True,
        "retry_count": 5,
        "max_retries": 3,
        "current_error": "Provider timeout.",
    }

    assert (
        route_after_error(
            exhausted_state
        )
        == END
    )


def test_recoverable_error_without_error_message_ends_workflow(
    base_state,
):
    invalid_state = {
        **base_state,
        "recoverable_error": True,
        "retry_count": 0,
        "current_error": None,
    }

    assert (
        route_after_error(invalid_state)
        == END
    )


def test_cancelled_workflow_ends_workflow(
    base_state,
):
    cancelled_state = {
        **base_state,
        "cancel_requested": True,
    }

    assert (
        route_after_error(
            cancelled_state
        )
        == END
    )


def test_terminal_completed_workflow_ends_workflow(
    base_state,
):
    completed_state = set_workflow_status(
        base_state,
        JobStatus.COMPLETED,
    )

    assert (
        route_after_error(
            completed_state
        )
        == END
    )


def test_terminal_failed_workflow_ends_workflow(
    base_state,
):
    failed_state = set_workflow_status(
        base_state,
        JobStatus.FAILED,
    )

    assert (
        route_after_error(failed_state)
        == END
    )


def test_terminal_cancelled_workflow_ends_workflow(
    base_state,
):
    cancelled_state = set_workflow_status(
        base_state,
        JobStatus.CANCELLED,
    )

    assert (
        route_after_error(
            cancelled_state
        )
        == END
    )


# ---------------------------------------------------------------------------
# Routing consistency
# ---------------------------------------------------------------------------


def test_ingestion_error_takes_precedence_over_requirements(
    base_state,
):
    state = {
        **base_state,
        "extracted_requirements": [
            "Create a dashboard"
        ],
    }

    failed_state = set_error(
        state,
        message="Ingestion failed.",
    )

    assert (
        route_after_ingestion(
            failed_state
        )
        == "handle_error"
    )


def test_validation_error_takes_precedence_over_success(
    base_state,
):
    state = {
        **base_state,
        "validation_outcome": "passed",
    }

    failed_state = set_error(
        state,
        message="Validation crashed.",
    )

    assert (
        route_after_validation(
            failed_state
        )
        == "handle_error"
    )


def test_approval_rejection_takes_precedence_over_approval_flag(
    base_state,
):
    rejected_state = {
        **base_state,
        "approval_required": True,
        "approval_status": ApprovalStatus.REJECTED,
        "review_status": "rejected",
    }

    assert (
        route_after_approval(
            rejected_state
        )
        == "handle_error"
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
