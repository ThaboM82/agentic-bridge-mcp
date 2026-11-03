"""
Comprehensive tests for workflow recovery and routing.

These tests cover workflow route resolution, error classification, retry
policies and backoff, recoverable and blocking failures, human-review
escalation, cancellation, and recovery-action decisions.
"""

from __future__ import annotations

import pytest

from app.core.enums import JobStatus, WorkflowStep
from app.core.exceptions import (
    AppError,
    AuthenticationError,
    AuthorizationError,
    ExternalProviderError,
    NotFoundError,
    ProviderTimeoutError,
    RateLimitError,
    ValidationError,
    WorkflowStateError,
)
from app.graph.recovery import (
    ERROR_CODE_AUTHORIZATION,
    ERROR_CODE_PROVIDER_TIMEOUT,
    ERROR_CODE_RATE_LIMIT,
    ERROR_CODE_UNEXPECTED,
    ERROR_CODE_VALIDATION,
    ERROR_CODE_WORKFLOW_STATE,
    calculate_retry_backoff,
    classify_exception,
    decide_recovery_action,
    get_retry_policy,
    prepare_cancellation,
    prepare_human_review,
    prepare_retry,
    record_blocking_failure,
    record_recoverable_failure,
    should_retry,
)
from app.graph.routing import (
    ROUTE_AWAITING_REVIEW,
    ROUTE_CANCELLED,
    ROUTE_COMPLETED,
    ROUTE_EXTRACT_REQUIREMENTS,
    ROUTE_FAILED,
    ROUTE_GENERATE_CODE,
    ROUTE_GENERATE_DESIGN,
    ROUTE_HUMAN_REVIEW,
    ROUTE_PUBLISH_ARTIFACTS,
    ROUTE_RETRY,
    ROUTE_VALIDATE_OUTPUT,
    get_workflow_routes,
    route_after_design,
    route_after_generation,
    route_after_publication,
    route_after_requirements,
    route_after_retrieval,
    route_after_review,
    route_after_validation,
    route_on_error,
    route_on_start,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state() -> dict:
    """Return a valid base workflow state for recovery and routing tests."""
    return {
        "execution_id": "wf_123",
        "brief_id": "brief_456",
        "raw_brief_text": "Build a dashboard.",
        "retry_count": 0,
        "max_retries": 3,
        "retry_backoff_seconds": 2.0,
        "workflow_status": JobStatus.PROCESSING,
        "current_step": WorkflowStep.INTAKE,
        "approval_required": False,
        "cancel_requested": False,
        "recovery_history": [],
    }


# ---------------------------------------------------------------------------
# Workflow route registry
# ---------------------------------------------------------------------------


def test_get_workflow_routes():
    routes = get_workflow_routes()

    assert routes["RETRIEVE_CONTEXT"] == "retrieve_context"
    assert routes["HUMAN_REVIEW"] == "human_review"
    assert routes["COMPLETED"] == "completed"

    expected_routes = {
        "START",
        "RETRIEVE_CONTEXT",
        "EXTRACT_REQUIREMENTS",
        "GENERATE_DESIGN",
        "GENERATE_CODE",
        "VALIDATE_OUTPUT",
        "HUMAN_REVIEW",
        "PUBLISH_ARTIFACTS",
        "AWAITING_REVIEW",
        "RETRY",
        "COMPLETED",
        "CANCELLED",
        "FAILED",
    }

    assert expected_routes.issubset(set(routes))


# ---------------------------------------------------------------------------
# Start routing
# ---------------------------------------------------------------------------


def test_route_on_start_success(state):
    assert (
        route_on_start(state)
        == ROUTE_EXTRACT_REQUIREMENTS
    )


def test_route_on_start_missing_brief():
    route = route_on_start(
        {
            "execution_id": "wf_1",
            "brief_id": "",
            "raw_brief_text": "Brief",
        }
    )

    assert route == ROUTE_FAILED


def test_route_on_start_missing_raw_brief():
    route = route_on_start(
        {
            "execution_id": "wf_1",
            "brief_id": "brief_1",
            "raw_brief_text": "",
        }
    )

    assert route == ROUTE_FAILED


def test_route_on_start_missing_execution_id():
    route = route_on_start(
        {
            "execution_id": "",
            "brief_id": "brief_1",
            "raw_brief_text": "Brief",
        }
    )

    assert route == ROUTE_FAILED


def test_route_on_start_cancelled(state):
    state["cancel_requested"] = True

    assert route_on_start(state) == ROUTE_CANCELLED


def test_route_on_start_failed_state(state):
    state["workflow_status"] = JobStatus.FAILED

    assert route_on_start(state) == ROUTE_FAILED


# ---------------------------------------------------------------------------
# Retrieval routing
# ---------------------------------------------------------------------------


def test_route_after_retrieval(state):
    assert (
        route_after_retrieval(state)
        == ROUTE_EXTRACT_REQUIREMENTS
    )


def test_route_after_retrieval_cancelled(state):
    state["cancel_requested"] = True

    assert (
        route_after_retrieval(state)
        == ROUTE_CANCELLED
    )


def test_route_after_retrieval_failed(state):
    state["workflow_status"] = JobStatus.FAILED

    assert (
        route_after_retrieval(state)
        == ROUTE_FAILED
    )


# ---------------------------------------------------------------------------
# Requirements routing
# ---------------------------------------------------------------------------


def test_route_after_requirements_success(state):
    state["extracted_requirements"] = [
        "Build a dashboard",
    ]

    assert (
        route_after_requirements(state)
        == ROUTE_GENERATE_DESIGN
    )


def test_route_after_requirements_missing_requirements(state):
    state["extracted_requirements"] = []

    assert (
        route_after_requirements(state)
        == ROUTE_HUMAN_REVIEW
    )


def test_route_after_requirements_none_requirements(state):
    state["extracted_requirements"] = None

    assert (
        route_after_requirements(state)
        == ROUTE_HUMAN_REVIEW
    )


def test_route_after_requirements_cancelled(state):
    state["extracted_requirements"] = [
        "Build a dashboard",
    ]
    state["cancel_requested"] = True

    assert (
        route_after_requirements(state)
        == ROUTE_CANCELLED
    )


# ---------------------------------------------------------------------------
# Design routing
# ---------------------------------------------------------------------------


def test_route_after_design_success(state):
    state["design_tokens"] = {
        "primary_color": "#000000",
    }

    assert (
        route_after_design(state)
        == ROUTE_GENERATE_CODE
    )


def test_route_after_design_missing_design(state):
    state["design_tokens"] = {}

    assert (
        route_after_design(state)
        == ROUTE_HUMAN_REVIEW
    )


def test_route_after_design_none_design(state):
    state["design_tokens"] = None

    assert (
        route_after_design(state)
        == ROUTE_HUMAN_REVIEW
    )


def test_route_after_design_cancelled(state):
    state["design_tokens"] = {
        "primary_color": "#000000",
    }
    state["cancel_requested"] = True

    assert (
        route_after_design(state)
        == ROUTE_CANCELLED
    )


# ---------------------------------------------------------------------------
# Generation routing
# ---------------------------------------------------------------------------


def test_route_after_generation_success(state):
    state["generated_code_files"] = {
        "app/main.py": "print('hello')",
    }

    assert (
        route_after_generation(state)
        == ROUTE_VALIDATE_OUTPUT
    )


def test_route_after_generation_missing_files(state):
    state["generated_code_files"] = {}

    assert (
        route_after_generation(state)
        == ROUTE_HUMAN_REVIEW
    )


def test_route_after_generation_none_files(state):
    state["generated_code_files"] = None

    assert (
        route_after_generation(state)
        == ROUTE_HUMAN_REVIEW
    )


def test_route_after_generation_cancelled(state):
    state["generated_code_files"] = {
        "app/main.py": "print('hello')",
    }
    state["cancel_requested"] = True

    assert (
        route_after_generation(state)
        == ROUTE_CANCELLED
    )


# ---------------------------------------------------------------------------
# Validation routing
# ---------------------------------------------------------------------------


def test_route_after_validation_passed(state):
    state["validation_outcome"] = "passed"

    assert (
        route_after_validation(state)
        == ROUTE_HUMAN_REVIEW
    )


def test_route_after_validation_retry(state):
    state.update(
        {
            "validation_outcome": "failed",
            "error_code": ERROR_CODE_PROVIDER_TIMEOUT,
            "recoverable_error": True,
        }
    )

    assert (
        route_after_validation(state)
        == ROUTE_RETRY
    )


def test_route_after_validation_human_review_when_retries_exhausted(
    state,
):
    state.update(
        {
            "validation_outcome": "failed",
            "error_code": ERROR_CODE_PROVIDER_TIMEOUT,
            "recoverable_error": True,
            "retry_count": 3,
        }
    )

    assert (
        route_after_validation(state)
        == ROUTE_HUMAN_REVIEW
    )


def test_route_after_validation_failure_without_retry(state):
    state.update(
        {
            "validation_outcome": "failed",
            "error_code": ERROR_CODE_UNEXPECTED,
            "recoverable_error": False,
        }
    )

    assert (
        route_after_validation(state)
        == ROUTE_HUMAN_REVIEW
    )


def test_route_after_validation_cancelled(state):
    state["validation_outcome"] = "passed"
    state["cancel_requested"] = True

    assert (
        route_after_validation(state)
        == ROUTE_CANCELLED
    )


# ---------------------------------------------------------------------------
# Review routing
# ---------------------------------------------------------------------------


def test_route_after_review_approved(state):
    state["review_status"] = "approved"

    assert (
        route_after_review(state)
        == ROUTE_PUBLISH_ARTIFACTS
    )


def test_route_after_review_awaiting(state):
    state["review_status"] = "pending_approval"
    state["approval_required"] = True

    assert (
        route_after_review(state)
        == ROUTE_AWAITING_REVIEW
    )


def test_route_after_review_rejected(state):
    state["review_status"] = "rejected"

    assert (
        route_after_review(state)
        == ROUTE_FAILED
    )


def test_route_after_review_changes_requested(state):
    state["review_status"] = "changes_requested"

    assert (
        route_after_review(state)
        == ROUTE_RETRY
    )


def test_route_after_review_cancelled(state):
    state["review_status"] = "approved"
    state["cancel_requested"] = True

    assert (
        route_after_review(state)
        == ROUTE_CANCELLED
    )


# ---------------------------------------------------------------------------
# Publication routing
# ---------------------------------------------------------------------------


def test_route_after_publication_completed(state):
    state["publishing_status"] = JobStatus.COMPLETED

    assert (
        route_after_publication(state)
        == ROUTE_COMPLETED
    )


def test_route_after_publication_failed(state):
    state["publishing_status"] = JobStatus.FAILED

    assert (
        route_after_publication(state)
        == ROUTE_FAILED
    )


def test_route_after_publication_processing(state):
    state["publishing_status"] = JobStatus.PROCESSING

    assert (
        route_after_publication(state)
        == ROUTE_AWAITING_REVIEW
    )


def test_route_after_publication_cancelled(state):
    state["publishing_status"] = JobStatus.COMPLETED
    state["cancel_requested"] = True

    assert (
        route_after_publication(state)
        == ROUTE_CANCELLED
    )


# ---------------------------------------------------------------------------
# Error routing
# ---------------------------------------------------------------------------


def test_route_on_error_retry(state):
    state.update(
        {
            "current_error": "Provider timeout",
            "error_code": ERROR_CODE_PROVIDER_TIMEOUT,
            "recoverable_error": True,
        }
    )

    assert route_on_error(state) == ROUTE_RETRY


def test_route_on_error_human_review(state):
    state.update(
        {
            "current_error": "Provider timeout",
            "error_code": ERROR_CODE_PROVIDER_TIMEOUT,
            "recoverable_error": True,
            "retry_count": 3,
        }
    )

    assert (
        route_on_error(state)
        == ROUTE_HUMAN_REVIEW
    )


def test_route_on_error_failed_for_blocking_error(state):
    state.update(
        {
            "current_error": "Invalid workflow state",
            "error_code": ERROR_CODE_WORKFLOW_STATE,
            "recoverable_error": False,
        }
    )

    assert route_on_error(state) == ROUTE_FAILED


def test_route_on_error_cancelled(state):
    state["cancel_requested"] = True

    assert route_on_error(state) == ROUTE_CANCELLED


# ---------------------------------------------------------------------------
# Exception classification
# ---------------------------------------------------------------------------


def test_classify_provider_timeout():
    error_code, recoverable, message = classify_exception(
        ProviderTimeoutError("Provider timed out")
    )

    assert (
        error_code == ERROR_CODE_PROVIDER_TIMEOUT
    )
    assert recoverable is True
    assert (
        message
        == "The language-model provider timed out."
    )


def test_classify_external_provider_error():
    error_code, recoverable, message = classify_exception(
        ExternalProviderError("Provider unavailable")
    )

    assert error_code == "external_provider_error"
    assert recoverable is True
    assert (
        message
        == "An external provider is temporarily unavailable."
    )


def test_classify_rate_limit_error():
    error_code, recoverable, message = classify_exception(
        RateLimitError("Too many requests")
    )

    assert error_code == ERROR_CODE_RATE_LIMIT
    assert recoverable is True
    assert (
        message
        == "The workflow exceeded its allowed request rate."
    )


def test_classify_validation_error():
    error_code, recoverable, message = classify_exception(
        ValidationError("Invalid brief")
    )

    assert error_code == ERROR_CODE_VALIDATION
    assert recoverable is False
    assert message == "Invalid brief"


def test_classify_authorization_error():
    error_code, recoverable, message = classify_exception(
        AuthorizationError("Access denied")
    )

    assert error_code == ERROR_CODE_AUTHORIZATION
    assert recoverable is False
    assert message == "Access denied"


def test_classify_authentication_error():
    error_code, recoverable, message = classify_exception(
        AuthenticationError("Authentication required")
    )

    assert error_code == "authentication_error"
    assert recoverable is False
    assert (
        message
        == "Workflow authentication is required."
    )


def test_classify_not_found_error():
    error_code, recoverable, message = classify_exception(
        NotFoundError("Workflow not found")
    )

    assert error_code == "not_found"
    assert recoverable is False
    assert message == "Workflow not found"


def test_classify_workflow_state_error():
    error_code, recoverable, message = classify_exception(
        WorkflowStateError("Invalid workflow state")
    )

    assert error_code == ERROR_CODE_WORKFLOW_STATE
    assert recoverable is False
    assert (
        message
        == "The workflow is in an invalid state."
    )


def test_classify_app_error_uses_safe_message():
    error_code, recoverable, message = classify_exception(
        AppError(
            "Custom workflow failure",
            code="custom_failure",
        )
    )

    assert error_code == "custom_failure"
    assert recoverable is False
    assert (
        message
        == "The workflow could not be completed."
    )


def test_classify_unexpected_error():
    error_code, recoverable, message = classify_exception(
        RuntimeError("secret internal detail")
    )

    assert error_code == ERROR_CODE_UNEXPECTED
    assert recoverable is False
    assert (
        message
        == "An unexpected workflow error occurred."
    )


def test_classify_none_exception():
    error_code, recoverable, message = classify_exception(
        None
    )

    assert error_code == ERROR_CODE_UNEXPECTED
    assert recoverable is False
    assert (
        message
        == "An unexpected workflow error occurred."
    )


# ---------------------------------------------------------------------------
# Retry decisions
# ---------------------------------------------------------------------------


def test_should_retry(state):
    state.update(
        {
            "recoverable_error": True,
            "error_code": ERROR_CODE_PROVIDER_TIMEOUT,
        }
    )

    assert should_retry(state) is True


def test_should_retry_exhausted(state):
    state.update(
        {
            "recoverable_error": True,
            "error_code": ERROR_CODE_PROVIDER_TIMEOUT,
            "retry_count": 3,
        }
    )

    assert should_retry(state) is False


def test_should_retry_non_recoverable_error(state):
    state.update(
        {
            "recoverable_error": False,
            "error_code": ERROR_CODE_UNEXPECTED,
        }
    )

    assert should_retry(state) is False


def test_should_retry_missing_error_code(state):
    state.update(
        {
            "recoverable_error": True,
            "error_code": None,
        }
    )

    assert should_retry(state) is False


def test_should_retry_cancelled_workflow(state):
    state.update(
        {
            "recoverable_error": True,
            "error_code": ERROR_CODE_PROVIDER_TIMEOUT,
            "cancel_requested": True,
        }
    )

    assert should_retry(state) is False


def test_get_retry_policy(state):
    policy = get_retry_policy(state)

    assert policy["retry_count"] == 0
    assert policy["max_retries"] == 3
    assert policy["remaining_attempts"] == 3
    assert policy["retries_exhausted"] is False


def test_get_retry_policy_after_retries(state):
    state["retry_count"] = 2

    policy = get_retry_policy(state)

    assert policy["retry_count"] == 2
    assert policy["remaining_attempts"] == 1
    assert policy["retries_exhausted"] is False


def test_get_retry_policy_when_exhausted(state):
    state["retry_count"] = 3

    policy = get_retry_policy(state)

    assert policy["retry_count"] == 3
    assert policy["remaining_attempts"] == 0
    assert policy["retries_exhausted"] is True


def test_get_retry_policy_bounds_invalid_max_retries(state):
    state["max_retries"] = -1

    policy = get_retry_policy(state)

    assert policy["max_retries"] == 0
    assert policy["remaining_attempts"] == 0
    assert policy["retries_exhausted"] is True


# ---------------------------------------------------------------------------
# Retry backoff
# ---------------------------------------------------------------------------


def test_calculate_retry_backoff_bounds():
    assert (
        calculate_retry_backoff(retry_count=0)
        == 2.0
    )

    assert (
        calculate_retry_backoff(retry_count=10)
        == 60.0
    )


def test_calculate_retry_backoff_progression():
    assert (
        calculate_retry_backoff(retry_count=0)
        == 2.0
    )

    assert (
        calculate_retry_backoff(retry_count=1)
        == 4.0
    )

    assert (
        calculate_retry_backoff(retry_count=2)
        == 8.0
    )


def test_calculate_retry_backoff_never_negative():
    assert (
        calculate_retry_backoff(retry_count=-5)
        >= 0
    )


def test_calculate_retry_backoff_respects_custom_base():
    assert (
        calculate_retry_backoff(
            retry_count=0,
            base_backoff_seconds=5.0,
        )
        == 5.0
    )


def test_calculate_retry_backoff_respects_custom_maximum():
    assert (
        calculate_retry_backoff(
            retry_count=10,
            base_backoff_seconds=2.0,
            max_backoff_seconds=10.0,
        )
        == 10.0
    )


# ---------------------------------------------------------------------------
# Recoverable failure recording
# ---------------------------------------------------------------------------


def test_record_recoverable_failure(state):
    updates = record_recoverable_failure(
        state,
        message="Provider timeout",
        error_step=WorkflowStep.REQUIREMENTS,
        error_code=ERROR_CODE_PROVIDER_TIMEOUT,
    )

    assert updates["recoverable_error"] is True
    assert (
        updates["workflow_status"]
        == JobStatus.PROCESSING
    )
    assert updates["recovery_history"]
    assert updates["current_error"]
    assert updates["error_code"] == ERROR_CODE_PROVIDER_TIMEOUT
    assert updates["error_step"] == WorkflowStep.REQUIREMENTS


def test_record_recoverable_failure_sanitizes_message(state):
    updates = record_recoverable_failure(
        state,
        message=(
            "Provider failed with api_key=sk-should-not-leak"
        ),
        error_step=WorkflowStep.REQUIREMENTS,
        error_code=ERROR_CODE_PROVIDER_TIMEOUT,
    )

    serialized_updates = str(updates)

    assert "sk-should-not-leak" not in serialized_updates


def test_record_recoverable_failure_appends_history(state):
    state["recovery_history"] = [
        {
            "event": "previous_failure",
        }
    ]

    updates = record_recoverable_failure(
        state,
        message="Provider timeout",
        error_step=WorkflowStep.REQUIREMENTS,
        error_code=ERROR_CODE_PROVIDER_TIMEOUT,
    )

    assert len(updates["recovery_history"]) == 2


# ---------------------------------------------------------------------------
# Blocking failure recording
# ---------------------------------------------------------------------------


def test_record_blocking_failure(state):
    updates = record_blocking_failure(
        state,
        message="Invalid workflow state",
        error_step=WorkflowStep.INITIALIZED,
        error_code="invalid_state",
    )

    assert updates["recoverable_error"] is False
    assert (
        updates["workflow_status"]
        == JobStatus.FAILED
    )
    assert updates["current_error"]
    assert updates["error_code"] == "invalid_state"
    assert updates["error_step"] == WorkflowStep.INITIALIZED


def test_record_blocking_failure_sanitizes_message(state):
    updates = record_blocking_failure(
        state,
        message=(
            "Database connection failed: "
            "postgresql://user:secret-password@localhost/db"
        ),
        error_step=WorkflowStep.INITIALIZED,
        error_code="database_failure",
    )

    serialized_updates = str(updates)

    assert "secret-password" not in serialized_updates


def test_record_blocking_failure_appends_history(state):
    state["recovery_history"] = []

    updates = record_blocking_failure(
        state,
        message="Invalid workflow state",
        error_step=WorkflowStep.INITIALIZED,
        error_code="invalid_state",
    )

    assert len(updates["recovery_history"]) == 1


# ---------------------------------------------------------------------------
# Retry preparation
# ---------------------------------------------------------------------------


def test_prepare_retry(state):
    state.update(
        {
            "error_code": ERROR_CODE_PROVIDER_TIMEOUT,
            "recoverable_error": True,
        }
    )

    updates = prepare_retry(
        state,
        error_step=WorkflowStep.REQUIREMENTS,
        error_code=ERROR_CODE_PROVIDER_TIMEOUT,
    )

    assert updates["retry_count"] == 1
    assert updates["current_error"] is None
    assert updates["recoverable_error"] is False
    assert updates["retry_backoff_seconds"] > 0
    assert updates["recovery_history"]


def test_prepare_retry_rejects_non_recoverable(state):
    state.update(
        {
            "error_code": "invalid_state",
            "recoverable_error": False,
        }
    )

    with pytest.raises(ValidationError):
        prepare_retry(
            state,
            error_step=WorkflowStep.REQUIREMENTS,
            error_code="invalid_state",
        )


def test_prepare_retry_rejects_exhausted_retries(state):
    state.update(
        {
            "error_code": ERROR_CODE_PROVIDER_TIMEOUT,
            "recoverable_error": True,
            "retry_count": 3,
        }
    )

    with pytest.raises(ValidationError):
        prepare_retry(
            state,
            error_step=WorkflowStep.REQUIREMENTS,
            error_code=ERROR_CODE_PROVIDER_TIMEOUT,
        )


def test_prepare_retry_rejects_cancelled_workflow(state):
    state.update(
        {
            "error_code": ERROR_CODE_PROVIDER_TIMEOUT,
            "recoverable_error": True,
            "cancel_requested": True,
        }
    )

    with pytest.raises(ValidationError):
        prepare_retry(
            state,
            error_step=WorkflowStep.REQUIREMENTS,
            error_code=ERROR_CODE_PROVIDER_TIMEOUT,
        )


# ---------------------------------------------------------------------------
# Human-review preparation
# ---------------------------------------------------------------------------


def test_prepare_human_review(state):
    updates = prepare_human_review(
        state,
        reason="Validation requires review.",
        error_step=WorkflowStep.VALIDATION,
        error_code="validation_failed",
    )

    assert updates["approval_required"] is True
    assert (
        updates["review_status"]
        == "pending_approval"
    )
    assert (
        updates["approval_status"]
        == "pending"
    )
    assert updates["current_error"]
    assert updates["error_code"] == "validation_failed"
    assert updates["error_step"] == WorkflowStep.VALIDATION
    assert updates["recovery_history"]


def test_prepare_human_review_sanitizes_reason(state):
    updates = prepare_human_review(
        state,
        reason=(
            "Validation failed with api_key=sk-should-not-leak"
        ),
        error_step=WorkflowStep.VALIDATION,
        error_code="validation_failed",
    )

    serialized_updates = str(updates)

    assert "sk-should-not-leak" not in serialized_updates


def test_prepare_human_review_rejects_empty_reason(state):
    with pytest.raises(ValidationError):
        prepare_human_review(
            state,
            reason="   ",
            error_step=WorkflowStep.VALIDATION,
            error_code="validation_failed",
        )


# ---------------------------------------------------------------------------
# Cancellation preparation
# ---------------------------------------------------------------------------


def test_prepare_cancellation(state):
    updates = prepare_cancellation(
        state,
        cancelled_by="user_1",
        reason="No longer needed.",
    )

    assert updates["cancel_requested"] is True
    assert (
        updates["workflow_status"]
        == JobStatus.CANCELLED
    )
    assert updates["cancelled_by"] == "user_1"
    assert updates["cancellation_reason"]
    assert updates["recovery_history"]


def test_prepare_cancellation_sanitizes_reason(state):
    updates = prepare_cancellation(
        state,
        cancelled_by="user_1",
        reason=(
            "Cancel because api_key=sk-should-not-leak"
        ),
    )

    serialized_updates = str(updates)

    assert "sk-should-not-leak" not in serialized_updates


def test_prepare_cancellation_rejects_empty_reason(state):
    with pytest.raises(ValidationError):
        prepare_cancellation(
            state,
            cancelled_by="user_1",
            reason="   ",
        )


def test_prepare_cancellation_rejects_missing_actor(state):
    with pytest.raises(ValidationError):
        prepare_cancellation(
            state,
            cancelled_by="",
            reason="No longer needed.",
        )


# ---------------------------------------------------------------------------
# Recovery-action decisions
# ---------------------------------------------------------------------------


def test_decide_recovery_action(state):
    state.update(
        {
            "error_code": ERROR_CODE_PROVIDER_TIMEOUT,
            "recoverable_error": True,
        }
    )

    assert (
        decide_recovery_action(state)
        == "retry"
    )

    state["retry_count"] = 3

    assert (
        decide_recovery_action(state)
        == "human_review"
    )


def test_decide_recovery_action_human_review_for_blocking_error(
    state,
):
    state.update(
        {
            "error_code": ERROR_CODE_WORKFLOW_STATE,
            "recoverable_error": False,
        }
    )

    assert (
        decide_recovery_action(state)
        == "human_review"
    )


def test_decide_recovery_action_cancelled_workflow(state):
    state.update(
        {
            "error_code": ERROR_CODE_PROVIDER_TIMEOUT,
            "recoverable_error": True,
            "cancel_requested": True,
        }
    )

    assert (
        decide_recovery_action(state)
        == "cancel"
    )
