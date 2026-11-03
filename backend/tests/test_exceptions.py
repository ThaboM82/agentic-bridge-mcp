"""
Comprehensive tests for AgenticBridge application exceptions.

These tests cover exception hierarchy, default error metadata, HTTP status
mapping, retryability, structured serialization, details handling, sensitive
data protection, and workflow-specific error behavior.
"""

from __future__ import annotations

import json

import pytest

from app.core.exceptions import (
    AppError,
    ApprovalRequiredError,
    AuthenticationError,
    AuthorizationError,
    ExternalProviderError,
    InvalidTokenError,
    NotFoundError,
    ProviderTimeoutError,
    RateLimitError,
    ValidationError,
    WorkflowNotFoundError,
    WorkflowStateError,
)


# ---------------------------------------------------------------------------
# Base application error
# ---------------------------------------------------------------------------


def test_base_error_defaults():
    error = AppError()

    assert error.code == "internal_error"
    assert error.http_status == 500
    assert error.retryable is False


def test_base_error_accepts_custom_message():
    error = AppError("Custom application failure")

    assert str(error) == "Custom application failure"
    assert error.message == "Custom application failure"


def test_base_error_accepts_custom_code():
    error = AppError(
        "Custom failure",
        code="custom_failure",
    )

    assert error.code == "custom_failure"


def test_base_error_accepts_custom_status():
    error = AppError(
        "Custom failure",
        http_status=418,
    )

    assert error.http_status == 418


def test_base_error_supports_details():
    error = AppError(
        "Custom failure",
        details={"resource": "workflow"},
    )

    assert error.details == {
        "resource": "workflow"
    }


def test_base_error_is_exception():
    error = AppError("Failure")

    assert isinstance(error, Exception)


def test_base_error_can_be_raised_and_caught():
    with pytest.raises(AppError) as exc_info:
        raise AppError("Expected failure")

    assert exc_info.value.message == "Expected failure"


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_validation_error_status():
    error = ValidationError(
        details={"field": "title"}
    )

    assert error.http_status == 422
    assert error.details == {"field": "title"}


def test_validation_error_defaults():
    error = ValidationError()

    assert error.code == "validation_error"
    assert error.http_status == 422
    assert error.retryable is False


def test_validation_error_accepts_custom_message():
    error = ValidationError("Invalid title.")

    assert error.message == "Invalid title."
    assert str(error) == "Invalid title."


def test_validation_error_supports_field_errors():
    details = {
        "field": "title",
        "issue": "required",
    }

    error = ValidationError(
        details=details
    )

    assert error.details == details


def test_validation_error_is_app_error():
    assert issubclass(
        ValidationError,
        AppError,
    )


def test_validation_error_serialization():
    error = ValidationError(
        message="Invalid title.",
        details={"field": "title"},
    )

    payload = error.to_dict()

    assert (
        payload["error"]["code"]
        == "validation_error"
    )
    assert (
        payload["error"]["message"]
        == "Invalid title."
    )
    assert payload["error"]["details"] == {
        "field": "title"
    }
    assert (
        payload["error"]["status"] == 422
    )


# ---------------------------------------------------------------------------
# Authentication and authorization errors
# ---------------------------------------------------------------------------


def test_authentication_error_defaults():
    error = AuthenticationError()

    assert error.code == "authentication_error"
    assert error.http_status == 401
    assert error.retryable is False


def test_authentication_error_is_app_error():
    assert issubclass(
        AuthenticationError,
        AppError,
    )


def test_authentication_error_serialization():
    error = AuthenticationError(
        "Authentication required."
    )

    payload = error.to_dict()

    assert (
        payload["error"]["code"]
        == "authentication_error"
    )
    assert (
        payload["error"]["message"]
        == "Authentication required."
    )
    assert (
        payload["error"]["status"] == 401
    )


def test_authorization_error_defaults():
    error = AuthorizationError()

    assert error.code == "authorization_error"
    assert error.http_status == 403
    assert error.retryable is False


def test_authorization_error_is_app_error():
    assert issubclass(
        AuthorizationError,
        AppError,
    )


def test_authorization_error_serialization():
    error = AuthorizationError(
        "Access denied."
    )

    payload = error.to_dict()

    assert (
        payload["error"]["code"]
        == "authorization_error"
    )
    assert (
        payload["error"]["message"]
        == "Access denied."
    )
    assert (
        payload["error"]["status"] == 403
    )


# ---------------------------------------------------------------------------
# Not-found errors
# ---------------------------------------------------------------------------


def test_not_found_error_defaults():
    error = NotFoundError()

    assert error.code == "not_found"
    assert error.http_status == 404
    assert error.retryable is False


def test_not_found_error_is_app_error():
    assert issubclass(
        NotFoundError,
        AppError,
    )


def test_not_found_error_serialization():
    error = NotFoundError("Resource not found.")

    payload = error.to_dict()

    assert (
        payload["error"]["code"]
        == "not_found"
    )
    assert (
        payload["error"]["message"]
        == "Resource not found."
    )
    assert (
        payload["error"]["status"] == 404
    )


def test_workflow_not_found_hierarchy():
    error = WorkflowNotFoundError()

    assert isinstance(error, NotFoundError)
    assert error.code == "workflow_not_found"
    assert error.http_status == 404


def test_workflow_not_found_serialization():
    error = WorkflowNotFoundError(
        "Workflow not found."
    )

    payload = error.to_dict()

    assert (
        payload["error"]["code"]
        == "workflow_not_found"
    )
    assert (
        payload["error"]["message"]
        == "Workflow not found."
    )
    assert (
        payload["error"]["status"] == 404
    )


# ---------------------------------------------------------------------------
# Rate-limit errors
# ---------------------------------------------------------------------------


def test_rate_limit_is_retryable():
    error = RateLimitError()

    assert error.http_status == 429
    assert error.retryable is True


def test_rate_limit_error_defaults():
    error = RateLimitError()

    assert error.code == "rate_limit_exceeded"
    assert error.http_status == 429


def test_rate_limit_error_supports_retry_after():
    error = RateLimitError(
        "Too many requests.",
        retry_after_seconds=30,
    )

    assert error.retry_after_seconds == 30


def test_rate_limit_error_defaults_retry_after():
    error = RateLimitError()

    assert error.retry_after_seconds == 60


def test_rate_limit_error_is_app_error():
    assert issubclass(
        RateLimitError,
        AppError,
    )


def test_rate_limit_error_serialization():
    error = RateLimitError(
        "Too many requests.",
        retry_after_seconds=45,
    )

    payload = error.to_dict()

    assert (
        payload["error"]["code"]
        == "rate_limit_exceeded"
    )
    assert (
        payload["error"]["status"] == 429
    )
    assert (
        payload["error"]["retryable"]
        is True
    )
    assert (
        payload["error"]["retry_after_seconds"]
        == 45
    )


# ---------------------------------------------------------------------------
# External provider errors
# ---------------------------------------------------------------------------


def test_provider_timeout_is_retryable():
    error = ProviderTimeoutError()

    assert isinstance(
        error,
        ExternalProviderError,
    )
    assert error.retryable is True
    assert error.http_status == 502


def test_provider_timeout_defaults():
    error = ProviderTimeoutError()

    assert error.code == "provider_timeout"
    assert error.http_status == 504


def test_provider_timeout_is_external_provider_error():
    assert issubclass(
        ProviderTimeoutError,
        ExternalProviderError,
    )


def test_external_provider_error_defaults():
    error = ExternalProviderError()

    assert error.code == "external_provider_error"
    assert error.http_status == 502
    assert error.retryable is True


def test_external_provider_error_is_app_error():
    assert issubclass(
        ExternalProviderError,
        AppError,
    )


def test_external_provider_error_serialization():
    error = ExternalProviderError(
        "Provider unavailable."
    )

    payload = error.to_dict()

    assert (
        payload["error"]["code"]
        == "external_provider_error"
    )
    assert (
        payload["error"]["status"] == 502
    )
    assert (
        payload["error"]["retryable"]
        is True
    )


def test_provider_timeout_serialization():
    error = ProviderTimeoutError(
        "Provider timed out."
    )

    payload = error.to_dict()

    assert (
        payload["error"]["code"]
        == "provider_timeout"
    )
    assert (
        payload["error"]["status"] == 504
    )
    assert (
        payload["error"]["retryable"]
        is True
    )


# ---------------------------------------------------------------------------
# Workflow state errors
# ---------------------------------------------------------------------------


def test_workflow_state_error_defaults():
    error = WorkflowStateError()

    assert error.code == "workflow_error"
    assert error.http_status == 400
    assert error.retryable is False


def test_workflow_state_error_is_app_error():
    assert issubclass(
        WorkflowStateError,
        AppError,
    )


def test_workflow_state_error_serialization():
    error = WorkflowStateError(
        "Invalid workflow transition."
    )

    payload = error.to_dict()

    assert (
        payload["error"]["code"]
        == "workflow_error"
    )
    assert (
        payload["error"]["message"]
        == "Invalid workflow transition."
    )
    assert (
        payload["error"]["status"] == 400
    )


# ---------------------------------------------------------------------------
# Approval-required errors
# ---------------------------------------------------------------------------


def test_approval_required_status():
    error = ApprovalRequiredError()

    assert error.http_status == 202
    assert error.code == "approval_required"


def test_approval_required_error_defaults():
    error = ApprovalRequiredError()

    assert error.retryable is False


def test_approval_required_error_is_app_error():
    assert issubclass(
        ApprovalRequiredError,
        AppError,
    )


def test_approval_required_error_serialization():
    error = ApprovalRequiredError(
        "Human approval is required."
    )

    payload = error.to_dict()

    assert (
        payload["error"]["code"]
        == "approval_required"
    )
    assert (
        payload["error"]["message"]
        == "Human approval is required."
    )
    assert (
        payload["error"]["status"] == 202
    )


# ---------------------------------------------------------------------------
# Token errors
# ---------------------------------------------------------------------------


def test_invalid_token_error_is_authentication_error():
    assert issubclass(
        InvalidTokenError,
        AuthenticationError,
    )


def test_invalid_token_error_defaults():
    error = InvalidTokenError()

    assert error.code == "authentication_error"
    assert error.http_status == 401


def test_invalid_token_error_serialization():
    error = InvalidTokenError(
        "Token has expired."
    )

    payload = error.to_dict()

    assert (
        payload["error"]["code"]
        == "authentication_error"
    )
    assert (
        payload["error"]["message"]
        == "Token has expired."
    )
    assert (
        payload["error"]["status"] == 401
    )


# ---------------------------------------------------------------------------
# Serialization behavior
# ---------------------------------------------------------------------------


def test_error_serialization_excludes_internal_data():
    error = ValidationError(
        message="Invalid title.",
        details={"field": "title"},
    )

    payload = error.to_dict()

    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["message"] == "Invalid title."
    assert payload["error"]["details"] == {"field": "title"}
    assert "traceback" not in payload
    assert "stack" not in payload
    assert "internal" not in payload


def test_error_serialization_is_json_safe():
    error = ValidationError(
        message="Invalid title.",
        details={
            "field": "title",
            "provided_length": 0,
        },
    )

    payload = error.to_dict()

    serialized_payload = json.dumps(payload)

    assert "validation_error" in serialized_payload
    assert "Invalid title." in serialized_payload


def test_error_serialization_omits_empty_details():
    error = ValidationError("Invalid title.")

    payload = error.to_dict()

    assert "details" not in payload["error"]


def test_error_serialization_includes_retry_metadata_for_retryable_errors():
    error = RateLimitError(
        retry_after_seconds=30
    )

    payload = error.to_dict()

    assert (
        payload["error"]["retryable"]
        is True
    )
    assert (
        payload["error"]["retry_after_seconds"]
        == 30
    )


def test_error_serialization_omits_retry_metadata_for_non_retryable_errors():
    error = ValidationError("Invalid title.")

    payload = error.to_dict()

    assert (
        payload["error"]["retryable"]
        is False
    )
    assert (
        "retry_after_seconds"
        not in payload["error"]
    )


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_all_domain_errors_inherit_app_error():
    domain_errors = [
        AuthenticationError(),
        AuthorizationError(),
        NotFoundError(),
        RateLimitError(),
        ValidationError(),
        ExternalProviderError(),
        ProviderTimeoutError(),
        WorkflowStateError(),
        ApprovalRequiredError(),
        WorkflowNotFoundError(),
        InvalidTokenError(),
    ]

    for error in domain_errors:
        assert isinstance(error, AppError)


def test_provider_timeout_is_external_provider_error():
    assert issubclass(
        ProviderTimeoutError,
        ExternalProviderError,
    )


def test_workflow_not_found_is_not_found_error():
    assert issubclass(
        WorkflowNotFoundError,
        NotFoundError,
    )


def test_invalid_token_is_authentication_error():
    assert issubclass(
        InvalidTokenError,
        AuthenticationError,
    )


# ---------------------------------------------------------------------------
# Error identity and messages
# ---------------------------------------------------------------------------


def test_error_message_is_preserved():
    message = "A specific workflow failure occurred."

    error = AppError(message)

    assert error.message == message
    assert str(error) == message


def test_error_code_is_stable():
    error = AppError(
        "Failure",
        code="stable_error_code",
    )

    assert error.code == "stable_error_code"


def test_error_details_are_not_required():
    error = AppError("Failure")

    assert error.details == {}


def test_error_details_can_be_nested():
    details = {
        "workflow": {
            "execution_id": "wf_123",
            "step": "validation",
        },
    }

    error = AppError(
        "Failure",
        details=details,
    )

    assert error.details == details


# ---------------------------------------------------------------------------
# Security-oriented serialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        ValidationError("Invalid input."),
        AuthenticationError("Authentication required."),
        AuthorizationError("Access denied."),
        NotFoundError("Resource not found."),
        RateLimitError("Too many requests."),
        ProviderTimeoutError("Provider timed out."),
        ExternalProviderError("Provider unavailable."),
        WorkflowStateError("Invalid workflow state."),
        ApprovalRequiredError("Approval required."),
    ],
)
def test_error_payloads_do_not_contain_stack_trace(error):
    payload = error.to_dict()

    serialized_payload = json.dumps(
        payload
    ).lower()

    assert "traceback" not in serialized_payload
    assert "stack trace" not in serialized_payload
    assert "exception:" not in serialized_payload


@pytest.mark.parametrize(
    "error",
    [
        ValidationError("Invalid input."),
        AuthenticationError("Authentication required."),
        AuthorizationError("Access denied."),
        NotFoundError("Resource not found."),
        RateLimitError("Too many requests."),
        ProviderTimeoutError("Provider timed out."),
        ExternalProviderError("Provider unavailable."),
        WorkflowStateError("Invalid workflow state."),
        ApprovalRequiredError("Approval required."),
    ],
)
def test_error_payloads_include_stable_error_contract(error):
    payload = error.to_dict()

    assert "code" in payload["error"]
    assert "message" in payload["error"]
    assert "status" in payload["error"]
    assert "retryable" in payload["error"]
