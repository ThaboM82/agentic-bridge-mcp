"""
Comprehensive tests for ingestion-job CRUD helpers.

These tests cover job ID normalization, status normalization, status-transition
validation, error-message truncation, redaction of sensitive material, and
boundary behavior.
"""

from __future__ import annotations

import pytest

from app.core.enums import JobStatus
from app.core.exceptions import (
    ValidationError,
    WorkflowStateError,
)
from app.crud.ingestion_job_crud import (
    _normalize_job_id,
    _normalize_job_status,
    _truncate_error_message,
    _validate_status_transition,
)


# ---------------------------------------------------------------------------
# Job ID normalization
# ---------------------------------------------------------------------------


def test_normalize_job_id():
    job_id = "123e4567-e89b-12d3-a456-426614174000"

    normalized_job_id = _normalize_job_id(job_id)

    assert normalized_job_id == job_id


def test_normalize_job_id_with_whitespace():
    job_id = "  123e4567-e89b-12d3-a456-426614174000  "

    normalized_job_id = _normalize_job_id(job_id)

    assert (
        normalized_job_id
        == "123e4567-e89b-12d3-a456-426614174000"
    )


def test_normalize_job_id_accepts_uuid_object():
    from uuid import UUID

    job_id = UUID(
        "123e4567-e89b-12d3-a456-426614174000"
    )

    normalized_job_id = _normalize_job_id(job_id)

    assert (
        normalized_job_id
        == "123e4567-e89b-12d3-a456-426614174000"
    )


def test_invalid_job_id():
    with pytest.raises(ValidationError):
        _normalize_job_id("not-a-uuid")


def test_empty_job_id_is_rejected():
    with pytest.raises(ValidationError):
        _normalize_job_id("")


def test_whitespace_only_job_id_is_rejected():
    with pytest.raises(ValidationError):
        _normalize_job_id("   ")


def test_none_job_id_is_rejected():
    with pytest.raises(ValidationError):
        _normalize_job_id(None)


def test_integer_job_id_is_rejected():
    with pytest.raises(ValidationError):
        _normalize_job_id(12345)


def test_oversized_job_id_is_rejected():
    oversized_job_id = "a" * 500

    with pytest.raises(ValidationError):
        _normalize_job_id(oversized_job_id)


# ---------------------------------------------------------------------------
# Job status normalization
# ---------------------------------------------------------------------------


def test_normalize_job_status():
    assert (
        _normalize_job_status("queued")
        == JobStatus.QUEUED
    )

    assert (
        _normalize_job_status(None)
        is None
    )


@pytest.mark.parametrize(
    ("raw_status", "expected_status"),
    [
        ("queued", JobStatus.QUEUED),
        ("QUEUED", JobStatus.QUEUED),
        (" Queued ", JobStatus.QUEUED),
        ("processing", JobStatus.PROCESSING),
        ("PROCESSING", JobStatus.PROCESSING),
        ("completed", JobStatus.COMPLETED),
        ("COMPLETED", JobStatus.COMPLETED),
        ("failed", JobStatus.FAILED),
        ("FAILED", JobStatus.FAILED),
        ("cancelled", JobStatus.CANCELLED),
        ("CANCELLED", JobStatus.CANCELLED),
    ],
)
def test_normalize_known_job_statuses(
    raw_status: str,
    expected_status: JobStatus,
):
    assert (
        _normalize_job_status(raw_status)
        == expected_status
    )


def test_invalid_job_status():
    with pytest.raises(ValidationError):
        _normalize_job_status("unknown-status")


def test_empty_job_status_is_rejected():
    with pytest.raises(ValidationError):
        _normalize_job_status("")


def test_whitespace_only_job_status_is_rejected():
    with pytest.raises(ValidationError):
        _normalize_job_status("   ")


def test_integer_job_status_is_rejected():
    with pytest.raises(ValidationError):
        _normalize_job_status(123)


def test_list_job_status_is_rejected():
    with pytest.raises(ValidationError):
        _normalize_job_status(["queued"])


# ---------------------------------------------------------------------------
# Status-transition validation
# ---------------------------------------------------------------------------


def test_valid_status_transition():
    current, next_status = _validate_status_transition(
        JobStatus.QUEUED,
        JobStatus.PROCESSING,
    )

    assert current == JobStatus.QUEUED
    assert next_status == JobStatus.PROCESSING


@pytest.mark.parametrize(
    ("current_status", "next_status"),
    [
        (
            JobStatus.QUEUED,
            JobStatus.PROCESSING,
        ),
        (
            JobStatus.QUEUED,
            JobStatus.CANCELLED,
        ),
        (
            JobStatus.PROCESSING,
            JobStatus.COMPLETED,
        ),
        (
            JobStatus.PROCESSING,
            JobStatus.FAILED,
        ),
        (
            JobStatus.PROCESSING,
            JobStatus.CANCELLED,
        ),
    ],
)
def test_allowed_status_transitions(
    current_status: JobStatus,
    next_status: JobStatus,
):
    validated_current, validated_next = (
        _validate_status_transition(
            current_status,
            next_status,
        )
    )

    assert validated_current == current_status
    assert validated_next == next_status


def test_invalid_status_transition():
    with pytest.raises(WorkflowStateError):
        _validate_status_transition(
            JobStatus.COMPLETED,
            JobStatus.PROCESSING,
        )


@pytest.mark.parametrize(
    ("current_status", "next_status"),
    [
        (
            JobStatus.COMPLETED,
            JobStatus.PROCESSING,
        ),
        (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
        ),
        (
            JobStatus.COMPLETED,
            JobStatus.CANCELLED,
        ),
        (
            JobStatus.FAILED,
            JobStatus.PROCESSING,
        ),
        (
            JobStatus.FAILED,
            JobStatus.COMPLETED,
        ),
        (
            JobStatus.CANCELLED,
            JobStatus.PROCESSING,
        ),
        (
            JobStatus.CANCELLED,
            JobStatus.COMPLETED,
        ),
    ],
)
def test_disallowed_status_transitions(
    current_status: JobStatus,
    next_status: JobStatus,
):
    with pytest.raises(WorkflowStateError):
        _validate_status_transition(
            current_status,
            next_status,
        )


def test_same_status_transition_is_rejected():
    with pytest.raises(WorkflowStateError):
        _validate_status_transition(
            JobStatus.PROCESSING,
            JobStatus.PROCESSING,
        )


def test_transition_from_none_status_is_rejected():
    with pytest.raises(WorkflowStateError):
        _validate_status_transition(
            None,
            JobStatus.PROCESSING,
        )


def test_transition_to_none_status_is_rejected():
    with pytest.raises(WorkflowStateError):
        _validate_status_transition(
            JobStatus.QUEUED,
            None,
        )


# ---------------------------------------------------------------------------
# Error-message truncation
# ---------------------------------------------------------------------------


def test_error_message_is_truncated():
    message = "x" * 5_000

    result = _truncate_error_message(message)

    assert result is not None
    assert len(result) <= 2_000


def test_error_message_preserves_short_message():
    message = "Document parsing failed"

    result = _truncate_error_message(message)

    assert result == message


def test_error_message_trims_whitespace():
    message = "  Document parsing failed  "

    result = _truncate_error_message(message)

    assert result == "Document parsing failed"


def test_empty_error_message_returns_none():
    assert (
        _truncate_error_message("   ")
        is None
    )

    assert (
        _truncate_error_message(None)
        is None
    )


def test_empty_string_error_message_returns_none():
    assert (
        _truncate_error_message("")
        is None
    )


def test_non_string_error_message_returns_none():
    assert (
        _truncate_error_message(12345)
        is None
    )

    assert (
        _truncate_error_message(["error"])
        is None
    )

    assert (
        _truncate_error_message({"error": True})
        is None
    )


def test_error_message_with_newlines_is_sanitized():
    message = "Line one\nLine two\nLine three"

    result = _truncate_error_message(message)

    assert result is not None
    assert "\n" not in result


def test_error_message_with_carriage_returns_is_sanitized():
    message = "Line one\r\nLine two"

    result = _truncate_error_message(message)

    assert result is not None
    assert "\r" not in result
    assert "\n" not in result


def test_error_message_with_control_characters_is_sanitized():
    message = "Failed\x00with\x1fcontrol characters"

    result = _truncate_error_message(message)

    assert result is not None
    assert "\x00" not in result
    assert "\x1f" not in result


def test_error_message_with_credentials_is_redacted():
    message = (
        "Connection failed using "
        "postgresql://user:super-secret-password@localhost:5432/app"
    )

    result = _truncate_error_message(message)

    assert result is not None
    assert "super-secret-password" not in result


def test_error_message_with_api_key_is_redacted():
    message = (
        "Provider request failed with api_key=sk-should-not-leak"
    )

    result = _truncate_error_message(message)

    assert result is not None
    assert "sk-should-not-leak" not in result


def test_error_message_with_bearer_token_is_redacted():
    message = (
        "Authorization failed: "
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secret"
    )

    result = _truncate_error_message(message)

    assert result is not None
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secret" not in result


def test_truncated_error_message_remains_bounded_after_redaction():
    message = (
        "postgresql://user:super-secret-password@localhost/db "
        + ("x" * 5_000)
    )

    result = _truncate_error_message(message)

    assert result is not None
    assert len(result) <= 2_000
    assert "super-secret-password" not in result
