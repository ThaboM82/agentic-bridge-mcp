"""
Comprehensive tests for approval and audit-log CRUD validation.

These tests cover approval IDs, statuses, decisions, comments, audit event
text, optional audit fields, metadata validation, sensitive-key rejection,
length limits, whitespace normalization, and fail-closed validation behavior.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.core.enums import (
    ApprovalDecision,
    ApprovalStatus,
)
from app.core.exceptions import ValidationError
from app.crud.approval_crud import (
    _normalize_approval_id,
    _normalize_comment,
    _normalize_decision,
    _normalize_status,
)
from app.crud.audit_log_crud import (
    _validate_metadata,
    _validate_optional_text,
    _validate_required_text,
)


# ---------------------------------------------------------------------------
# Approval-ID normalization
# ---------------------------------------------------------------------------


def test_normalize_approval_id():
    approval_id = "123e4567-e89b-12d3-a456-426614174000"

    assert _normalize_approval_id(
        approval_id
    )


def test_normalize_approval_id_accepts_uuid_object():
    approval_id = uuid4()

    assert (
        _normalize_approval_id(
            approval_id
        )
        == str(approval_id)
    )


def test_normalize_approval_id_strips_whitespace():
    approval_id = str(uuid4())

    assert (
        _normalize_approval_id(
            f"  {approval_id}  "
        )
        == approval_id
    )


def test_normalize_approval_id_normalizes_case():
    approval_id = uuid4()

    assert (
        _normalize_approval_id(
            str(approval_id).upper()
        )
        == str(approval_id)
    )


@pytest.mark.parametrize(
    "invalid_approval_id",
    [
        "",
        "   ",
        "not-a-uuid",
        "123",
        "123e4567-e89b-12d3-a456",
        "123e4567-e89b-12d3-a456-42661417400",
        "123e4567-e89b-12d3-a456-4266141740000",
        "zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz",
        None,
    ],
)
def test_invalid_approval_id(
    invalid_approval_id,
):
    with pytest.raises(ValidationError):
        _normalize_approval_id(
            invalid_approval_id
        )


# ---------------------------------------------------------------------------
# Approval-status normalization
# ---------------------------------------------------------------------------


def test_normalize_approval_status():
    assert (
        _normalize_status("pending")
        == ApprovalStatus.PENDING
    )
    assert (
        _normalize_status(None) is None
    )


@pytest.mark.parametrize(
    "status,expected",
    [
        ("pending", ApprovalStatus.PENDING),
        ("PENDING", ApprovalStatus.PENDING),
        (
            "  pending  ",
            ApprovalStatus.PENDING,
        ),
        (
            "approved",
            ApprovalStatus.APPROVED,
        ),
        (
            "APPROVED",
            ApprovalStatus.APPROVED,
        ),
        (
            "rejected",
            ApprovalStatus.REJECTED,
        ),
        (
            "changes_requested",
            ApprovalStatus.CHANGES_REQUESTED,
        ),
        (
            "expired",
            ApprovalStatus.EXPIRED,
        ),
    ],
)
def test_normalize_approval_status_accepts_supported_statuses(
    status: str,
    expected: ApprovalStatus,
):
    assert (
        _normalize_status(status)
        == expected
    )


@pytest.mark.parametrize(
    "invalid_status",
    [
        "",
        "   ",
        "unknown-status",
        "waiting",
        "complete",
        "approved ",
        "123",
        [],
        {},
    ],
)
def test_invalid_approval_status(
    invalid_status,
):
    with pytest.raises(ValidationError):
        _normalize_status(
            invalid_status
        )


def test_normalize_approval_status_none_returns_none():
    assert (
        _normalize_status(None) is None
    )


# ---------------------------------------------------------------------------
# Approval-decision normalization
# ---------------------------------------------------------------------------


def test_normalize_decision():
    assert (
        _normalize_decision("approved")
        == ApprovalDecision.APPROVED
    )


@pytest.mark.parametrize(
    "decision,expected",
    [
        (
            "approved",
            ApprovalDecision.APPROVED,
        ),
        (
            "APPROVED",
            ApprovalDecision.APPROVED,
        ),
        (
            "  approved  ",
            ApprovalDecision.APPROVED,
        ),
        (
            "rejected",
            ApprovalDecision.REJECTED,
        ),
        (
            "REJECTED",
            ApprovalDecision.REJECTED,
        ),
        (
            "request_changes",
            ApprovalDecision.REQUEST_CHANGES,
        ),
        (
            "REQUEST_CHANGES",
            ApprovalDecision.REQUEST_CHANGES,
        ),
    ],
)
def test_normalize_decision_accepts_supported_decisions(
    decision: str,
    expected: ApprovalDecision,
):
    assert (
        _normalize_decision(decision)
        == expected
    )


@pytest.mark.parametrize(
    "invalid_decision",
    [
        "",
        "   ",
        "unknown-decision",
        "approve",
        "reject",
        "changes",
        "maybe",
        "123",
        [],
        {},
    ],
)
def test_invalid_decision(invalid_decision):
    with pytest.raises(ValidationError):
        _normalize_decision(
            invalid_decision
        )


# ---------------------------------------------------------------------------
# Approval-comment normalization
# ---------------------------------------------------------------------------


def test_comment_is_bounded():
    assert (
        _normalize_comment(
            "  Looks good.  "
        )
        == "Looks good."
    )
    assert (
        _normalize_comment("   ")
        is None
    )


def test_comment_preserves_internal_whitespace():
    assert (
        _normalize_comment(
            "Looks   good."
        )
        == "Looks   good."
    )


def test_comment_handles_empty_string():
    assert (
        _normalize_comment("")
        is None
    )


def test_comment_handles_none():
    assert (
        _normalize_comment(None)
        is None
    )


@pytest.mark.parametrize(
    "valid_comment",
    [
        "Approved.",
        "Rejected due to accessibility issues.",
        "Please clarify the metrics.",
        "Comment with\nnewline",
        "Comment with\ttab",
    ],
)
def test_comment_accepts_valid_values(
    valid_comment: str,
):
    assert (
        _normalize_comment(
            valid_comment
        )
        == valid_comment.strip()
    )


def test_invalid_comment_length():
    with pytest.raises(ValidationError):
        _normalize_comment(
            "x" * 3_000
        )


def test_comment_length_boundary_is_enforced():
    valid_comment = "x" * 2_000

    assert (
        _normalize_comment(
            valid_comment
        )
        == valid_comment
    )

    with pytest.raises(ValidationError):
        _normalize_comment(
            "x" * 2_001
        )


def test_comment_rejects_non_string_values():
    with pytest.raises(ValidationError):
        _normalize_comment(123)

    with pytest.raises(ValidationError):
        _normalize_comment(
            ["comment"]
        )

    with pytest.raises(ValidationError):
        _normalize_comment(
            {"comment": "value"}
        )


# ---------------------------------------------------------------------------
# Required audit text
# ---------------------------------------------------------------------------


def test_required_audit_text():
    assert _validate_required_text(
        "workflow_published",
        "event_type",
        100,
    ) == "workflow_published"


def test_required_audit_text_strips_whitespace():
    assert _validate_required_text(
        "  workflow_published  ",
        "event_type",
        100,
    ) == "workflow_published"


@pytest.mark.parametrize(
    "invalid_value",
    [
        "",
        "   ",
        None,
    ],
)
def test_invalid_required_audit_text(
    invalid_value,
):
    with pytest.raises(ValidationError):
        _validate_required_text(
            invalid_value,
            "event_type",
            100,
        )


def test_required_audit_text_rejects_overlength_value():
    with pytest.raises(ValidationError):
        _validate_required_text(
            "x" * 101,
            "event_type",
            100,
        )


def test_required_audit_text_enforces_length_boundary():
    valid_value = "x" * 100

    assert (
        _validate_required_text(
            valid_value,
            "event_type",
            100,
        )
        == valid_value
    )

    with pytest.raises(ValidationError):
        _validate_required_text(
            "x" * 101,
            "event_type",
            100,
        )


def test_required_audit_text_rejects_non_string_values():
    with pytest.raises(ValidationError):
        _validate_required_text(
            123,
            "event_type",
            100,
        )

    with pytest.raises(ValidationError):
        _validate_required_text(
            ["event"],
            "event_type",
            100,
        )


# ---------------------------------------------------------------------------
# Optional audit text
# ---------------------------------------------------------------------------


def test_optional_audit_text():
    assert _validate_optional_text(
        "  value  ",
        "action",
        100,
    ) == "value"
    assert (
        _validate_optional_text(
            None,
            "action",
            100,
        )
        is None
    )


def test_optional_audit_text_handles_empty_string():
    assert (
        _validate_optional_text(
            "",
            "action",
            100,
        )
        is None
    )


def test_optional_audit_text_handles_whitespace():
    assert (
        _validate_optional_text(
            "   ",
            "action",
            100,
        )
        is None
    )


def test_optional_audit_text_preserves_internal_whitespace():
    assert (
        _validate_optional_text(
            "workflow   published",
            "action",
            100,
        )
        == "workflow   published"
    )


def test_optional_audit_text_rejects_overlength_value():
    with pytest.raises(ValidationError):
        _validate_optional_text(
            "x" * 101,
            "action",
            100,
        )


def test_optional_audit_text_enforces_length_boundary():
    valid_value = "x" * 100

    assert (
        _validate_optional_text(
            valid_value,
            "action",
            100,
        )
        == valid_value
    )

    with pytest.raises(ValidationError):
        _validate_optional_text(
            "x" * 101,
            "action",
            100,
        )


def test_optional_audit_text_rejects_non_string_values():
    with pytest.raises(ValidationError):
        _validate_optional_text(
            123,
            "action",
            100,
        )

    with pytest.raises(ValidationError):
        _validate_optional_text(
            ["action"],
            "action",
            100,
        )


# ---------------------------------------------------------------------------
# Audit-metadata validation
# ---------------------------------------------------------------------------


def test_audit_metadata_rejects_sensitive_keys():
    with pytest.raises(ValidationError):
        _validate_metadata(
            {"api_key": "secret"}
        )


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "api_key",
        "apikey",
        "API_KEY",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "credential",
        "cookie",
        "private_key",
        "database_url",
        "connection_string",
        "prompt",
        "completion",
        "messages",
        "embedding",
        "vector",
        "generated_code",
        "signed_url",
        "storage_url",
        "file_url",
        "download_url",
    ],
)
def test_audit_metadata_rejects_common_sensitive_keys(
    sensitive_key: str,
):
    with pytest.raises(ValidationError):
        _validate_metadata(
            {
                "safe": "value",
                sensitive_key: "should-not-leak",
            }
        )


def test_audit_metadata_accepts_safe_data():
    metadata = {
        "document_count": 3,
        "provider": "openai",
    }

    assert (
        _validate_metadata(metadata)
        == metadata
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"document_count": 3},
        {"provider": "openai"},
        {"workflow_step": "validation"},
        {"tags": ["design", "accessibility"]},
    ],
)
def test_audit_metadata_accepts_safe_metadata(
    metadata: dict,
):
    assert (
        _validate_metadata(metadata)
        == metadata
    )


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        "not-a-dictionary",
        123,
        [],
        None,
    ],
)
def test_audit_metadata_rejects_non_dict_metadata(
    invalid_metadata,
):
    with pytest.raises(ValidationError):
        _validate_metadata(
            invalid_metadata
        )


def test_audit_metadata_rejects_nested_sensitive_values():
    with pytest.raises(ValidationError):
        _validate_metadata(
            {
                "context": {
                    "api_key": "should-not-leak"
                }
            }
        )


def test_audit_metadata_rejects_overly_large_metadata():
    metadata = {
        f"key_{index}": f"value-{index}"
        for index in range(100)
    }

    with pytest.raises(ValidationError):
        _validate_metadata(metadata)


# ---------------------------------------------------------------------------
# Cross-field consistency
# ---------------------------------------------------------------------------


def test_approval_status_and_decision_are_independent():
    status = _normalize_status(
        "approved"
    )
    decision = _normalize_decision(
        "approved"
    )

    assert (
        status == ApprovalStatus.APPROVED
    )
    assert (
        decision
        == ApprovalDecision.APPROVED
    )


def test_none_status_and_comment_are_allowed():
    assert (
        _normalize_status(None) is None
    )
    assert (
        _normalize_comment(None)
        is None
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
