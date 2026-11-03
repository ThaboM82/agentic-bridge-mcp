"""
Comprehensive tests for document CRUD normalization and validation helpers.

These tests cover document types and statuses, tag normalization, search-term
handling, whitespace normalization, case-insensitive deduplication, length and
count limits, and fail-closed validation behavior.
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.enums import (
    DocumentStatus,
    DocumentType,
)
from app.core.exceptions import ValidationError
from app.crud.document_crud import (
    _normalize_document_status,
    _normalize_document_type,
    _normalize_search_term,
    _normalize_tags,
)


# ---------------------------------------------------------------------------
# Document-type normalization
# ---------------------------------------------------------------------------


def test_normalize_document_type():
    assert (
        _normalize_document_type(
            "requirements"
        )
        == DocumentType.REQUIREMENTS
    )
    assert (
        _normalize_document_type(None)
        is None
    )


@pytest.mark.parametrize(
    "document_type,expected",
    [
        (
            "requirements",
            DocumentType.REQUIREMENTS,
        ),
        (
            "REQUIREMENTS",
            DocumentType.REQUIREMENTS,
        ),
        (
            "  requirements  ",
            DocumentType.REQUIREMENTS,
        ),
        (
            "architecture",
            DocumentType.ARCHITECTURE,
        ),
        (
            "Architecture",
            DocumentType.ARCHITECTURE,
        ),
        (
            "design",
            DocumentType.DESIGN,
        ),
        (
            "DESIGN",
            DocumentType.DESIGN,
        ),
        (
            "test_plan",
            DocumentType.TEST_PLAN,
        ),
        (
            "test-plan",
            DocumentType.TEST_PLAN,
        ),
    ],
)
def test_normalize_document_type_accepts_supported_types(
    document_type: str,
    expected: DocumentType,
):
    assert (
        _normalize_document_type(
            document_type
        )
        == expected
    )


@pytest.mark.parametrize(
    "invalid_document_type",
    [
        "",
        "   ",
        "unknown-type",
        "specification",
        "contract",
        "invoice",
        "123",
        [],
        {},
    ],
)
def test_invalid_document_type(
    invalid_document_type,
):
    with pytest.raises(ValidationError):
        _normalize_document_type(
            invalid_document_type
        )


def test_normalize_document_type_none_returns_none():
    assert (
        _normalize_document_type(None)
        is None
    )


# ---------------------------------------------------------------------------
# Document-status normalization
# ---------------------------------------------------------------------------


def test_normalize_document_status():
    assert (
        _normalize_document_status(
            "indexed"
        )
        == DocumentStatus.INDEXED
    )
    assert (
        _normalize_document_status(None)
        is None
    )


@pytest.mark.parametrize(
    "document_status,expected",
    [
        (
            "uploaded",
            DocumentStatus.UPLOADED,
        ),
        (
            "UPLOADING",
            DocumentStatus.UPLOADING,
        ),
        (
            "  processing  ",
            DocumentStatus.PROCESSING,
        ),
        (
            "processed",
            DocumentStatus.PROCESSED,
        ),
        (
            "indexed",
            DocumentStatus.INDEXED,
        ),
        (
            "failed",
            DocumentStatus.FAILED,
        ),
        (
            "archived",
            DocumentStatus.ARCHIVED,
        ),
    ],
)
def test_normalize_document_status_accepts_supported_statuses(
    document_status: str,
    expected: DocumentStatus,
):
    assert (
        _normalize_document_status(
            document_status
        )
        == expected
    )


@pytest.mark.parametrize(
    "invalid_document_status",
    [
        "",
        "   ",
        "unknown-status",
        "pending",
        "complete",
        "error",
        "deleted",
        "123",
        [],
        {},
    ],
)
def test_invalid_document_status(
    invalid_document_status,
):
    with pytest.raises(ValidationError):
        _normalize_document_status(
            invalid_document_status
        )


def test_normalize_document_status_none_returns_none():
    assert (
        _normalize_document_status(None)
        is None
    )


# ---------------------------------------------------------------------------
# Tag normalization
# ---------------------------------------------------------------------------


def test_normalize_tags():
    assert _normalize_tags(
        ["  API ", "api", "Design"]
    ) == ["api", "design"]


def test_normalize_tags_preserves_order():
    assert _normalize_tags(
        ["design", "api", "backend"]
    ) == [
        "design",
        "api",
        "backend",
    ]


def test_normalize_tags_deduplicates_case_insensitively():
    assert _normalize_tags(
        [
            "api",
            "API",
            "Api",
            "design",
            "DESIGN",
        ]
    ) == ["api", "design"]


def test_normalize_tags_strips_whitespace():
    assert _normalize_tags(
        [
            "  api  ",
            "\tdesign\t",
            "\nbackend\n",
        ]
    ) == [
        "api",
        "design",
        "backend",
    ]


def test_normalize_tags_removes_empty_values():
    assert _normalize_tags(
        ["", "   ", "api", None]
    ) == ["api"]


def test_normalize_tags_handles_empty_list():
    assert _normalize_tags([]) == []


def test_normalize_tags_handles_none():
    assert _normalize_tags(None) == []


@pytest.mark.parametrize(
    "invalid_tags",
    [
        "not-a-list",
        123,
        {},
        [123],
        [True],
        [{"tag": "api"}],
        [["api"]],
    ],
)
def test_normalize_tags_rejects_non_string_tags(
    invalid_tags,
):
    with pytest.raises(ValidationError):
        _normalize_tags(invalid_tags)


def test_invalid_tag_count():
    with pytest.raises(ValidationError):
        _normalize_tags(
            [
                f"tag-{index}"
                for index in range(30)
            ]
        )


def test_tag_count_boundary_is_enforced():
    valid_tags = [
        f"tag-{index}"
        for index in range(20)
    ]

    assert (
        _normalize_tags(valid_tags)
        == valid_tags
    )

    with pytest.raises(ValidationError):
        _normalize_tags(
            [
                f"tag-{index}"
                for index in range(21)
            ]
        )


def test_invalid_tag_length():
    with pytest.raises(ValidationError):
        _normalize_tags(["x" * 100])


def test_tag_length_boundary_is_enforced():
    valid_tag = "x" * 50

    assert _normalize_tags(
        [valid_tag]
    ) == [valid_tag]

    with pytest.raises(ValidationError):
        _normalize_tags(["x" * 51])


# ---------------------------------------------------------------------------
# Search-term normalization
# ---------------------------------------------------------------------------


def test_normalize_search_term():
    assert (
        _normalize_search_term(
            "  workflow  "
        )
        == "workflow"
    )
    assert (
        _normalize_search_term("   ")
        is None
    )
    assert (
        _normalize_search_term(None)
        is None
    )


def test_normalize_search_term_collapses_internal_whitespace():
    assert (
        _normalize_search_term(
            "workflow   execution   status"
        )
        == "workflow execution status"
    )


def test_normalize_search_term_handles_newlines_and_tabs():
    assert (
        _normalize_search_term(
            "workflow\texecution\nstatus"
        )
        == "workflow execution status"
    )


def test_normalize_search_term_handles_empty_string():
    assert (
        _normalize_search_term("")
        is None
    )


@pytest.mark.parametrize(
    "valid_search_term",
    [
        "workflow",
        "architecture brief",
        "API documentation",
        "accessibility requirements",
    ],
)
def test_normalize_search_term_accepts_valid_terms(
    valid_search_term: str,
):
    assert (
        _normalize_search_term(
            valid_search_term
        )
        == " ".join(
            valid_search_term.split()
        )
    )


def test_invalid_search_term():
    with pytest.raises(ValidationError):
        _normalize_search_term(
            "x" * 500
        )


def test_search_term_length_boundary_is_enforced():
    valid_search_term = "x" * 200

    assert (
        _normalize_search_term(
            valid_search_term
        )
        == valid_search_term
    )

    with pytest.raises(ValidationError):
        _normalize_search_term(
            "x" * 201
        )


def test_normalize_search_term_rejects_non_string_values():
    with pytest.raises(ValidationError):
        _normalize_search_term(123)

    with pytest.raises(ValidationError):
        _normalize_search_term(
            ["workflow"]
        )

    with pytest.raises(ValidationError):
        _normalize_search_term(
            {"query": "workflow"}
        )


# ---------------------------------------------------------------------------
# Cross-field consistency
# ---------------------------------------------------------------------------


def test_document_type_and_status_normalization_are_independent():
    document_type = (
        _normalize_document_type(
            "requirements"
        )
    )
    document_status = (
        _normalize_document_status(
            "indexed"
        )
    )

    assert (
        document_type
        == DocumentType.REQUIREMENTS
    )
    assert (
        document_status
        == DocumentStatus.INDEXED
    )


def test_none_document_type_and_status_are_allowed():
    assert (
        _normalize_document_type(None)
        is None
    )
    assert (
        _normalize_document_status(
            None
        )
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
