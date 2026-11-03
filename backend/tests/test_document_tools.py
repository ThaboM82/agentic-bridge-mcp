"""
Comprehensive tests for document tool helpers.

These tests cover document and tenant ID coercion, document batch validation,
deduplication and batching, document summaries, sensitive-metadata redaction,
tag normalization, and document tool call-ID generation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import ValidationError
from app.graph.tools.document_tools import (
    _coerce_document_id,
    _coerce_document_ids,
    _coerce_tenant_id,
    generate_document_tool_call_id,
    get_document_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tenant_id() -> str:
    return str(uuid4())


@pytest.fixture
def document_id() -> UUID:
    return uuid4()


@pytest.fixture
def document_timestamp() -> datetime:
    return datetime(
        2026,
        1,
        1,
        12,
        0,
        0,
        tzinfo=timezone.utc,
    )


class FakeDocument:
    """Deterministic document-like object for summary tests."""

    def __init__(
        self,
        *,
        document_id: UUID | None = None,
        title: str = "Architecture Brief",
        summary: str = (
            "Internal architecture overview."
        ),
        status: str = "processed",
        document_type: str = "pdf",
        mime_type: str = "application/pdf",
        file_size_bytes: int = 1024,
        page_count: int = 3,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        self.id = document_id or uuid4()
        self.title = title
        self.summary = summary
        self.status = status
        self.document_type = document_type
        self.mime_type = mime_type
        self.file_size_bytes = file_size_bytes
        self.page_count = page_count
        self.tags = tags or []
        self.metadata = metadata or {}
        self.created_at = created_at
        self.updated_at = updated_at


# ---------------------------------------------------------------------------
# Document-ID coercion
# ---------------------------------------------------------------------------


def test_coerce_document_id_accepts_uuid_string():
    document_id = uuid4()

    assert (
        _coerce_document_id(
            str(document_id)
        )
        == document_id
    )


def test_coerce_document_id_accepts_uuid_object():
    document_id = uuid4()

    assert (
        _coerce_document_id(
            document_id
        )
        == document_id
    )


def test_coerce_document_id_strips_whitespace():
    document_id = uuid4()

    assert (
        _coerce_document_id(
            f"  {document_id}  "
        )
        == document_id
    )


def test_coerce_document_id_normalizes_case():
    document_id = uuid4()

    assert (
        _coerce_document_id(
            str(document_id).upper()
        )
        == document_id
    )


@pytest.mark.parametrize(
    "invalid_document_id",
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
def test_coerce_document_id_rejects_invalid_value(
    invalid_document_id,
):
    with pytest.raises(ValidationError):
        _coerce_document_id(
            invalid_document_id
        )


# ---------------------------------------------------------------------------
# Tenant-ID coercion
# ---------------------------------------------------------------------------


def test_coerce_tenant_id_accepts_uuid_string():
    tenant_id = uuid4()

    assert (
        _coerce_tenant_id(
            str(tenant_id)
        )
        == str(tenant_id)
    )


def test_coerce_tenant_id_accepts_uuid_object():
    tenant_id = uuid4()

    assert (
        _coerce_tenant_id(tenant_id)
        == str(tenant_id)
    )


def test_coerce_tenant_id_strips_whitespace():
    tenant_id = uuid4()

    assert (
        _coerce_tenant_id(
            f"  {tenant_id}  "
        )
        == str(tenant_id)
    )


def test_coerce_tenant_id_normalizes_case():
    tenant_id = uuid4()

    assert (
        _coerce_tenant_id(
            str(tenant_id).upper()
        )
        == str(tenant_id)
    )


@pytest.mark.parametrize(
    "invalid_tenant_id",
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
def test_coerce_tenant_id_rejects_invalid_value(
    invalid_tenant_id,
):
    with pytest.raises(ValidationError):
        _coerce_tenant_id(
            invalid_tenant_id
        )


# ---------------------------------------------------------------------------
# Document-ID batch coercion
# ---------------------------------------------------------------------------


def test_coerce_document_ids_deduplicates_and_bounds():
    first_id = uuid4()
    second_id = uuid4()

    document_ids = _coerce_document_ids(
        [
            str(first_id),
            first_id,
            str(second_id),
        ]
    )

    assert document_ids == [
        first_id,
        second_id,
    ]


def test_coerce_document_ids_preserves_order():
    first_id = uuid4()
    second_id = uuid4()
    third_id = uuid4()

    document_ids = _coerce_document_ids(
        [
            str(third_id),
            str(first_id),
            str(second_id),
        ]
    )

    assert document_ids == [
        third_id,
        first_id,
        second_id,
    ]


def test_coerce_document_ids_accepts_uuid_objects():
    first_id = uuid4()
    second_id = uuid4()

    document_ids = _coerce_document_ids(
        [first_id, second_id]
    )

    assert document_ids == [
        first_id,
        second_id,
    ]


def test_coerce_document_ids_strips_whitespace():
    first_id = uuid4()
    second_id = uuid4()

    document_ids = _coerce_document_ids(
        [
            f"  {first_id}  ",
            f"  {second_id}  ",
        ]
    )

    assert document_ids == [
        first_id,
        second_id,
    ]


def test_coerce_document_ids_rejects_empty_batch():
    with pytest.raises(ValidationError):
        _coerce_document_ids([])


def test_coerce_document_ids_rejects_none_batch():
    with pytest.raises(ValidationError):
        _coerce_document_ids(None)


@pytest.mark.parametrize(
    "invalid_batch",
    [
        "not-a-list",
        123,
        {},
        [uuid4(), "not-a-uuid"],
        [uuid4(), ""],
        [uuid4(), None],
    ],
)
def test_coerce_document_ids_rejects_invalid_batches(
    invalid_batch,
):
    with pytest.raises(ValidationError):
        _coerce_document_ids(
            invalid_batch
        )


def test_coerce_document_ids_rejects_oversized_batch():
    oversized_batch = [
        str(uuid4()) for _ in range(101)
    ]

    with pytest.raises(ValidationError):
        _coerce_document_ids(
            oversized_batch
        )


# ---------------------------------------------------------------------------
# Document summaries
# ---------------------------------------------------------------------------


def test_document_summary_excludes_sensitive_metadata():
    class FakeDocument:
        id = uuid4()
        title = "Architecture Brief"
        summary = (
            "Internal architecture overview."
        )
        status = "processed"
        document_type = "pdf"
        mime_type = "application/pdf"
        file_size_bytes = 1024
        page_count = 3
        tags = [
            "architecture",
            "architecture",
            "backend",
        ]
        metadata = {
            "source_system": "sharepoint",
            "api_key": "should-not-leak",
            "signed_url": "should-not-leak",
        }
        created_at = None
        updated_at = None

    summary = get_document_summary(
        FakeDocument()
    )

    assert summary["title"] == (
        "Architecture Brief"
    )
    assert summary["tags"] == [
        "architecture",
        "backend",
    ]
    assert (
        "api_key"
        not in summary["metadata"]
    )
    assert (
        "signed_url"
        not in summary["metadata"]
    )
    assert summary["metadata"][
        "source_system"
    ] == "sharepoint"


def test_document_summary_includes_core_fields(
    document_id: UUID,
    document_timestamp: datetime,
):
    document = FakeDocument(
        document_id=document_id,
        created_at=document_timestamp,
        updated_at=document_timestamp,
    )

    summary = get_document_summary(
        document
    )

    assert summary["id"] == str(
        document_id
    )
    assert summary["title"] == (
        "Architecture Brief"
    )
    assert summary["summary"] == (
        "Internal architecture overview."
    )
    assert (
        summary["status"]
        == "processed"
    )
    assert (
        summary["document_type"]
        == "pdf"
    )
    assert summary["mime_type"] == (
        "application/pdf"
    )
    assert (
        summary["file_size_bytes"]
        == 1024
    )
    assert summary["page_count"] == 3
    assert summary["created_at"] == (
        document_timestamp
    )
    assert summary["updated_at"] == (
        document_timestamp
    )


def test_document_summary_redacts_sensitive_metadata_keys():
    document = FakeDocument(
        metadata={
            "source_system": "sharepoint",
            "api_key": "should-not-leak",
            "apikey": "should-not-leak",
            "password": "should-not-leak",
            "secret": "should-not-leak",
            "token": "should-not-leak",
            "authorization": "should-not-leak",
            "credential": "should-not-leak",
            "private_key": "should-not-leak",
            "database_url": "should-not-leak",
            "connection_string": "should-not-leak",
            "prompt": "should-not-leak",
            "completion": "should-not-leak",
            "messages": "should-not-leak",
            "embedding": "should-not-leak",
            "vector": "should-not-leak",
            "generated_code": "should-not-leak",
            "signed_url": "should-not-leak",
            "storage_url": "should-not-leak",
            "file_url": "should-not-leak",
            "download_url": "should-not-leak",
        }
    )

    summary = get_document_summary(
        document
    )

    metadata = summary["metadata"]

    assert metadata["source_system"] == (
        "sharepoint"
    )

    for sensitive_key in (
        "api_key",
        "apikey",
        "password",
        "secret",
        "token",
        "authorization",
        "credential",
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
    ):
        assert sensitive_key not in metadata


def test_document_summary_deduplicates_tags():
    document = FakeDocument(
        tags=[
            "architecture",
            "Architecture",
            "ARCHITECTURE",
            "backend",
            "Backend",
        ]
    )

    summary = get_document_summary(
        document
    )

    assert summary["tags"] == [
        "architecture",
        "backend",
    ]


def test_document_summary_handles_empty_tags():
    document = FakeDocument(tags=[])

    summary = get_document_summary(
        document
    )

    assert summary["tags"] == []


def test_document_summary_handles_none_tags():
    document = FakeDocument(
        tags=None
    )

    summary = get_document_summary(
        document
    )

    assert summary["tags"] == []


def test_document_summary_handles_empty_metadata():
    document = FakeDocument(
        metadata={}
    )

    summary = get_document_summary(
        document
    )

    assert summary["metadata"] == {}


def test_document_summary_handles_none_metadata():
    document = FakeDocument(
        metadata=None
    )

    summary = get_document_summary(
        document
    )

    assert summary["metadata"] == {}


def test_document_summary_does_not_leak_sensitive_metadata():
    document = FakeDocument(
        metadata={
            "source_system": "sharepoint",
            "api_key": "sk-should-not-leak",
            "signed_url": "https://storage.example.com/signed?signature=secret",
        }
    )

    summary = get_document_summary(
        document
    )

    serialized_summary = str(summary)

    assert (
        "sk-should-not-leak"
        not in serialized_summary
    )
    assert (
        "signature=secret"
        not in serialized_summary
    )


# ---------------------------------------------------------------------------
# Tool call IDs
# ---------------------------------------------------------------------------


def test_generate_document_tool_call_id():
    tool_call_id = (
        generate_document_tool_call_id()
    )

    assert tool_call_id.startswith(
        "doc_tool_"
    )
    assert (
        tool_call_id
        != generate_document_tool_call_id()
    )


def test_document_tool_call_id_is_url_safe():
    tool_call_id = (
        generate_document_tool_call_id()
    )

    assert " " not in tool_call_id
    assert "/" not in tool_call_id
    assert "+" not in tool_call_id


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
