"""
Comprehensive tests for ingestion schemas.

These tests cover document creation, metadata validation, batch ingestion
configuration, job results, job responses, progress events, terminal-state
behavior, and security-oriented validation boundaries.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.ingestion import (
    BatchIngestionRequest,
    IngestionBatchError,
    IngestionDocumentCreate,
    IngestionJobCreatedResponse,
    IngestionJobResponse,
    IngestionJobResult,
    IngestionJobStatus,
    IngestionProgressEvent,
    IngestionSourceType,
)


VALID_DOCUMENT = {
    "title": "Enterprise Password Reset Specification",
    "content": (
        "Users must be able to securely reset their password."
    ),
}


def valid_document(
    **overrides,
) -> dict:
    """Return a valid ingestion document payload."""
    return {
        **VALID_DOCUMENT,
        **overrides,
    }


def valid_job_response(
    **overrides,
) -> dict:
    """Return a valid ingestion job response payload."""
    now = datetime.now(timezone.utc)

    base = {
        "job_id": "ingest_1234567890ab",
        "status": "processing",
        "total_documents": 10,
        "processed_documents": 4,
        "failed_documents": 1,
        "current_batch": 2,
        "total_batches": 5,
        "progress_percent": 40,
        "created_at": now,
        "updated_at": now,
    }

    return {
        **base,
        **overrides,
    }


# ---------------------------------------------------------------------------
# Ingestion document creation
# ---------------------------------------------------------------------------


def test_ingestion_document_defaults_to_manual_source():
    document = IngestionDocumentCreate.model_validate(
        VALID_DOCUMENT
    )

    assert (
        document.metadata.source_type
        == IngestionSourceType.MANUAL
    )
    assert document.metadata.tags == []


def test_ingestion_document_accepts_valid_document():
    document = IngestionDocumentCreate.model_validate(
        valid_document()
    )

    assert (
        document.title
        == "Enterprise Password Reset Specification"
    )
    assert document.content.startswith(
        "Users must be able"
    )


def test_ingestion_document_accepts_metadata():
    document = IngestionDocumentCreate.model_validate(
        valid_document(
            metadata={
                "source_type": "google_docs",
                "source_url": (
                    "https://docs.google.com/document/d/example"
                ),
                "source_id": "google-doc-123",
                "tenant_id": "tenant_001",
                "tags": ["requirements"],
            }
        )
    )

    assert (
        document.metadata.source_type
        == IngestionSourceType.GOOGLE_DOCS
    )
    assert (
        document.metadata.source_id
        == "google-doc-123"
    )
    assert document.metadata.tags == [
        "requirements"
    ]


def test_ingestion_document_normalizes_whitespace():
    document = IngestionDocumentCreate.model_validate(
        valid_document(
            title="  Enterprise Password Reset  ",
            content=(
                "  Users must reset passwords securely.  "
            ),
        )
    )

    assert (
        document.title
        == "Enterprise Password Reset"
    )
    assert document.content == (
        "Users must reset passwords securely."
    )


def test_ingestion_document_deduplicates_tags():
    document = IngestionDocumentCreate.model_validate(
        valid_document(
            metadata={
                "source_type": "manual",
                "tags": [
                    "requirements",
                    " Requirements ",
                    "security",
                ],
            }
        )
    )

    assert document.metadata.tags == [
        "requirements",
        "security",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", ""),
        ("title", "   "),
        ("title", None),
        ("title", 123),
        ("title", "x" * 256),
        ("content", ""),
        ("content", "   "),
        ("content", None),
        ("content", 123),
    ],
)
def test_ingestion_document_rejects_invalid_core_fields(
    field: str,
    value,
):
    with pytest.raises(ValidationError):
        IngestionDocumentCreate.model_validate(
            valid_document(
                **{field: value}
            )
        )


def test_ingestion_document_rejects_invalid_source_url():
    with pytest.raises(ValidationError):
        IngestionDocumentCreate.model_validate(
            valid_document(
                metadata={
                    "source_type": "google_docs",
                    "source_url": "not-a-url",
                }
            )
        )


def test_ingestion_document_rejects_unsupported_source_type():
    with pytest.raises(ValidationError):
        IngestionDocumentCreate.model_validate(
            valid_document(
                metadata={
                    "source_type": "unknown_source",
                }
            )
        )


def test_ingestion_document_rejects_unknown_metadata_fields():
    with pytest.raises(ValidationError) as exc_info:
        IngestionDocumentCreate.model_validate(
            valid_document(
                metadata={
                    "source_type": "manual",
                    "unexpected_field": "value",
                }
            )
        )

    assert any(
        error["type"] == "extra_forbidden"
        for error in exc_info.value.errors()
    )


def test_ingestion_document_rejects_too_many_tags():
    with pytest.raises(ValidationError):
        IngestionDocumentCreate.model_validate(
            valid_document(
                metadata={
                    "source_type": "manual",
                    "tags": [
                        f"tag-{index}"
                        for index in range(51)
                    ],
                }
            )
        )


def test_ingestion_document_rejects_oversized_tag():
    with pytest.raises(ValidationError):
        IngestionDocumentCreate.model_validate(
            valid_document(
                metadata={
                    "source_type": "manual",
                    "tags": ["x" * 101],
                }
            )
        )


def test_ingestion_document_rejects_oversized_source_id():
    with pytest.raises(ValidationError):
        IngestionDocumentCreate.model_validate(
            valid_document(
                metadata={
                    "source_type": "google_docs",
                    "source_id": "x" * 256,
                }
            )
        )


def test_ingestion_document_metadata_is_not_required():
    document = IngestionDocumentCreate.model_validate(
        valid_document()
    )

    assert document.metadata is not None
    assert (
        document.metadata.source_type
        == IngestionSourceType.MANUAL
    )


# ---------------------------------------------------------------------------
# Batch ingestion requests
# ---------------------------------------------------------------------------


def test_batch_ingestion_request_uses_sensible_defaults():
    request = BatchIngestionRequest.model_validate(
        {
            "documents": [
                valid_document()
            ]
        }
    )

    assert request.batch_size == 10
    assert request.max_concurrent_batches == 2
    assert request.continue_on_error is True


def test_batch_ingestion_request_accepts_multiple_documents():
    request = BatchIngestionRequest.model_validate(
        {
            "documents": [
                valid_document(),
                valid_document(
                    title="Second document"
                ),
                valid_document(
                    title="Third document"
                ),
            ]
        }
    )

    assert len(request.documents) == 3


def test_batch_ingestion_request_rejects_empty_document_list():
    with pytest.raises(ValidationError):
        BatchIngestionRequest.model_validate(
            {"documents": []}
        )


def test_batch_ingestion_request_rejects_missing_documents():
    with pytest.raises(ValidationError):
        BatchIngestionRequest.model_validate(
            {}
        )


@pytest.mark.parametrize(
    "batch_size",
    [0, -1, 101],
)
def test_batch_ingestion_request_rejects_invalid_batch_size(
    batch_size: int,
):
    with pytest.raises(ValidationError):
        BatchIngestionRequest.model_validate(
            {
                "documents": [
                    valid_document()
                ],
                "batch_size": batch_size,
            }
        )


@pytest.mark.parametrize(
    "max_concurrent_batches",
    [0, -1, 11],
)
def test_batch_ingestion_request_rejects_invalid_concurrency(
    max_concurrent_batches: int,
):
    with pytest.raises(ValidationError):
        BatchIngestionRequest.model_validate(
            {
                "documents": [
                    valid_document()
                ],
                "max_concurrent_batches": (
                    max_concurrent_batches
                ),
            }
        )


def test_batch_ingestion_request_accepts_valid_configuration():
    request = BatchIngestionRequest.model_validate(
        {
            "documents": [
                valid_document()
            ],
            "batch_size": 25,
            "max_concurrent_batches": 5,
            "continue_on_error": False,
        }
    )

    assert request.batch_size == 25
    assert request.max_concurrent_batches == 5
    assert request.continue_on_error is False


def test_batch_ingestion_request_rejects_invalid_document():
    with pytest.raises(ValidationError):
        BatchIngestionRequest.model_validate(
            {
                "documents": [
                    valid_document(
                        title=""
                    )
                ]
            }
        )


# ---------------------------------------------------------------------------
# Ingestion batch errors
# ---------------------------------------------------------------------------


def test_ingestion_batch_error_accepts_valid_payload():
    error = IngestionBatchError(
        batch_number=2,
        document_indexes=[3, 4],
        error_code="embedding_provider_unavailable",
        message="Embedding provider unavailable.",
        retryable=True,
    )

    assert error.batch_number == 2
    assert error.document_indexes == [3, 4]
    assert (
        error.error_code
        == "embedding_provider_unavailable"
    )
    assert error.retryable is True


def test_ingestion_batch_error_rejects_invalid_batch_number():
    with pytest.raises(ValidationError):
        IngestionBatchError(
            batch_number=0,
            document_indexes=[1],
            error_code="embedding_provider_unavailable",
            message="Embedding provider unavailable.",
            retryable=True,
        )


def test_ingestion_batch_error_rejects_empty_message():
    with pytest.raises(ValidationError):
        IngestionBatchError(
            batch_number=1,
            document_indexes=[1],
            error_code="embedding_provider_unavailable",
            message="",
            retryable=True,
        )


def test_ingestion_batch_error_rejects_empty_error_code():
    with pytest.raises(ValidationError):
        IngestionBatchError(
            batch_number=1,
            document_indexes=[1],
            error_code="",
            message="Embedding provider unavailable.",
            retryable=True,
        )


# ---------------------------------------------------------------------------
# Ingestion job results
# ---------------------------------------------------------------------------


def test_ingestion_job_result_supports_partial_success():
    result = IngestionJobResult(
        created_document_ids=[1, 2],
        created_document_count=2,
        failed_document_count=1,
        errors=[
            IngestionBatchError(
                batch_number=2,
                document_indexes=[3],
                error_code="embedding_provider_unavailable",
                message=(
                    "Embedding provider unavailable."
                ),
                retryable=True,
            )
        ],
    )

    assert result.created_document_count == 2
    assert result.failed_document_count == 1
    assert result.errors[0].retryable is True


def test_ingestion_job_result_supports_full_success():
    result = IngestionJobResult(
        created_document_ids=[1, 2, 3],
        created_document_count=3,
        failed_document_count=0,
        errors=[],
    )

    assert result.created_document_count == 3
    assert result.failed_document_count == 0
    assert result.errors == []


def test_ingestion_job_result_supports_full_failure():
    result = IngestionJobResult(
        created_document_ids=[],
        created_document_count=0,
        failed_document_count=2,
        errors=[
            IngestionBatchError(
                batch_number=1,
                document_indexes=[1, 2],
                error_code="validation_failed",
                message="Documents failed validation.",
                retryable=False,
            )
        ],
    )

    assert result.created_document_count == 0
    assert result.failed_document_count == 2
    assert result.errors[0].retryable is False


def test_ingestion_job_result_rejects_negative_counts():
    with pytest.raises(ValidationError):
        IngestionJobResult(
            created_document_ids=[],
            created_document_count=-1,
            failed_document_count=0,
            errors=[],
        )


def test_ingestion_job_result_rejects_negative_failure_count():
    with pytest.raises(ValidationError):
        IngestionJobResult(
            created_document_ids=[],
            created_document_count=0,
            failed_document_count=-1,
            errors=[],
        )


# ---------------------------------------------------------------------------
# Ingestion job responses
# ---------------------------------------------------------------------------


def test_ingestion_job_response_accepts_processing_state():
    job = IngestionJobResponse.model_validate(
        valid_job_response()
    )

    assert (
        job.status
        == IngestionJobStatus.PROCESSING
    )
    assert job.is_terminal is False


def test_ingestion_job_response_accepts_completed_state():
    job = IngestionJobResponse.model_validate(
        valid_job_response(
            status="completed",
            processed_documents=10,
            progress_percent=100,
        )
    )

    assert (
        job.status
        == IngestionJobStatus.COMPLETED
    )
    assert job.is_terminal is True


def test_ingestion_job_response_accepts_failed_state():
    job = IngestionJobResponse.model_validate(
        valid_job_response(
            status="failed",
            progress_percent=40,
        )
    )

    assert (
        job.status == IngestionJobStatus.FAILED
    )
    assert job.is_terminal is True


def test_ingestion_job_response_accepts_cancelled_state():
    job = IngestionJobResponse.model_validate(
        valid_job_response(
            status="cancelled",
            progress_percent=20,
        )
    )

    assert (
        job.status
        == IngestionJobStatus.CANCELLED
    )
    assert job.is_terminal is True


def test_ingestion_job_response_accepts_queued_state():
    job = IngestionJobResponse.model_validate(
        valid_job_response(
            status="queued",
            processed_documents=0,
            failed_documents=0,
            current_batch=0,
            progress_percent=0,
        )
    )

    assert (
        job.status == IngestionJobStatus.QUEUED
    )
    assert job.is_terminal is False


def test_ingestion_job_response_rejects_progress_above_100():
    with pytest.raises(ValidationError):
        IngestionJobResponse.model_validate(
            valid_job_response(
                progress_percent=101
            )
        )


def test_ingestion_job_response_rejects_negative_progress():
    with pytest.raises(ValidationError):
        IngestionJobResponse.model_validate(
            valid_job_response(
                progress_percent=-1
            )
        )


def test_ingestion_job_response_rejects_unknown_status():
    with pytest.raises(ValidationError):
        IngestionJobResponse.model_validate(
            valid_job_response(
                status="unknown-status"
            )
        )


def test_ingestion_job_response_rejects_invalid_job_id():
    with pytest.raises(ValidationError):
        IngestionJobResponse.model_validate(
            valid_job_response(
                job_id="invalid job id"
            )
        )


def test_ingestion_job_response_rejects_updated_at_before_created_at():
    created_at = datetime.now(timezone.utc)
    updated_at = created_at - timedelta(
        seconds=1
    )

    with pytest.raises(ValidationError):
        IngestionJobResponse.model_validate(
            valid_job_response(
                created_at=created_at,
                updated_at=updated_at,
            )
        )


# ---------------------------------------------------------------------------
# Ingestion job creation responses
# ---------------------------------------------------------------------------


def test_ingestion_job_created_response():
    response = IngestionJobCreatedResponse(
        job_id="ingest_1234567890ab",
        total_documents=25,
        total_batches=3,
        status_endpoint=(
            "/api/v1/documents/ingest/jobs/"
            "ingest_1234567890ab"
        ),
        stream_endpoint=(
            "/api/v1/documents/ingest/jobs/"
            "ingest_1234567890ab/stream"
        ),
    )

    assert (
        response.status
        == IngestionJobStatus.QUEUED
    )
    assert response.progress_percent == 0
    assert response.total_documents == 25
    assert response.total_batches == 3


def test_ingestion_job_created_response_rejects_invalid_job_id():
    with pytest.raises(ValidationError):
        IngestionJobCreatedResponse(
            job_id="invalid job id",
            total_documents=25,
            total_batches=3,
            status_endpoint="/status",
            stream_endpoint="/stream",
        )


def test_ingestion_job_created_response_rejects_zero_documents():
    with pytest.raises(ValidationError):
        IngestionJobCreatedResponse(
            job_id="ingest_1234567890ab",
            total_documents=0,
            total_batches=0,
            status_endpoint="/status",
            stream_endpoint="/stream",
        )


def test_ingestion_job_created_response_rejects_invalid_endpoints():
    with pytest.raises(ValidationError):
        IngestionJobCreatedResponse(
            job_id="ingest_1234567890ab",
            total_documents=25,
            total_batches=3,
            status_endpoint="not-a-path",
            stream_endpoint="/stream",
        )


# ---------------------------------------------------------------------------
# Ingestion progress events
# ---------------------------------------------------------------------------


def test_progress_event_supports_progress_update():
    event = IngestionProgressEvent(
        event="progress",
        job_id="ingest_1234567890ab",
        status=IngestionJobStatus.PROCESSING,
        processed_documents=6,
        failed_documents=1,
        progress_percent=60,
        current_batch=3,
        total_batches=10,
    )

    assert event.event == "progress"
    assert event.progress_percent == 60
    assert event.processed_documents == 6
    assert event.failed_documents == 1
    assert event.current_batch == 3
    assert event.total_batches == 10


def test_progress_event_supports_completion():
    event = IngestionProgressEvent(
        event="completed",
        job_id="ingest_1234567890ab",
        status=IngestionJobStatus.COMPLETED,
        processed_documents=10,
        failed_documents=0,
        progress_percent=100,
        current_batch=5,
        total_batches=5,
    )

    assert event.event == "completed"
    assert (
        event.status
        == IngestionJobStatus.COMPLETED
    )
    assert event.progress_percent == 100


def test_progress_event_supports_failure():
    event = IngestionProgressEvent(
        event="failed",
        job_id="ingest_1234567890ab",
        status=IngestionJobStatus.FAILED,
        processed_documents=4,
        failed_documents=6,
        progress_percent=40,
        current_batch=3,
        total_batches=5,
    )

    assert event.event == "failed"
    assert (
        event.status == IngestionJobStatus.FAILED
    )
    assert event.failed_documents == 6


def test_progress_event_rejects_invalid_progress():
    with pytest.raises(ValidationError):
        IngestionProgressEvent(
            event="progress",
            job_id="ingest_1234567890ab",
            status=IngestionJobStatus.PROCESSING,
            processed_documents=6,
            failed_documents=1,
            progress_percent=101,
            current_batch=3,
            total_batches=10,
        )


def test_progress_event_rejects_unknown_event_type():
    with pytest.raises(ValidationError):
        IngestionProgressEvent(
            event="unknown-event",
            job_id="ingest_1234567890ab",
            status=IngestionJobStatus.PROCESSING,
            processed_documents=6,
            failed_documents=1,
            progress_percent=60,
            current_batch=3,
            total_batches=10,
        )


def test_progress_event_rejects_invalid_job_id():
    with pytest.raises(ValidationError):
        IngestionProgressEvent(
            event="progress",
            job_id="invalid job id",
            status=IngestionJobStatus.PROCESSING,
            processed_documents=6,
            failed_documents=1,
            progress_percent=60,
            current_batch=3,
            total_batches=10,
        )


# ---------------------------------------------------------------------------
# Security-oriented validation
# ---------------------------------------------------------------------------


def test_document_schema_does_not_accept_credentials():
    with pytest.raises(ValidationError):
        IngestionDocumentCreate.model_validate(
            valid_document(
                metadata={
                    "source_type": "manual",
                    "api_key": (
                        "sk-should-not-leak"
                    ),
                }
            )
        )


def test_document_schema_does_not_accept_authorization_headers():
    with pytest.raises(ValidationError):
        IngestionDocumentCreate.model_validate(
            valid_document(
                metadata={
                    "source_type": "manual",
                    "authorization": (
                        "Bearer secret-token"
                    ),
                }
            )
        )


def test_job_response_serialization_is_json_safe():
    job = IngestionJobResponse.model_validate(
        valid_job_response()
    )

    serialized_job = job.model_dump_json()

    assert "ingest_1234567890ab" in serialized_job
    assert "Traceback" not in serialized_job
    assert "object at 0x" not in serialized_job
