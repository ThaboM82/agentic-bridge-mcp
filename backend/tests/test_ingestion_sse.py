"""
Comprehensive API tests for the ingestion SSE streaming endpoint.

These tests cover connection behavior, terminal and non-terminal job states,
progress events, partial completion, failures, cancellation, unknown jobs,
invalid job IDs, event ordering, response headers, and safe serialization.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.ingestion import IngestionJobStatus
from app.services.document_service import batch_ingestion_service


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """Return an authenticated test client for the FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


def parse_sse_events(
    response_text: str,
) -> list[dict[str, Any]]:
    """
    Parse Server-Sent Events from a response body.

    Args:
        response_text: Raw SSE response text.

    Returns:
        A list of decoded JSON event payloads.
    """
    events: list[dict[str, Any]] = []

    for line in response_text.splitlines():
        if not line.startswith("data: "):
            continue

        payload = line.removeprefix(
            "data: "
        ).strip()

        if payload:
            events.append(
                json.loads(payload)
            )

    return events


def seed_ingestion_job(
    job_id: str,
    status: str,
    progress_percent: int,
    *,
    total_documents: int = 4,
    processed_documents: int = 2,
    failed_documents: int = 0,
    skipped_documents: int = 0,
    current_batch: int = 1,
    total_batches: int = 2,
    error_message: str | None = None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Seed an ingestion job into the in-memory batch ingestion service.

    Args:
        job_id: Ingestion job identifier.
        status: Job status.
        progress_percent: Job progress percentage.
        total_documents: Total documents in the job.
        processed_documents: Number of processed documents.
        failed_documents: Number of failed documents.
        skipped_documents: Number of skipped documents.
        current_batch: Current batch number.
        total_batches: Total batch count.
        error_message: Optional job error message.
        result: Optional job result payload.

    Returns:
        The seeded job record.
    """
    now = datetime.now(timezone.utc)

    job = {
        "job_id": job_id,
        "status": status,
        "total_documents": total_documents,
        "processed_documents": processed_documents,
        "failed_documents": failed_documents,
        "skipped_documents": skipped_documents,
        "current_batch": current_batch,
        "total_batches": total_batches,
        "progress_percent": progress_percent,
        "error_message": error_message,
        "result": result,
        "created_at": now,
        "updated_at": now,
    }

    batch_ingestion_service._jobs[job_id] = job

    return job


def clear_ingestion_jobs() -> None:
    """Clear all in-memory ingestion jobs after a test."""
    batch_ingestion_service._jobs.clear()


@pytest.fixture(autouse=True)
def clean_ingestion_job_store():
    """Ensure each test starts and ends with an empty in-memory job store."""
    clear_ingestion_jobs()

    yield

    clear_ingestion_jobs()


# ---------------------------------------------------------------------------
# SSE connection behavior
# ---------------------------------------------------------------------------


def test_sse_stream_returns_event_stream_content_type(client):
    job_id = "ingest_sse_content_type"
    seed_ingestion_job(
        job_id,
        "completed",
        100,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        assert response.status_code == 200
        assert response.headers[
            "content-type"
        ].startswith("text/event-stream")


def test_sse_stream_emits_connected_event_first(client):
    job_id = "ingest_sse_connected_first"
    seed_ingestion_job(
        job_id,
        "processing",
        50,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    assert events
    assert events[0]["event"] == "connected"
    assert events[0]["job_id"] == job_id


def test_sse_stream_emits_terminal_event_last(client):
    job_id = "ingest_sse_terminal_last"
    seed_ingestion_job(
        job_id,
        "completed",
        100,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    assert len(events) >= 2
    assert events[-1]["event"] in {
        "progress",
        "completed",
        "failed",
        "cancelled",
        "partially_completed",
    }


def test_sse_stream_includes_job_id_in_events(client):
    job_id = "ingest_sse_job_id"
    seed_ingestion_job(
        job_id,
        "processing",
        50,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    assert events

    for event in events:
        if event.get("event") != "connected":
            assert event.get("job_id") == job_id


# ---------------------------------------------------------------------------
# Terminal job states
# ---------------------------------------------------------------------------


def test_sse_stream_emits_connected_and_terminal_events(client):
    job_id = "ingest_sse_completed"
    seed_ingestion_job(
        job_id,
        "completed",
        100,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        assert response.status_code == 200
        assert response.headers[
            "content-type"
        ].startswith("text/event-stream")

        events = parse_sse_events(
            response.text
        )

    assert events[0]["event"] == "connected"
    assert (
        events[-1]["status"]
        == IngestionJobStatus.COMPLETED
    )
    assert events[-1]["progress_percent"] == 100


def test_sse_stream_emits_failure_event(client):
    job_id = "ingest_sse_failed"
    seed_ingestion_job(
        job_id,
        "failed",
        25,
        error_message="Document parsing failed.",
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    assert events[-1][
        "status"
    ] == IngestionJobStatus.FAILED
    assert events[-1]["progress_percent"] == 25


def test_sse_stream_emits_cancelled_event(client):
    job_id = "ingest_sse_cancelled"
    seed_ingestion_job(
        job_id,
        "cancelled",
        40,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    assert events[-1][
        "status"
    ] == IngestionJobStatus.CANCELLED
    assert events[-1]["progress_percent"] == 40


def test_sse_stream_emits_partial_completion_event(client):
    job_id = "ingest_sse_partial"
    seed_ingestion_job(
        job_id,
        "partially_completed",
        75,
        processed_documents=3,
        failed_documents=1,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    assert events[-1]["status"] == (
        IngestionJobStatus.PARTIALLY_COMPLETED
    )
    assert events[-1]["progress_percent"] == 75


# ---------------------------------------------------------------------------
# Non-terminal job states
# ---------------------------------------------------------------------------


def test_sse_stream_emits_progress_event(client):
    job_id = "ingest_sse_processing"
    seed_ingestion_job(
        job_id,
        "processing",
        50,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    progress_events = [
        event
        for event in events
        if event.get("event") == "progress"
    ]

    assert progress_events
    assert (
        progress_events[0]["progress_percent"]
        == 50
    )


def test_sse_stream_emits_queued_progress_event(client):
    job_id = "ingest_sse_queued"
    seed_ingestion_job(
        job_id,
        "queued",
        0,
        processed_documents=0,
        current_batch=0,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    progress_events = [
        event
        for event in events
        if event.get("event") == "progress"
    ]

    assert progress_events
    assert (
        progress_events[0]["status"]
        == IngestionJobStatus.QUEUED
    )
    assert (
        progress_events[0]["progress_percent"]
        == 0
    )


def test_sse_stream_progress_event_contains_batch_metadata(client):
    job_id = "ingest_sse_batch_metadata"
    seed_ingestion_job(
        job_id,
        "processing",
        50,
        current_batch=1,
        total_batches=2,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    progress_events = [
        event
        for event in events
        if event.get("event") == "progress"
    ]

    assert progress_events
    assert progress_events[0]["current_batch"] == 1
    assert progress_events[0]["total_batches"] == 2


def test_sse_stream_progress_event_contains_document_counts(client):
    job_id = "ingest_sse_document_counts"
    seed_ingestion_job(
        job_id,
        "processing",
        50,
        total_documents=4,
        processed_documents=2,
        failed_documents=0,
        skipped_documents=0,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    progress_events = [
        event
        for event in events
        if event.get("event") == "progress"
    ]

    assert progress_events
    assert (
        progress_events[0]["processed_documents"]
        == 2
    )
    assert (
        progress_events[0]["failed_documents"]
        == 0
    )
    assert (
        progress_events[0]["skipped_documents"]
        == 0
    )


# ---------------------------------------------------------------------------
# Unknown and invalid jobs
# ---------------------------------------------------------------------------


def test_sse_stream_reports_unknown_job(client):
    with client.stream(
        "GET",
        "/api/v1/documents/ingest/jobs/ingest_does_not_exist/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    assert events[0]["event"] == "connected"
    assert events[-1] == {
        "error": "Ingestion job not found"
    }


def test_sse_stream_rejects_invalid_job_id(client):
    response = client.get(
        "/api/v1/documents/ingest/jobs/invalid%20job%20id/stream",
    )

    assert response.status_code in {
        400,
        404,
        422,
    }


def test_sse_stream_rejects_empty_job_id(client):
    response = client.get(
        "/api/v1/documents/ingest/jobs//stream",
    )

    assert response.status_code in {
        400,
        404,
        405,
        422,
    }


# ---------------------------------------------------------------------------
# Event payload safety
# ---------------------------------------------------------------------------


def test_sse_events_are_valid_json(client):
    job_id = "ingest_sse_valid_json"
    seed_ingestion_job(
        job_id,
        "completed",
        100,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    assert events

    for event in events:
        assert isinstance(event, dict)


def test_sse_events_do_not_expose_sensitive_fields(client):
    job_id = "ingest_sse_security"
    seeded_job = seed_ingestion_job(
        job_id,
        "failed",
        25,
        error_message=(
            "Provider failed with api_key=sk-should-not-leak"
        ),
    )

    seeded_job["internal_debug"] = {
        "api_key": "sk-should-not-leak",
        "database_url": (
            "postgresql://user:secret@localhost/db"
        ),
    }

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    serialized_events = json.dumps(events).lower()

    assert "sk-should-not-leak" not in serialized_events
    assert "postgresql://user:secret@localhost/db" not in serialized_events
    assert "internal_debug" not in serialized_events


def test_sse_events_do_not_expose_stack_traces(client):
    job_id = "ingest_sse_no_traceback"
    seed_ingestion_job(
        job_id,
        "failed",
        25,
        error_message="Document parsing failed.",
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    serialized_events = json.dumps(
        events
    ).lower()

    assert "traceback" not in serialized_events
    assert "runtimeerror" not in serialized_events


# ---------------------------------------------------------------------------
# Event ordering and completeness
# ---------------------------------------------------------------------------


def test_sse_stream_event_sequence_is_stable(client):
    job_id = "ingest_sse_sequence"
    seed_ingestion_job(
        job_id,
        "completed",
        100,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    assert len(events) >= 2
    assert events[0]["event"] == "connected"

    terminal_event = events[-1]

    assert terminal_event["status"] == (
        IngestionJobStatus.COMPLETED
    )
    assert terminal_event["progress_percent"] == 100


def test_sse_stream_emits_only_expected_event_types(client):
    job_id = "ingest_sse_expected_events"
    seed_ingestion_job(
        job_id,
        "completed",
        100,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    event_names = {
        event.get("event")
        for event in events
    }

    allowed_event_names = {
        "connected",
        "progress",
        "completed",
        "failed",
        "cancelled",
        "partially_completed",
        "error",
    }

    assert event_names.issubset(
        allowed_event_names
    )


def test_sse_stream_does_not_emit_duplicate_connected_events(client):
    job_id = "ingest_sse_single_connected"
    seed_ingestion_job(
        job_id,
        "completed",
        100,
    )

    with client.stream(
        "GET",
        f"/api/v1/documents/ingest/jobs/{job_id}/stream",
    ) as response:
        events = parse_sse_events(
            response.text
        )

    connected_events = [
        event
        for event in events
        if event.get("event") == "connected"
    ]

    assert len(connected_events) == 1
