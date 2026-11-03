from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from app.core.auth import get_current_user
from app.core.database import get_db
from app.main import app
from app.schemas.ingestion import (
    IngestionJobCreatedResponse,
    IngestionJobResponse,
    IngestionJobStatus,
)
from app.services.document_service import batch_ingestion_service, document_service


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_current_user():
    class FakeUser:
        id = "user_123"
        subject = "user_123"
        email = "tester@example.com"
        roles = ["editor"]
        scopes = [
            "documents:read",
            "documents:write",
        ]

        def has_role(self, *roles):
            return any(role in self.roles for role in roles)

        def has_scope(self, *scopes):
            return any(scope in self.scopes for scope in scopes)

        def to_dict(self):
            return {
                "id": self.id,
                "email": self.email,
                "roles": self.roles,
                "scopes": self.scopes,
            }

    return FakeUser()


@pytest.fixture
def client(mock_current_user):
    """
    Router integration client.

    Overrides authentication and provides a no-op database dependency.
    Service methods are mocked in individual tests, so no real database
    or external provider is contacted.
    """
    async def override_current_user():
        return mock_current_user

    async def override_get_db():
        yield AsyncMock()

    app.dependency_overrides[get_current_user] = override_current_user
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def valid_batch_payload():
    return {
        "documents": [
            {
                "title": "Password Reset Specification",
                "content": "Users must be able to securely reset their password.",
                "metadata": {
                    "source_type": "manual",
                    "tenant_id": "tenant_001",
                    "tags": ["requirements"],
                },
            },
            {
                "title": "Design System Guidelines",
                "content": "Use the primary brand color consistently across components.",
                "metadata": {
                    "source_type": "figma",
                    "source_url": "https://www.figma.com/file/example",
                    "tags": ["design-system"],
                },
            },
        ],
        "batch_size": 2,
        "max_concurrent_batches": 1,
        "continue_on_error": True,
    }


# ---------------------------------------------------------------------------
# Single document ingestion
# ---------------------------------------------------------------------------


def test_ingest_document_returns_created_document(client, monkeypatch):
    created_document = {
        "id": 42,
        "title": "Password Reset Specification",
        "content": "Users must be able to securely reset their password.",
        "created_at": datetime.now(timezone.utc),
        "updated_at": None,
    }

    mock_create = AsyncMock(return_value=created_document)
    monkeypatch.setattr(
        document_service,
        "create_document",
        mock_create,
    )

    response = client.post(
        "/api/v1/documents",
        json={
            "title": "Password Reset Specification",
            "content": "Users must be able to securely reset their password.",
        },
    )

    assert response.status_code == 201
    body = response.json()

    assert body["id"] == 42
    assert body["title"] == "Password Reset Specification"
    mock_create.assert_awaited_once()


def test_ingest_document_returns_422_for_empty_content(client):
    response = client.post(
        "/api/v1/documents",
        json={
            "title": "Invalid document",
            "content": "",
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_ingest_document_maps_embedding_failure_to_502(
    client,
    monkeypatch,
):
    from app.services.document_service import EmbeddingError

    mock_create = AsyncMock(
        side_effect=EmbeddingError(
            "Unable to generate document embedding."
        )
    )

    monkeypatch.setattr(
        document_service,
        "create_document",
        mock_create,
    )

    response = client.post(
        "/api/v1/documents",
        json={
            "title": "Failing document",
            "content": "This document should fail during embedding.",
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Unable to generate document embedding."
    )


def test_ingest_document_requires_authentication(client, monkeypatch):
    app.dependency_overrides.pop(get_current_user, None)

    response = client.post(
        "/api/v1/documents",
        json={
            "title": "Unauthorized document",
            "content": "This request has no authenticated user.",
        },
    )

    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


def test_semantic_search_returns_response_envelope(client, monkeypatch):
    search_results = [
        {
            "id": 1,
            "title": "Password Reset Specification",
            "content": "Users must be able to securely reset their password.",
            "similarity_score": 0.93,
        },
        {
            "id": 2,
            "title": "Design System Guidelines",
            "content": "Use the primary brand color consistently.",
            "similarity_score": 0.81,
        },
    ]

    mock_search = AsyncMock(return_value=search_results)
    monkeypatch.setattr(
        document_service,
        "search_documents",
        mock_search,
    )

    response = client.post(
        "/api/v1/documents/search",
        json={
            "query": "How should users reset their password?",
            "limit": 2,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["query"] == "How should users reset their password?"
    assert body["total_results"] == 2
    assert body["results"][0]["id"] == 1
    assert body["results"][0]["similarity_score"] == 0.93

    mock_search.assert_awaited_once()


def test_semantic_search_rejects_blank_query(client):
    response = client.post(
        "/api/v1/documents/search",
        json={
            "query": "   ",
            "limit": 5,
        },
    )

    assert response.status_code == 422


def test_semantic_search_rejects_invalid_limit(client):
    response = client.post(
        "/api/v1/documents/search",
        json={
            "query": "password reset",
            "limit": 1000,
        },
    )

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Document listing and statistics
# ---------------------------------------------------------------------------


def test_list_documents_returns_paginated_documents(client, monkeypatch):
    documents = [
        {
            "id": 1,
            "title": "Document 1",
            "content": "First document",
            "created_at": datetime.now(timezone.utc),
            "updated_at": None,
        },
        {
            "id": 2,
            "title": "Document 2",
            "content": "Second document",
            "created_at": datetime.now(timezone.utc),
            "updated_at": None,
        },
    ]

    class FakeResult:
        def scalars(self):
            return self

        def all(self):
            return documents

    class FakeSession:
        async def execute(self, statement):
            return FakeResult()

    async def override_get_db():
        yield FakeSession()

    app.dependency_overrides[get_db] = override_get_db

    response = client.get(
        "/api/v1/documents",
        params={"skip": 0, "limit": 2},
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body) == 2
    assert body[0]["id"] == 1
    assert body[1]["id"] == 2


def test_list_documents_rejects_invalid_pagination(client):
    response = client.get(
        "/api/v1/documents",
        params={"skip": -1, "limit": 10},
    )

    assert response.status_code == 422

    response = client.get(
        "/api/v1/documents",
        params={"skip": 0, "limit": 0},
    )

    assert response.status_code == 422


def test_get_document_stats_returns_metadata(client, monkeypatch):
    class FakeResult:
        def scalar_one(self):
            return 128

    class FakeSession:
        async def execute(self, statement):
            return FakeResult()

    async def override_get_db():
        yield FakeSession()

    app.dependency_overrides[get_db] = override_get_db

    response = client.get("/api/v1/documents/stats")

    assert response.status_code == 200

    body = response.json()

    assert body["total_documents"] == 128
    assert body["vector_index"] == "pgvector"
    assert body["embedding_model"] == "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Batch ingestion
# ---------------------------------------------------------------------------


def test_start_batch_ingestion_returns_job_handle(
    client,
    valid_batch_payload,
    monkeypatch,
):
    job = {
        "job_id": "ingest_testjob123",
        "status": "queued",
        "total_documents": 2,
        "processed_documents": 0,
        "failed_documents": 0,
        "skipped_documents": 0,
        "current_batch": 0,
        "total_batches": 1,
        "progress_percent": 0,
        "error_message": None,
        "result": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }

    mock_create_job = AsyncMock(return_value=job)
    monkeypatch.setattr(
        batch_ingestion_service,
        "create_job",
        mock_create_job,
    )

    response = client.post(
        "/api/v1/documents/ingest/batch",
        json=valid_batch_payload,
    )

    assert response.status_code == 202

    body = response.json()

    assert body["job_id"] == "ingest_testjob123"
    assert body["status"] == IngestionJobStatus.QUEUED
    assert body["total_documents"] == 2
    assert body["total_batches"] == 1
    assert body["status_endpoint"] == (
        "/api/v1/documents/ingest/jobs/ingest_testjob123"
    )
    assert body["stream_endpoint"] == (
        "/api/v1/documents/ingest/jobs/ingest_testjob123/stream"
    )

    mock_create_job.assert_awaited_once()


def test_start_batch_ingestion_rejects_empty_batch(
    client,
    valid_batch_payload,
):
    valid_batch_payload["documents"] = []

    response = client.post(
        "/api/v1/documents/ingest/batch",
        json=valid_batch_payload,
    )

    assert response.status_code == 422


def test_start_batch_ingestion_rejects_invalid_batch_size(
    client,
    valid_batch_payload,
):
    valid_batch_payload["batch_size"] = 0

    response = client.post(
        "/api/v1/documents/ingest/batch",
        json=valid_batch_payload,
    )

    assert response.status_code == 422


def test_get_ingestion_job_returns_progress(
    client,
    monkeypatch,
):
    job = {
        "job_id": "ingest_progress123",
        "status": "processing",
        "total_documents": 10,
        "processed_documents": 4,
        "failed_documents": 1,
        "skipped_documents": 0,
        "current_batch": 2,
        "total_batches": 5,
        "progress_percent": 40,
        "error_message": None,
        "result": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }

    mock_get_job = AsyncMock(return_value=job)
    monkeypatch.setattr(
        batch_ingestion_service,
        "get_job",
        mock_get_job,
    )

    response = client.get(
        "/api/v1/documents/ingest/jobs/ingest_progress123",
    )

    assert response.status_code == 200

    body = response.json()

    assert body["job_id"] == "ingest_progress123"
    assert body["status"] == IngestionJobStatus.PROCESSING
    assert body["progress_percent"] == 40
    assert body["processed_documents"] == 4
    assert body["failed_documents"] == 1


def test_get_unknown_ingestion_job_returns_404(client, monkeypatch):
    mock_get_job = AsyncMock(side_effect=KeyError("Job not found."))
    monkeypatch.setattr(
        batch_ingestion_service,
        "get_job",
        mock_get_job,
    )

    response = client.get(
        "/api/v1/documents/ingest/jobs/ingest_unknown",
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

