"""
Comprehensive tests for the ingestion workflow node.

These tests cover brief normalization, document ID normalization, ingestion
success, validation failures, recovery metadata, execution logging, tenant and
user identity preservation, and security boundaries.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.core.enums import JobStatus, WorkflowStep
from app.core.exceptions import WorkflowStateError
from app.graph.nodes.ingestion_node import (
    _normalize_brief_text,
    _normalize_document_ids,
    ingestion_node,
)
from app.graph.state import create_initial_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state() -> dict:
    """Return a valid initial workflow state for ingestion tests."""
    return create_initial_state(
        execution_id="wf_123",
        brief_id="brief_456",
        raw_brief_text=(
            "  Build   an internal analytics dashboard.  "
        ),
        tenant_id="tenant_789",
        user_id="user_001",
    )


@pytest.fixture
def state_with_documents(state: dict) -> dict:
    """Return workflow state containing duplicate document IDs."""
    return {
        **state,
        "document_ids": [
            "doc_1",
            "doc_1",
            "doc_2",
        ],
    }


@pytest.fixture
def mock_document_service():
    """Patch DocumentService used by the ingestion node."""
    with patch(
        "app.services.document_service.DocumentService"
    ) as mock_service_class:
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        yield mock_service


# ---------------------------------------------------------------------------
# Brief normalization
# ---------------------------------------------------------------------------


def test_normalize_brief_text():
    assert (
        _normalize_brief_text(
            "  Build   an app.  "
        )
        == "Build an app."
    )


def test_normalize_brief_text_preserves_single_spaces():
    assert (
        _normalize_brief_text(
            "Build a secure dashboard."
        )
        == "Build a secure dashboard."
    )


def test_normalize_brief_text_handles_newlines():
    assert (
        _normalize_brief_text(
            "Build\n\nan\napp."
        )
        == "Build an app."
    )


def test_normalize_brief_text_handles_tabs():
    assert (
        _normalize_brief_text(
            "\tBuild\tan\tapp.\t"
        )
        == "Build an app."
    )


def test_empty_brief_is_rejected():
    with pytest.raises(WorkflowStateError):
        _normalize_brief_text("   ")

    with pytest.raises(WorkflowStateError):
        _normalize_brief_text(None)


def test_empty_string_brief_is_rejected():
    with pytest.raises(WorkflowStateError):
        _normalize_brief_text("")


def test_non_string_brief_is_rejected():
    with pytest.raises(WorkflowStateError):
        _normalize_brief_text(12345)

    with pytest.raises(WorkflowStateError):
        _normalize_brief_text(["Build an app"])


def test_oversized_brief_is_rejected():
    with pytest.raises(WorkflowStateError):
        _normalize_brief_text(
            "x" * 2_000_000
        )


def test_brief_at_maximum_length_is_accepted():
    maximum_length = 20_000

    normalized_brief = _normalize_brief_text(
        "x" * maximum_length
    )

    assert len(normalized_brief) == maximum_length


# ---------------------------------------------------------------------------
# Document ID normalization
# ---------------------------------------------------------------------------


def test_document_ids_are_deduplicated():
    result = _normalize_document_ids(
        [
            "doc_1",
            " doc_1 ",
            "doc_2",
            "",
        ]
    )

    assert result == ["doc_1", "doc_2"]


def test_document_ids_preserve_order():
    result = _normalize_document_ids(
        [
            "doc_3",
            "doc_1",
            "doc_2",
        ]
    )

    assert result == [
        "doc_3",
        "doc_1",
        "doc_2",
    ]


def test_document_ids_accept_none():
    assert (
        _normalize_document_ids(None)
        == []
    )


def test_document_ids_accept_empty_list():
    assert (
        _normalize_document_ids([])
        == []
    )


def test_document_ids_reject_non_list_input():
    with pytest.raises(WorkflowStateError):
        _normalize_document_ids("doc_1")

    with pytest.raises(WorkflowStateError):
        _normalize_document_ids(123)

    with pytest.raises(WorkflowStateError):
        _normalize_document_ids(
            {"document_id": "doc_1"}
        )


def test_document_ids_reject_non_string_items():
    with pytest.raises(WorkflowStateError):
        _normalize_document_ids(
            ["doc_1", 123]
        )


def test_document_ids_reject_blank_items():
    with pytest.raises(WorkflowStateError):
        _normalize_document_ids(
            ["doc_1", "   "]
        )


def test_too_many_documents_are_rejected():
    with pytest.raises(WorkflowStateError):
        _normalize_document_ids(
            [
                f"doc_{index}"
                for index in range(101)
            ]
        )


def test_maximum_document_count_is_accepted():
    document_ids = [
        f"doc_{index}"
        for index in range(100)
    ]

    normalized_document_ids = (
        _normalize_document_ids(document_ids)
    )

    assert len(normalized_document_ids) == 100


def test_document_ids_do_not_leak_raw_whitespace():
    result = _normalize_document_ids(
        [
            "  doc_1  ",
            "doc_2",
        ]
    )

    assert result == ["doc_1", "doc_2"]


# ---------------------------------------------------------------------------
# Successful ingestion
# ---------------------------------------------------------------------------


async def test_ingestion_node_completes(
    state_with_documents: dict,
):
    result = await ingestion_node(
        state_with_documents
    )

    assert (
        result["current_step"]
        == WorkflowStep.INGESTION
    )
    assert (
        result["ingestion_status"]
        == JobStatus.COMPLETED
    )
    assert result["document_ids"] == [
        "doc_1",
        "doc_2",
    ]
    assert result["current_error"] is None
    assert result["execution_logs"]


async def test_ingestion_node_completes_without_documents(
    state: dict,
):
    result = await ingestion_node(state)

    assert (
        result["current_step"]
        == WorkflowStep.INGESTION
    )
    assert (
        result["ingestion_status"]
        == JobStatus.COMPLETED
    )
    assert result["document_ids"] == []
    assert result["current_error"] is None


async def test_ingestion_node_preserves_execution_identity(
    state_with_documents: dict,
):
    result = await ingestion_node(
        state_with_documents
    )

    assert result["execution_id"] == "wf_123"
    assert result["brief_id"] == "brief_456"
    assert result["tenant_id"] == "tenant_789"
    assert result["user_id"] == "user_001"


async def test_ingestion_node_normalizes_brief(
    state: dict,
):
    result = await ingestion_node(state)

    assert (
        result["normalized_brief_text"]
        == "Build an internal analytics dashboard."
    )


async def test_ingestion_node_normalizes_document_ids(
    state_with_documents: dict,
):
    result = await ingestion_node(
        state_with_documents
    )

    assert result["document_ids"] == [
        "doc_1",
        "doc_2",
    ]


async def test_ingestion_node_records_execution_logs(
    state_with_documents: dict,
):
    result = await ingestion_node(
        state_with_documents
    )

    execution_logs = result["execution_logs"]

    assert isinstance(execution_logs, list)
    assert execution_logs


async def test_ingestion_node_does_not_expose_raw_brief(
    state: dict,
):
    result = await ingestion_node(state)

    serialized_result = str(result)

    assert (
        "Build   an internal analytics dashboard."
        not in serialized_result
    )


async def test_ingestion_node_does_not_expose_document_contents(
    state_with_documents: dict,
    mock_document_service: AsyncMock,
):
    mock_document_service.get_document_content.return_value = {
        "content": "Secret document contents",
    }

    result = await ingestion_node(
        state_with_documents
    )

    serialized_result = str(result)

    assert "Secret document contents" not in serialized_result


# ---------------------------------------------------------------------------
# Ingestion validation failures
# ---------------------------------------------------------------------------


async def test_ingestion_node_fails_on_empty_brief():
    state = create_initial_state(
        execution_id="wf_123",
        brief_id="brief_456",
        raw_brief_text="   ",
    )

    result = await ingestion_node(state)

    assert (
        result["ingestion_status"]
        == JobStatus.FAILED
    )
    assert result["current_error"]
    assert result["recoverable_error"] is False


async def test_ingestion_node_fails_on_missing_brief():
    state = create_initial_state(
        execution_id="wf_123",
        brief_id="brief_456",
        raw_brief_text="Build a dashboard.",
    )

    state["raw_brief_text"] = None

    result = await ingestion_node(state)

    assert (
        result["ingestion_status"]
        == JobStatus.FAILED
    )
    assert result["current_error"]
    assert result["recoverable_error"] is False


async def test_ingestion_node_fails_on_invalid_document_ids(
    state: dict,
):
    invalid_state = {
        **state,
        "document_ids": "not-a-list",
    }

    result = await ingestion_node(invalid_state)

    assert (
        result["ingestion_status"]
        == JobStatus.FAILED
    )
    assert result["current_error"]
    assert result["recoverable_error"] is False


async def test_ingestion_node_fails_on_too_many_documents(
    state: dict,
):
    invalid_state = {
        **state,
        "document_ids": [
            f"doc_{index}"
            for index in range(101)
        ],
    }

    result = await ingestion_node(invalid_state)

    assert (
        result["ingestion_status"]
        == JobStatus.FAILED
    )
    assert result["current_error"]
    assert result["recoverable_error"] is False


async def test_ingestion_failure_does_not_leak_sensitive_details(
    state: dict,
):
    invalid_state = {
        **state,
        "document_ids": [
            "doc_1",
            "   ",
        ],
    }

    result = await ingestion_node(invalid_state)

    serialized_result = str(result).lower()

    assert "api_key" not in serialized_result
    assert "secret" not in serialized_result
    assert "password" not in serialized_result
    assert "traceback" not in serialized_result


# ---------------------------------------------------------------------------
# Recovery metadata
# ---------------------------------------------------------------------------


async def test_successful_ingestion_has_no_recovery_error(
    state_with_documents: dict,
):
    result = await ingestion_node(
        state_with_documents
    )

    assert result["current_error"] is None
    assert result["recoverable_error"] is False
    assert result["error_code"] is None


async def test_failed_ingestion_sets_error_metadata(
    state: dict,
):
    invalid_state = {
        **state,
        "raw_brief_text": None,
    }

    result = await ingestion_node(invalid_state)

    assert result["current_error"]
    assert result["error_code"]
    assert result["error_step"] == WorkflowStep.INGESTION
    assert result["recoverable_error"] is False


async def test_failed_ingestion_records_recovery_history(
    state: dict,
):
    invalid_state = {
        **state,
        "raw_brief_text": None,
    }

    result = await ingestion_node(invalid_state)

    recovery_history = result.get(
        "recovery_history",
        [],
    )

    assert isinstance(recovery_history, list)


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


async def test_ingestion_node_sets_ingestion_step(
    state_with_documents: dict,
):
    result = await ingestion_node(
        state_with_documents
    )

    assert (
        result["current_step"]
        == WorkflowStep.INGESTION
    )


async def test_successful_ingestion_sets_completed_status(
    state_with_documents: dict,
):
    result = await ingestion_node(
        state_with_documents
    )

    assert (
        result["ingestion_status"]
        == JobStatus.COMPLETED
    )


async def test_failed_ingestion_sets_failed_status(
    state: dict,
):
    invalid_state = {
        **state,
        "raw_brief_text": None,
    }

    result = await ingestion_node(invalid_state)

    assert (
        result["ingestion_status"]
        == JobStatus.FAILED
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
