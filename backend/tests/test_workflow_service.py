"""
Comprehensive tests for the workflow service.

These tests cover synchronous execution, asynchronous job creation, execution
status retrieval, streaming metadata, tenant isolation, payload validation,
repository persistence, cancellation, and missing-workflow behavior.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.core.enums import JobStatus, WorkflowStep
from app.core.exceptions import ValidationError
from app.services.workflow_service import (
    InMemoryWorkflowRepository,
    WorkflowNotFoundError,
    WorkflowService,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeWorkflow:
    async def ainvoke(self, state):
        return {
            **state,
            "current_step": "completed",
            "extracted_requirements": [
                "User can reset password",
            ],
        }


class FailingWorkflow:
    async def ainvoke(self, state):
        raise RuntimeError("Workflow execution failed")


class SlowWorkflow:
    async def ainvoke(self, state):
        await asyncio.sleep(10)

        return {
            **state,
            "current_step": "completed",
        }


class RecordingWorkflow:
    def __init__(self):
        self.invoked_states = []

    async def ainvoke(self, state):
        self.invoked_states.append(state)

        return {
            **state,
            "current_step": "completed",
            "extracted_requirements": [
                "User can reset password",
            ],
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def repository():
    return InMemoryWorkflowRepository()


@pytest.fixture
def service(repository):
    return WorkflowService(
        workflow=FakeWorkflow(),
        repository=repository,
    )


@pytest.fixture
def valid_payload():
    return {
        "brief_id": f"brief_{uuid4()}",
        "raw_brief_text": (
            "Build a secure password reset experience."
        ),
        "tenant_id": str(uuid4()),
        "user_id": str(uuid4()),
    }


# ---------------------------------------------------------------------------
# Synchronous execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_sync_returns_final_state():
    service = WorkflowService(
        workflow=FakeWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    payload = {
        "brief_id": "brief_123",
        "raw_brief_text": (
            "Build a secure password reset experience."
        ),
    }

    result = await service.execute_sync(
        payload
    )

    assert (
        result["brief_id"] == "brief_123"
    )
    assert (
        result["current_step"]
        == "completed"
    )
    assert result[
        "extracted_requirements"
    ] == ["User can reset password"]


@pytest.mark.asyncio
async def test_execute_sync_generates_execution_id(
    valid_payload,
):
    service = WorkflowService(
        workflow=FakeWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    result = await service.execute_sync(
        valid_payload
    )

    assert result["execution_id"]
    assert result["execution_id"].startswith(
        "wf_"
    )


@pytest.mark.asyncio
async def test_execute_sync_preserves_brief_text(
    valid_payload,
):
    service = WorkflowService(
        workflow=FakeWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    result = await service.execute_sync(
        valid_payload
    )

    assert (
        result["raw_brief_text"]
        == valid_payload["raw_brief_text"]
    )


@pytest.mark.asyncio
async def test_execute_sync_preserves_tenant_context(
    valid_payload,
):
    service = WorkflowService(
        workflow=FakeWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    result = await service.execute_sync(
        valid_payload
    )

    assert (
        result["tenant_id"]
        == valid_payload["tenant_id"]
    )
    assert (
        result["user_id"]
        == valid_payload["user_id"]
    )


@pytest.mark.asyncio
async def test_execute_sync_rejects_missing_brief_id():
    service = WorkflowService(
        workflow=FakeWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    with pytest.raises(ValidationError):
        await service.execute_sync(
            {
                "raw_brief_text": (
                    "Build a dashboard."
                )
            }
        )


@pytest.mark.asyncio
async def test_execute_sync_rejects_empty_brief_id():
    service = WorkflowService(
        workflow=FakeWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    with pytest.raises(ValidationError):
        await service.execute_sync(
            {
                "brief_id": "",
                "raw_brief_text": (
                    "Build a dashboard."
                ),
            }
        )


@pytest.mark.asyncio
async def test_execute_sync_rejects_missing_brief_text():
    service = WorkflowService(
        workflow=FakeWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    with pytest.raises(ValidationError):
        await service.execute_sync(
            {"brief_id": "brief_123"}
        )


@pytest.mark.asyncio
async def test_execute_sync_rejects_empty_brief_text():
    service = WorkflowService(
        workflow=FakeWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    with pytest.raises(ValidationError):
        await service.execute_sync(
            {
                "brief_id": "brief_123",
                "raw_brief_text": "",
            }
        )


@pytest.mark.asyncio
async def test_execute_sync_rejects_whitespace_brief_text():
    service = WorkflowService(
        workflow=FakeWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    with pytest.raises(ValidationError):
        await service.execute_sync(
            {
                "brief_id": "brief_123",
                "raw_brief_text": "   ",
            }
        )


@pytest.mark.asyncio
async def test_execute_sync_does_not_mutate_payload(
    valid_payload,
):
    service = WorkflowService(
        workflow=FakeWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    original_payload = {
        **valid_payload
    }

    await service.execute_sync(
        valid_payload
    )

    assert valid_payload == original_payload


# ---------------------------------------------------------------------------
# Asynchronous execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_async_creates_queued_job():
    service = WorkflowService(
        workflow=FakeWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    payload = {
        "brief_id": "brief_456",
        "raw_brief_text": (
            "Create an enterprise onboarding workflow."
        ),
    }

    result = await service.start_async(
        payload
    )

    assert result["status"] == "queued"
    assert (
        result["brief_id"] == "brief_456"
    )
    assert (
        "/status/" in result["poll_endpoint"]
    )
    assert (
        "/stream/" in result["stream_endpoint"]
    )


@pytest.mark.asyncio
async def test_start_async_returns_execution_id(
    valid_payload,
):
    service = WorkflowService(
        workflow=FakeWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    result = await service.start_async(
        valid_payload
    )

    assert result["execution_id"]
    assert result["execution_id"].startswith(
        "wf_"
    )


@pytest.mark.asyncio
async def test_start_async_returns_unique_execution_ids(
    valid_payload,
):
    service = WorkflowService(
        workflow=FakeWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    first_result = (
        await service.start_async(
            valid_payload
        )
    )
    second_result = (
        await service.start_async(
            valid_payload
        )
    )

    assert (
        first_result["execution_id"]
        != second_result["execution_id"]
    )


@pytest.mark.asyncio
async def test_start_async_persists_queued_execution(
    service,
    valid_payload,
):
    result = await service.start_async(
        valid_payload
    )

    execution = await service.get_execution(
        result["execution_id"]
    )

    assert execution["execution_id"] == (
        result["execution_id"]
    )
    assert (
        execution["workflow_status"]
        == JobStatus.QUEUED
    )
    assert (
        execution["brief_id"]
        == valid_payload["brief_id"]
    )


@pytest.mark.asyncio
async def test_start_async_returns_status_endpoint_with_execution_id(
    service,
    valid_payload,
):
    result = await service.start_async(
        valid_payload
    )

    assert (
        result["execution_id"]
        in result["poll_endpoint"]
    )


@pytest.mark.asyncio
async def test_start_async_returns_stream_endpoint_with_execution_id(
    service,
    valid_payload,
):
    result = await service.start_async(
        valid_payload
    )

    assert (
        result["execution_id"]
        in result["stream_endpoint"]
    )


@pytest.mark.asyncio
async def test_start_async_rejects_missing_brief_id(
    service,
):
    with pytest.raises(ValidationError):
        await service.start_async(
            {
                "raw_brief_text": (
                    "Build a dashboard."
                )
            }
        )


@pytest.mark.asyncio
async def test_start_async_rejects_missing_brief_text(
    service,
):
    with pytest.raises(ValidationError):
        await service.start_async(
            {"brief_id": "brief_123"}
        )


@pytest.mark.asyncio
async def test_start_async_rejects_empty_brief_text(
    service,
):
    with pytest.raises(ValidationError):
        await service.start_async(
            {
                "brief_id": "brief_123",
                "raw_brief_text": "",
            }
        )


# ---------------------------------------------------------------------------
# Execution retrieval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_execution_raises_for_unknown_id():
    service = WorkflowService(
        workflow=FakeWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    with pytest.raises(
        WorkflowNotFoundError
    ):
        await service.get_execution(
            "does-not-exist"
        )


@pytest.mark.asyncio
async def test_get_execution_returns_started_execution(
    service,
    valid_payload,
):
    started = await service.start_async(
        valid_payload
    )

    execution = await service.get_execution(
        started["execution_id"]
    )

    assert (
        execution["execution_id"]
        == started["execution_id"]
    )
    assert (
        execution["workflow_status"]
        == JobStatus.QUEUED
    )


@pytest.mark.asyncio
async def test_get_execution_rejects_invalid_id(
    service,
):
    with pytest.raises(ValidationError):
        await service.get_execution(
            "not-a-uuid"
        )


@pytest.mark.asyncio
async def test_get_execution_rejects_empty_id(
    service,
):
    with pytest.raises(ValidationError):
        await service.get_execution("")


@pytest.mark.asyncio
async def test_get_execution_rejects_none_id(
    service,
):
    with pytest.raises(ValidationError):
        await service.get_execution(None)


# ---------------------------------------------------------------------------
# Repository behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_memory_repository_saves_execution(
    repository,
    valid_payload,
):
    execution_id = f"wf_{uuid4()}"

    execution = {
        "execution_id": execution_id,
        "brief_id": valid_payload["brief_id"],
        "workflow_status": JobStatus.QUEUED,
    }

    await repository.save(execution)

    stored = await repository.get(
        execution_id
    )

    assert stored == execution


@pytest.mark.asyncio
async def test_in_memory_repository_updates_execution(
    repository,
    valid_payload,
):
    execution_id = f"wf_{uuid4()}"

    execution = {
        "execution_id": execution_id,
        "brief_id": valid_payload["brief_id"],
        "workflow_status": JobStatus.QUEUED,
    }

    await repository.save(execution)

    updated_execution = {
        **execution,
        "workflow_status": JobStatus.PROCESSING,
    }

    await repository.save(updated_execution)

    stored = await repository.get(
        execution_id
    )

    assert (
        stored["workflow_status"]
        == JobStatus.PROCESSING
    )


@pytest.mark.asyncio
async def test_in_memory_repository_raises_for_missing_execution(
    repository,
):
    with pytest.raises(
        WorkflowNotFoundError
    ):
        await repository.get(
            f"wf_{uuid4()}"
        )


@pytest.mark.asyncio
async def test_in_memory_repository_isolates_executions(
    repository,
    valid_payload,
):
    first_id = f"wf_{uuid4()}"
    second_id = f"wf_{uuid4()}"

    first_execution = {
        "execution_id": first_id,
        "brief_id": valid_payload["brief_id"],
        "workflow_status": JobStatus.QUEUED,
    }

    second_execution = {
        "execution_id": second_id,
        "brief_id": valid_payload["brief_id"],
        "workflow_status": JobStatus.QUEUED,
    }

    await repository.save(first_execution)
    await repository.save(second_execution)

    first_stored = await repository.get(
        first_id
    )
    second_stored = await repository.get(
        second_id
    )

    assert (
        first_stored["execution_id"]
        == first_id
    )
    assert (
        second_stored["execution_id"]
        == second_id
    )


# ---------------------------------------------------------------------------
# Workflow invocation behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_invokes_workflow_with_initialized_state(
    valid_payload,
):
    workflow = RecordingWorkflow()

    service = WorkflowService(
        workflow=workflow,
        repository=InMemoryWorkflowRepository(),
    )

    await service.execute_sync(
        valid_payload
    )

    assert len(
        workflow.invoked_states
    ) == 1
    assert (
        workflow.invoked_states[0][
            "brief_id"
        ]
        == valid_payload["brief_id"]
    )
    assert (
        workflow.invoked_states[0][
            "raw_brief_text"
        ]
        == valid_payload["raw_brief_text"]
    )


@pytest.mark.asyncio
async def test_service_does_not_expose_raw_brief_in_result_metadata(
    service,
    valid_payload,
):
    started = await service.start_async(
        valid_payload
    )

    assert "raw_brief_text" not in started


@pytest.mark.asyncio
async def test_service_failure_raises_workflow_error(
    valid_payload,
):
    service = WorkflowService(
        workflow=FailingWorkflow(),
        repository=InMemoryWorkflowRepository(),
    )

    with pytest.raises(RuntimeError):
        await service.execute_sync(
            valid_payload
        )


@pytest.mark.asyncio
async def test_service_failure_does_not_leave_queued_execution(
    valid_payload,
):
    repository = InMemoryWorkflowRepository()

    service = WorkflowService(
        workflow=FailingWorkflow(),
        repository=repository,
    )

    with pytest.raises(RuntimeError):
        await service.execute_sync(
            valid_payload
        )

    assert repository._executions == {}


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_execution_marks_cancelled(
    service,
    valid_payload,
):
    started = await service.start_async(
        valid_payload
    )

    result = await service.cancel_execution(
        started["execution_id"]
    )

    assert (
        result["workflow_status"]
        == JobStatus.CANCELLED
    )


@pytest.mark.asyncio
async def test_cancel_unknown_execution_raises_not_found(
    service,
):
    with pytest.raises(
        WorkflowNotFoundError
    ):
        await service.cancel_execution(
            f"wf_{uuid4()}"
        )


@pytest.mark.asyncio
async def test_cancel_invalid_execution_raises_validation_error(
    service,
):
    with pytest.raises(ValidationError):
        await service.cancel_execution(
            "not-a-uuid"
        )


@pytest.mark.asyncio
async def test_cancelled_execution_is_persisted(
    service,
    valid_payload,
):
    started = await service.start_async(
        valid_payload
    )

    await service.cancel_execution(
        started["execution_id"]
    )

    execution = await service.get_execution(
        started["execution_id"]
    )

    assert (
        execution["workflow_status"]
        == JobStatus.CANCELLED
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
