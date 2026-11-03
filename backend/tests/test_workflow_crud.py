"""
Comprehensive tests for workflow CRUD normalization and validation helpers.

These tests cover workflow-ID parsing, status and workflow-step validation,
UUID canonicalization, enum normalization, whitespace handling, and
missing-workflow lookup behavior.
"""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from app.core.enums import JobStatus, WorkflowStep
from app.core.exceptions import (
    ValidationError,
    WorkflowNotFoundError,
)
from app.crud.workflow_crud import (
    _parse_workflow_id,
    _validate_status,
    _validate_workflow_step,
    get_workflow_execution_or_404,
)


# ---------------------------------------------------------------------------
# Workflow-ID parsing
# ---------------------------------------------------------------------------


def test_parse_valid_workflow_id():
    workflow_id = "123e4567-e89b-12d3-a456-426614174000"

    assert _parse_workflow_id(
        workflow_id
    )


def test_parse_workflow_id_returns_uuid():
    workflow_id = uuid4()

    parsed = _parse_workflow_id(
        str(workflow_id)
    )

    assert isinstance(parsed, UUID)
    assert parsed == workflow_id


def test_parse_workflow_id_accepts_uuid_object():
    workflow_id = uuid4()

    assert (
        _parse_workflow_id(workflow_id)
        == workflow_id
    )


def test_parse_workflow_id_strips_whitespace():
    workflow_id = uuid4()

    parsed = _parse_workflow_id(
        f"  {workflow_id}  "
    )

    assert parsed == workflow_id


def test_parse_workflow_id_normalizes_case():
    workflow_id = uuid4()

    parsed = _parse_workflow_id(
        str(workflow_id).upper()
    )

    assert parsed == workflow_id


@pytest.mark.parametrize(
    "invalid_workflow_id",
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
def test_parse_invalid_workflow_id(
    invalid_workflow_id,
):
    with pytest.raises(ValidationError):
        _parse_workflow_id(
            invalid_workflow_id
        )


# ---------------------------------------------------------------------------
# Status validation
# ---------------------------------------------------------------------------


def test_validate_status():
    assert (
        _validate_status("completed")
        == JobStatus.COMPLETED
    )
    assert (
        _validate_status(None) is None
    )


@pytest.mark.parametrize(
    "status,expected",
    [
        ("queued", JobStatus.QUEUED),
        ("processing", JobStatus.PROCESSING),
        ("completed", JobStatus.COMPLETED),
        ("failed", JobStatus.FAILED),
        ("cancelled", JobStatus.CANCELLED),
        ("QUEUED", JobStatus.QUEUED),
        ("Processing", JobStatus.PROCESSING),
        ("COMPLETED", JobStatus.COMPLETED),
        ("  failed  ", JobStatus.FAILED),
    ],
)
def test_validate_status_accepts_supported_statuses(
    status: str,
    expected: JobStatus,
):
    assert (
        _validate_status(status)
        == expected
    )


@pytest.mark.parametrize(
    "invalid_status",
    [
        "",
        "   ",
        "unknown-status",
        "pending",
        "running",
        "success",
        "error",
        "in_progress",
        "123",
        [],
        {},
    ],
)
def test_invalid_status(invalid_status):
    with pytest.raises(ValidationError):
        _validate_status(
            invalid_status
        )


def test_validate_status_is_case_insensitive():
    assert (
        _validate_status("CoMpLeTeD")
        == JobStatus.COMPLETED
    )


def test_validate_status_none_returns_none():
    assert (
        _validate_status(None) is None
    )


# ---------------------------------------------------------------------------
# Workflow-step validation
# ---------------------------------------------------------------------------


def test_validate_workflow_step():
    assert (
        _validate_workflow_step("design")
        == WorkflowStep.DESIGN
    )
    assert (
        _validate_workflow_step(None)
        is None
    )


@pytest.mark.parametrize(
    "workflow_step,expected",
    [
        ("initialized", WorkflowStep.INITIALIZED),
        ("ingestion", WorkflowStep.INGESTION),
        ("requirements", WorkflowStep.REQUIREMENTS),
        ("design", WorkflowStep.DESIGN),
        ("validation", WorkflowStep.VALIDATION),
        ("review", WorkflowStep.REVIEW),
        ("INITIALIZED", WorkflowStep.INITIALIZED),
        ("Ingestion", WorkflowStep.INGESTION),
        ("DESIGN", WorkflowStep.DESIGN),
        ("  validation  ", WorkflowStep.VALIDATION),
    ],
)
def test_validate_workflow_step_accepts_supported_steps(
    workflow_step: str,
    expected: WorkflowStep,
):
    assert (
        _validate_workflow_step(
            workflow_step
        )
        == expected
    )


@pytest.mark.parametrize(
    "invalid_workflow_step",
    [
        "",
        "   ",
        "unknown-step",
        "start",
        "finish",
        "complete",
        "deployment",
        "123",
        [],
        {},
    ],
)
def test_invalid_workflow_step(
    invalid_workflow_step,
):
    with pytest.raises(ValidationError):
        _validate_workflow_step(
            invalid_workflow_step
        )


def test_validate_workflow_step_is_case_insensitive():
    assert (
        _validate_workflow_step(
            "DeSiGn"
        )
        == WorkflowStep.DESIGN
    )


def test_validate_workflow_step_none_returns_none():
    assert (
        _validate_workflow_step(None)
        is None
    )


# ---------------------------------------------------------------------------
# Missing workflow lookup
# ---------------------------------------------------------------------------


def test_missing_workflow_raises_not_found():
    class FakeResult:
        def scalar_one_or_none(self):
            return None

    class FakeSession:
        async def execute(
            self,
            *_args,
            **_kwargs,
        ):
            return FakeResult()

    async def run():
        await get_workflow_execution_or_404(
            FakeSession(),
            "123e4567-e89b-12d3-a456-426614174000",
        )

    with pytest.raises(
        WorkflowNotFoundError
    ):
        asyncio.run(run())


async def test_missing_workflow_error_contains_workflow_id():
    class FakeResult:
        def scalar_one_or_none(self):
            return None

    class FakeSession:
        async def execute(
            self,
            *_args,
            **_kwargs,
        ):
            return FakeResult()

    workflow_id = str(uuid4())

    with pytest.raises(
        WorkflowNotFoundError
    ) as exc_info:
        await get_workflow_execution_or_404(
            FakeSession(),
            workflow_id,
        )

    assert workflow_id in str(
        exc_info.value
    )


async def test_missing_workflow_lookup_rejects_invalid_id():
    class FakeSession:
        async def execute(
            self,
            *_args,
            **_kwargs,
        ):
            raise AssertionError(
                "Database should not be queried for an invalid workflow ID."
            )

    with pytest.raises(ValidationError):
        await get_workflow_execution_or_404(
            FakeSession(),
            "not-a-uuid",
        )


# ---------------------------------------------------------------------------
# Cross-field consistency
# ---------------------------------------------------------------------------


def test_status_and_step_validation_are_independent():
    status = _validate_status(
        "completed"
    )
    workflow_step = (
        _validate_workflow_step(
            "design"
        )
    )

    assert status == JobStatus.COMPLETED
    assert (
        workflow_step
        == WorkflowStep.DESIGN
    )


def test_none_status_and_step_are_allowed():
    assert (
        _validate_status(None) is None
    )
    assert (
        _validate_workflow_step(None)
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
