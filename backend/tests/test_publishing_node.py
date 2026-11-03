"""
Tests for the publishing workflow node.

These tests cover approval gating, generated-file validation, dangerous-code
detection, auto-approval behavior, artifact preparation, and workflow-state
transitions.
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.enums import (
    ApprovalStatus,
    JobStatus,
    WorkflowStep,
)
from app.core.exceptions import WorkflowStateError
from app.graph.nodes.publishing_node import publishing_node
from app.graph.state import create_initial_state


@pytest.fixture
def state():
    """Return a valid initial workflow state."""
    return create_initial_state(
        execution_id="wf_123",
        brief_id="brief_456",
        raw_brief_text="Build a dashboard.",
        approval_required=True,
    )


@pytest.fixture
def state_with_safe_files(state):
    """Return workflow state containing safe generated files."""
    return {
        **state,
        "generated_code_files": {
            "app/main.py": "print('hello')",
        },
    }


@pytest.fixture
def state_without_approval(state_with_safe_files):
    """Return workflow state where approval is not required."""
    return {
        **state_with_safe_files,
        "approval_required": False,
    }


# ---------------------------------------------------------------------------
# Validation and state guards
# ---------------------------------------------------------------------------


def test_publishing_requires_generated_files(state):
    with pytest.raises(WorkflowStateError):
        asyncio.run(publishing_node(state))


async def test_publishing_rejects_empty_generated_files(state):
    state_with_empty_files = {
        **state,
        "generated_code_files": {},
    }

    with pytest.raises(WorkflowStateError):
        await publishing_node(state_with_empty_files)


async def test_publishing_rejects_none_generated_files(state):
    state_with_none_files = {
        **state,
        "generated_code_files": None,
    }

    with pytest.raises(WorkflowStateError):
        await publishing_node(state_with_none_files)


async def test_publishing_rejects_invalid_generated_files_type(state):
    state_with_invalid_files = {
        **state,
        "generated_code_files": ["not-a-dict"],
    }

    with pytest.raises(WorkflowStateError):
        await publishing_node(state_with_invalid_files)


# ---------------------------------------------------------------------------
# Approval-required publishing
# ---------------------------------------------------------------------------


async def test_publishing_prepares_approval(
    state_with_safe_files,
):
    result = await publishing_node(
        state_with_safe_files
    )

    assert result["review_status"] == "pending_approval"
    assert (
        result["approval_status"]
        == ApprovalStatus.PENDING
    )
    assert (
        result["publishing_status"]
        == JobStatus.PROCESSING
    )
    assert (
        result["current_step"]
        == WorkflowStep.PUBLISHING
    )
    assert result["artifacts"]
    assert not result["current_error"]
    assert not result["validation_issues"]


async def test_publishing_creates_artifact_metadata(
    state_with_safe_files,
):
    result = await publishing_node(
        state_with_safe_files
    )

    artifacts = result["artifacts"]

    assert isinstance(artifacts, list)
    assert len(artifacts) >= 1

    artifact = artifacts[0]

    assert artifact["path"] == "app/main.py"
    assert artifact["artifact_type"] == "code"
    assert artifact["status"] == "ready_for_review"


async def test_publishing_does_not_publish_when_approval_required(
    state_with_safe_files,
):
    result = await publishing_node(
        state_with_safe_files
    )

    assert result["publishing_status"] != JobStatus.COMPLETED
    assert result["approval_status"] == ApprovalStatus.PENDING


# ---------------------------------------------------------------------------
# Auto-approval publishing
# ---------------------------------------------------------------------------


async def test_publishing_auto_approves_when_not_required(
    state_without_approval,
):
    result = await publishing_node(
        state_without_approval
    )

    assert result["review_status"] == "approved"
    assert (
        result["approval_status"]
        == ApprovalStatus.APPROVED
    )
    assert (
        result["publishing_status"]
        == JobStatus.COMPLETED
    )
    assert (
        result["current_step"]
        == WorkflowStep.PUBLISHING
    )
    assert result["artifacts"]
    assert not result["current_error"]


async def test_auto_approval_publishes_generated_files(
    state_without_approval,
):
    result = await publishing_node(
        state_without_approval
    )

    artifacts = result["artifacts"]

    assert artifacts
    assert artifacts[0]["status"] == "published"


# ---------------------------------------------------------------------------
# Dangerous generated code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dangerous_code",),
    [
        ("eval('print(1)')",),
        ("exec('print(1)')",),
        ("__import__('os').system('ls')",),
        ("import subprocess",),
        ("open('/etc/passwd')",),
    ],
)
async def test_publishing_flags_dangerous_code(
    state,
    dangerous_code,
):
    state_with_dangerous_code = {
        **state,
        "generated_code_files": {
            "app/dangerous.py": dangerous_code,
        },
    }

    result = await publishing_node(
        state_with_dangerous_code
    )

    assert (
        result["publishing_status"]
        == JobStatus.FAILED
    )
    assert result["current_error"]
    assert result["validation_issues"]
    assert result["review_status"] != "approved"
    assert (
        result["approval_status"]
        != ApprovalStatus.APPROVED
    )


async def test_dangerous_code_blocks_auto_approval(state):
    state_with_dangerous_code = {
        **state,
        "approval_required": False,
        "generated_code_files": {
            "app/dangerous.py": "eval('print(1)')",
        },
    }

    result = await publishing_node(
        state_with_dangerous_code
    )

    assert (
        result["publishing_status"]
        == JobStatus.FAILED
    )
    assert result["current_error"]
    assert result["validation_issues"]


# ---------------------------------------------------------------------------
# Validation issues
# ---------------------------------------------------------------------------


async def test_publishing_records_validation_issues(
    state,
):
    state_with_dangerous_code = {
        **state,
        "generated_code_files": {
            "app/dangerous.py": "eval('print(1)')",
        },
    }

    result = await publishing_node(
        state_with_dangerous_code
    )

    validation_issues = result[
        "validation_issues"
    ]

    assert isinstance(validation_issues, list)
    assert validation_issues


async def test_safe_code_has_no_validation_issues(
    state_with_safe_files,
):
    result = await publishing_node(
        state_with_safe_files
    )

    assert not result["validation_issues"]
    assert not result["current_error"]


# ---------------------------------------------------------------------------
# Workflow state transitions
# ---------------------------------------------------------------------------


async def test_publishing_sets_current_step(
    state_with_safe_files,
):
    result = await publishing_node(
        state_with_safe_files
    )

    assert (
        result["current_step"]
        == WorkflowStep.PUBLISHING
    )


async def test_publishing_preserves_execution_identity(
    state_with_safe_files,
):
    result = await publishing_node(
        state_with_safe_files
    )

    assert result["execution_id"] == "wf_123"
    assert result["brief_id"] == "brief_456"


async def test_publishing_does_not_expose_generated_file_contents(
    state_with_safe_files,
):
    result = await publishing_node(
        state_with_safe_files
    )

    serialized_result = str(result)

    assert "print('hello')" not in serialized_result
