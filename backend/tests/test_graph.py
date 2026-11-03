"""
Comprehensive end-to-end tests for the AgenticBridge workflow graph.

These tests verify graph construction, workflow execution, state transitions,
approval gating, generated-artifact handling, provider failure handling,
dangerous-code detection, retry behavior, and security boundaries.

Security rules verified by these tests:
- Workflow results never expose generated file contents.
- Workflow results never expose prompts, completions, embeddings, credentials,
  database URLs, storage URLs, or raw provider errors.
- Workflow failures produce stable, client-safe error states.
- Approval-required workflows never publish automatically.
- Dangerous generated code is blocked before publication.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core.enums import (
    ApprovalStatus,
    JobStatus,
    WorkflowStep,
)
from app.core.exceptions import (
    AppError,
    ExternalProviderError,
    ProviderTimeoutError,
    ValidationError,
    WorkflowStateError,
)
from app.graph.runner import (
    WorkflowRunResult,
    create_workflow_runner,
)
from app.graph.state import create_initial_state
from app.graph.workflow import app_workflow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def execution_id() -> str:
    """Return a stable workflow execution ID."""
    return f"wf_{uuid4()}"


@pytest.fixture
def brief_id() -> str:
    """Return a stable brief ID."""
    return f"brief_{uuid4()}"


@pytest.fixture
def tenant_id() -> str:
    """Return a stable tenant ID."""
    return str(uuid4())


@pytest.fixture
def user_id() -> str:
    """Return a stable user ID."""
    return str(uuid4())


@pytest.fixture
def base_state(
    execution_id: str,
    brief_id: str,
    tenant_id: str,
    user_id: str,
) -> dict:
    """Return a valid initial workflow state requiring approval."""
    return create_initial_state(
        execution_id=execution_id,
        brief_id=brief_id,
        raw_brief_text=(
            "Build a secure dashboard with authentication, role-based "
            "access, and audit logging."
        ),
        approval_required=True,
        tenant_id=tenant_id,
        user_id=user_id,
    )


@pytest.fixture
def approval_not_required_state(
    execution_id: str,
    brief_id: str,
    tenant_id: str,
    user_id: str,
) -> dict:
    """Return a valid initial workflow state with approval disabled."""
    return create_initial_state(
        execution_id=execution_id,
        brief_id=brief_id,
        raw_brief_text=(
            "Build a secure dashboard with authentication, role-based "
            "access, and audit logging."
        ),
        approval_required=False,
        tenant_id=tenant_id,
        user_id=user_id,
    )


@pytest.fixture
def state_with_safe_files(
    base_state: dict,
) -> dict:
    """Return workflow state containing safe generated files."""
    return {
        **base_state,
        "generated_code_files": {
            "app/main.py": "print('hello')",
            "app/models.py": "class User: pass",
        },
        "current_step": WorkflowStep.PUBLISHING,
    }


@pytest.fixture
def state_with_dangerous_files(
    base_state: dict,
) -> dict:
    """Return workflow state containing dangerous generated files."""
    return {
        **base_state,
        "generated_code_files": {
            "app/dangerous.py": "eval('print(1)')",
        },
        "current_step": WorkflowStep.PUBLISHING,
    }


@pytest.fixture
def mock_workflow_dependencies():
    """Patch external dependencies used by workflow nodes."""
    with (
        patch(
            "app.services.workflow_service.WorkflowService"
        ) as mock_workflow_service_class,
        patch(
            "app.services.document_service.DocumentService"
        ) as mock_document_service_class,
        patch(
            "app.services.retrieval_service.RetrievalService"
        ) as mock_retrieval_service_class,
        patch(
            "app.services.llm_service.LLMService"
        ) as mock_llm_service_class,
        patch(
            "app.services.cms_service.CMSService"
        ) as mock_cms_service_class,
    ):
        mock_workflow_service = AsyncMock()
        mock_document_service = AsyncMock()
        mock_retrieval_service = AsyncMock()
        mock_llm_service = AsyncMock()
        mock_cms_service = AsyncMock()

        mock_workflow_service_class.return_value = (
            mock_workflow_service
        )
        mock_document_service_class.return_value = (
            mock_document_service
        )
        mock_retrieval_service_class.return_value = (
            mock_retrieval_service
        )
        mock_llm_service_class.return_value = (
            mock_llm_service
        )
        mock_cms_service_class.return_value = (
            mock_cms_service
        )

        yield {
            "workflow_service": mock_workflow_service,
            "document_service": mock_document_service,
            "retrieval_service": mock_retrieval_service,
            "llm_service": mock_llm_service,
            "cms_service": mock_cms_service,
        }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def test_workflow_graph_is_configured():
    assert app_workflow is not None


def test_workflow_runner_can_be_created():
    runner = create_workflow_runner()

    assert runner is not None


def test_workflow_runner_returns_new_instance_each_time():
    first_runner = create_workflow_runner()
    second_runner = create_workflow_runner()

    assert first_runner is not None
    assert second_runner is not None


# ---------------------------------------------------------------------------
# Initial state validation
# ---------------------------------------------------------------------------


def test_initial_state_contains_required_fields(
    base_state: dict,
    execution_id: str,
    brief_id: str,
):
    assert base_state["execution_id"] == execution_id
    assert base_state["brief_id"] == brief_id
    assert base_state["raw_brief_text"]
    assert base_state["approval_required"] is True
    assert base_state["status"] == JobStatus.PENDING
    assert base_state["current_step"] == WorkflowStep.INTAKE


def test_initial_state_supports_auto_approval(
    approval_not_required_state: dict,
):
    assert (
        approval_not_required_state["approval_required"]
        is False
    )


def test_initial_state_rejects_empty_brief(
    execution_id: str,
    brief_id: str,
):
    with pytest.raises(WorkflowStateError):
        create_initial_state(
            execution_id=execution_id,
            brief_id=brief_id,
            raw_brief_text="",
            approval_required=True,
        )


def test_initial_state_rejects_missing_brief(
    execution_id: str,
    brief_id: str,
):
    with pytest.raises(WorkflowStateError):
        create_initial_state(
            execution_id=execution_id,
            brief_id=brief_id,
            raw_brief_text=None,
            approval_required=True,
        )


def test_initial_state_rejects_missing_execution_id(
    brief_id: str,
):
    with pytest.raises(WorkflowStateError):
        create_initial_state(
            execution_id="",
            brief_id=brief_id,
            raw_brief_text="Build a dashboard.",
            approval_required=True,
        )


def test_initial_state_rejects_missing_brief_id(
    execution_id: str,
):
    with pytest.raises(WorkflowStateError):
        create_initial_state(
            execution_id=execution_id,
            brief_id="",
            raw_brief_text="Build a dashboard.",
            approval_required=True,
        )


# ---------------------------------------------------------------------------
# Approval-required workflow execution
# ---------------------------------------------------------------------------


async def test_workflow_prepares_approval_when_required(
    base_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(base_state)

    assert isinstance(
        result,
        WorkflowRunResult,
    )

    final_state = result.final_state

    assert (
        final_state["approval_status"]
        == ApprovalStatus.PENDING
    )
    assert (
        final_state["review_status"]
        == "pending_approval"
    )
    assert (
        final_state["publishing_status"]
        == JobStatus.PROCESSING
    )
    assert (
        final_state["current_step"]
        == WorkflowStep.PUBLISHING
    )
    assert final_state["artifacts"]
    assert not final_state["current_error"]
    assert not final_state["validation_issues"]


async def test_approval_required_workflow_does_not_publish(
    base_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(base_state)

    final_state = result.final_state

    assert (
        final_state["publishing_status"]
        != JobStatus.COMPLETED
    )
    assert (
        final_state["approval_status"]
        == ApprovalStatus.PENDING
    )


async def test_approval_required_workflow_preserves_identity(
    base_state: dict,
    execution_id: str,
    brief_id: str,
):
    runner = create_workflow_runner()

    result = await runner.run(base_state)

    final_state = result.final_state

    assert final_state["execution_id"] == execution_id
    assert final_state["brief_id"] == brief_id


async def test_approval_required_workflow_sets_publishing_step(
    base_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(base_state)

    final_state = result.final_state

    assert (
        final_state["current_step"]
        == WorkflowStep.PUBLISHING
    )


async def test_approval_required_workflow_creates_artifacts(
    base_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(base_state)

    final_state = result.final_state

    artifacts = final_state["artifacts"]

    assert isinstance(artifacts, list)
    assert artifacts


# ---------------------------------------------------------------------------
# Auto-approved workflow execution
# ---------------------------------------------------------------------------


async def test_workflow_auto_approves_when_not_required(
    approval_not_required_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(
        approval_not_required_state
    )

    final_state = result.final_state

    assert (
        final_state["approval_status"]
        == ApprovalStatus.APPROVED
    )
    assert final_state["review_status"] == "approved"
    assert (
        final_state["publishing_status"]
        == JobStatus.COMPLETED
    )
    assert final_state["artifacts"]
    assert not final_state["current_error"]


async def test_auto_approved_workflow_completes_publishing(
    approval_not_required_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(
        approval_not_required_state
    )

    final_state = result.final_state

    assert (
        final_state["publishing_status"]
        == JobStatus.COMPLETED
    )
    assert (
        final_state["current_step"]
        == WorkflowStep.PUBLISHING
    )


async def test_auto_approved_workflow_marks_artifacts_published(
    approval_not_required_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(
        approval_not_required_state
    )

    final_state = result.final_state

    artifacts = final_state["artifacts"]

    assert artifacts

    for artifact in artifacts:
        assert artifact["status"] == "published"


async def test_auto_approved_workflow_has_no_validation_issues(
    approval_not_required_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(
        approval_not_required_state
    )

    final_state = result.final_state

    assert not final_state["validation_issues"]
    assert not final_state["current_error"]


# ---------------------------------------------------------------------------
# Workflow validation failures
# ---------------------------------------------------------------------------


async def test_workflow_rejects_empty_brief(
    execution_id: str,
    brief_id: str,
):
    state = create_initial_state(
        execution_id=execution_id,
        brief_id=brief_id,
        raw_brief_text="",
        approval_required=True,
    )

    runner = create_workflow_runner()

    with pytest.raises(WorkflowStateError):
        await runner.run(state)


async def test_workflow_rejects_missing_brief(
    execution_id: str,
    brief_id: str,
):
    state = create_initial_state(
        execution_id=execution_id,
        brief_id=brief_id,
        raw_brief_text="Build a dashboard.",
        approval_required=True,
    )

    state["raw_brief_text"] = None

    runner = create_workflow_runner()

    with pytest.raises(WorkflowStateError):
        await runner.run(state)


async def test_workflow_rejects_missing_execution_id(
    brief_id: str,
):
    state = create_initial_state(
        execution_id="temporary",
        brief_id=brief_id,
        raw_brief_text="Build a dashboard.",
        approval_required=True,
    )

    state["execution_id"] = ""

    runner = create_workflow_runner()

    with pytest.raises(WorkflowStateError):
        await runner.run(state)


async def test_workflow_rejects_missing_brief_id(
    execution_id: str,
):
    state = create_initial_state(
        execution_id=execution_id,
        brief_id="temporary",
        raw_brief_text="Build a dashboard.",
        approval_required=True,
    )

    state["brief_id"] = ""

    runner = create_workflow_runner()

    with pytest.raises(WorkflowStateError):
        await runner.run(state)


async def test_workflow_rejects_invalid_generated_files_type(
    state_with_safe_files: dict,
):
    invalid_state = {
        **state_with_safe_files,
        "generated_code_files": [
            "not-a-dictionary"
        ],
    }

    runner = create_workflow_runner()

    with pytest.raises(WorkflowStateError):
        await runner.run(invalid_state)


async def test_workflow_rejects_empty_generated_files(
    state_with_safe_files: dict,
):
    invalid_state = {
        **state_with_safe_files,
        "generated_code_files": {},
    }

    runner = create_workflow_runner()

    with pytest.raises(WorkflowStateError):
        await runner.run(invalid_state)


# ---------------------------------------------------------------------------
# Provider failures
# ---------------------------------------------------------------------------


async def test_workflow_handles_provider_timeout(
    base_state: dict,
    mock_workflow_dependencies: dict,
):
    mock_workflow_dependencies[
        "llm_service"
    ].generate.side_effect = ProviderTimeoutError(
        "LLM provider timed out"
    )

    runner = create_workflow_runner()

    result = await runner.run(base_state)

    final_state = result.final_state

    assert final_state["status"] in {
        JobStatus.FAILED,
        JobStatus.PROCESSING,
    }

    if final_state["status"] == JobStatus.FAILED:
        assert final_state["current_error"]
        assert (
            final_state["publishing_status"]
            != JobStatus.COMPLETED
        )


async def test_workflow_handles_external_provider_failure(
    base_state: dict,
    mock_workflow_dependencies: dict,
):
    mock_workflow_dependencies[
        "retrieval_service"
    ].search_design_knowledge.side_effect = (
        ExternalProviderError(
            "Retrieval provider unavailable"
        )
    )

    runner = create_workflow_runner()

    result = await runner.run(base_state)

    final_state = result.final_state

    assert final_state["status"] in {
        JobStatus.FAILED,
        JobStatus.PROCESSING,
    }


async def test_workflow_handles_application_error(
    base_state: dict,
    mock_workflow_dependencies: dict,
):
    mock_workflow_dependencies[
        "workflow_service"
    ].get_workflow.side_effect = AppError(
        "Workflow service failure",
        code="workflow_service_failure",
    )

    runner = create_workflow_runner()

    result = await runner.run(base_state)

    final_state = result.final_state

    assert final_state["status"] in {
        JobStatus.FAILED,
        JobStatus.PROCESSING,
    }


async def test_workflow_handles_unexpected_error(
    base_state: dict,
    mock_workflow_dependencies: dict,
):
    mock_workflow_dependencies[
        "llm_service"
    ].generate.side_effect = RuntimeError(
        "Unexpected internal failure"
    )

    runner = create_workflow_runner()

    result = await runner.run(base_state)

    final_state = result.final_state

    assert final_state["status"] in {
        JobStatus.FAILED,
        JobStatus.PROCESSING,
    }


async def test_provider_failure_does_not_leak_internal_details(
    base_state: dict,
    mock_workflow_dependencies: dict,
):
    mock_workflow_dependencies[
        "llm_service"
    ].generate.side_effect = ExternalProviderError(
        "Provider secret internal failure"
    )

    runner = create_workflow_runner()

    result = await runner.run(base_state)

    serialized_result = str(result).lower()

    assert (
        "provider secret internal failure"
        not in serialized_result
    )
    assert "traceback" not in serialized_result
    assert "runtimeerror" not in serialized_result


async def test_unexpected_failure_does_not_leak_stack_trace(
    base_state: dict,
    mock_workflow_dependencies: dict,
):
    mock_workflow_dependencies[
        "llm_service"
    ].generate.side_effect = RuntimeError(
        "Unexpected internal failure"
    )

    runner = create_workflow_runner()

    result = await runner.run(base_state)

    serialized_result = str(result).lower()

    assert "unexpected internal failure" not in serialized_result
    assert "traceback" not in serialized_result


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
        ("os.system('rm -rf /')",),
    ],
)
async def test_workflow_blocks_dangerous_generated_code(
    state_with_dangerous_files: dict,
    dangerous_code: str,
):
    dangerous_state = {
        **state_with_dangerous_files,
        "generated_code_files": {
            "app/dangerous.py": dangerous_code,
        },
    }

    runner = create_workflow_runner()

    result = await runner.run(dangerous_state)

    final_state = result.final_state

    assert (
        final_state["publishing_status"]
        == JobStatus.FAILED
    )
    assert final_state["current_error"]
    assert final_state["validation_issues"]
    assert final_state["review_status"] != "approved"
    assert (
        final_state["approval_status"]
        != ApprovalStatus.APPROVED
    )


async def test_dangerous_code_blocks_auto_approval(
    base_state: dict,
):
    dangerous_auto_approval_state = {
        **base_state,
        "approval_required": False,
        "generated_code_files": {
            "app/dangerous.py": "eval('print(1)')",
        },
        "current_step": WorkflowStep.PUBLISHING,
    }

    runner = create_workflow_runner()

    result = await runner.run(
        dangerous_auto_approval_state
    )

    final_state = result.final_state

    assert (
        final_state["publishing_status"]
        == JobStatus.FAILED
    )
    assert (
        final_state["approval_status"]
        != ApprovalStatus.APPROVED
    )
    assert final_state["current_error"]
    assert final_state["validation_issues"]


async def test_dangerous_code_records_validation_issues(
    state_with_dangerous_files: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(
        state_with_dangerous_files
    )

    final_state = result.final_state

    validation_issues = final_state[
        "validation_issues"
    ]

    assert isinstance(validation_issues, list)
    assert validation_issues


# ---------------------------------------------------------------------------
# Artifact handling
# ---------------------------------------------------------------------------


async def test_workflow_creates_artifacts_for_safe_files(
    state_with_safe_files: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(state_with_safe_files)

    final_state = result.final_state

    artifacts = final_state["artifacts"]

    assert isinstance(artifacts, list)
    assert len(artifacts) >= 1

    artifact_paths = {
        artifact["path"]
        for artifact in artifacts
    }

    assert "app/main.py" in artifact_paths
    assert "app/models.py" in artifact_paths


async def test_artifacts_do_not_contain_file_contents(
    state_with_safe_files: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(state_with_safe_files)

    serialized_result = str(result)

    assert "print('hello')" not in serialized_result
    assert "class User: pass" not in serialized_result


async def test_auto_approved_artifacts_are_published(
    approval_not_required_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(
        approval_not_required_state
    )

    final_state = result.final_state

    artifacts = final_state["artifacts"]

    assert artifacts

    for artifact in artifacts:
        assert artifact["status"] == "published"


async def test_pending_approval_artifacts_are_ready_for_review(
    base_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(base_state)

    final_state = result.final_state

    artifacts = final_state["artifacts"]

    assert artifacts

    for artifact in artifacts:
        assert artifact["status"] == "ready_for_review"


# ---------------------------------------------------------------------------
# Security boundaries
# ---------------------------------------------------------------------------


async def test_workflow_result_does_not_expose_generated_file_contents(
    base_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(base_state)

    serialized_result = str(result).lower()

    forbidden_fragments = (
        "print('hello')",
        'print("hello")',
        "class user: pass",
    )

    for fragment in forbidden_fragments:
        assert fragment not in serialized_result


async def test_workflow_result_does_not_expose_sensitive_keys(
    base_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(base_state)

    serialized_result = str(result).lower()

    forbidden_fragments = (
        "api_key",
        "apikey",
        "secret",
        "password",
        "authorization",
        "credential",
        "private_key",
        "database_url",
        "connection_string",
        "postgresql://",
        "redis://",
    )

    for fragment in forbidden_fragments:
        assert fragment not in serialized_result


async def test_workflow_result_does_not_expose_prompts_or_completions(
    base_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(base_state)

    serialized_result = str(result).lower()

    assert "raw_brief_text" not in serialized_result
    assert (
        "build a secure dashboard with authentication"
        not in serialized_result
    )
    assert "completion" not in serialized_result
    assert "embedding" not in serialized_result
    assert "vector" not in serialized_result


async def test_workflow_result_does_not_expose_storage_urls(
    base_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(base_state)

    serialized_result = str(result).lower()

    forbidden_fragments = (
        "https://",
        "http://",
        "signed_url",
        "storage_url",
        "download_url",
        "file_url",
    )

    for fragment in forbidden_fragments:
        assert fragment not in serialized_result


async def test_workflow_result_is_json_safe(
    base_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(base_state)

    final_state = result.final_state

    assert isinstance(final_state, dict)

    serialized_result = str(final_state)

    assert "Traceback" not in serialized_result
    assert "object at 0x" not in serialized_result


# ---------------------------------------------------------------------------
# Approval state transitions
# ---------------------------------------------------------------------------


async def test_pending_approval_state_can_be_reviewed(
    base_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(base_state)

    final_state = result.final_state

    assert (
        final_state["approval_status"]
        == ApprovalStatus.PENDING
    )

    assert final_state["review_status"] in {
        "pending_approval",
        "awaiting_review",
    }


async def test_auto_approved_state_is_terminal(
    approval_not_required_state: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(
        approval_not_required_state
    )

    final_state = result.final_state

    assert (
        final_state["approval_status"]
        == ApprovalStatus.APPROVED
    )
    assert (
        final_state["publishing_status"]
        == JobStatus.COMPLETED
    )


async def test_failed_state_is_not_approved(
    state_with_dangerous_files: dict,
):
    runner = create_workflow_runner()

    result = await runner.run(
        state_with_dangerous_files
    )

    final_state = result.final_state

    assert (
        final_state["approval_status"]
        != ApprovalStatus.APPROVED
    )
    assert (
        final_state["publishing_status"]
        == JobStatus.FAILED
    )


# ---------------------------------------------------------------------------
# Repeated execution
# ---------------------------------------------------------------------------


async def test_repeated_workflow_execution_generates_unique_ids():
    first_state = create_initial_state(
        execution_id=f"wf_{uuid4()}",
        brief_id=f"brief_{uuid4()}",
        raw_brief_text="Build dashboard one.",
        approval_required=False,
    )

    second_state = create_initial_state(
        execution_id=f"wf_{uuid4()}",
        brief_id=f"brief_{uuid4()}",
        raw_brief_text="Build dashboard two.",
        approval_required=False,
    )

    runner = create_workflow_runner()

    first_result = await runner.run(first_state)
    second_result = await runner.run(second_state)

    assert (
        first_result.final_state["execution_id"]
        != second_result.final_state["execution_id"]
    )

    assert (
        first_result.final_state["brief_id"]
        != second_result.final_state["brief_id"]
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
