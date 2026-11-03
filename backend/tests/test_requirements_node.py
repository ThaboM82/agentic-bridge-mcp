"""
Comprehensive tests for the requirements-extraction workflow node.

These tests cover brief-text construction, requirement normalization, output
schema validation, prompt construction, successful extraction, provider
failures, malformed model output, retry metadata, execution logging, and
security boundaries.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from app.core.enums import JobStatus, WorkflowStep
from app.core.exceptions import (
    ExternalProviderError,
    ProviderTimeoutError,
    WorkflowStateError,
)
from app.graph.nodes.requirements_node import (
    _build_brief_text,
    _normalize_requirement,
    requirements_node,
)
from app.graph.prompts.requirements import (
    REQUIREMENTS_EXTRACTION_PROMPT_VERSION,
    RequirementsExtractionOutputSchema,
    build_requirements_extraction_messages,
)
from app.graph.state import create_initial_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state() -> dict:
    """Return a valid initial workflow state for requirements extraction."""
    return create_initial_state(
        execution_id="wf_123",
        brief_id="brief_456",
        raw_brief_text=(
            "Build an internal analytics dashboard."
        ),
        tenant_id="tenant_789",
        user_id="user_001",
    )


@pytest.fixture
def state_with_documents(state: dict) -> dict:
    """Return workflow state containing source document IDs."""
    return {
        **state,
        "document_ids": [
            "doc_1",
            "doc_2",
        ],
        "normalized_brief_text": (
            "Build an internal analytics dashboard."
        ),
    }


@pytest.fixture
def mock_llm_service():
    """Patch the LLM service used by the requirements node."""
    with patch(
        "app.services.llm_service.LLMService"
    ) as mock_service_class:
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        yield mock_service


# ---------------------------------------------------------------------------
# Brief-text construction
# ---------------------------------------------------------------------------


def test_build_brief_text():
    assert (
        _build_brief_text(
            "  Build   an app.  "
        )
        == "Build   an app."
    )


def test_build_brief_text_preserves_internal_formatting():
    assert (
        _build_brief_text(
            "Build\na\ndashboard."
        )
        == "Build\na\ndashboard."
    )


def test_build_brief_text_trims_outer_whitespace():
    assert (
        _build_brief_text(
            "  Build a dashboard.  "
        )
        == "Build a dashboard."
    )


def test_empty_brief_is_rejected():
    with pytest.raises(WorkflowStateError):
        _build_brief_text("   ")

    with pytest.raises(WorkflowStateError):
        _build_brief_text(None)


def test_empty_string_brief_is_rejected():
    with pytest.raises(WorkflowStateError):
        _build_brief_text("")


def test_non_string_brief_is_rejected():
    with pytest.raises(WorkflowStateError):
        _build_brief_text(123)

    with pytest.raises(WorkflowStateError):
        _build_brief_text(
            ["Build a dashboard"]
        )


def test_oversized_brief_is_rejected():
    with pytest.raises(WorkflowStateError):
        _build_brief_text(
            "x" * 2_000_000
        )


# ---------------------------------------------------------------------------
# Requirement normalization
# ---------------------------------------------------------------------------


def test_normalize_requirement():
    assert (
        _normalize_requirement(
            "  Create a dashboard.  "
        )
        == "Create a dashboard."
    )

    assert (
        _normalize_requirement("   ")
        is None
    )

    assert (
        _normalize_requirement(123)
        is None
    )


def test_normalize_requirement_handles_none():
    assert (
        _normalize_requirement(None)
        is None
    )


def test_normalize_requirement_handles_empty_string():
    assert (
        _normalize_requirement("")
        is None
    )


def test_normalize_requirement_preserves_sentence():
    requirement = (
        "Users must authenticate before viewing metrics."
    )

    assert (
        _normalize_requirement(requirement)
        == requirement
    )


def test_normalize_requirement_trims_whitespace():
    assert (
        _normalize_requirement(
            "  Show key metrics.  "
        )
        == "Show key metrics."
    )


# ---------------------------------------------------------------------------
# Requirements output schema
# ---------------------------------------------------------------------------


def test_requirements_schema_normalizes_values():
    output = RequirementsExtractionOutputSchema(
        requirements=[
            "Create a dashboard",
            "",
            "Show key metrics",
        ],
        assumptions=[
            "Use responsive layout."
        ],
        missing_information=[
            "Define key metrics."
        ],
        confidence=0.8,
    )

    assert output.requirements == [
        "Create a dashboard",
        "Show key metrics",
    ]
    assert output.confidence == 0.8


def test_requirements_schema_supports_full_output():
    output = RequirementsExtractionOutputSchema(
        requirements=[
            "Build a dashboard",
            "Show key metrics",
        ],
        assumptions=[
            "Use responsive layout."
        ],
        missing_information=[
            "Define key metrics."
        ],
        confidence=0.85,
    )

    assert output.requirements == [
        "Build a dashboard",
        "Show key metrics",
    ]
    assert output.assumptions == [
        "Use responsive layout."
    ]
    assert output.missing_information == [
        "Define key metrics."
    ]
    assert output.confidence == 0.85


def test_requirements_schema_defaults_optional_lists():
    output = RequirementsExtractionOutputSchema(
        requirements=[
            "Build a dashboard"
        ],
        confidence=0.9,
    )

    assert output.assumptions == []
    assert output.missing_information == []


def test_requirements_schema_rejects_missing_requirements():
    with pytest.raises(ValidationError):
        RequirementsExtractionOutputSchema(
            confidence=0.8,
        )


def test_requirements_schema_rejects_missing_confidence():
    with pytest.raises(ValidationError):
        RequirementsExtractionOutputSchema(
            requirements=[
                "Build a dashboard"
            ],
        )


def test_requirements_schema_rejects_confidence_below_zero():
    with pytest.raises(ValidationError):
        RequirementsExtractionOutputSchema(
            requirements=[
                "Build a dashboard"
            ],
            confidence=-0.1,
        )


def test_requirements_schema_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        RequirementsExtractionOutputSchema(
            requirements=[
                "Build a dashboard"
            ],
            confidence=1.1,
        )


def test_requirements_schema_rejects_too_many_requirements():
    with pytest.raises(ValidationError):
        RequirementsExtractionOutputSchema(
            requirements=[
                f"Requirement {index}"
                for index in range(101)
            ]
        )


def test_requirements_schema_accepts_maximum_requirements():
    requirements = [
        f"Requirement {index}"
        for index in range(100)
    ]

    output = RequirementsExtractionOutputSchema(
        requirements=requirements,
        confidence=0.8,
    )

    assert len(output.requirements) == 100


def test_requirements_schema_rejects_non_string_requirement():
    with pytest.raises(ValidationError):
        RequirementsExtractionOutputSchema(
            requirements=[
                "Build a dashboard",
                123,
            ],
            confidence=0.8,
        )


def test_requirements_schema_rejects_unknown_fields():
    with pytest.raises(ValidationError) as exc_info:
        RequirementsExtractionOutputSchema(
            requirements=[
                "Build a dashboard"
            ],
            confidence=0.8,
            unexpected_field="value",
        )

    assert any(
        error["type"] == "extra_forbidden"
        for error in exc_info.value.errors()
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_build_messages():
    messages = (
        build_requirements_extraction_messages(
            brief_text=(
                "Build an analytics dashboard."
            ),
            format_instructions=(
                "Return JSON."
            ),
        )
    )

    assert len(messages) == 2
    assert (
        "Build an analytics dashboard."
        in messages[1].content
    )


def test_build_messages_includes_format_instructions():
    messages = (
        build_requirements_extraction_messages(
            brief_text=(
                "Build an analytics dashboard."
            ),
            format_instructions=(
                "Return valid JSON only."
            ),
        )
    )

    assert (
        "Return valid JSON only."
        in messages[0].content
    )


def test_build_messages_includes_prompt_version():
    messages = (
        build_requirements_extraction_messages(
            brief_text=(
                "Build an analytics dashboard."
            ),
            format_instructions=(
                "Return JSON."
            ),
        )
    )

    assert (
        REQUIREMENTS_EXTRACTION_PROMPT_VERSION
        is not None
    )


def test_build_messages_rejects_empty_brief():
    with pytest.raises(WorkflowStateError):
        build_requirements_extraction_messages(
            brief_text="",
            format_instructions=(
                "Return JSON."
            ),
        )


def test_build_messages_rejects_missing_brief():
    with pytest.raises(WorkflowStateError):
        build_requirements_extraction_messages(
            brief_text=None,
            format_instructions=(
                "Return JSON."
            ),
        )


def test_build_messages_rejects_empty_format_instructions():
    with pytest.raises(WorkflowStateError):
        build_requirements_extraction_messages(
            brief_text=(
                "Build an analytics dashboard."
            ),
            format_instructions="",
        )


# ---------------------------------------------------------------------------
# Successful requirements extraction
# ---------------------------------------------------------------------------


async def test_requirements_node_extracts_requirements(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.return_value = (
        RequirementsExtractionOutputSchema(
            requirements=[
                "Build a dashboard",
                "Show key metrics",
            ],
            assumptions=[
                "Use responsive layout."
            ],
            missing_information=[],
            confidence=0.9,
        )
    )

    result = await requirements_node(
        state_with_documents
    )

    assert (
        result["current_step"]
        == WorkflowStep.REQUIREMENTS
    )
    assert result["requirements"] == [
        "Build a dashboard",
        "Show key metrics",
    ]
    assert result["assumptions"] == [
        "Use responsive layout."
    ]
    assert result["missing_information"] == []
    assert result["confidence"] == 0.9
    assert (
        result["requirements_status"]
        == JobStatus.COMPLETED
    )
    assert result["current_error"] is None
    assert result["execution_logs"]


async def test_requirements_node_preserves_execution_identity(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.return_value = (
        RequirementsExtractionOutputSchema(
            requirements=[
                "Build a dashboard"
            ],
            confidence=0.9,
        )
    )

    result = await requirements_node(
        state_with_documents
    )

    assert result["execution_id"] == "wf_123"
    assert result["brief_id"] == "brief_456"
    assert result["tenant_id"] == "tenant_789"
    assert result["user_id"] == "user_001"


async def test_requirements_node_records_prompt_version(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.return_value = (
        RequirementsExtractionOutputSchema(
            requirements=[
                "Build a dashboard"
            ],
            confidence=0.9,
        )
    )

    result = await requirements_node(
        state_with_documents
    )

    assert (
        result["requirements_prompt_version"]
        == REQUIREMENTS_EXTRACTION_PROMPT_VERSION
    )


async def test_requirements_node_records_execution_logs(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.return_value = (
        RequirementsExtractionOutputSchema(
            requirements=[
                "Build a dashboard"
            ],
            confidence=0.9,
        )
    )

    result = await requirements_node(
        state_with_documents
    )

    execution_logs = result[
        "execution_logs"
    ]

    assert isinstance(execution_logs, list)
    assert execution_logs


async def test_requirements_node_does_not_expose_brief_text(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.return_value = (
        RequirementsExtractionOutputSchema(
            requirements=[
                "Build a dashboard"
            ],
            confidence=0.9,
        )
    )

    result = await requirements_node(
        state_with_documents
    )

    serialized_result = str(result)

    assert (
        "Build an internal analytics dashboard."
        not in serialized_result
    )


# ---------------------------------------------------------------------------
# Requirements validation failures
# ---------------------------------------------------------------------------


async def test_requirements_node_requires_brief():
    state = create_initial_state(
        execution_id="wf_123",
        brief_id="brief_456",
        raw_brief_text="   ",
    )

    result = await requirements_node(state)

    assert (
        result["current_step"]
        == WorkflowStep.REQUIREMENTS
    )
    assert result["current_error"]
    assert result["recoverable_error"] is False
    assert (
        result["requirements_status"]
        == JobStatus.FAILED
    )


async def test_requirements_node_fails_on_missing_brief(
    state_with_documents: dict,
):
    invalid_state = {
        **state_with_documents,
        "raw_brief_text": None,
    }

    result = await requirements_node(
        invalid_state
    )

    assert (
        result["requirements_status"]
        == JobStatus.FAILED
    )
    assert result["current_error"]
    assert result["recoverable_error"] is False


async def test_requirements_node_fails_on_empty_requirements(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.return_value = (
        RequirementsExtractionOutputSchema(
            requirements=[],
            confidence=0.4,
        )
    )

    result = await requirements_node(
        state_with_documents
    )

    assert (
        result["requirements_status"]
        == JobStatus.FAILED
    )
    assert result["current_error"]
    assert result["recoverable_error"] is False


async def test_requirements_node_fails_on_low_confidence(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.return_value = (
        RequirementsExtractionOutputSchema(
            requirements=[
                "Build a dashboard"
            ],
            missing_information=[
                "Define key metrics."
            ],
            confidence=0.2,
        )
    )

    result = await requirements_node(
        state_with_documents
    )

    assert (
        result["requirements_status"]
        == JobStatus.FAILED
    )
    assert result["current_error"]
    assert result["recoverable_error"] is False


# ---------------------------------------------------------------------------
# Provider failures
# ---------------------------------------------------------------------------


async def test_requirements_node_handles_provider_failure(
    state: dict,
):
    result = await requirements_node(state)

    assert (
        result["current_step"]
        == WorkflowStep.REQUIREMENTS
    )
    assert result["current_error"]


async def test_requirements_node_handles_provider_timeout(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.side_effect = ProviderTimeoutError(
        "LLM provider timed out"
    )

    result = await requirements_node(
        state_with_documents
    )

    assert (
        result["requirements_status"]
        == JobStatus.FAILED
    )
    assert result["current_error"]
    assert result["recoverable_error"] is True


async def test_requirements_node_handles_external_provider_failure(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.side_effect = ExternalProviderError(
        "LLM provider unavailable"
    )

    result = await requirements_node(
        state_with_documents
    )

    assert (
        result["requirements_status"]
        == JobStatus.FAILED
    )
    assert result["current_error"]
    assert result["recoverable_error"] is True


async def test_requirements_node_handles_malformed_model_output(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.side_effect = ValidationError(
        "Model returned invalid structured output"
    )

    result = await requirements_node(
        state_with_documents
    )

    assert (
        result["requirements_status"]
        == JobStatus.FAILED
    )
    assert result["current_error"]
    assert result["recoverable_error"] is False


async def test_provider_failure_does_not_leak_internal_details(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.side_effect = RuntimeError(
        "internal secret database password"
    )

    result = await requirements_node(
        state_with_documents
    )

    serialized_result = str(result).lower()

    assert "internal secret" not in serialized_result
    assert "database password" not in serialized_result
    assert "traceback" not in serialized_result


# ---------------------------------------------------------------------------
# Recovery metadata
# ---------------------------------------------------------------------------


async def test_successful_requirements_have_no_error_metadata(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.return_value = (
        RequirementsExtractionOutputSchema(
            requirements=[
                "Build a dashboard"
            ],
            confidence=0.9,
        )
    )

    result = await requirements_node(
        state_with_documents
    )

    assert result["current_error"] is None
    assert result["recoverable_error"] is False
    assert result["error_code"] is None


async def test_failed_requirements_set_error_metadata(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.side_effect = ProviderTimeoutError(
        "LLM provider timed out"
    )

    result = await requirements_node(
        state_with_documents
    )

    assert result["current_error"]
    assert result["error_code"]
    assert (
        result["error_step"]
        == WorkflowStep.REQUIREMENTS
    )
    assert result["recoverable_error"] is True


async def test_failed_requirements_record_recovery_history(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.side_effect = ProviderTimeoutError(
        "LLM provider timed out"
    )

    result = await requirements_node(
        state_with_documents
    )

    recovery_history = result.get(
        "recovery_history",
        [],
    )

    assert isinstance(recovery_history, list)


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


async def test_requirements_node_sets_requirements_step(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.return_value = (
        RequirementsExtractionOutputSchema(
            requirements=[
                "Build a dashboard"
            ],
            confidence=0.9,
        )
    )

    result = await requirements_node(
        state_with_documents
    )

    assert (
        result["current_step"]
        == WorkflowStep.REQUIREMENTS
    )


async def test_successful_requirements_set_completed_status(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.return_value = (
        RequirementsExtractionOutputSchema(
            requirements=[
                "Build a dashboard"
            ],
            confidence=0.9,
        )
    )

    result = await requirements_node(
        state_with_documents
    )

    assert (
        result["requirements_status"]
        == JobStatus.COMPLETED
    )


async def test_failed_requirements_set_failed_status(
    state_with_documents: dict,
    mock_llm_service: AsyncMock,
):
    mock_llm_service.generate_structured.side_effect = ProviderTimeoutError(
        "LLM provider timed out"
    )

    result = await requirements_node(
        state_with_documents
    )

    assert (
        result["requirements_status"]
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
