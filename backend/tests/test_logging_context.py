"""
Comprehensive tests for structured logging context helpers.

These tests cover request, workflow, job, document, and LLM context binding;
context isolation; sensitive-value redaction; invalid identifier handling;
context clearing; and safe logging metadata.
"""

from __future__ import annotations

import logging
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import ValidationError
from app.core.logging_context import (
    bind_document_context,
    bind_job_context,
    bind_llm_context,
    bind_request_context,
    bind_workflow_context,
    bind_workflow_step,
    clear_logging_context,
    generate_request_id,
    get_document_id,
    get_job_id,
    get_logging_context,
    get_request_id,
    get_workflow_id,
    get_workflow_step,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_logging_context():
    """Ensure every test starts and ends with a clean logging context."""
    clear_logging_context()

    yield

    clear_logging_context()


# ---------------------------------------------------------------------------
# Request ID generation
# ---------------------------------------------------------------------------


def test_generate_request_id_returns_prefixed_id():
    request_id = generate_request_id()

    assert request_id.startswith("req_")


def test_generate_request_id_is_unique():
    first_request_id = generate_request_id()
    second_request_id = generate_request_id()

    assert first_request_id != second_request_id


def test_generate_request_id_has_expected_format():
    request_id = generate_request_id()

    assert len(request_id) > len("req_")
    assert request_id[4:].replace("-", "").isalnum()


# ---------------------------------------------------------------------------
# Request context
# ---------------------------------------------------------------------------


def test_request_context_is_bound():
    request_id = generate_request_id()

    bind_request_context(
        request_id,
        user_id="user_123",
        tenant_id="tenant_456",
    )

    assert get_request_id() == request_id
    assert get_logging_context()["user_id"] == "user_123"
    assert get_logging_context()["tenant_id"] == "tenant_456"


def test_request_context_accepts_uuid_request_id():
    request_id = uuid4()

    bind_request_context(
        request_id,
        user_id="user_123",
        tenant_id="tenant_456",
    )

    assert get_request_id() == str(request_id)


def test_request_context_generates_id_when_missing():
    bind_request_context(
        user_id="user_123",
        tenant_id="tenant_456",
    )

    request_id = get_request_id()

    assert request_id
    assert request_id.startswith("req_")


def test_request_context_rejects_invalid_user_id():
    with pytest.raises(ValidationError):
        bind_request_context(
            generate_request_id(),
            user_id="",
            tenant_id="tenant_456",
        )


def test_request_context_rejects_invalid_tenant_id():
    with pytest.raises(ValidationError):
        bind_request_context(
            generate_request_id(),
            user_id="user_123",
            tenant_id="",
        )


def test_request_context_rejects_oversized_request_id():
    oversized_request_id = "x" * 500

    with pytest.raises(ValidationError):
        bind_request_context(
            oversized_request_id,
            user_id="user_123",
            tenant_id="tenant_456",
        )


def test_request_context_redacts_sensitive_user_id():
    bind_request_context(
        generate_request_id(),
        user_id="api_key=sk-should-not-leak",
        tenant_id="tenant_456",
    )

    context = get_logging_context()

    assert "sk-should-not-leak" not in str(context)


def test_request_context_redacts_sensitive_tenant_id():
    bind_request_context(
        generate_request_id(),
        user_id="user_123",
        tenant_id="password=should-not-leak",
    )

    context = get_logging_context()

    assert "should-not-leak" not in str(context)


# ---------------------------------------------------------------------------
# Invalid request IDs
# ---------------------------------------------------------------------------


def test_invalid_request_id_is_rejected():
    with pytest.raises(ValidationError):
        bind_request_context(
            "bad request id with spaces"
        )


def test_empty_request_id_is_rejected():
    with pytest.raises(ValidationError):
        bind_request_context("")


def test_whitespace_request_id_is_rejected():
    with pytest.raises(ValidationError):
        bind_request_context("   ")


def test_none_request_id_is_rejected():
    with pytest.raises(ValidationError):
        bind_request_context(None)


def test_integer_request_id_is_rejected():
    with pytest.raises(ValidationError):
        bind_request_context(12345)


# ---------------------------------------------------------------------------
# Workflow context
# ---------------------------------------------------------------------------


def test_workflow_context_is_bound():
    bind_workflow_context(
        "wf_123",
        workflow_step="design",
        tenant_id="tenant_456",
    )

    assert get_workflow_id() == "wf_123"
    assert get_workflow_step() == "design"
    assert (
        get_logging_context()["tenant_id"]
        == "tenant_456"
    )


def test_workflow_context_generates_id_when_missing():
    bind_workflow_context()

    workflow_id = get_workflow_id()

    assert workflow_id
    assert workflow_id.startswith("wf_")


def test_workflow_context_accepts_uuid_workflow_id():
    workflow_id = uuid4()

    bind_workflow_context(workflow_id)

    assert get_workflow_id() == str(workflow_id)


def test_workflow_context_rejects_invalid_workflow_id():
    with pytest.raises(ValidationError):
        bind_workflow_context(
            "invalid workflow id"
        )


def test_workflow_context_rejects_empty_workflow_id():
    with pytest.raises(ValidationError):
        bind_workflow_context("")


def test_workflow_context_rejects_oversized_workflow_id():
    with pytest.raises(ValidationError):
        bind_workflow_context("x" * 500)


def test_workflow_context_redacts_sensitive_tenant_id():
    bind_workflow_context(
        "wf_123",
        tenant_id="api_key=sk-should-not-leak",
    )

    context = get_logging_context()

    assert "sk-should-not-leak" not in str(context)


# ---------------------------------------------------------------------------
# Workflow step context
# ---------------------------------------------------------------------------


def test_workflow_step_can_be_updated():
    bind_workflow_context("wf_123")
    bind_workflow_step("code_generation")

    assert (
        get_workflow_step()
        == "code_generation"
    )


@pytest.mark.parametrize(
    "workflow_step",
    [
        "intake",
        "ingestion",
        "retrieve_context",
        "extract_requirements",
        "generate_design",
        "generate_code",
        "validate_output",
        "human_review",
        "publishing",
        "completed",
        "failed",
    ],
)
def test_workflow_step_accepts_known_steps(
    workflow_step: str,
):
    bind_workflow_context("wf_123")
    bind_workflow_step(workflow_step)

    assert get_workflow_step() == workflow_step


def test_workflow_step_rejects_empty_value():
    bind_workflow_context("wf_123")

    with pytest.raises(ValidationError):
        bind_workflow_step("")


def test_workflow_step_rejects_whitespace_only_value():
    bind_workflow_context("wf_123")

    with pytest.raises(ValidationError):
        bind_workflow_step("   ")


def test_workflow_step_rejects_invalid_value():
    bind_workflow_context("wf_123")

    with pytest.raises(ValidationError):
        bind_workflow_step(
            "invalid workflow step"
        )


def test_workflow_step_rejects_none_value():
    bind_workflow_context("wf_123")

    with pytest.raises(ValidationError):
        bind_workflow_step(None)


def test_workflow_step_rejects_oversized_value():
    bind_workflow_context("wf_123")

    with pytest.raises(ValidationError):
        bind_workflow_step("x" * 500)


# ---------------------------------------------------------------------------
# Job and document context
# ---------------------------------------------------------------------------


def test_job_and_document_context():
    bind_job_context(
        "job_123",
        workflow_id="wf_456",
    )
    bind_document_context("doc_789")

    assert get_job_id() == "job_123"
    assert get_document_id() == "doc_789"
    assert (
        get_logging_context()["workflow_id"]
        == "wf_456"
    )


def test_job_context_accepts_uuid_job_id():
    job_id = uuid4()

    bind_job_context(job_id)

    assert get_job_id() == str(job_id)


def test_job_context_generates_id_when_missing():
    bind_job_context()

    job_id = get_job_id()

    assert job_id
    assert job_id.startswith("job_")


def test_job_context_rejects_invalid_job_id():
    with pytest.raises(ValidationError):
        bind_job_context("invalid job id")


def test_job_context_rejects_empty_job_id():
    with pytest.raises(ValidationError):
        bind_job_context("")


def test_job_context_rejects_invalid_workflow_id():
    with pytest.raises(ValidationError):
        bind_job_context(
            "job_123",
            workflow_id="invalid workflow id",
        )


def test_document_context_accepts_uuid_document_id():
    document_id = uuid4()

    bind_document_context(document_id)

    assert get_document_id() == str(document_id)


def test_document_context_generates_id_when_missing():
    bind_document_context()

    document_id = get_document_id()

    assert document_id
    assert document_id.startswith("doc_")


def test_document_context_rejects_invalid_document_id():
    with pytest.raises(ValidationError):
        bind_document_context(
            "invalid document id"
        )


def test_document_context_rejects_empty_document_id():
    with pytest.raises(ValidationError):
        bind_document_context("")


def test_document_context_rejects_oversized_document_id():
    with pytest.raises(ValidationError):
        bind_document_context("x" * 500)


# ---------------------------------------------------------------------------
# LLM context
# ---------------------------------------------------------------------------


def test_llm_context():
    bind_llm_context(
        "openai",
        model="gpt-4o-mini",
    )

    context = get_logging_context()

    assert context["provider"] == "openai"
    assert context["model"] == "gpt-4o-mini"


def test_llm_context_accepts_optional_operation():
    bind_llm_context(
        "openai",
        model="gpt-4o-mini",
        operation="generate_code",
    )

    context = get_logging_context()

    assert context["provider"] == "openai"
    assert context["model"] == "gpt-4o-mini"
    assert (
        context["operation"]
        == "generate_code"
    )


def test_llm_context_rejects_empty_provider():
    with pytest.raises(ValidationError):
        bind_llm_context("")


def test_llm_context_rejects_invalid_provider():
    with pytest.raises(ValidationError):
        bind_llm_context("invalid provider")


def test_llm_context_rejects_empty_model():
    with pytest.raises(ValidationError):
        bind_llm_context(
            "openai",
            model="",
        )


def test_llm_context_rejects_invalid_model():
    with pytest.raises(ValidationError):
        bind_llm_context(
            "openai",
            model="invalid model",
        )


def test_llm_context_redacts_sensitive_provider():
    bind_llm_context(
        "openai api_key=sk-should-not-leak",
        model="gpt-4o-mini",
    )

    context = get_logging_context()

    assert "sk-should-not-leak" not in str(context)


def test_llm_context_redacts_sensitive_model():
    bind_llm_context(
        "openai",
        model="gpt-4o-mini password=should-not-leak",
    )

    context = get_logging_context()

    assert "should-not-leak" not in str(context)


# ---------------------------------------------------------------------------
# Context clearing
# ---------------------------------------------------------------------------


def test_clear_context_resets_values():
    bind_request_context(
        generate_request_id(),
        user_id="user_123",
        tenant_id="tenant_456",
    )
    clear_logging_context()

    context = get_logging_context()

    assert all(
        value == ""
        for value in context.values()
    )


def test_clear_context_is_idempotent():
    clear_logging_context()
    clear_logging_context()

    context = get_logging_context()

    assert all(
        value == ""
        for value in context.values()
    )


def test_clear_context_removes_workflow_values():
    bind_workflow_context(
        "wf_123",
        workflow_step="design",
    )
    clear_logging_context()

    assert get_workflow_id() == ""
    assert get_workflow_step() == ""


def test_clear_context_removes_job_and_document_values():
    bind_job_context("job_123")
    bind_document_context("doc_789")
    clear_logging_context()

    assert get_job_id() == ""
    assert get_document_id() == ""


# ---------------------------------------------------------------------------
# Context replacement
# ---------------------------------------------------------------------------


def test_request_context_can_be_replaced():
    first_request_id = generate_request_id()

    bind_request_context(
        first_request_id,
        user_id="user_123",
        tenant_id="tenant_456",
    )

    second_request_id = generate_request_id()

    bind_request_context(
        second_request_id,
        user_id="user_456",
        tenant_id="tenant_789",
    )

    assert get_request_id() == second_request_id
    assert (
        get_logging_context()["user_id"]
        == "user_456"
    )
    assert (
        get_logging_context()["tenant_id"]
        == "tenant_789"
    )


def test_workflow_context_can_be_replaced():
    bind_workflow_context(
        "wf_123",
        workflow_step="design",
    )

    bind_workflow_context(
        "wf_456",
        workflow_step="code_generation",
    )

    assert get_workflow_id() == "wf_456"
    assert (
        get_workflow_step()
        == "code_generation"
    )


# ---------------------------------------------------------------------------
# Structured logging integration
# ---------------------------------------------------------------------------


def test_logging_context_is_available_to_logger(
    caplog,
):
    request_id = generate_request_id()

    bind_request_context(
        request_id,
        user_id="user_123",
        tenant_id="tenant_456",
    )

    logger = logging.getLogger(
        "agentic_bridge.test"
    )

    with caplog.at_level(logging.INFO):
        logger.info("test_event")

    assert "test_event" in caplog.text


def test_sensitive_metadata_is_not_logged(
    caplog,
):
    request_id = generate_request_id()

    bind_request_context(
        request_id,
        user_id="api_key=sk-should-not-leak",
        tenant_id="tenant_456",
    )

    logger = logging.getLogger(
        "agentic_bridge.test"
    )

    with caplog.at_level(logging.INFO):
        logger.info(
            "test_event",
            extra={
                "api_key": "sk-should-not-leak",
            },
        )

    assert "sk-should-not-leak" not in caplog.text


def test_context_does_not_leak_credentials_between_tests():
    bind_request_context(
        generate_request_id(),
        user_id="user_123",
        tenant_id="tenant_456",
    )

    clear_logging_context()

    context = get_logging_context()

    serialized_context = str(context).lower()

    assert "user_123" not in serialized_context
    assert "tenant_456" not in serialized_context
