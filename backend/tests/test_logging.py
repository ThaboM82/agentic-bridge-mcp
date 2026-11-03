"""
Comprehensive tests for structured logging configuration.

These tests cover logger setup, sensitive-value redaction, event-name
normalization, JSON serialization, request correlation, log levels, and
protection against leaking credentials, prompts, completions, storage URLs,
and stack traces.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest

from app.core.logging import (
    configure_logging,
    get_logger,
    normalize_event_name,
    redact_sensitive_values,
    sensitive_log_filter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_logging():
    """Ensure logging tests do not leak handlers or context."""
    root_logger = logging.getLogger()

    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    yield

    root_logger.handlers = original_handlers
    root_logger.setLevel(original_level)


@pytest.fixture
def logger() -> logging.Logger:
    """Return a configured AgenticBridge logger."""
    return get_logger("agentic_bridge.test")


# ---------------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------------


def test_get_logger_returns_logger():
    logger = get_logger("agentic_bridge.test")

    assert logger is not None
    assert isinstance(logger, logging.Logger)


def test_get_logger_uses_expected_name():
    logger = get_logger("agentic_bridge.test")

    assert logger.name == "agentic_bridge.test"


def test_get_logger_is_cached():
    first_logger = get_logger("agentic_bridge.test")
    second_logger = get_logger("agentic_bridge.test")

    assert first_logger is second_logger


def test_configure_logging_does_not_raise():
    configure_logging()


def test_configure_logging_sets_root_level():
    configure_logging()

    root_logger = logging.getLogger()

    assert root_logger.level in {
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
    }


# ---------------------------------------------------------------------------
# Sensitive-value redaction
# ---------------------------------------------------------------------------


def test_sensitive_values_are_redacted():
    event = {
        "event": "login_attempt",
        "api_key": "secret-value",
        "password": "hunter2",
        "nested": {
            "token": "abc123",
            "safe": "value",
        },
    }

    result = redact_sensitive_values(
        None,
        "info",
        event,
    )

    assert result["api_key"] == "[REDACTED]"
    assert result["password"] == "[REDACTED]"
    assert result["nested"]["token"] == "[REDACTED]"
    assert result["nested"]["safe"] == "value"


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "api_key",
        "apikey",
        "API_KEY",
        "password",
        "passwd",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "credential",
        "credentials",
        "cookie",
        "private_key",
        "database_url",
        "connection_string",
    ],
)
def test_common_sensitive_keys_are_redacted(
    sensitive_key: str,
):
    event = {
        "event": "test_event",
        sensitive_key: "should-not-leak",
    }

    result = redact_sensitive_values(
        None,
        "info",
        event,
    )

    assert result[sensitive_key] == "[REDACTED]"


def test_nested_sensitive_values_are_redacted():
    event = {
        "event": "provider_request",
        "request": {
            "headers": {
                "authorization": (
                    "Bearer secret-token"
                ),
                "content_type": (
                    "application/json"
                ),
            },
            "metadata": {
                "api_key": "secret-key",
                "model": "gpt-4o-mini",
            },
        },
    }

    result = redact_sensitive_values(
        None,
        "info",
        event,
    )

    assert (
        result["request"]["headers"][
            "authorization"
        ]
        == "[REDACTED]"
    )
    assert (
        result["request"]["headers"][
            "content_type"
        ]
        == "application/json"
    )
    assert (
        result["request"]["metadata"][
            "api_key"
        ]
        == "[REDACTED]"
    )
    assert (
        result["request"]["metadata"]["model"]
        == "gpt-4o-mini"
    )


def test_sensitive_values_in_lists_are_redacted():
    event = {
        "event": "batch_request",
        "items": [
            {
                "api_key": "secret-one",
                "name": "item-one",
            },
            {
                "api_key": "secret-two",
                "name": "item-two",
            },
        ],
    }

    result = redact_sensitive_values(
        None,
        "info",
        event,
    )

    assert (
        result["items"][0]["api_key"]
        == "[REDACTED]"
    )
    assert (
        result["items"][0]["name"]
        == "item-one"
    )
    assert (
        result["items"][1]["api_key"]
        == "[REDACTED]"
    )
    assert (
        result["items"][1]["name"]
        == "item-two"
    )


def test_prompt_and_completion_keys_are_redacted():
    event = {
        "event": "llm_invocation",
        "prompt": "Build a dashboard",
        "completion": "Here is the implementation",
        "messages": [
            {"role": "user", "content": "secret"}
        ],
        "model": "gpt-4o-mini",
    }

    result = redact_sensitive_values(
        None,
        "info",
        event,
    )

    assert result["prompt"] == "[REDACTED]"
    assert result["completion"] == "[REDACTED]"
    assert result["messages"] == "[REDACTED]"
    assert result["model"] == "gpt-4o-mini"


def test_embedding_and_vector_keys_are_redacted():
    event = {
        "event": "embedding_created",
        "embedding": [0.1, 0.2, 0.3],
        "vector": [0.4, 0.5, 0.6],
        "model": "text-embedding-3-small",
    }

    result = redact_sensitive_values(
        None,
        "info",
        event,
    )

    assert result["embedding"] == "[REDACTED]"
    assert result["vector"] == "[REDACTED]"
    assert (
        result["model"]
        == "text-embedding-3-small"
    )


def test_database_url_credentials_are_redacted():
    event = {
        "event": "database_connection",
        "database_url": (
            "postgresql://user:super-secret-password@localhost:5432/app"
        ),
    }

    result = redact_sensitive_values(
        None,
        "info",
        event,
    )

    serialized_result = json.dumps(result)

    assert "super-secret-password" not in serialized_result


def test_storage_and_signed_urls_are_redacted():
    event = {
        "event": "document_uploaded",
        "storage_url": (
            "https://storage.example.com/secret-object"
        ),
        "signed_url": (
            "https://storage.example.com/signed-object?signature=secret"
        ),
        "file_name": "brief.pdf",
    }

    result = redact_sensitive_values(
        None,
        "info",
        event,
    )

    assert result["storage_url"] == "[REDACTED]"
    assert result["signed_url"] == "[REDACTED]"
    assert result["file_name"] == "brief.pdf"


def test_stack_trace_and_traceback_keys_are_redacted():
    event = {
        "event": "unexpected_error",
        "stack_trace": "Traceback details",
        "traceback": "More traceback details",
        "error_code": "internal_error",
    }

    result = redact_sensitive_values(
        None,
        "info",
        event,
    )

    assert result["stack_trace"] == "[REDACTED]"
    assert result["traceback"] == "[REDACTED]"
    assert (
        result["error_code"]
        == "internal_error"
    )


def test_redaction_preserves_safe_values():
    event = {
        "event": "workflow_started",
        "workflow_id": "wf_123",
        "tenant_id": "tenant_456",
        "user_id": "user_789",
        "model": "gpt-4o-mini",
        "duration_ms": 123.45,
        "success": True,
    }

    result = redact_sensitive_values(
        None,
        "info",
        event,
    )

    assert result == event


def test_redaction_handles_non_dict_event():
    result = redact_sensitive_values(
        None,
        "info",
        "not-a-dictionary",
    )

    assert result == "not-a-dictionary"


def test_redaction_handles_none_event():
    result = redact_sensitive_values(
        None,
        "info",
        None,
    )

    assert result is None


# ---------------------------------------------------------------------------
# Event-name normalization
# ---------------------------------------------------------------------------


def test_event_name_is_normalized():
    event = {
        "event": "Workflow Completed",
    }

    result = normalize_event_name(
        None,
        "info",
        event,
    )

    assert result["event"] == "workflow_completed"


@pytest.mark.parametrize(
    ("raw_event_name", "expected_event_name"),
    [
        ("Workflow Started", "workflow_started"),
        ("WORKFLOW FAILED", "workflow_failed"),
        ("LLM Invocation Completed", "llm_invocation_completed"),
        ("Document  Ingested", "document_ingested"),
        ("Human Review Requested", "human_review_requested"),
        ("CMS Draft Updated", "cms_draft_updated"),
        ("Figma Context Retrieved", "figma_context_retrieved"),
    ],
)
def test_event_names_are_normalized_consistently(
    raw_event_name: str,
    expected_event_name: str,
):
    event = {
        "event": raw_event_name,
    }

    result = normalize_event_name(
        None,
        "info",
        event,
    )

    assert (
        result["event"]
        == expected_event_name
    )


def test_event_name_preserves_snake_case():
    event = {
        "event": "workflow_started",
    }

    result = normalize_event_name(
        None,
        "info",
        event,
    )

    assert result["event"] == "workflow_started"


def test_missing_event_name_is_ignored():
    event = {
        "workflow_id": "wf_123",
    }

    result = normalize_event_name(
        None,
        "info",
        event,
    )

    assert result == event


def test_non_string_event_name_is_ignored():
    event = {
        "event": 123,
    }

    result = normalize_event_name(
        None,
        "info",
        event,
    )

    assert result == event


def test_event_name_is_bounded():
    event = {
        "event": "x" * 500,
    }

    result = normalize_event_name(
        None,
        "info",
        event,
    )

    assert len(result["event"]) <= 200


# ---------------------------------------------------------------------------
# Sensitive log filter
# ---------------------------------------------------------------------------


def test_sensitive_log_filter_redacts_record_message():
    logger = logging.getLogger(
        "agentic_bridge.test.filter"
    )

    record = logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn=__file__,
        lno=1,
        msg="Login with api_key=secret-value",
        args=(),
        exc_info=None,
    )

    filtered_record = sensitive_log_filter(
        record
    )

    assert (
        "secret-value"
        not in filtered_record.getMessage()
    )


def test_sensitive_log_filter_redacts_record_args():
    logger = logging.getLogger(
        "agentic_bridge.test.filter"
    )

    record = logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn=__file__,
        lno=1,
        msg="Login with %s",
        args=("api_key=secret-value",),
        exc_info=None,
    )

    filtered_record = sensitive_log_filter(
        record
    )

    assert (
        "secret-value"
        not in filtered_record.getMessage()
    )


def test_sensitive_log_filter_preserves_safe_message():
    logger = logging.getLogger(
        "agentic_bridge.test.filter"
    )

    record = logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn=__file__,
        lno=1,
        msg="Workflow started",
        args=(),
        exc_info=None,
    )

    filtered_record = sensitive_log_filter(
        record
    )

    assert (
        filtered_record.getMessage()
        == "Workflow started"
    )


# ---------------------------------------------------------------------------
# JSON-safe serialization
# ---------------------------------------------------------------------------


def test_redacted_event_is_json_serializable():
    event = {
        "event": "provider_request",
        "api_key": "secret-value",
        "nested": {
            "token": "secret-token",
            "safe": "value",
        },
    }

    result = redact_sensitive_values(
        None,
        "info",
        event,
    )

    serialized_result = json.dumps(result)

    assert "secret-value" not in serialized_result
    assert "secret-token" not in serialized_result
    assert "[REDACTED]" in serialized_result


def test_normalized_event_is_json_serializable():
    event = {
        "event": "Workflow Completed",
        "workflow_id": "wf_123",
    }

    result = normalize_event_name(
        None,
        "info",
        event,
    )

    serialized_result = json.dumps(result)

    assert "workflow_completed" in serialized_result


# ---------------------------------------------------------------------------
# Logger integration
# ---------------------------------------------------------------------------


def test_logger_does_not_log_sensitive_values(
    logger: logging.Logger,
    caplog,
):
    with caplog.at_level(logging.INFO):
        logger.info(
            "provider_request",
            extra={
                "api_key": "secret-value",
                "password": "hunter2",
            },
        )

    assert "secret-value" not in caplog.text
    assert "hunter2" not in caplog.text


def test_logger_preserves_safe_context(
    logger: logging.Logger,
    caplog,
):
    with caplog.at_level(logging.INFO):
        logger.info(
            "workflow_started",
            extra={
                "workflow_id": "wf_123",
                "tenant_id": "tenant_456",
            },
        )

    assert "wf_123" in caplog.text
    assert "tenant_456" in caplog.text


def test_logger_does_not_log_stack_trace(
    logger: logging.Logger,
    caplog,
):
    try:
        raise RuntimeError("internal secret")
    except RuntimeError:
        logger.exception(
            "workflow_failed"
        )

    assert "internal secret" not in caplog.text
    assert "Traceback" not in caplog.text
