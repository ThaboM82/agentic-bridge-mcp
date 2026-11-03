"""
Comprehensive tests for workflow callbacks and checkpointing.

These tests cover callback metadata sanitization, sensitive-value redaction,
string bounding, synchronous and asynchronous LangChain callback behavior,
callback factory helpers, run-ID generation, memory checkpointing, backend
selection, and configuration validation.

Security rules verified by these tests:
- Callback metadata never exposes credentials, tokens, prompts, completions,
  embeddings, storage URLs, or raw provider errors.
- Callback logs contain only stable event names and bounded metadata.
- Checkpoint backend configuration fails closed for unsupported backends.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import ConfigurationError
from app.graph.callbacks import (
    AsyncWorkflowCallbackHandler,
    WorkflowCallbackHandler,
    create_async_workflow_callbacks,
    create_workflow_callbacks,
    generate_callback_run_id,
    sanitize_callback_metadata,
    sanitize_callback_value,
)
from app.graph.checkpoints import (
    create_checkpointer,
    create_memory_checkpointer,
    get_checkpoint_backend_name,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tenant_id() -> str:
    """Return a stable tenant ID."""
    return str(uuid4())


@pytest.fixture
def execution_id() -> str:
    """Return a stable workflow execution ID."""
    return f"wf_{uuid4()}"


@pytest.fixture
def callback_handler(
    execution_id: str,
    tenant_id: str,
) -> WorkflowCallbackHandler:
    """Return a synchronous workflow callback handler."""
    return WorkflowCallbackHandler(
        execution_id=execution_id,
        workflow_step="requirements",
        tenant_id=tenant_id,
    )


@pytest.fixture
def async_callback_handler(
    execution_id: str,
    tenant_id: str,
) -> AsyncWorkflowCallbackHandler:
    """Return an asynchronous workflow callback handler."""
    return AsyncWorkflowCallbackHandler(
        execution_id=execution_id,
        workflow_step="requirements",
        tenant_id=tenant_id,
    )


# ---------------------------------------------------------------------------
# Callback metadata sanitization
# ---------------------------------------------------------------------------


def test_sanitize_callback_metadata_redacts_sensitive_keys():
    metadata = {
        "api_key": "super-secret-value",
        "password": "hunter2",
        "tenant_id": "tenant-123",
        "model_name": "gpt-4.1",
    }

    safe_metadata = sanitize_callback_metadata(
        metadata
    )

    assert (
        safe_metadata["api_key"]
        == "[REDACTED]"
    )
    assert (
        safe_metadata["password"]
        == "[REDACTED]"
    )
    assert (
        safe_metadata["tenant_id"]
        == "tenant-123"
    )
    assert (
        safe_metadata["model_name"]
        == "gpt-4.1"
    )


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
        "cookie",
        "private_key",
        "database_url",
        "connection_string",
        "prompt",
        "completion",
        "messages",
        "input",
        "output",
        "embedding",
        "vector",
        "generated_code",
        "signed_url",
        "storage_url",
        "file_url",
        "download_url",
    ],
)
def test_sanitize_callback_metadata_redacts_common_sensitive_keys(
    sensitive_key: str,
):
    metadata = {
        "safe": "value",
        sensitive_key: "should-not-leak",
    }

    safe_metadata = sanitize_callback_metadata(
        metadata
    )

    assert safe_metadata["safe"] == "value"
    assert (
        safe_metadata[sensitive_key]
        == "[REDACTED]"
    )


def test_sanitize_callback_metadata_handles_nested_values():
    metadata = {
        "request": {
            "api_key": "should-not-leak",
            "model": "gpt-4.1",
        },
        "tags": ["requirements", "security"],
    }

    safe_metadata = sanitize_callback_metadata(
        metadata
    )

    assert (
        safe_metadata["request"]["api_key"]
        == "[REDACTED]"
    )
    assert (
        safe_metadata["request"]["model"]
        == "gpt-4.1"
    )
    assert safe_metadata["tags"] == [
        "requirements",
        "security",
    ]


def test_sanitize_callback_metadata_bounds_metadata_keys():
    metadata = {
        f"key_{index}": f"value-{index}"
        for index in range(100)
    }

    safe_metadata = sanitize_callback_metadata(
        metadata
    )

    assert len(safe_metadata) <= 50


def test_sanitize_callback_metadata_handles_none():
    assert (
        sanitize_callback_metadata(None)
        == {}
    )


def test_sanitize_callback_metadata_handles_empty_dict():
    assert (
        sanitize_callback_metadata({})
        == {}
    )


def test_sanitize_callback_metadata_handles_non_dict():
    assert (
        sanitize_callback_metadata(
            "not-a-dictionary"
        )
        == {}
    )


def test_sanitize_callback_metadata_handles_non_string_keys():
    metadata = {
        123: "numeric-key-value",
        "safe": "value",
    }

    safe_metadata = sanitize_callback_metadata(
        metadata
    )

    assert safe_metadata == {
        "safe": "value"
    }


# ---------------------------------------------------------------------------
# Callback value sanitization
# ---------------------------------------------------------------------------


def test_sanitize_callback_value_redacts_secret_patterns():
    value = (
        "Authorization: Bearer abcdef123456"
    )

    sanitized = sanitize_callback_value(
        value
    )

    assert "abcdef123456" not in sanitized
    assert "[REDACTED]" in sanitized


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "api_key=sk-should-not-leak",
        "apikey=sk-should-not-leak",
        "password=hunter2",
        "token=abc123",
        "Bearer abcdef123456",
        "postgresql://user:secret@localhost/db",
        "postgres://user:secret@localhost/db",
        "redis://user:secret@localhost:6379/0",
        "https://storage.example.com/signed?signature=secret",
    ],
)
def test_sanitize_callback_value_redacts_sensitive_patterns(
    unsafe_value: str,
):
    sanitized = sanitize_callback_value(
        unsafe_value
    )

    assert "should-not-leak" not in sanitized
    assert "hunter2" not in sanitized
    assert "abc123" not in sanitized
    assert "abcdef123456" not in sanitized
    assert "secret@localhost" not in sanitized
    assert "signature=secret" not in sanitized


def test_sanitize_callback_value_bounds_strings():
    sanitized = sanitize_callback_value(
        "x" * 5_000,
        max_length=100,
    )

    assert len(sanitized) <= 103
    assert sanitized.startswith("x" * 100)


def test_sanitize_callback_value_preserves_safe_strings():
    assert (
        sanitize_callback_value(
            "Build a dashboard"
        )
        == "Build a dashboard"
    )


def test_sanitize_callback_value_handles_none():
    assert (
        sanitize_callback_value(None)
        is None
    )


def test_sanitize_callback_value_handles_primitives():
    assert (
        sanitize_callback_value(True)
        is True
    )
    assert sanitize_callback_value(42) == 42
    assert (
        sanitize_callback_value(3.14)
        == 3.14
    )


def test_sanitize_callback_value_handles_uuid():
    value = uuid4()

    assert (
        sanitize_callback_value(value)
        == str(value)
    )


def test_sanitize_callback_value_handles_unserializable_objects():
    class Unsupported:
        pass

    sanitized = sanitize_callback_value(
        Unsupported()
    )

    assert (
        sanitized == "[UNSERIALIZABLE]"
    )


def test_sanitize_callback_value_handles_lists():
    sanitized = sanitize_callback_value(
        [
            "one",
            "two",
            "three",
        ]
    )

    assert sanitized == [
        "one",
        "two",
        "three",
    ]


def test_sanitize_callback_value_bounds_list_items():
    sanitized = sanitize_callback_value(
        [
            f"item-{index}"
            for index in range(100)
        ]
    )

    assert len(sanitized) <= 50


# ---------------------------------------------------------------------------
# Synchronous callback handler
# ---------------------------------------------------------------------------


def test_workflow_callback_handler_does_not_raise(
    callback_handler: WorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.INFO):
        callback_handler.on_llm_start(
            {"name": "test-model"},
            ["secret prompt"],
            run_id=uuid4(),
        )

    assert (
        "workflow_llm_started"
        in caplog.text
    )
    assert (
        "secret prompt"
        not in caplog.text
    )


def test_workflow_callback_handler_logs_model_name(
    callback_handler: WorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.INFO):
        callback_handler.on_llm_start(
            {"name": "gpt-4.1-mini"},
            ["secret prompt"],
            run_id=uuid4(),
        )

    assert "gpt-4.1-mini" in caplog.text
    assert "secret prompt" not in caplog.text


def test_workflow_callback_handler_logs_completion(
    callback_handler: WorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.INFO):
        callback_handler.on_llm_end(
            {"output": "secret completion"},
            run_id=uuid4(),
        )

    assert (
        "workflow_llm_completed"
        in caplog.text
    )
    assert (
        "secret completion"
        not in caplog.text
    )


def test_workflow_callback_handler_logs_error(
    callback_handler: WorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.ERROR):
        callback_handler.on_llm_error(
            RuntimeError("provider failed"),
            run_id=uuid4(),
        )

    assert (
        "workflow_llm_failed"
        in caplog.text
    )
    assert (
        "provider failed"
        not in caplog.text
    )


def test_workflow_callback_handler_logs_chain_start(
    callback_handler: WorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.INFO):
        callback_handler.on_chain_start(
            {"name": "workflow-chain"},
            {"secret": "input"},
            run_id=uuid4(),
        )

    assert (
        "workflow_chain_started"
        in caplog.text
    )
    assert "secret" not in caplog.text


def test_workflow_callback_handler_logs_chain_end(
    callback_handler: WorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.INFO):
        callback_handler.on_chain_end(
            {"secret": "output"},
            run_id=uuid4(),
        )

    assert (
        "workflow_chain_completed"
        in caplog.text
    )
    assert "secret" not in caplog.text


def test_workflow_callback_handler_logs_chain_error(
    callback_handler: WorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.ERROR):
        callback_handler.on_chain_error(
            RuntimeError("chain failed"),
            run_id=uuid4(),
        )

    assert (
        "workflow_chain_failed"
        in caplog.text
    )
    assert (
        "chain failed"
        not in caplog.text
    )


def test_workflow_callback_handler_logs_tool_start(
    callback_handler: WorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.INFO):
        callback_handler.on_tool_start(
            {"name": "search_documents"},
            "secret tool input",
            run_id=uuid4(),
        )

    assert (
        "workflow_tool_started"
        in caplog.text
    )
    assert (
        "secret tool input"
        not in caplog.text
    )


def test_workflow_callback_handler_logs_tool_end(
    callback_handler: WorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.INFO):
        callback_handler.on_tool_end(
            "secret tool output",
            run_id=uuid4(),
        )

    assert (
        "workflow_tool_completed"
        in caplog.text
    )
    assert (
        "secret tool output"
        not in caplog.text
    )


def test_workflow_callback_handler_logs_tool_error(
    callback_handler: WorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.ERROR):
        callback_handler.on_tool_error(
            RuntimeError("tool failed"),
            run_id=uuid4(),
        )

    assert (
        "workflow_tool_failed"
        in caplog.text
    )
    assert (
        "tool failed"
        not in caplog.text
    )


def test_workflow_callback_handler_handles_missing_serialization(
    callback_handler: WorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.INFO):
        callback_handler.on_llm_start(
            {"name": "test-model"},
            [object()],
            run_id=uuid4(),
        )

    assert (
        "workflow_llm_started"
        in caplog.text
    )


# ---------------------------------------------------------------------------
# Asynchronous callback handler
# ---------------------------------------------------------------------------


async def test_async_workflow_callback_handler_does_not_raise(
    async_callback_handler: AsyncWorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.INFO):
        await async_callback_handler.on_llm_error(
            RuntimeError("provider failed"),
            run_id=uuid4(),
        )

    assert (
        "workflow_llm_failed"
        in caplog.text
    )
    assert (
        "provider failed"
        not in caplog.text
    )


async def test_async_workflow_callback_handler_logs_start(
    async_callback_handler: AsyncWorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.INFO):
        await async_callback_handler.on_llm_start(
            {"name": "gpt-4.1-mini"},
            ["secret prompt"],
            run_id=uuid4(),
        )

    assert (
        "workflow_llm_started"
        in caplog.text
    )
    assert (
        "secret prompt"
        not in caplog.text
    )


async def test_async_workflow_callback_handler_logs_end(
    async_callback_handler: AsyncWorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.INFO):
        await async_callback_handler.on_llm_end(
            {"output": "secret completion"},
            run_id=uuid4(),
        )

    assert (
        "workflow_llm_completed"
        in caplog.text
    )
    assert (
        "secret completion"
        not in caplog.text
    )


async def test_async_workflow_callback_handler_logs_chain_events(
    async_callback_handler: AsyncWorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.INFO):
        await async_callback_handler.on_chain_start(
            {"name": "workflow-chain"},
            {"secret": "input"},
            run_id=uuid4(),
        )

        await async_callback_handler.on_chain_end(
            {"secret": "output"},
            run_id=uuid4(),
        )

    assert (
        "workflow_chain_started"
        in caplog.text
    )
    assert (
        "workflow_chain_completed"
        in caplog.text
    )
    assert "secret" not in caplog.text


async def test_async_workflow_callback_handler_logs_tool_events(
    async_callback_handler: AsyncWorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.INFO):
        await async_callback_handler.on_tool_start(
            {"name": "search_documents"},
            "secret tool input",
            run_id=uuid4(),
        )

        await async_callback_handler.on_tool_end(
            "secret tool output",
            run_id=uuid4(),
        )

    assert (
        "workflow_tool_started"
        in caplog.text
    )
    assert (
        "workflow_tool_completed"
        in caplog.text
    )
    assert (
        "secret tool input"
        not in caplog.text
    )
    assert (
        "secret tool output"
        not in caplog.text
    )


async def test_async_workflow_callback_handler_handles_unserializable_input(
    async_callback_handler: AsyncWorkflowCallbackHandler,
    caplog,
):
    with caplog.at_level(logging.INFO):
        await async_callback_handler.on_llm_start(
            {"name": "test-model"},
            [object()],
            run_id=uuid4(),
        )

    assert (
        "workflow_llm_started"
        in caplog.text
    )


# ---------------------------------------------------------------------------
# Callback factories
# ---------------------------------------------------------------------------


def test_create_workflow_callbacks():
    callbacks = create_workflow_callbacks(
        execution_id="wf_123",
        workflow_step="validation",
    )

    assert len(callbacks) == 1
    assert isinstance(
        callbacks[0],
        WorkflowCallbackHandler,
    )


def test_create_workflow_callbacks_with_tenant():
    tenant_id = str(uuid4())

    callbacks = create_workflow_callbacks(
        execution_id="wf_123",
        workflow_step="validation",
        tenant_id=tenant_id,
    )

    assert callbacks[0].tenant_id == tenant_id


def test_create_async_workflow_callbacks():
    callbacks = create_async_workflow_callbacks(
        execution_id="wf_123",
        workflow_step="validation",
    )

    assert len(callbacks) == 1
    assert isinstance(
        callbacks[0],
        AsyncWorkflowCallbackHandler,
    )


def test_create_async_workflow_callbacks_with_tenant():
    tenant_id = str(uuid4())

    callbacks = create_async_workflow_callbacks(
        execution_id="wf_123",
        workflow_step="validation",
        tenant_id=tenant_id,
    )

    assert callbacks[0].tenant_id == tenant_id


# ---------------------------------------------------------------------------
# Callback run IDs
# ---------------------------------------------------------------------------


def test_generate_callback_run_id():
    run_id = generate_callback_run_id()

    assert run_id.startswith("callback_")
    assert (
        run_id != generate_callback_run_id()
    )


def test_generate_callback_run_id_is_url_safe():
    run_id = generate_callback_run_id()

    assert " " not in run_id
    assert "/" not in run_id
    assert "+" not in run_id


# ---------------------------------------------------------------------------
# Memory checkpointing
# ---------------------------------------------------------------------------


def test_create_memory_checkpointer():
    checkpointer = create_memory_checkpointer()

    assert checkpointer is not None


def test_create_memory_checkpointer_returns_new_instance():
    first_checkpointer = (
        create_memory_checkpointer()
    )

    second_checkpointer = (
        create_memory_checkpointer()
    )

    assert first_checkpointer is not None
    assert second_checkpointer is not None


def test_create_checkpointer_memory_backend():
    checkpointer = create_checkpointer(
        backend="memory"
    )

    assert checkpointer is not None


def test_create_checkpointer_rejects_unknown_backend():
    with pytest.raises(ConfigurationError):
        create_checkpointer(
            backend="redis"
        )


def test_create_checkpointer_rejects_empty_backend():
    with pytest.raises(ConfigurationError):
        create_checkpointer(backend="")


def test_create_checkpointer_rejects_none_backend():
    with pytest.raises(ConfigurationError):
        create_checkpointer(backend=None)


def test_create_checkpointer_rejects_non_string_backend():
    with pytest.raises(ConfigurationError):
        create_checkpointer(backend=123)


# ---------------------------------------------------------------------------
# Checkpoint backend metadata
# ---------------------------------------------------------------------------


def test_get_checkpoint_backend_name():
    assert (
        get_checkpoint_backend_name(
            backend="memory"
        )
        == "memory"
    )

    with pytest.raises(ConfigurationError):
        get_checkpoint_backend_name(
            backend="unknown"
        )


def test_get_checkpoint_backend_name_rejects_empty_backend():
    with pytest.raises(ConfigurationError):
        get_checkpoint_backend_name(
            backend=""
        )


def test_get_checkpoint_backend_name_rejects_none_backend():
    with pytest.raises(ConfigurationError):
        get_checkpoint_backend_name(
            backend=None
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
