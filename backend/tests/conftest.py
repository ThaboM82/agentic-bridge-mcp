"""
Shared pytest configuration and fixtures for the AgenticBridge backend.

This conftest:
- Ensures the backend root is importable in every test run.
- Provides deterministic UUID, tenant, user, execution, and brief fixtures.
- Provides reusable workflow-state and MCP-principal fixtures.
- Provides async helpers and event-loop configuration.
- Registers custom markers for test selection.
- Prevents accidental network access and external side effects.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Import-path configuration
# ---------------------------------------------------------------------------

BACKEND_ROOT = Path(
    __file__
).resolve().parents[1]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(
        0, str(BACKEND_ROOT)
    )

# Ensure tests never accidentally read local developer secrets.
os.environ.setdefault(
    "ENVIRONMENT", "test"
)
os.environ.setdefault(
    "APP_ENV", "test"
)
os.environ.setdefault(
    "LOG_LEVEL", "INFO"
)
os.environ.setdefault(
    "DISABLE_EXTERNAL_CALLS", "true"
)

# ---------------------------------------------------------------------------
# Pytest configuration
# ---------------------------------------------------------------------------


def pytest_configure(
    config: pytest.Config,
) -> None:
    """Register project-specific test markers."""
    config.addinivalue_line(
        "markers",
        "unit: Fast isolated tests with no external dependencies.",
    )
    config.addinivalue_line(
        "markers",
        "integration: Tests that exercise multiple internal modules together.",
    )
    config.addinivalue_line(
        "markers",
        "security: Tests focused on authentication, authorization, sanitization, or secret prevention.",
    )
    config.addinivalue_line(
        "markers",
        "workflow: Tests for workflow state, graph routing, nodes, or execution.",
    )
    config.addinivalue_line(
        "markers",
        "mcp: Tests for MCP server modules, tools, schemas, or transports.",
    )
    config.addinivalue_line(
        "markers",
        "slow: Tests that are comparatively expensive and can be excluded during fast local runs.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Apply conservative default markers based on test location."""
    for item in items:
        module_path = str(
            item.fspath
        )

        if "/tests/unit/" in module_path:
            item.add_marker(
                pytest.mark.unit
            )

        if (
            "/tests/integration/"
            in module_path
        ):
            item.add_marker(
                pytest.mark.integration
            )

        if "/tests/security/" in module_path:
            item.add_marker(
                pytest.mark.security
            )

        if "/tests/graph/" in module_path:
            item.add_marker(
                pytest.mark.workflow
            )

        if (
            "/tests/mcp_server/"
            in module_path
        ):
            item.add_marker(
                pytest.mark.mcp
            )


# ---------------------------------------------------------------------------
# Event-loop configuration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    """Return the event-loop policy used by async tests."""
    return asyncio.get_event_loop_policy()


@pytest_asyncio.fixture
async def event_loop() -> AsyncIterator[asyncio.AbstractEventLoop]:
    """Provide a function-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    yield loop

    loop.close()
    asyncio.set_event_loop(
        asyncio.new_event_loop()
    )


# ---------------------------------------------------------------------------
# Identifier fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def uuid_value() -> UUID:
    """Return a random UUID object."""
    return uuid4()


@pytest.fixture
def uuid_string() -> str:
    """Return a random UUID string."""
    return str(uuid4())


@pytest.fixture
def tenant_id() -> str:
    """Return a stable tenant ID."""
    return str(uuid4())


@pytest.fixture
def user_id() -> str:
    """Return a stable user ID."""
    return str(uuid4())


@pytest.fixture
def brief_id() -> str:
    """Return a stable brief ID."""
    return f"brief_{uuid4()}"


@pytest.fixture
def execution_id() -> str:
    """Return a stable workflow execution ID."""
    return f"wf_{uuid4()}"


@pytest.fixture
def document_id() -> str:
    """Return a stable document ID."""
    return str(uuid4())


@pytest.fixture
def embedding_id() -> str:
    """Return a stable embedding ID."""
    return str(uuid4())


@pytest.fixture
def reviewer_id() -> str:
    """Return a stable reviewer ID."""
    return str(uuid4())


@pytest.fixture
def request_id() -> str:
    """Return a stable MCP request ID."""
    return str(uuid4())


@pytest.fixture
def correlation_id() -> str:
    """Return a stable MCP correlation ID."""
    return str(uuid4())


# ---------------------------------------------------------------------------
# Workflow-state fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_brief_text() -> str:
    """Return a representative raw brief."""
    return (
        "Build an accessible internal analytics "
        "dashboard with key metrics and export."
    )


@pytest.fixture
def workflow_state(
    execution_id: str,
    brief_id: str,
    tenant_id: str,
    user_id: str,
    raw_brief_text: str,
) -> dict[str, Any]:
    """Return a valid initial workflow state."""
    from app.graph.state import (
        create_initial_state,
    )

    return create_initial_state(
        execution_id=execution_id,
        brief_id=brief_id,
        raw_brief_text=raw_brief_text,
        tenant_id=tenant_id,
        user_id=user_id,
        approval_required=True,
    )


@pytest.fixture
def workflow_state_factory(
    tenant_id: str,
    user_id: str,
    raw_brief_text: str,
) -> Callable[..., dict[str, Any]]:
    """Return a factory for creating customized workflow states."""

    def _factory(
        **overrides: Any,
    ) -> dict[str, Any]:
        from app.graph.state import (
            create_initial_state,
        )

        state = create_initial_state(
            execution_id=f"wf_{uuid4()}",
            brief_id=f"brief_{uuid4()}",
            raw_brief_text=raw_brief_text,
            tenant_id=tenant_id,
            user_id=user_id,
            approval_required=True,
        )

        return {
            **state,
            **overrides,
        }

    return _factory


@pytest.fixture
def recoverable_error_state(
    workflow_state: dict[str, Any],
) -> dict[str, Any]:
    """Return a workflow state containing a recoverable error."""
    from app.core.enums import (
        WorkflowStep,
    )
    from app.graph.state import (
        set_error,
    )

    return set_error(
        workflow_state,
        message="Provider timeout.",
        error_step=WorkflowStep.DESIGN,
        error_code="provider_timeout",
        recoverable=True,
    )


@pytest.fixture
def blocking_error_state(
    workflow_state: dict[str, Any],
) -> dict[str, Any]:
    """Return a workflow state containing a non-recoverable error."""
    from app.core.enums import (
        WorkflowStep,
    )
    from app.graph.state import (
        set_error,
    )

    return set_error(
        workflow_state,
        message="Invalid workflow state.",
        error_step=WorkflowStep.VALIDATION,
        error_code="invalid_state",
        recoverable=False,
    )


# ---------------------------------------------------------------------------
# MCP principal and context fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def principal(
    user_id: str,
    tenant_id: str,
) -> Any:
    """Return a fully authorized MCP principal."""
    from app.mcp_server.auth import (
        MCPPrincipal,
    )

    return MCPPrincipal(
        user_id=user_id,
        tenant_id=tenant_id,
        role="developer",
        scopes=frozenset(
            {
                "workflows:read",
                "workflows:write",
                "workflows:cancel",
                "workflows:review",
                "documents:read",
            }
        ),
    )


@pytest.fixture
def viewer_principal(
    user_id: str,
    tenant_id: str,
) -> Any:
    """Return a read-only MCP principal."""
    from app.mcp_server.auth import (
        MCPPrincipal,
    )

    return MCPPrincipal(
        user_id=user_id,
        tenant_id=tenant_id,
        role="viewer",
        scopes=frozenset(
            {"workflows:read"}
        ),
    )


@pytest.fixture
def mcp_context(
    principal: Any,
) -> Any:
    """Return an authenticated MCP request context."""
    from app.mcp_server.context import (
        create_request_context,
    )

    return create_request_context(
        principal=principal,
    )


@pytest.fixture
def mcp_context_factory(
    principal: Any,
) -> Callable[..., Any]:
    """Return a factory for customized MCP request contexts."""
    from app.mcp_server.context import (
        create_request_context,
    )

    def _factory(
        **overrides: Any,
    ) -> Any:
        return create_request_context(
            principal=principal,
            **overrides,
        )

    return _factory


# ---------------------------------------------------------------------------
# Repository and service fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_workflow_repository() -> Any:
    """Return a fresh in-memory workflow repository."""
    from app.services.workflow_service import (
        InMemoryWorkflowRepository,
    )

    return InMemoryWorkflowRepository()


@pytest.fixture
def fake_workflow() -> Any:
    """Return a deterministic fake workflow graph."""

    class FakeWorkflow:
        async def ainvoke(
            self,
            state: dict[str, Any],
        ) -> dict[str, Any]:
            return {
                **state,
                "current_step": "completed",
                "extracted_requirements": [
                    "User can reset password"
                ],
            }

    return FakeWorkflow()


@pytest.fixture
def workflow_service(
    fake_workflow: Any,
    in_memory_workflow_repository: Any,
) -> Any:
    """Return a workflow service backed by deterministic doubles."""
    from app.services.workflow_service import (
        WorkflowService,
    )

    return WorkflowService(
        workflow=fake_workflow,
        repository=in_memory_workflow_repository,
    )


@pytest.fixture
def workflow_payload(
    brief_id: str,
    tenant_id: str,
    user_id: str,
    raw_brief_text: str,
) -> dict[str, Any]:
    """Return a valid workflow-start payload."""
    return {
        "brief_id": brief_id,
        "raw_brief_text": raw_brief_text,
        "tenant_id": tenant_id,
        "user_id": user_id,
    }


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def async_mock() -> AsyncMock:
    """Return a configured asynchronous mock."""
    return AsyncMock()


@pytest.fixture
def sync_mock() -> MagicMock:
    """Return a configured synchronous mock."""
    return MagicMock()


@pytest.fixture
def fake_session() -> MagicMock:
    """Return a database-session-like mock for repository tests."""
    session = MagicMock()
    session.execute = AsyncMock()

    return session


# ---------------------------------------------------------------------------
# Security-oriented fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sensitive_metadata() -> dict[str, Any]:
    """Return metadata containing common sensitive keys."""
    return {
        "api_key": "should-not-leak",
        "apikey": "should-not-leak",
        "password": "hunter2",
        "secret": "should-not-leak",
        "token": "should-not-leak",
        "authorization": "Bearer should-not-leak",
        "credential": "should-not-leak",
        "private_key": "should-not-leak",
        "database_url": "postgresql://user:password@localhost/db",
        "connection_string": "postgresql://user:password@localhost/db",
        "prompt": "secret user prompt",
        "completion": "secret model completion",
        "messages": [
            "secret user prompt"
        ],
        "embedding": [0.1, 0.2, 0.3],
        "vector": [0.1, 0.2, 0.3],
        "generated_code": "print('secret')",
        "signed_url": "https://storage.example.com/signed?signature=secret",
        "storage_url": "s3://bucket/secret-object",
        "file_url": "https://files.example.com/secret",
        "download_url": "https://files.example.com/download",
    }


@pytest.fixture
def safe_metadata() -> dict[str, Any]:
    """Return metadata containing only non-sensitive values."""
    return {
        "tenant_id": "tenant_123",
        "model_name": "gpt-4.1-mini",
        "workflow_step": "requirements",
        "source": "confluence",
        "tags": ["design", "accessibility"],
    }


@pytest.fixture
def secret_patterns() -> list[str]:
    """Return strings that must never appear in sanitized output."""
    return [
        "should-not-leak",
        "hunter2",
        "sk-should-not-leak",
        "abcdef123456",
        "postgresql://user:password@localhost/db",
        "postgres://user:password@localhost/db",
        "redis://user:password@localhost:6379/0",
        "signature=secret",
    ]


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def assert_no_secrets() -> Callable[..., None]:
    """Return a helper that asserts sensitive fragments are absent."""

    def _assert_no_secrets(
        value: Any,
        forbidden_fragments: list[
            str
        ]
        | None = None,
    ) -> None:
        serialized_value = str(
            value
        ).lower()

        fragments = (
            forbidden_fragments
            or [
                "api_key",
                "apikey",
                "password",
                "secret",
                "token",
                "authorization",
                "postgresql://",
                "postgres://",
                "redis://",
            ]
        )

        for fragment in fragments:
            assert (
                fragment.lower()
                not in serialized_value
            ), (
                f"Sensitive fragment "
                f"{fragment!r} leaked into "
                f"serialized value."
            )

    return _assert_no_secrets


@pytest.fixture
def assert_valid_uuid() -> Callable[
    [str], None
]:
    """Return a helper that asserts a value is a UUID string."""

    def _assert_valid_uuid(
        value: str,
    ) -> None:
        assert isinstance(
            value, str
        )
        UUID(value)

    return _assert_valid_uuid


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_asyncio_policy() -> Iterator[
    None
]:
    """Ensure each test starts with a clean asyncio policy."""
    original_policy = (
        asyncio.get_event_loop_policy()
    )

    yield

    asyncio.set_event_loop_policy(
        original_policy
    )
