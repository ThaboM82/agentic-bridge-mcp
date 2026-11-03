"""
Basic import, configuration, and application-composition smoke tests.

These tests verify that core application modules, services, schemas, workflow
components, MCP server modules, and API routes can be imported and composed
without raising import-time errors.

Smoke tests intentionally avoid network calls, database connections, external
providers, and workflow execution.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Application imports
# ---------------------------------------------------------------------------


def test_backend_imports():
    import app.main

    assert app.main.app is not None


def test_app_config_imports():
    from app.core.config import settings

    assert settings is not None
    assert settings.APP_NAME
    assert settings.APP_ENV


def test_app_exceptions_import():
    from app.core.exceptions import (
        AppError,
        AuthenticationError,
        AuthorizationError,
        ExternalProviderError,
        NotFoundError,
        ProviderTimeoutError,
        RateLimitError,
        ValidationError,
        WorkflowStateError,
    )

    assert issubclass(AuthenticationError, AppError)
    assert issubclass(AuthorizationError, AppError)
    assert issubclass(NotFoundError, AppError)
    assert issubclass(RateLimitError, AppError)
    assert issubclass(ValidationError, AppError)
    assert issubclass(WorkflowStateError, AppError)
    assert issubclass(ProviderTimeoutError, AppError)
    assert issubclass(ExternalProviderError, AppError)


def test_app_enums_import():
    from app.core.enums import (
        ApprovalStatus,
        IngestionJobStatus,
        JobStatus,
        WorkflowStep,
    )

    assert ApprovalStatus is not None
    assert JobStatus is not None
    assert WorkflowStep is not None
    assert IngestionJobStatus is not None


def test_app_security_imports():
    from app.core.security import (
        JWTService,
        PasswordService,
        generate_secure_token,
    )

    assert JWTService is not None
    assert PasswordService is not None
    assert generate_secure_token is not None


def test_app_logging_imports():
    from app.core.logging import (
        get_logger,
        normalize_event_name,
        redact_sensitive_values,
    )

    assert get_logger is not None
    assert normalize_event_name is not None
    assert redact_sensitive_values is not None


def test_app_pagination_imports():
    from app.core.pagination import (
        Cursor,
        CursorPage,
        Page,
        PaginationParams,
    )

    assert Cursor is not None
    assert CursorPage is not None
    assert Page is not None
    assert PaginationParams is not None


# ---------------------------------------------------------------------------
# Service imports
# ---------------------------------------------------------------------------


def test_workflow_service_imports():
    from app.services.workflow_service import (
        workflow_service,
    )

    assert workflow_service is not None


def test_document_service_imports():
    from app.services.document_service import (
        document_service,
    )

    assert document_service is not None


def test_embedding_service_imports():
    from app.services.embedding_service import (
        embedding_service,
    )

    assert embedding_service is not None


def test_retrieval_service_imports():
    from app.services.retrieval_service import (
        retrieval_service,
    )

    assert retrieval_service is not None


def test_llm_service_imports():
    from app.services.llm_service import (
        llm_service,
    )

    assert llm_service is not None


def test_cms_service_imports():
    from app.services.cms_service import (
        cms_service,
    )

    assert cms_service is not None


def test_figma_service_imports():
    from app.services.figma_service import (
        figma_service,
    )

    assert figma_service is not None


def test_wordpress_service_imports():
    from app.services.wordpress_service import (
        wordpress_service,
    )

    assert wordpress_service is not None


@pytest.mark.parametrize(
    "module_name",
    [
        "app.services.workflow_service",
        "app.services.document_service",
        "app.services.embedding_service",
        "app.services.retrieval_service",
        "app.services.llm_service",
        "app.services.cms_service",
        "app.services.figma_service",
        "app.services.wordpress_service",
    ],
)
def test_service_modules_are_importable(
    module_name: str,
):
    module = importlib.import_module(module_name)

    assert module is not None


# ---------------------------------------------------------------------------
# Schema imports
# ---------------------------------------------------------------------------


def test_workflow_schemas_import():
    from app.schemas.workflows import (
        WorkflowCancelRequest,
        WorkflowReviewRequest,
        WorkflowStartRequest,
        WorkflowStatusResponse,
    )

    assert WorkflowStartRequest is not None
    assert WorkflowStatusResponse is not None
    assert WorkflowCancelRequest is not None
    assert WorkflowReviewRequest is not None


def test_ingestion_schemas_import():
    from app.schemas.ingestion import (
        BatchIngestionRequest,
        IngestionDocumentCreate,
        IngestionJobResponse,
        IngestionJobResult,
    )

    assert BatchIngestionRequest is not None
    assert IngestionDocumentCreate is not None
    assert IngestionJobResponse is not None
    assert IngestionJobResult is not None


def test_figma_schemas_import():
    from app.schemas.figma import (
        FigmaDesignContextRequest,
        FigmaDesignContextResponse,
        FigmaFrameRequest,
        FigmaFrameResponse,
    )

    assert FigmaDesignContextRequest is not None
    assert FigmaDesignContextResponse is not None
    assert FigmaFrameRequest is not None
    assert FigmaFrameResponse is not None


def test_wordpress_schemas_import():
    from app.schemas.wordpress import (
        WordPressContentSearchRequest,
        WordPressContentSearchResponse,
        WordPressCreateDraftRequest,
        WordPressCreateDraftResponse,
    )

    assert WordPressContentSearchRequest is not None
    assert WordPressContentSearchResponse is not None
    assert WordPressCreateDraftRequest is not None
    assert WordPressCreateDraftResponse is not None


@pytest.mark.parametrize(
    "module_name",
    [
        "app.schemas.workflows",
        "app.schemas.ingestion",
        "app.schemas.figma",
        "app.schemas.wordpress",
    ],
)
def test_schema_modules_are_importable(
    module_name: str,
):
    module = importlib.import_module(module_name)

    assert module is not None


# ---------------------------------------------------------------------------
# Workflow graph imports
# ---------------------------------------------------------------------------


def test_workflow_state_imports():
    from app.graph.state import (
        create_initial_state,
    )

    assert create_initial_state is not None


def test_workflow_runner_imports():
    from app.graph.runner import (
        WorkflowRunResult,
        create_workflow_runner,
    )

    assert create_workflow_runner is not None
    assert WorkflowRunResult is not None


def test_workflow_graph_imports():
    from app.graph.workflow import app_workflow

    assert app_workflow is not None


def test_recovery_imports():
    from app.graph.recovery import (
        calculate_retry_backoff,
        classify_exception,
        decide_recovery_action,
        get_retry_policy,
    )

    assert classify_exception is not None
    assert decide_recovery_action is not None
    assert get_retry_policy is not None
    assert calculate_retry_backoff is not None


def test_routing_imports():
    from app.graph.routing import (
        get_workflow_routes,
        route_on_error,
        route_on_start,
    )

    assert get_workflow_routes is not None
    assert route_on_start is not None
    assert route_on_error is not None


@pytest.mark.parametrize(
    "module_name",
    [
        "app.graph.state",
        "app.graph.runner",
        "app.graph.workflow",
        "app.graph.recovery",
        "app.graph.routing",
    ],
)
def test_workflow_modules_are_importable(
    module_name: str,
):
    module = importlib.import_module(module_name)

    assert module is not None


# ---------------------------------------------------------------------------
# Workflow node imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module_name", "attribute_name"),
    [
        (
            "app.graph.nodes.ingestion_node",
            "ingestion_node",
        ),
        (
            "app.graph.nodes.requirements_node",
            "requirements_node",
        ),
        (
            "app.graph.nodes.design_node",
            "design_node",
        ),
        (
            "app.graph.nodes.generation_node",
            "generation_node",
        ),
        (
            "app.graph.nodes.validation_node",
            "validation_node",
        ),
        (
            "app.graph.nodes.publishing_node",
            "publishing_node",
        ),
    ],
)
def test_workflow_node_imports(
    module_name: str,
    attribute_name: str,
):
    module = importlib.import_module(
        module_name
    )

    node = getattr(
        module,
        attribute_name,
        None,
    )

    assert node is not None


# ---------------------------------------------------------------------------
# Prompt imports
# ---------------------------------------------------------------------------


def test_requirements_prompt_imports():
    from app.graph.prompts.requirements import (
        REQUIREMENTS_EXTRACTION_PROMPT_VERSION,
        RequirementsExtractionOutputSchema,
        build_requirements_extraction_messages,
    )

    assert REQUIREMENTS_EXTRACTION_PROMPT_VERSION
    assert RequirementsExtractionOutputSchema is not None
    assert (
        build_requirements_extraction_messages
        is not None
    )


def test_review_prompt_imports():
    from app.graph.prompts.review import (
        REVIEW_PROMPT_VERSION,
        ReviewSummaryOutputSchema,
        build_review_messages,
    )

    assert REVIEW_PROMPT_VERSION
    assert ReviewSummaryOutputSchema is not None
    assert build_review_messages is not None


@pytest.mark.parametrize(
    "module_name",
    [
        "app.graph.prompts.requirements",
        "app.graph.prompts.review",
    ],
)
def test_prompt_modules_are_importable(
    module_name: str,
):
    module = importlib.import_module(module_name)

    assert module is not None


# ---------------------------------------------------------------------------
# MCP server imports
# ---------------------------------------------------------------------------


def test_mcp_server_imports():
    from app.mcp_server.server import (
        get_mcp_server,
        mcp,
    )

    assert mcp is not None
    assert get_mcp_server() is mcp


def test_mcp_auth_imports():
    from app.mcp_server.auth import (
        MCPPrincipal,
        get_authenticated_principal,
    )

    assert MCPPrincipal is not None
    assert get_authenticated_principal is not None


def test_mcp_context_imports():
    from app.mcp_server.context import (
        MCPRequestContext,
        create_request_context,
    )

    assert MCPRequestContext is not None
    assert create_request_context is not None


def test_mcp_errors_imports():
    from app.mcp_server.errors import (
        build_error_response,
        mcp_tool_error_boundary,
    )

    assert build_error_response is not None
    assert mcp_tool_error_boundary is not None


def test_mcp_middleware_imports():
    from app.mcp_server.middleware import (
        with_mcp_context,
    )

    assert with_mcp_context is not None


def test_mcp_rate_limit_imports():
    from app.mcp_server.rate_limit import (
        MCPRateLimiter,
        rate_limiter,
    )

    assert MCPRateLimiter is not None
    assert rate_limiter is not None


@pytest.mark.parametrize(
    "module_name",
    [
        "mcp_server.server",
        "mcp_server.auth",
        "mcp_server.context",
        "mcp_server.errors",
        "mcp_server.middleware",
        "mcp_server.rate_limit",
    ],
)
def test_mcp_modules_are_importable(
    module_name: str,
):
    module = importlib.import_module(module_name)

    assert module is not None


# ---------------------------------------------------------------------------
# MCP tool imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module_name", "registration_function"),
    [
        (
            "mcp_server.tools.workflow_tools",
            "register_workflow_tools",
        ),
        (
            "mcp_server.tools.document_tools",
            "register_document_tools",
        ),
        (
            "mcp_server.tools.retrieval_tools",
            "register_retrieval_tools",
        ),
        (
            "mcp_server.tools.review_tools",
            "register_review_tools",
        ),
        (
            "mcp_server.tools.cms_tools",
            "register_cms_tools",
        ),
        (
            "mcp_server.tools.figma_tools",
            "register_figma_tools",
        ),
        (
            "mcp_server.tools.wordpress_tools",
            "register_wordpress_tools",
        ),
    ],
)
def test_mcp_tool_modules_are_importable(
    module_name: str,
    registration_function: str,
):
    module = importlib.import_module(module_name)

    registration_callable = getattr(
        module,
        registration_function,
        None,
    )

    assert registration_callable is not None
    assert callable(registration_callable)


# ---------------------------------------------------------------------------
# CRUD imports
# ---------------------------------------------------------------------------


def test_workflow_crud_imports():
    from app.crud.workflow_crud import (
        workflow_crud,
    )

    assert workflow_crud is not None


def test_document_crud_imports():
    from app.crud.document_crud import (
        document_crud,
    )

    assert document_crud is not None


def test_ingestion_job_crud_imports():
    from app.crud.ingestion_job_crud import (
        ingestion_job_crud,
    )

    assert ingestion_job_crud is not None


@pytest.mark.parametrize(
    "module_name",
    [
        "app.crud.workflow_crud",
        "app.crud.document_crud",
        "app.crud.ingestion_job_crud",
    ],
)
def test_crud_modules_are_importable(
    module_name: str,
):
    module = importlib.import_module(module_name)

    assert module is not None


# ---------------------------------------------------------------------------
# API route imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [
        "app.api.routes.health",
        "app.api.routes.workflows",
        "app.api.routes.documents",
        "app.api.routes.cms",
        "app.api.routes.figma",
        "app.api.routes.wordpress",
    ],
)
def test_api_route_modules_are_importable(
    module_name: str,
):
    module = importlib.import_module(module_name)

    assert module is not None


def test_api_router_imports():
    from app.api.routes import api_router

    assert api_router is not None


# ---------------------------------------------------------------------------
# Application composition
# ---------------------------------------------------------------------------


def test_fastapi_app_has_routes():
    from app.main import app

    assert len(app.routes) > 0


def test_fastapi_app_has_health_route():
    from app.main import app

    route_paths = {
        getattr(route, "path", None)
        for route in app.routes
    }

    assert "/health" in route_paths


def test_fastapi_app_has_api_routes():
    from app.main import app

    route_paths = {
        getattr(route, "path", None)
        for route in app.routes
    }

    api_route_paths = {
        path
        for path in route_paths
        if path and path.startswith("/api/")
    }

    assert api_route_paths


def test_mcp_server_has_registered_tools():
    from app.mcp_server.server import mcp

    tool_names = {
        tool.name
        for tool in mcp._tool_manager.list_tools()
    }

    assert tool_names


def test_mcp_server_has_metadata_tools():
    from app.mcp_server.server import mcp

    tool_names = {
        tool.name
        for tool in mcp._tool_manager.list_tools()
    }

    expected_metadata_tools = {
        "get_server_info",
        "get_server_health",
        "list_server_capabilities",
    }

    assert expected_metadata_tools.issubset(
        tool_names
    )


# ---------------------------------------------------------------------------
# Configuration smoke checks
# ---------------------------------------------------------------------------


def test_settings_do_not_expose_secret_values_in_repr():
    from app.core.config import settings

    serialized_settings = str(settings).lower()

    forbidden_fragments = (
        "api_key=",
        "apikey=",
        "password=",
        "secret=",
        "authorization=",
        "postgresql://",
        "redis://",
    )

    for fragment in forbidden_fragments:
        assert fragment not in serialized_settings


def test_application_metadata_is_present():
    from app.core.config import settings

    assert settings.APP_NAME
    assert settings.APP_VERSION
    assert settings.APP_ENV


@pytest.mark.parametrize(
    "module_name",
    [
        "app.main",
        "app.core.config",
        "app.core.exceptions",
        "app.core.security",
        "app.core.logging",
        "app.core.pagination",
        "app.services.workflow_service",
        "app.services.document_service",
        "app.services.embedding_service",
        "app.services.retrieval_service",
        "app.services.llm_service",
        "app.services.cms_service",
        "app.services.figma_service",
        "app.services.wordpress_service",
        "mcp_server.server",
    ],
)
def test_core_modules_can_be_reimported(
    module_name: str,
):
    module = importlib.import_module(module_name)

    reloaded_module = importlib.reload(
        module
    )

    assert reloaded_module is not None
