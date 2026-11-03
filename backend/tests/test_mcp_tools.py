"""
Integration-style tests for AgenticBridge MCP tools.

These tests verify tool registration, authentication, authorization, tenant
isolation, argument validation, rate limiting, and safe result handling across
the workflow, document, retrieval, review, CMS, Figma, and WordPress tool
families.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.mcp_server.auth import MCPPrincipal
from app.mcp_server.server import mcp
from app.mcp_server.tools.cms_tools import register_cms_tools
from app.mcp_server.tools.document_tools import (
    register_document_tools,
)
from app.mcp_server.tools.figma_tools import (
    register_figma_tools,
)
from app.mcp_server.tools.retrieval_tools import (
    register_retrieval_tools,
)
from app.mcp_server.tools.review_tools import (
    register_review_tools,
)
from app.mcp_server.tools.workflow_tools import (
    register_workflow_tools,
)
from app.mcp_server.tools.wordpress_tools import (
    register_wordpress_tools,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def principal() -> MCPPrincipal:
    """Return an authenticated MCP principal with broad read/write scopes."""
    return MCPPrincipal(
        user_id=str(uuid4()),
        tenant_id=str(uuid4()),
        role="developer",
        scopes=frozenset(
            {
                "workflows:read",
                "workflows:write",
                "workflows:cancel",
                "workflows:review",
                "documents:read",
                "retrieval:read",
                "cms:read",
                "cms:write",
                "cms:publish",
                "figma:read",
                "wordpress:read",
                "wordpress:write",
                "wordpress:publish",
            }
        ),
    )


@pytest.fixture
def readonly_principal() -> MCPPrincipal:
    """Return an authenticated MCP principal with read-only scopes."""
    return MCPPrincipal(
        user_id=str(uuid4()),
        tenant_id=str(uuid4()),
        role="viewer",
        scopes=frozenset(
            {
                "workflows:read",
                "documents:read",
                "retrieval:read",
                "cms:read",
                "figma:read",
                "wordpress:read",
            }
        ),
    )


@pytest.fixture
def mock_workflow_service():
    """Patch WorkflowService used by workflow MCP tools."""
    with patch(
        "app.services.workflow_service.WorkflowService"
    ) as mock_service_class:
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        yield mock_service


@pytest.fixture
def mock_document_service():
    """Patch DocumentService used by document MCP tools."""
    with patch(
        "app.services.document_service.DocumentService"
    ) as mock_service_class:
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        yield mock_service


@pytest.fixture
def mock_retrieval_service():
    """Patch RetrievalService used by retrieval MCP tools."""
    with patch(
        "app.services.retrieval_service.RetrievalService"
    ) as mock_service_class:
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        yield mock_service


@pytest.fixture
def mock_cms_service():
    """Patch CMSService used by CMS MCP tools."""
    with patch(
        "app.services.cms_service.CMSService"
    ) as mock_service_class:
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        yield mock_service


@pytest.fixture
def mock_figma_service():
    """Patch FigmaService used by Figma MCP tools."""
    with patch(
        "app.services.figma_service.FigmaService"
    ) as mock_service_class:
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        yield mock_service


@pytest.fixture
def mock_wordpress_service():
    """Patch WordPressService used by WordPress MCP tools."""
    with patch(
        "app.services.wordpress_service.WordPressService"
    ) as mock_service_class:
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        yield mock_service


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_all_tool_families_register_successfully():
    register_workflow_tools(mcp)
    register_document_tools(mcp)
    register_retrieval_tools(mcp)
    register_review_tools(mcp)
    register_cms_tools(mcp)
    register_figma_tools(mcp)
    register_wordpress_tools(mcp)

    tool_names = [
        tool.name
        for tool in mcp._tool_manager.list_tools()
    ]

    expected_tools = {
        # Workflow
        "start_generation_workflow",
        "get_workflow_status",
        "list_workflow_executions",
        "cancel_workflow",
        "retrieve_workflow_context",
        # Review
        "approve_workflow",
        "reject_workflow",
        "request_workflow_changes",
        # Documents
        "get_document_summary",
        # Retrieval
        "search_design_knowledge",
        "search_domain_knowledge",
        # CMS
        "search_cms_content",
        "create_cms_draft",
        "update_cms_draft",
        "request_cms_publication",
        # Figma
        "get_figma_design_context",
        "get_figma_frame_summary",
        # WordPress
        "search_wordpress_content",
        "create_wordpress_draft",
        "update_wordpress_draft",
        "request_wordpress_publication",
    }

    missing_tools = expected_tools - set(tool_names)

    assert not missing_tools


def test_server_metadata_tools_are_registered():
    tool_names = [
        tool.name
        for tool in mcp._tool_manager.list_tools()
    ]

    assert "get_server_info" in tool_names
    assert "get_server_health" in tool_names
    assert "list_server_capabilities" in tool_names


# ---------------------------------------------------------------------------
# Workflow tools
# ---------------------------------------------------------------------------


async def test_start_generation_workflow_requires_authentication():
    with pytest.raises(Exception):
        await mcp.call_tool(
            "start_generation_workflow",
            {
                "brief_text": "Build a dashboard",
            },
        )


async def test_start_generation_workflow_rejects_empty_brief(
    principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "start_generation_workflow",
            {
                "brief_text": "",
                "principal": principal,
            },
        )


async def test_start_generation_workflow_returns_safe_result(
    principal,
    mock_workflow_service,
):
    execution_id = f"wf_{uuid4()}"

    mock_workflow_service.start_workflow.return_value = {
        "execution_id": execution_id,
        "status": "queued",
        "current_step": "intake",
    }

    result = await mcp.call_tool(
        "start_generation_workflow",
        {
            "brief_text": "Build a dashboard",
            "principal": principal,
        },
    )

    assert result["execution_id"] == execution_id
    assert result["status"] == "queued"


async def test_get_workflow_status_enforces_tenant_isolation(
    principal,
    mock_workflow_service,
):
    other_tenant_id = str(uuid4())

    with pytest.raises(Exception):
        await mcp.call_tool(
            "get_workflow_status",
            {
                "execution_id": f"wf_{uuid4()}",
                "tenant_id": other_tenant_id,
                "principal": principal,
            },
        )


async def test_cancel_workflow_requires_cancel_scope(
    readonly_principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "cancel_workflow",
            {
                "execution_id": f"wf_{uuid4()}",
                "principal": readonly_principal,
            },
        )


# ---------------------------------------------------------------------------
# Review tools
# ---------------------------------------------------------------------------


async def test_approve_workflow_requires_review_scope(
    readonly_principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "approve_workflow",
            {
                "execution_id": f"wf_{uuid4()}",
                "principal": readonly_principal,
            },
        )


async def test_reject_workflow_requires_review_scope(
    readonly_principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "reject_workflow",
            {
                "execution_id": f"wf_{uuid4()}",
                "principal": readonly_principal,
            },
        )


async def test_request_workflow_changes_requires_review_scope(
    readonly_principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "request_workflow_changes",
            {
                "execution_id": f"wf_{uuid4()}",
                "change_requests": "Add authentication",
                "principal": readonly_principal,
            },
        )


# ---------------------------------------------------------------------------
# Document tools
# ---------------------------------------------------------------------------


async def test_get_document_summary_requires_authentication():
    with pytest.raises(Exception):
        await mcp.call_tool(
            "get_document_summary",
            {
                "document_id": f"doc_{uuid4()}",
            },
        )


async def test_get_document_summary_rejects_invalid_id(
    principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "get_document_summary",
            {
                "document_id": "",
                "principal": principal,
            },
        )


async def test_get_document_summary_returns_safe_metadata(
    principal,
    mock_document_service,
):
    document_id = f"doc_{uuid4()}"

    mock_document_service.get_document_summary.return_value = {
        "document_id": document_id,
        "title": "Product brief",
        "status": "processed",
        "citation_count": 3,
    }

    result = await mcp.call_tool(
        "get_document_summary",
        {
            "document_id": document_id,
            "principal": principal,
        },
    )

    assert result["document_id"] == document_id
    assert result["status"] == "processed"
    assert "content" not in result
    assert "signed_url" not in result
    assert "storage_url" not in result


# ---------------------------------------------------------------------------
# Retrieval tools
# ---------------------------------------------------------------------------


async def test_search_design_knowledge_requires_authentication():
    with pytest.raises(Exception):
        await mcp.call_tool(
            "search_design_knowledge",
            {
                "query": "dashboard layout",
            },
        )


async def test_search_design_knowledge_rejects_empty_query(
    principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "search_design_knowledge",
            {
                "query": "",
                "principal": principal,
            },
        )


async def test_search_design_knowledge_returns_sanitized_results(
    principal,
    mock_retrieval_service,
):
    mock_retrieval_service.search_design_knowledge.return_value = {
        "results": [
            {
                "title": "Dashboard layout",
                "summary": "Use a responsive grid.",
                "score": 0.92,
            }
        ],
        "query": "dashboard layout",
    }

    result = await mcp.call_tool(
        "search_design_knowledge",
        {
            "query": "dashboard layout",
            "principal": principal,
        },
    )

    assert result["query"] == "dashboard layout"
    assert result["results"][0]["title"] == "Dashboard layout"
    assert "embedding" not in result
    assert "vector" not in result


async def test_search_domain_knowledge_rejects_invalid_limit(
    principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "search_domain_knowledge",
            {
                "query": "authentication",
                "limit": 10_000,
                "principal": principal,
            },
        )


# ---------------------------------------------------------------------------
# CMS tools
# ---------------------------------------------------------------------------


async def test_create_cms_draft_requires_write_scope(
    readonly_principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "create_cms_draft",
            {
                "title": "Draft title",
                "content": "Draft content",
                "principal": readonly_principal,
            },
        )


async def test_create_cms_draft_rejects_empty_title(
    principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "create_cms_draft",
            {
                "title": "",
                "content": "Draft content",
                "principal": principal,
            },
        )


async def test_create_cms_draft_returns_safe_result(
    principal,
    mock_cms_service,
):
    content_id = f"cms_{uuid4()}"

    mock_cms_service.create_draft.return_value = {
        "content_id": content_id,
        "title": "Draft title",
        "status": "draft",
    }

    result = await mcp.call_tool(
        "create_cms_draft",
        {
            "title": "Draft title",
            "content": "Draft content",
            "principal": principal,
        },
    )

    assert result["content_id"] == content_id
    assert result["status"] == "draft"


async def test_request_cms_publication_requires_publish_scope(
    readonly_principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "request_cms_publication",
            {
                "content_id": f"cms_{uuid4()}",
                "principal": readonly_principal,
            },
        )


# ---------------------------------------------------------------------------
# Figma tools
# ---------------------------------------------------------------------------


async def test_get_figma_design_context_requires_authentication():
    with pytest.raises(Exception):
        await mcp.call_tool(
            "get_figma_design_context",
            {
                "figma_file_key": "figma-file-key",
            },
        )


async def test_get_figma_design_context_requires_read_scope(
    principal,
    mock_figma_service,
):
    mock_figma_service.get_design_context.return_value = {
        "figma_file_key": "figma-file-key",
        "items": [],
        "truncated": False,
    }

    result = await mcp.call_tool(
        "get_figma_design_context",
        {
            "figma_file_key": "figma-file-key",
            "principal": principal,
        },
    )

    assert result["figma_file_key"] == "figma-file-key"
    assert result["items"] == []


async def test_get_figma_frame_summary_rejects_empty_node_id(
    principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "get_figma_frame_summary",
            {
                "figma_file_key": "figma-file-key",
                "figma_node_id": "",
                "principal": principal,
            },
        )


# ---------------------------------------------------------------------------
# WordPress tools
# ---------------------------------------------------------------------------


async def test_search_wordpress_content_requires_authentication():
    with pytest.raises(Exception):
        await mcp.call_tool(
            "search_wordpress_content",
            {
                "query": "release notes",
            },
        )


async def test_search_wordpress_content_rejects_empty_query(
    principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "search_wordpress_content",
            {
                "query": "",
                "principal": principal,
            },
        )


async def test_create_wordpress_draft_requires_write_scope(
    readonly_principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "create_wordpress_draft",
            {
                "title": "Release notes",
                "content": "Initial draft",
                "principal": readonly_principal,
            },
        )


async def test_create_wordpress_draft_returns_safe_result(
    principal,
    mock_wordpress_service,
):
    wordpress_content_id = f"wp_{uuid4()}"

    mock_wordpress_service.create_draft.return_value = {
        "wordpress_content_id": wordpress_content_id,
        "title": "Release notes",
        "status": "draft",
    }

    result = await mcp.call_tool(
        "create_wordpress_draft",
        {
            "title": "Release notes",
            "content": "Initial draft",
            "principal": principal,
        },
    )

    assert (
        result["wordpress_content_id"]
        == wordpress_content_id
    )
    assert result["status"] == "draft"


async def test_request_wordpress_publication_requires_publish_scope(
    readonly_principal,
):
    with pytest.raises(Exception):
        await mcp.call_tool(
            "request_wordpress_publication",
            {
                "wordpress_content_id": (
                    f"wp_{uuid4()}"
                ),
                "principal": readonly_principal,
            },
        )


# ---------------------------------------------------------------------------
# Security and result sanitization
# ---------------------------------------------------------------------------


async def test_tool_results_do_not_expose_sensitive_keys(
    principal,
    mock_workflow_service,
):
    execution_id = f"wf_{uuid4()}"

    mock_workflow_service.start_workflow.return_value = {
        "execution_id": execution_id,
        "status": "queued",
        "api_key": "should-not-leak",
        "prompt": "should-not-leak",
        "storage_url": "https://example.com/secret",
    }

    result = await mcp.call_tool(
        "start_generation_workflow",
        {
            "brief_text": "Build a dashboard",
            "principal": principal,
        },
    )

    serialized_result = str(result).lower()

    assert "should-not-leak" not in serialized_result
    assert "https://example.com/secret" not in serialized_result


async def test_tool_results_are_json_safe(
    principal,
    mock_workflow_service,
):
    execution_id = f"wf_{uuid4()}"

    mock_workflow_service.start_workflow.return_value = {
        "execution_id": execution_id,
        "status": "queued",
        "generated_code_files": {
            "app/main.py": "print('secret')"
        },
    }

    result = await mcp.call_tool(
        "start_generation_workflow",
        {
            "brief_text": "Build a dashboard",
            "principal": principal,
        },
    )

    serialized_result = str(result)

    assert "print('secret')" not in serialized_result


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


async def test_start_generation_workflow_is_rate_limited(
    principal,
    mock_workflow_service,
):
    mock_workflow_service.start_workflow.return_value = {
        "execution_id": f"wf_{uuid4()}",
        "status": "queued",
    }

    from app.mcp_server.rate_limit import (
        MCPRateLimiter,
        rate_limiter,
    )

    original_limiter = rate_limiter

    test_limiter = MCPRateLimiter(
        default_requests_per_minute=100,
        default_burst=1,
    )

    test_limiter.configure_tool(
        tool_name="start_generation_workflow",
        requests_per_minute=10,
        burst=1,
        window_seconds=60,
    )

    with patch(
        "mcp_server.middleware.rate_limiter",
        test_limiter,
    ):
        await mcp.call_tool(
            "start_generation_workflow",
            {
                "brief_text": "Build dashboard one",
                "principal": principal,
            },
        )

        with pytest.raises(Exception):
            await mcp.call_tool(
                "start_generation_workflow",
                {
                    "brief_text": (
                        "Build dashboard two"
                    ),
                    "principal": principal,
                },
            )

    rate_limiter = original_limiter


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
