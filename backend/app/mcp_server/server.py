"""
AgenticBridge MCP server.

This module exposes a secure Model Context Protocol (MCP) interface to the
AgenticBridge workflow platform. MCP clients can start workflows, inspect
execution status, retrieve workflow-safe context, inspect document metadata,
search CMS content, inspect Figma design context, manage WordPress drafts,
and resolve human-review interrupts.

Security and architecture rules:
- Authenticate and authorize every tool invocation.
- Enforce tenant isolation on every workflow, document, CMS, retrieval,
  Figma, and WordPress operation.
- Treat all MCP tool arguments as untrusted input.
- Validate and bound every argument before it reaches a service or workflow.
- Never return raw workflow state, generated file contents, prompts, model
  completions, embeddings, credentials, database URLs, or customer PII.
- Never return raw Figma API payloads, image URLs, source file contents, or
  provider credentials.
- Never return raw WordPress API payloads, credentials, database URLs,
  storage URLs, or customer PII.
- Never log tool arguments or results that may contain sensitive material.
- Keep this server thin: business logic belongs in application services and
  workflow nodes.
- Human approval remains required before publication when configured.
"""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import FastMCP

from app.core.config import settings
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


logger = logging.getLogger("agentic_bridge.mcp_server")

MCP_SERVER_NAME = "AgenticBridge"
MCP_SERVER_VERSION = "1.2.0"

AVAILABLE_TOOL_FAMILIES: tuple[str, ...] = (
    "workflow",
    "documents",
    "retrieval",
    "review",
    "cms",
    "figma",
    "wordpress",
)

SECURITY_BOUNDARIES: tuple[str, ...] = (
    "tenant_isolation",
    "scope_authorization",
    "rate_limiting",
    "request_correlation",
    "audit_logging",
    "result_sanitization",
    "human_approval_for_publication",
)

MCP_SERVER_INSTRUCTIONS = """
AgenticBridge helps authorized users turn product briefs, source documents,
design context, and CMS content into reviewed, validated implementation
artifacts.

Available capabilities:
- Start and inspect generation workflows.
- Retrieve tenant-scoped design and domain context.
- Inspect document metadata and citations.
- Search CMS content.
- Create and update CMS drafts.
- Request CMS publication.
- Inspect workflow-safe Figma design context and frame summaries.
- Search WordPress content.
- Create and update WordPress drafts.
- Request WordPress publication.
- Approve, reject, or request changes during human review.

Security boundaries:
- Every operation is tenant-scoped and scope-authorized.
- File contents, embeddings, storage URLs, credentials, prompts, completions,
  and generated file contents are never exposed.
- Raw Figma API payloads, image URLs, and source file contents are never
  exposed.
- Raw WordPress API payloads, credentials, and storage URLs are never exposed.
- Human approval is required before publication when configured.
"""


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------


mcp = FastMCP(
    name=MCP_SERVER_NAME,
    version=MCP_SERVER_VERSION,
    instructions=MCP_SERVER_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register_all_tools() -> None:
    """
    Register all AgenticBridge MCP tools.

    Each tool family is owned by a dedicated module:
    - Workflow execution and status
    - Document metadata and citations
    - Semantic retrieval
    - Human review and approval
    - CMS content operations
    - Figma design context
    - WordPress content operations
    """
    register_workflow_tools(mcp)
    register_document_tools(mcp)
    register_retrieval_tools(mcp)
    register_review_tools(mcp)
    register_cms_tools(mcp)
    register_figma_tools(mcp)
    register_wordpress_tools(mcp)

    logger.info(
        "mcp_tools_registered",
        server_name=MCP_SERVER_NAME,
        server_version=MCP_SERVER_VERSION,
        tool_families=AVAILABLE_TOOL_FAMILIES,
    )


# ---------------------------------------------------------------------------
# Server metadata tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_server_info(
    context: Any | None = None,
) -> dict[str, Any]:
    """
    Get non-sensitive AgenticBridge MCP server information.

    Args:
        context: MCP request context.

    Returns:
        Server name, version, environment category, available tool families,
        and security boundaries. No credentials, database URLs, provider keys,
        or internal configuration values are returned.
    """
    return {
        "server_name": MCP_SERVER_NAME,
        "server_version": MCP_SERVER_VERSION,
        "application_name": settings.APP_NAME,
        "application_version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "workflow_approval_required_by_default": True,
        "available_tool_families": list(
            AVAILABLE_TOOL_FAMILIES
        ),
        "security_boundaries": list(
            SECURITY_BOUNDARIES
        ),
    }


@mcp.tool()
async def get_server_health(
    context: Any | None = None,
) -> dict[str, Any]:
    """
    Get non-sensitive MCP server health information.

    Args:
        context: MCP request context.

    Returns:
        A bounded health response. This endpoint does not expose dependency
        connection strings, internal errors, provider status, or detailed
        infrastructure state.
    """
    return {
        "status": "healthy",
        "server_name": MCP_SERVER_NAME,
        "server_version": MCP_SERVER_VERSION,
        "environment": settings.APP_ENV,
        "checks": {
            "server": "healthy",
            "tool_registry": "healthy",
            "security_policy": "healthy",
        },
    }


@mcp.tool()
async def list_server_capabilities(
    context: Any | None = None,
) -> dict[str, Any]:
    """
    List high-level MCP capabilities without exposing implementation details.

    Args:
        context: MCP request context.

    Returns:
        A bounded capability catalog describing available tool families and
        their intended use.
    """
    return {
        "server_name": MCP_SERVER_NAME,
        "server_version": MCP_SERVER_VERSION,
        "capabilities": [
            {
                "family": "workflow",
                "description": (
                    "Start, inspect, and manage generation workflow "
                    "executions."
                ),
                "mutation_supported": True,
            },
            {
                "family": "documents",
                "description": (
                    "Inspect tenant-scoped document metadata and citations."
                ),
                "mutation_supported": False,
            },
            {
                "family": "retrieval",
                "description": (
                    "Search workflow-safe design and domain knowledge."
                ),
                "mutation_supported": False,
            },
            {
                "family": "review",
                "description": (
                    "Resolve human-review interrupts through approval, "
                    "rejection, or change requests."
                ),
                "mutation_supported": True,
            },
            {
                "family": "cms",
                "description": (
                    "Search CMS content and manage reviewed CMS drafts."
                ),
                "mutation_supported": True,
            },
            {
                "family": "figma",
                "description": (
                    "Retrieve workflow-safe Figma design context and frame "
                    "summaries."
                ),
                "mutation_supported": False,
            },
            {
                "family": "wordpress",
                "description": (
                    "Search WordPress content and manage reviewed WordPress "
                    "drafts."
                ),
                "mutation_supported": True,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


def get_mcp_server() -> FastMCP:
    """
    Return the configured MCP server instance.

    Returns:
        The configured `FastMCP` server.
    """
    return mcp


def get_registered_tool_families() -> tuple[str, ...]:
    """
    Return the configured MCP tool families.

    Returns:
        A tuple of tool-family names.
    """
    return AVAILABLE_TOOL_FAMILIES


async def run_mcp_server(
    *,
    transport: str = "stdio",
) -> None:
    """
    Run the MCP server using the configured transport.

    The transport adapter and authentication middleware are responsible for
    verifying caller identity and attaching a trusted `MCPPrincipal` to each
    request context.

    Args:
        transport: MCP transport name. Supported values are `stdio`, `sse`,
            and `streamable-http`, depending on the installed FastMCP version.

    Raises:
        ValueError: If the requested transport is unsupported.
    """
    register_all_tools()

    logger.info(
        "mcp_server_starting",
        server_name=MCP_SERVER_NAME,
        server_version=MCP_SERVER_VERSION,
        app_name=settings.APP_NAME,
        app_env=settings.APP_ENV,
        app_version=settings.APP_VERSION,
        transport=transport,
        tool_families=AVAILABLE_TOOL_FAMILIES,
    )

    if transport == "stdio":
        await mcp.run_async()
    elif transport == "sse":
        await mcp.run_sse_async()
    elif transport == "streamable-http":
        await mcp.run_http_async()
    else:
        raise ValueError(
            "Unsupported MCP transport. "
            "Use 'stdio', 'sse', or 'streamable-http'."
        )


# Register tools at import time so MCP clients and test harnesses can discover
# the complete tool surface without starting the server.
register_all_tools()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_mcp_server())