"""
Dependency factories for the AgenticBridge MCP server.

This module keeps MCP tools thin by centralizing construction of services,
workflow runners, and authenticated request contexts.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.graph.runner import WorkflowRunner, create_workflow_runner
from app.graph.workflow import app_workflow
from app.services.workflow_service import WorkflowService
from app.mcp_server.auth import MCPPrincipal
from app.mcp_server.context import MCPRequestContext, create_request_context


@lru_cache
def get_workflow_service() -> WorkflowService:
    """Return the shared workflow service instance."""
    return WorkflowService()


@lru_cache
def get_workflow_runner() -> WorkflowRunner:
    """Return the shared workflow runner instance."""
    return create_workflow_runner(
        app_workflow,
    )


def build_request_context(
    *,
    principal: MCPPrincipal,
    request_id: str | None = None,
    correlation_id: str | None = None,
) -> MCPRequestContext:
    """
    Build a request-scoped MCP context.

    Args:
        principal: Authenticated MCP principal.
        request_id: Optional request correlation ID.
        correlation_id: Optional business correlation ID.

    Returns:
        A request-scoped MCP context.
    """
    return create_request_context(
        principal=principal,
        request_id=request_id,
        correlation_id=correlation_id,
    )


def get_tool_dependencies() -> dict[str, Any]:
    """
    Return shared dependencies available to MCP tools.

    Returns:
        A dictionary of shared service instances.
    """
    return {
        "workflow_service": get_workflow_service(),
        "workflow_runner": get_workflow_runner(),
    }