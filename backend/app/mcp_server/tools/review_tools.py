"""
MCP human-review tools.

These tools allow authorized reviewers to approve, reject, or request changes
for a workflow. They do not make approval decisions autonomously.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from app.core.enums import ApprovalDecision
from app.mcp_server.auth import (
    get_authenticated_principal,
    require_scope,
)
from app.mcp_server.dependencies import get_workflow_service
from app.mcp_server.errors import mcp_tool_error_boundary
from app.mcp_server.middleware import with_mcp_context
from app.mcp_server.schemas import WorkflowReviewRequest


def register_review_tools(
    mcp: FastMCP,
) -> None:
    """Register human-review MCP tools."""

    @mcp.tool()
    @mcp_tool_error_boundary
    async def approve_workflow(
        execution_id: str,
        comment: str | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """Approve a workflow for publication."""
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope="workflows:review",
        )

        request = WorkflowReviewRequest(
            execution_id=execution_id,
            decision="approved",
            comment=comment,
        )

        async def approve() -> Any:
            return await workflow_service.resolve_review(
                request=WorkflowReviewRequest(
                    execution_id=request.execution_id,
                    decision=ApprovalDecision.APPROVED,
                    comment=request.comment,
                ),
                tenant_id=principal.tenant_id,
                reviewer_id=principal.user_id,
            )

        result = await with_mcp_context(
            approve,
            principal=principal,
            tool_name="approve_workflow",
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def request_workflow_changes(
        execution_id: str,
        comment: str,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """Request changes during human review."""
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope="workflows:review",
        )

        request = WorkflowReviewRequest(
            execution_id=execution_id,
            decision="request_changes",
            comment=comment,
        )

        async def request_changes() -> Any:
            return await workflow_service.resolve_review(
                request=WorkflowReviewRequest(
                    execution_id=request.execution_id,
                    decision=ApprovalDecision.REQUEST_CHANGES,
                    comment=request.comment,
                ),
                tenant_id=principal.tenant_id,
                reviewer_id=principal.user_id,
            )

        result = await with_mcp_context(
            request_changes,
            principal=principal,
            tool_name="request_workflow_changes",
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def reject_workflow(
        execution_id: str,
        comment: str,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """Reject a workflow execution."""
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope="workflows:review",
        )

        request = WorkflowReviewRequest(
            execution_id=execution_id,
            decision="rejected",
            comment=comment,
        )

        async def reject() -> Any:
            return await workflow_service.resolve_review(
                request=WorkflowReviewRequest(
                    execution_id=request.execution_id,
                    decision=ApprovalDecision.REJECTED,
                    comment=request.comment,
                ),
                tenant_id=principal.tenant_id,
                reviewer_id=principal.user_id,
            )

        result = await with_mcp_context(
            reject,
            principal=principal,
            tool_name="reject_workflow",
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )