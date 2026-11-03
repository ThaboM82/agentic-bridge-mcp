"""
MCP workflow tools.

These tools expose a narrow, authenticated interface for starting, inspecting,
listing, and cancelling AgenticBridge workflow executions.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import Context, FastMCP

from app.core.enums import JobStatus, WorkflowStep
from app.core.exceptions import AuthorizationError
from app.graph.runner import WorkflowRunResult
from app.mcp_server.auth import (
    get_authenticated_principal,
    require_scope,
)
from app.mcp_server.dependencies import (
    get_workflow_runner,
    get_workflow_service,
)
from app.mcp_server.errors import mcp_tool_error_boundary
from app.mcp_server.middleware import with_mcp_context
from app.mcp_server.rate_limit import rate_limiter
from app.mcp_server.schemas import (
    WorkflowCancelRequest,
    WorkflowListRequest,
    WorkflowStartRequest,
    WorkflowStatusRequest,
)


def register_workflow_tools(
    mcp: FastMCP,
) -> None:
    """Register workflow MCP tools."""

    @mcp.tool()
    @mcp_tool_error_boundary
    async def start_generation_workflow(
        brief_text: str,
        document_ids: list[str] | None = None,
        workflow_name: str | None = None,
        approval_required: bool = True,
        metadata: dict[str, Any] | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """Start an AgenticBridge generation workflow."""
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope="workflows:write",
        )

        request = WorkflowStartRequest(
            brief_text=brief_text,
            document_ids=document_ids or [],
            workflow_name=workflow_name or "design_to_code",
            approval_required=approval_required,
            metadata={"values": metadata or {}},
        )

        execution_id = uuid4()

        result: WorkflowRunResult = (
            await with_mcp_context(
                workflow_runner.run,
                principal=principal,
                tool_name="start_generation_workflow",
                execution_id=execution_id,
                brief_id=uuid4(),
                raw_brief_text=request.brief_text,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                workflow_name=request.workflow_name,
                approval_required=request.approval_required,
                initial_state_overrides={
                    "document_ids": [
                        str(document_id)
                        for document_id in request.document_ids
                    ],
                    "mcp_metadata": request.metadata.values,
                },
            )
        )

        return result.to_dict()

    @mcp.tool()
    @mcp_tool_error_boundary
    async def get_workflow_status(
        execution_id: str,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """Get safe workflow execution status."""
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope="workflows:read",
        )

        request = WorkflowStatusRequest(
            execution_id=execution_id,
        )

        async def load_status() -> Any:
            return await workflow_service.get_workflow_status(
                execution_id=request.execution_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )

        result = await with_mcp_context(
            load_status,
            principal=principal,
            tool_name="get_workflow_status",
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def list_workflow_executions(
        status: str | None = None,
        current_step: str | None = None,
        page: int | None = None,
        page_size: int | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """List tenant-visible workflow executions."""
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope="workflows:read",
        )

        request = WorkflowListRequest(
            status=status,
            current_step=current_step,
            page=page or 1,
            page_size=page_size or 20,
        )

        async def load_executions() -> Any:
            return await workflow_service.list_workflow_executions(
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                status=(
                    JobStatus(request.status)
                    if request.status
                    else None
                ),
                current_step=(
                    WorkflowStep(request.current_step)
                    if request.current_step
                    else None
                ),
                page=request.page,
                page_size=request.page_size,
            )

        result = await with_mcp_context(
            load_executions,
            principal=principal,
            tool_name="list_workflow_executions",
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )

    @mcp.tool()
    @mcp_tool_error_boundary
    async def cancel_workflow(
        execution_id: str,
        reason: str | None = None,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """Request cancellation of a workflow execution."""
        principal = await get_authenticated_principal(context)

        require_scope(
            principal,
            scope="workflows:cancel",
        )

        request = WorkflowCancelRequest(
            execution_id=execution_id,
            reason=reason,
        )

        async def cancel() -> Any:
            return await workflow_service.cancel_workflow(
                request=request,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
            )

        result = await with_mcp_context(
            cancel,
            principal=principal,
            tool_name="cancel_workflow",
        )

        return result.model_dump(
            mode="json",
            exclude_none=True,
        )


# Configure a conservative default limit for workflow-start operations.
rate_limiter.configure_tool(
    tool_name="start_generation_workflow",
    requests_per_minute=10,
    burst=5,
)