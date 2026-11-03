"""
Workflow execution API routes.

These routes provide tenant-scoped workflow creation, listing, retrieval,
status transitions, cancellation, step inspection, approvals, artifacts,
retry handling, and workflow analytics. All operations enforce tenant
isolation and explicit authorization scopes.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from pydantic import (
    BaseModel,
    Field,
)

from app.api.dependencies import (
    CurrentPrincipal,
    DatabaseSession,
    Pagination,
    require_scoped_access,
)
from app.api.pagination import (
    build_paginated_response,
)
from app.api.schemas import (
    WorkflowCreateRequest,
    WorkflowExecutionResponse,
    WorkflowStatusUpdateRequest,
)
from app.core.exceptions import ValidationError
from app.repositories.approval_repository import (
    ApprovalRepository,
)
from app.repositories.artifact_repository import (
    ArtifactRepository,
)
from app.repositories.audit_log_repository import (
    AuditLogRepository,
)
from app.repositories.workflow_repository import (
    WorkflowRepository,
)


logger = logging.getLogger(
    "agentic_bridge.api.workflow_router"
)


router = APIRouter(
    prefix="/workflows",
    tags=["Workflows"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class WorkflowRetryRequest(BaseModel):
    """Request to retry a failed workflow execution."""

    reason: str | None = Field(
        default=None,
        max_length=1_000,
        description="Optional reason for retrying the workflow.",
    )


class WorkflowStepResponse(BaseModel):
    """Workflow step execution detail."""

    step_name: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Workflow collection
# ---------------------------------------------------------------------------


@router.get(
    "",
    summary="List workflow executions",
    description=(
        "List tenant-scoped workflow executions with optional filters."
    ),
)
async def list_workflows(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
    workflow_status: Annotated[
        str | None,
        Query(
            max_length=50,
            description=(
                "Filter by queued, processing, waiting_approval, completed, "
                "failed, or cancelled."
            ),
        ),
    ] = None,
    workflow_type: Annotated[
        str | None,
        Query(
            max_length=100,
            description="Filter by workflow type.",
        ),
    ] = None,
    current_step: Annotated[
        str | None,
        Query(
            max_length=100,
            description="Filter by current workflow step.",
        ),
    ] = None,
    created_after: Annotated[
        datetime | None,
        Query(
            description=(
                "Return workflows created on or after this time."
            ),
        ),
    ] = None,
    created_before: Annotated[
        datetime | None,
        Query(
            description=(
                "Return workflows created on or before this time."
            ),
        ),
    ] = None,
):
    """List tenant workflow executions."""
    workflows, total_count = (
        await WorkflowRepository.list_executions(
            db,
            tenant_id=principal.tenant_id,
            skip=pagination.skip,
            limit=pagination.limit,
            status_filter=workflow_status,
            workflow_type_filter=workflow_type,
            current_step_filter=current_step,
            created_after=created_after,
            created_before=created_before,
        )
    )

    await db.commit()

    return build_paginated_response(
        workflows,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


@router.post(
    "",
    response_model=WorkflowExecutionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create workflow execution",
    description=(
        "Create and queue a tenant-scoped workflow execution."
    ),
)
async def create_workflow(
    payload: WorkflowCreateRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:write"
            )
        ),
    ],
    db: DatabaseSession,
    request: Request,
) -> WorkflowExecutionResponse:
    """Create a tenant workflow execution."""
    try:
        workflow = (
            await WorkflowRepository.create_workflow(
                db,
                tenant_id=principal.tenant_id,
                created_by=principal.user_id,
                workflow_type=payload.workflow_type,
                title=payload.title,
                input_data=payload.input_data,
                metadata=payload.metadata,
            )
        )
    except ValidationError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="workflow_created",
        actor_id=principal.user_id,
        execution_id=workflow.execution_id,
        resource_type="workflow",
        resource_id=workflow.execution_id,
        action="create_workflow",
        outcome=workflow.status,
        metadata={
            "workflow_type": workflow.workflow_type,
            "title": workflow.title,
        },
    )

    await db.commit()

    logger.info(
        "Workflow created.",
        extra={
            "execution_id": str(
                workflow.execution_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
            "workflow_type": workflow.workflow_type,
            "request_id": request.headers.get(
                "X-Request-ID"
            ),
        },
    )

    return workflow


# ---------------------------------------------------------------------------
# Individual workflows
# ---------------------------------------------------------------------------


@router.get(
    "/{execution_id}",
    response_model=WorkflowExecutionResponse,
    summary="Get workflow execution",
    description=(
        "Retrieve a tenant-scoped workflow execution by execution ID."
    ),
)
async def get_workflow(
    execution_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:read"
            )
        ),
    ],
    db: DatabaseSession,
) -> WorkflowExecutionResponse:
    """Retrieve a tenant workflow execution."""
    workflow = (
        await WorkflowRepository.get_by_execution_id(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow execution not found.",
        )

    await db.commit()

    return workflow


@router.get(
    "/{execution_id}/steps",
    response_model=list[WorkflowStepResponse],
    summary="Get workflow steps",
    description=(
        "Retrieve step-level execution details for a workflow."
    ),
)
async def get_workflow_steps(
    execution_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:read"
            )
        ),
    ],
    db: DatabaseSession,
) -> list[WorkflowStepResponse]:
    """Retrieve workflow step details."""
    workflow = (
        await WorkflowRepository.get_by_execution_id(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow execution not found.",
        )

    steps = (
        await WorkflowRepository.get_workflow_steps(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
        )
    )

    await db.commit()

    return [
        WorkflowStepResponse(
            step_name=step.get(
                "step_name",
                "unknown",
            ),
            status=step.get(
                "status",
                "unknown",
            ),
            started_at=step.get(
                "started_at"
            ),
            completed_at=step.get(
                "completed_at"
            ),
            error_message=step.get(
                "error_message"
            ),
            metadata=step.get(
                "metadata",
                {},
            ),
        )
        for step in steps
    ]


# ---------------------------------------------------------------------------
# Workflow lifecycle
# ---------------------------------------------------------------------------


@router.post(
    "/{execution_id}/status",
    response_model=WorkflowExecutionResponse,
    summary="Update workflow status",
    description=(
        "Explicitly update a tenant workflow execution status. Requires the "
        "`workflows:write` scope."
    ),
)
async def update_workflow_status(
    execution_id: UUID,
    payload: WorkflowStatusUpdateRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:write"
            )
        ),
    ],
    db: DatabaseSession,
) -> WorkflowExecutionResponse:
    """Update a tenant workflow execution status."""
    try:
        workflow = (
            await WorkflowRepository.set_workflow_status(
                db,
                execution_id=execution_id,
                tenant_id=principal.tenant_id,
                status=payload.status,
            )
        )
    except ValidationError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow execution not found.",
        )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="workflow_status_changed",
        actor_id=principal.user_id,
        execution_id=workflow.execution_id,
        resource_type="workflow",
        resource_id=workflow.execution_id,
        action="update_workflow_status",
        outcome=workflow.status,
        metadata={
            "workflow_status": workflow.status,
        },
    )

    await db.commit()

    logger.info(
        "Workflow status updated.",
        extra={
            "execution_id": str(
                workflow.execution_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
            "workflow_status": workflow.status,
        },
    )

    return workflow


@router.post(
    "/{execution_id}/cancel",
    response_model=WorkflowExecutionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Cancel workflow execution",
    description=(
        "Request cancellation of a tenant workflow execution."
    ),
)
async def cancel_workflow(
    execution_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:write"
            )
        ),
    ],
    db: DatabaseSession,
) -> WorkflowExecutionResponse:
    """Request cancellation of a tenant workflow execution."""
    workflow = (
        await WorkflowRepository.mark_cancel_requested(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow execution not found.",
        )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="workflow_cancel_requested",
        actor_id=principal.user_id,
        execution_id=workflow.execution_id,
        resource_type="workflow",
        resource_id=workflow.execution_id,
        action="cancel_workflow",
        outcome=workflow.status,
        metadata={
            "cancel_requested": workflow.cancel_requested,
        },
    )

    await db.commit()

    logger.info(
        "Workflow cancellation requested.",
        extra={
            "execution_id": str(
                workflow.execution_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
        },
    )

    return workflow


@router.post(
    "/{execution_id}/retry",
    response_model=WorkflowExecutionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Retry workflow execution",
    description=(
        "Retry a failed or cancelled tenant workflow execution."
    ),
)
async def retry_workflow(
    execution_id: UUID,
    payload: WorkflowRetryRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:write"
            )
        ),
    ],
    db: DatabaseSession,
) -> WorkflowExecutionResponse:
    """Retry a failed or cancelled workflow execution."""
    existing_workflow = (
        await WorkflowRepository.get_by_execution_id(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not existing_workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow execution not found.",
        )

    if existing_workflow.status not in {
        "failed",
        "cancelled",
    }:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Only failed or cancelled workflow executions can be retried."
            ),
        )

    try:
        workflow = (
            await WorkflowRepository.retry_workflow(
                db,
                execution_id=execution_id,
                tenant_id=principal.tenant_id,
                requested_by=principal.user_id,
                reason=payload.reason,
            )
        )
    except ValidationError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="workflow_retry_requested",
        actor_id=principal.user_id,
        execution_id=workflow.execution_id,
        resource_type="workflow",
        resource_id=workflow.execution_id,
        action="retry_workflow",
        outcome=workflow.status,
        metadata={
            "reason": payload.reason,
        },
    )

    await db.commit()

    logger.info(
        "Workflow retry requested.",
        extra={
            "execution_id": str(
                workflow.execution_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
        },
    )

    return workflow


# ---------------------------------------------------------------------------
# Related resources
# ---------------------------------------------------------------------------


@router.get(
    "/{execution_id}/approvals",
    summary="List workflow approvals",
    description=(
        "List approval requests associated with a workflow execution."
    ),
)
async def list_workflow_approvals(
    execution_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
):
    """List approvals for a workflow execution."""
    workflow = (
        await WorkflowRepository.get_by_execution_id(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow execution not found.",
        )

    approvals, total_count = (
        await ApprovalRepository.list_approvals_for_execution(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
            skip=pagination.skip,
            limit=pagination.limit,
        )
    )

    await db.commit()

    return build_paginated_response(
        approvals,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


@router.get(
    "/{execution_id}/artifacts",
    summary="List workflow artifacts",
    description=(
        "List generated artifacts associated with a workflow execution."
    ),
)
async def list_workflow_artifacts(
    execution_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
):
    """List artifacts for a workflow execution."""
    workflow = (
        await WorkflowRepository.get_by_execution_id(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not workflow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow execution not found.",
        )

    artifacts, total_count = (
        await ArtifactRepository.list_artifacts_for_execution(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
            skip=pagination.skip,
            limit=pagination.limit,
        )
    )

    await db.commit()

    return build_paginated_response(
        artifacts,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


@router.get(
    "/analytics",
    summary="Workflow analytics",
    description=(
        "Return aggregate workflow execution analytics for the tenant."
    ),
)
async def get_workflow_analytics(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:read"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Return tenant workflow analytics."""
    analytics = (
        await WorkflowRepository.get_execution_analytics(
            db,
            tenant_id=principal.tenant_id,
        )
    )

    await db.commit()

    return analytics