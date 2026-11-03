"""
Tenant administration API routes.

These routes provide tenant-scoped operational visibility and administrative
actions for workflow executions, approvals, documents, usage, and audit
events. All endpoints require explicit admin scopes and enforce tenant
isolation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
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
from app.core.enums import JobStatus
from app.core.exceptions import ValidationError
from app.repositories.approval_repository import (
    ApprovalRepository,
)
from app.repositories.audit_log_repository import (
    AuditLogRepository,
)
from app.repositories.document_repository import (
    DocumentRepository,
)
from app.repositories.usage_repository import (
    UsageRepository,
)
from app.repositories.workflow_repository import (
    WorkflowRepository,
)


logger = logging.getLogger(
    "agentic_bridge.api.admin"
)


router = APIRouter(
    prefix="/admin",
    tags=["Administration"],
)


# ---------------------------------------------------------------------------
# Operational overview
# ---------------------------------------------------------------------------


@router.get("/overview")
async def get_admin_overview(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "admin:read"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Return tenant-level workflow, approval, document, and usage overview."""
    workflow_analytics = (
        await WorkflowRepository.get_execution_analytics(
            db,
            tenant_id=principal.tenant_id,
        )
    )

    pending_approvals = (
        await ApprovalRepository.count_approvals(
            db,
            tenant_id=principal.tenant_id,
            status="pending",
        )
    )

    total_documents = (
        await DocumentRepository.list_documents(
            db,
            tenant_id=principal.tenant_id,
            skip=0,
            limit=1,
        )
    )[1]

    usage_summary = (
        await UsageRepository.summarize_usage_by_tenant(
            db,
            tenant_id=principal.tenant_id,
        )
    )

    await db.commit()

    return {
        "tenant_id": str(
            principal.tenant_id
        ),
        "workflow": workflow_analytics,
        "approvals": {
            "pending_approvals": pending_approvals,
        },
        "documents": {
            "total_documents": total_documents,
        },
        "usage": usage_summary,
    }


# ---------------------------------------------------------------------------
# Workflow administration
# ---------------------------------------------------------------------------


@router.get("/executions")
async def list_all_executions(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "admin:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
    workflow_status: Annotated[
        str | None,
        Query(
            description=(
                "Filter by queued, processing, completed, failed, or cancelled."
            )
        ),
    ] = None,
    current_step: Annotated[
        str | None,
        Query(
            description="Filter by current workflow step."
        ),
    ] = None,
    created_after: Annotated[
        datetime | None,
        Query(
            description="Return executions created on or after this time."
        ),
    ] = None,
    created_before: Annotated[
        datetime | None,
        Query(
            description="Return executions created on or before this time."
        ),
    ] = None,
):
    """List all tenant workflow executions with optional filters."""
    try:
        executions, total_count = (
            await WorkflowRepository.list_executions(
                db,
                tenant_id=principal.tenant_id,
                skip=pagination.skip,
                limit=pagination.limit,
                status_filter=workflow_status,
                current_step_filter=current_step,
                created_after=created_after,
                created_before=created_before,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await db.commit()

    return build_paginated_response(
        executions,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


@router.get("/executions/{execution_id}")
async def get_execution(
    execution_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "admin:read"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Retrieve a tenant workflow execution."""
    execution = (
        await WorkflowRepository.get_by_execution_id(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow execution not found.",
        )

    await db.commit()

    return execution


@router.get(
    "/executions/{execution_id}/audit"
)
async def get_execution_audit_events(
    execution_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "admin:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
):
    """Retrieve audit events for a workflow execution."""
    events, total_count = (
        await AuditLogRepository.get_events_for_execution(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
            skip=pagination.skip,
            limit=pagination.limit,
        )
    )

    await db.commit()

    return build_paginated_response(
        events,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


@router.post(
    "/executions/{execution_id}/cancel",
    status_code=status.HTTP_202_ACCEPTED,
)
async def cancel_execution(
    execution_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "admin:write"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Request cancellation of a tenant workflow execution."""
    execution = (
        await WorkflowRepository.mark_cancel_requested(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow execution not found.",
        )

    await db.commit()

    logger.info(
        "Admin requested workflow cancellation.",
        extra={
            "execution_id": str(
                execution_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
        },
    )

    return {
        "execution_id": execution.execution_id,
        "status": execution.status,
        "cancel_requested": execution.cancel_requested,
    }


@router.post(
    "/executions/{execution_id}/status",
    status_code=status.HTTP_200_OK,
)
async def set_execution_status(
    execution_id: UUID,
    payload: dict,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "admin:write"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Explicitly set a tenant workflow execution status."""
    requested_status = payload.get(
        "status"
    )

    if not requested_status:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A workflow status is required.",
        )

    try:
        execution = (
            await WorkflowRepository.set_workflow_status(
                db,
                execution_id=execution_id,
                tenant_id=principal.tenant_id,
                status=requested_status,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow execution not found.",
        )

    await db.commit()

    logger.info(
        "Admin updated workflow status.",
        extra={
            "execution_id": str(
                execution_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
            "status": execution.status,
        },
    )

    return {
        "execution_id": execution.execution_id,
        "status": execution.status,
    }


# ---------------------------------------------------------------------------
# Approval administration
# ---------------------------------------------------------------------------


@router.get("/approvals")
async def list_approvals(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "admin:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
    approval_status: Annotated[
        str | None,
        Query(
            description="Filter by approval status."
        ),
    ] = None,
):
    """List tenant approvals."""
    approvals, total_count = (
        await ApprovalRepository.list_for_tenant(
            db,
            tenant_id=principal.tenant_id,
            skip=pagination.skip,
            limit=pagination.limit,
        )
    )

    if approval_status:
        normalized_status = (
            approval_status.strip().lower()
        )

        approvals = [
            approval
            for approval in approvals
            if approval.status
            == normalized_status
        ]

    await db.commit()

    return build_paginated_response(
        approvals,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


@router.get("/approvals/pending")
async def list_pending_approvals(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "admin:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
):
    """List pending tenant approvals."""
    approvals, total_count = (
        await ApprovalRepository.list_pending_approvals(
            db,
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


# ---------------------------------------------------------------------------
# Document administration
# ---------------------------------------------------------------------------


@router.get("/documents")
async def list_documents(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "admin:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
    document_status: Annotated[
        str | None,
        Query(
            description="Filter by document processing status."
        ),
    ] = None,
    document_type: Annotated[
        str | None,
        Query(
            description="Filter by document type."
        ),
    ] = None,
    search: Annotated[
        str | None,
        Query(
            max_length=200,
            description="Search document titles."
        ),
    ] = None,
):
    """List tenant documents."""
    try:
        documents, total_count = (
            await DocumentRepository.list_documents(
                db,
                tenant_id=principal.tenant_id,
                skip=pagination.skip,
                limit=pagination.limit,
                status_filter=document_status,
                document_type_filter=document_type,
                search_term=search,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await db.commit()

    return build_paginated_response(
        documents,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


# ---------------------------------------------------------------------------
# Usage administration
# ---------------------------------------------------------------------------


@router.get("/usage/summary")
async def get_usage_summary(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "admin:read"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Return tenant-level provider usage summary."""
    usage_summary = (
        await UsageRepository.summarize_usage_by_tenant(
            db,
            tenant_id=principal.tenant_id,
        )
    )

    await db.commit()

    return usage_summary


@router.get("/usage/models")
async def get_usage_by_model(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "admin:read"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Return tenant usage grouped by provider and model."""
    usage_by_model = (
        await UsageRepository.summarize_usage_by_model(
            db,
            tenant_id=principal.tenant_id,
        )
    )

    await db.commit()

    return {
        "tenant_id": str(
            principal.tenant_id
        ),
        "models": usage_by_model,
    }


# ---------------------------------------------------------------------------
# Audit administration
# ---------------------------------------------------------------------------


@router.get("/audit/events")
async def list_audit_events(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "admin:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
    event_type: Annotated[
        str | None,
        Query(
            max_length=100,
            description="Filter audit events by type."
        ),
    ] = None,
):
    """List tenant audit events."""
    try:
        if event_type:
            events, total_count = (
                await AuditLogRepository.list_events_by_event_type(
                    db,
                    tenant_id=principal.tenant_id,
                    event_type=event_type,
                    skip=pagination.skip,
                    limit=pagination.limit,
                )
            )
        else:
            events, total_count = (
                await AuditLogRepository.list_for_tenant(
                    db,
                    tenant_id=principal.tenant_id,
                    skip=pagination.skip,
                    limit=pagination.limit,
                )
            )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await db.commit()

    return build_paginated_response(
        events,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )