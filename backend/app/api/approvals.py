"""
Human-approval API routes.

These routes expose tenant-scoped approval queues, approval details,
resolution, history, and workflow-specific approval access. All mutating
operations require the `workflows:review` scope and an authenticated principal.
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
from app.api.schemas import (
    ApprovalResolveRequest,
)
from app.core.enums import ApprovalStatus
from app.core.exceptions import ValidationError
from app.repositories.approval_repository import (
    ApprovalRepository,
)
from app.repositories.audit_log_repository import (
    AuditLogRepository,
)
from app.repositories.workflow_repository import (
    WorkflowRepository,
)


logger = logging.getLogger(
    "agentic_bridge.api.approvals"
)


router = APIRouter(
    prefix="/approvals",
    tags=["Approvals"],
)


# ---------------------------------------------------------------------------
# Approval collection
# ---------------------------------------------------------------------------


@router.get("")
async def list_approvals(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:review"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
    approval_status: Annotated[
        str | None,
        Query(
            description=(
                "Filter by pending, approved, rejected, "
                "changes_requested, or expired."
            )
        ),
    ] = None,
):
    """List tenant-scoped approvals with optional status filtering."""
    try:
        approvals, total_count = (
            await ApprovalRepository.list_for_tenant(
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


@router.get("/pending")
async def list_pending_approvals(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:review"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
):
    """List pending approvals for the authenticated tenant."""
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


@router.get("/count")
async def count_approvals(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:review"
            )
        ),
    ],
    db: DatabaseSession,
    approval_status: Annotated[
        str | None,
        Query(
            description=(
                "Optional approval status to count."
            )
        ),
    ] = None,
):
    """Count tenant-scoped approvals, optionally by status."""
    try:
        total_count = (
            await ApprovalRepository.count_approvals(
                db,
                tenant_id=principal.tenant_id,
                status=approval_status,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await db.commit()

    return {
        "tenant_id": str(
            principal.tenant_id
        ),
        "status": approval_status,
        "count": total_count,
    }


# ---------------------------------------------------------------------------
# Individual approvals
# ---------------------------------------------------------------------------


@router.get("/{approval_id}")
async def get_approval(
    approval_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:review"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Retrieve a tenant-scoped approval by ID."""
    approval = (
        await ApprovalRepository.get_approval(
            db,
            approval_id=approval_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not approval:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Approval not found.",
        )

    await db.commit()

    return approval


@router.get(
    "/executions/{execution_id}"
)
async def list_approvals_for_execution(
    execution_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:review"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
):
    """List approval history for a workflow execution."""
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

    approvals, total_count = (
        await ApprovalRepository.list_approvals_for_execution(
            db,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
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
# Approval resolution
# ---------------------------------------------------------------------------


@router.post(
    "/{approval_id}/resolve",
    status_code=status.HTTP_200_OK,
)
async def resolve_approval(
    approval_id: UUID,
    payload: ApprovalResolveRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:review"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Resolve a pending approval with a reviewer decision."""
    try:
        approval = (
            await ApprovalRepository.resolve_approval(
                db,
                approval_id=approval_id,
                tenant_id=principal.tenant_id,
                reviewer_id=principal.user_id,
                decision=payload.decision,
                comment=payload.comment,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    if not approval:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Approval not found.",
        )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="approval_resolved",
        actor_id=principal.user_id,
        execution_id=approval.execution_id,
        resource_type="approval",
        resource_id=approval.id,
        action="resolve_approval",
        outcome=approval.status,
        metadata={
            "decision": approval.decision,
            "workflow_step": approval.workflow_step,
        },
    )

    await db.commit()

    logger.info(
        "Approval resolved.",
        extra={
            "approval_id": str(
                approval.id
            ),
            "execution_id": str(
                approval.execution_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "reviewer_id": str(
                principal.user_id
            ),
            "decision": approval.decision,
        },
    )

    return approval


@router.post(
    "/{approval_id}/approve",
    status_code=status.HTTP_200_OK,
)
async def approve_approval(
    approval_id: UUID,
    payload: ApprovalResolveRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:review"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Approve a pending approval."""
    return await _resolve_with_decision(
        db=db,
        approval_id=approval_id,
        principal=principal,
        decision="approved",
        comment=payload.comment,
    )


@router.post(
    "/{approval_id}/reject",
    status_code=status.HTTP_200_OK,
)
async def reject_approval(
    approval_id: UUID,
    payload: ApprovalResolveRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:review"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Reject a pending approval."""
    return await _resolve_with_decision(
        db=db,
        approval_id=approval_id,
        principal=principal,
        decision="rejected",
        comment=payload.comment,
    )


@router.post(
    "/{approval_id}/request-changes",
    status_code=status.HTTP_200_OK,
)
async def request_approval_changes(
    approval_id: UUID,
    payload: ApprovalResolveRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:review"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Request changes for a pending approval."""
    return await _resolve_with_decision(
        db=db,
        approval_id=approval_id,
        principal=principal,
        decision="request_changes",
        comment=payload.comment,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _resolve_with_decision(
    *,
    db,
    approval_id: UUID,
    principal,
    decision: str,
    comment: str | None,
):
    """Resolve an approval and record an audit event."""
    try:
        approval = (
            await ApprovalRepository.resolve_approval(
                db,
                approval_id=approval_id,
                tenant_id=principal.tenant_id,
                reviewer_id=principal.user_id,
                decision=decision,
                comment=comment,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    if not approval:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Approval not found.",
        )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="approval_resolved",
        actor_id=principal.user_id,
        execution_id=approval.execution_id,
        resource_type="approval",
        resource_id=approval.id,
        action="resolve_approval",
        outcome=approval.status,
        metadata={
            "decision": approval.decision,
            "workflow_step": approval.workflow_step,
        },
    )

    await db.commit()

    logger.info(
        "Approval resolved through dedicated endpoint.",
        extra={
            "approval_id": str(
                approval.id
            ),
            "execution_id": str(
                approval.execution_id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "reviewer_id": str(
                principal.user_id
            ),
            "decision": approval.decision,
        },
    )

    return approval