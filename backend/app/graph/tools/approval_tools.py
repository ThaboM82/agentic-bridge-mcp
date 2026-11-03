"""
Approval tools for workflow nodes.

Provides a narrow interface for creating and resolving human approval requests.
Approval policy belongs in the workflow service, not in these helpers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import ApprovalStatus
from app.core.exceptions import ValidationError
from app.crud.approval_crud import (
    approve_approval_request,
    create_approval_request,
    get_pending_approval_for_workflow,
    reject_approval_request,
)


logger = logging.getLogger("agentic_bridge.graph.tools.approval")


async def create_workflow_approval(
    db: AsyncSession,
    *,
    workflow_id: str | UUID,
    workflow_step: str,
    tenant_id: str | None = None,
    requested_by: str | None = None,
    approver_role: str | None = None,
    approver_user_id: str | None = None,
    payload: dict | None = None,
    expires_in_minutes: int = 1440,
) -> dict[str, Any]:
    """Create an approval request for a workflow step."""
    if expires_in_minutes <= 0:
        raise ValidationError("Approval expiration must be positive.")

    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=expires_in_minutes
    )

    approval = await create_approval_request(
        db,
        workflow_id=workflow_id,
        workflow_step=workflow_step,
        requested_by=requested_by,
        tenant_id=tenant_id,
        approver_role=approver_role,
        approver_user_id=approver_user_id,
        payload=payload or {},
        expires_at=expires_at,
    )

    logger.info(
        "workflow_approval_created",
        approval_id=str(approval.id),
        workflow_id=str(workflow_id),
        workflow_step=workflow_step,
    )

    return {
        "approval_id": str(approval.id),
        "status": approval.status,
        "expires_at": approval.expires_at.isoformat(),
    }


async def get_active_workflow_approval(
    db: AsyncSession,
    workflow_id: str | UUID,
    *,
    workflow_step: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any] | None:
    """Retrieve the active pending approval for a workflow."""
    approval = await get_pending_approval_for_workflow(
        db,
        workflow_id,
        workflow_step=workflow_step,
        tenant_id=tenant_id,
    )

    if approval is None:
        return None

    return {
        "approval_id": str(approval.id),
        "status": approval.status,
        "workflow_step": approval.workflow_step,
        "expires_at": (
            approval.expires_at.isoformat()
            if approval.expires_at
            else None
        ),
    }


async def resolve_workflow_approval(
    db: AsyncSession,
    approval_id: str | UUID,
    *,
    decision: str,
    decided_by: str,
    comment: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """Approve or reject a workflow approval request."""
    normalized_decision = decision.strip().lower()

    if normalized_decision == "approved":
        approval = await approve_approval_request(
            db,
            approval_id,
            decided_by=decided_by,
            comment=comment,
            tenant_id=tenant_id,
        )

    elif normalized_decision == "rejected":
        approval = await reject_approval_request(
            db,
            approval_id,
            decided_by=decided_by,
            comment=comment,
            tenant_id=tenant_id,
        )

    else:
        raise ValidationError(
            "Approval decision must be either approved or rejected."
        )

    logger.info(
        "workflow_approval_resolved",
        approval_id=str(approval.id),
        workflow_id=str(approval.workflow_id),
        decision=approval.status,
    )

    return {
        "approval_id": str(approval.id),
        "workflow_id": str(approval.workflow_id),
        "status": approval.status,
        "decision": approval.decision,
    }


def is_approval_terminal(status: str | ApprovalStatus) -> bool:
    """Determine whether an approval status is final."""
    normalized_status = ApprovalStatus(status)

    return normalized_status is not ApprovalStatus.PENDING