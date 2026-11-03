"""
Creative brief API routes.

These routes provide tenant-scoped creation, retrieval, listing, filtering,
updating, deletion, status management, and workflow execution for creative
briefs. Brief content is validated before persistence, and all mutations are
recorded in the tenant audit trail.
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
from app.core.exceptions import ValidationError
from app.repositories.audit_log_repository import (
    AuditLogRepository,
)
from app.repositories.brief_repository import (
    BriefRepository,
)
from app.repositories.workflow_repository import (
    WorkflowRepository,
)


logger = logging.getLogger(
    "agentic_bridge.api.briefs"
)


router = APIRouter(
    prefix="/briefs",
    tags=["Briefs"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class BriefCreateRequest(BaseModel):
    """Request to create a creative brief."""

    title: str = Field(
        min_length=1,
        max_length=200,
    )
    objective: str = Field(
        min_length=1,
        max_length=5_000,
    )
    audience: str | None = Field(
        default=None,
        max_length=2_000,
    )
    tone: str | None = Field(
        default=None,
        max_length=500,
    )
    deliverables: list[str] = Field(
        default_factory=list,
        description="Requested creative deliverables.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Creative or regulatory constraints.",
    )
    metadata: dict = Field(
        default_factory=dict
    )


class BriefUpdateRequest(BaseModel):
    """Request to update a creative brief."""

    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    objective: str | None = Field(
        default=None,
        min_length=1,
        max_length=5_000,
    )
    audience: str | None = Field(
        default=None,
        max_length=2_000,
    )
    tone: str | None = Field(
        default=None,
        max_length=500,
    )
    deliverables: list[str] | None = None
    constraints: list[str] | None = None
    metadata: dict | None = None


class BriefStatusRequest(BaseModel):
    """Request to update a brief status."""

    brief_status: str = Field(
        min_length=1,
        max_length=50,
        description=(
            "New brief status, such as draft, ready, or archived."
        ),
    )


class BriefRunRequest(BaseModel):
    """Request to start a workflow from a brief."""

    workflow_type: str = Field(
        min_length=1,
        max_length=100,
        description="Workflow template or type to execute.",
    )
    metadata: dict = Field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# Brief collection
# ---------------------------------------------------------------------------


@router.get("")
async def list_briefs(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "briefs:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
    brief_status: Annotated[
        str | None,
        Query(
            max_length=50,
            description="Filter by brief status."
        ),
    ] = None,
    search: Annotated[
        str | None,
        Query(
            max_length=200,
            description="Search brief titles and objectives."
        ),
    ] = None,
    created_after: Annotated[
        datetime | None,
        Query(
            description="Return briefs created on or after this time."
        ),
    ] = None,
    created_before: Annotated[
        datetime | None,
        Query(
            description="Return briefs created on or before this time."
        ),
    ] = None,
):
    """List tenant-scoped creative briefs with optional filters."""
    briefs, total_count = (
        await BriefRepository.list_for_tenant(
            db,
            tenant_id=principal.tenant_id,
            skip=pagination.skip,
            limit=pagination.limit,
        )
    )

    if brief_status:
        normalized_status = (
            brief_status.strip().lower()
        )

        briefs = [
            brief
            for brief in briefs
            if brief.brief_status
            == normalized_status
        ]

    if search:
        normalized_search = (
            search.strip().lower()
        )

        briefs = [
            brief
            for brief in briefs
            if normalized_search
            in brief.title.lower()
            or normalized_search
            in brief.objective.lower()
        ]

    if created_after:
        briefs = [
            brief
            for brief in briefs
            if brief.created_at
            >= created_after
        ]

    if created_before:
        briefs = [
            brief
            for brief in briefs
            if brief.created_at
            <= created_before
        ]

    await db.commit()

    return build_paginated_response(
        briefs,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
)
async def create_brief(
    payload: BriefCreateRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "briefs:write"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Create a tenant-scoped creative brief."""
    try:
        brief = (
            await BriefRepository.create_brief(
                db,
                tenant_id=principal.tenant_id,
                created_by=principal.user_id,
                title=payload.title,
                objective=payload.objective,
                audience=payload.audience,
                tone=payload.tone,
                deliverables=payload.deliverables,
                constraints=payload.constraints,
                metadata=payload.metadata,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="brief_created",
        actor_id=principal.user_id,
        resource_type="brief",
        resource_id=brief.id,
        action="create_brief",
        outcome=brief.brief_status,
        metadata={
            "title": brief.title,
            "deliverable_count": len(
                brief.deliverables or []
            ),
        },
    )

    await db.commit()

    logger.info(
        "Creative brief created.",
        extra={
            "brief_id": str(
                brief.id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
        },
    )

    return brief


# ---------------------------------------------------------------------------
# Individual briefs
# ---------------------------------------------------------------------------


@router.get("/{brief_id}")
async def get_brief(
    brief_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "briefs:read"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Retrieve a tenant-scoped creative brief."""
    brief = (
        await BriefRepository.get_by_id(
            db,
            record_id=brief_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not brief:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Brief not found.",
        )

    await db.commit()

    return brief


@router.patch(
    "/{brief_id}",
    status_code=status.HTTP_200_OK,
)
async def update_brief(
    brief_id: UUID,
    payload: BriefUpdateRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "briefs:write"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Update a tenant-scoped creative brief."""
    existing_brief = (
        await BriefRepository.get_by_id(
            db,
            record_id=brief_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not existing_brief:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Brief not found.",
        )

    update_data = payload.model_dump(
        exclude_unset=True
    )

    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one field must be provided.",
        )

    try:
        brief = (
            await BriefRepository.update_brief(
                db,
                brief_id=brief_id,
                tenant_id=principal.tenant_id,
                update_data=update_data,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="brief_updated",
        actor_id=principal.user_id,
        resource_type="brief",
        resource_id=brief.id,
        action="update_brief",
        outcome=brief.brief_status,
        metadata={
            "updated_fields": list(
                update_data.keys()
            ),
        },
    )

    await db.commit()

    logger.info(
        "Creative brief updated.",
        extra={
            "brief_id": str(
                brief.id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
            "updated_fields": list(
                update_data.keys()
            ),
        },
    )

    return brief


@router.delete(
    "/{brief_id}",
    status_code=status.HTTP_200_OK,
)
async def delete_brief(
    brief_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "briefs:write"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Delete a tenant-scoped creative brief."""
    existing_brief = (
        await BriefRepository.get_by_id(
            db,
            record_id=brief_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not existing_brief:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Brief not found.",
        )

    deleted = (
        await BriefRepository.delete_by_id(
            db,
            record_id=brief_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Brief not found.",
        )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="brief_deleted",
        actor_id=principal.user_id,
        resource_type="brief",
        resource_id=brief_id,
        action="delete_brief",
        outcome="deleted",
        metadata={
            "title": existing_brief.title,
        },
    )

    await db.commit()

    logger.info(
        "Creative brief deleted.",
        extra={
            "brief_id": str(
                brief_id
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
        "brief_id": str(
            brief_id
        ),
        "deleted": True,
    }


# ---------------------------------------------------------------------------
# Brief status
# ---------------------------------------------------------------------------


@router.post(
    "/{brief_id}/status",
    status_code=status.HTTP_200_OK,
)
async def set_brief_status(
    brief_id: UUID,
    payload: BriefStatusRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "briefs:write"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Update the lifecycle status of a creative brief."""
    try:
        brief = (
            await BriefRepository.set_brief_status(
                db,
                brief_id=brief_id,
                tenant_id=principal.tenant_id,
                brief_status=payload.brief_status,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    if not brief:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Brief not found.",
        )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="brief_status_changed",
        actor_id=principal.user_id,
        resource_type="brief",
        resource_id=brief.id,
        action="set_brief_status",
        outcome=brief.brief_status,
        metadata={
            "brief_status": brief.brief_status,
        },
    )

    await db.commit()

    logger.info(
        "Creative brief status updated.",
        extra={
            "brief_id": str(
                brief.id
            ),
            "tenant_id": str(
                principal.tenant_id
            ),
            "actor_id": str(
                principal.user_id
            ),
            "brief_status": brief.brief_status,
        },
    )

    return brief


# ---------------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------------


@router.post(
    "/{brief_id}/run",
    status_code=status.HTTP_202_ACCEPTED,
)
async def run_workflow_from_brief(
    brief_id: UUID,
    payload: BriefRunRequest,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "workflows:write"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Start a workflow execution using a creative brief."""
    brief = (
        await BriefRepository.get_by_id(
            db,
            record_id=brief_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not brief:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Brief not found.",
        )

    try:
        execution = (
            await WorkflowRepository.create_workflow(
                db,
                tenant_id=principal.tenant_id,
                created_by=principal.user_id,
                workflow_type=payload.workflow_type,
                title=brief.title,
                input_data={
                    "brief_id": str(
                        brief.id
                    ),
                    "objective": brief.objective,
                    "audience": brief.audience,
                    "tone": brief.tone,
                    "deliverables": brief.deliverables,
                    "constraints": brief.constraints,
                },
                metadata=payload.metadata,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await BriefRepository.set_brief_status(
        db,
        brief_id=brief_id,
        tenant_id=principal.tenant_id,
        brief_status="in_workflow",
    )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="brief_workflow_started",
        actor_id=principal.user_id,
        execution_id=execution.execution_id,
        resource_type="brief",
        resource_id=brief.id,
        action="run_workflow_from_brief",
        outcome=execution.status,
        metadata={
            "workflow_type": payload.workflow_type,
        },
    )

    await db.commit()

    logger.info(
        "Workflow started from creative brief.",
        extra={
            "brief_id": str(
                brief.id
            ),
            "execution_id": str(
                execution.execution_id
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
        "brief_id": str(
            brief.id
        ),
        "execution_id": str(
            execution.execution_id
        ),
        "status": execution.status,
    }