"""
Read-only audit trail API routes.

These routes expose tenant-scoped audit events, workflow execution audit
history, actor activity, resource activity, event summaries, and audit-event
counts. They are intentionally read-only: audit records cannot be created,
edited, or deleted through this API.
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
from app.core.exceptions import ValidationError
from app.repositories.audit_log_repository import (
    AuditLogRepository,
)
from app.repositories.workflow_repository import (
    WorkflowRepository,
)


logger = logging.getLogger(
    "agentic_bridge.api.audit"
)


router = APIRouter(
    prefix="/audit",
    tags=["Audit"],
)


# ---------------------------------------------------------------------------
# Tenant audit events
# ---------------------------------------------------------------------------


@router.get("/events")
async def list_audit_events(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "audit:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
    event_type: Annotated[
        str | None,
        Query(
            max_length=100,
            description="Filter audit events by event type."
        ),
    ] = None,
    resource_type: Annotated[
        str | None,
        Query(
            max_length=100,
            description="Filter audit events by resource type."
        ),
    ] = None,
    actor_id: Annotated[
        UUID | None,
        Query(
            description="Filter audit events by actor ID."
        ),
    ] = None,
    created_after: Annotated[
        datetime | None,
        Query(
            description="Return events created on or after this time."
        ),
    ] = None,
    created_before: Annotated[
        datetime | None,
        Query(
            description="Return events created on or before this time."
        ),
    ] = None,
):
    """List tenant-scoped audit events with optional filters."""
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

    if resource_type:
        normalized_resource_type = (
            resource_type.strip().lower()
        )

        events = [
            event
            for event in events
            if event.resource_type
            == normalized_resource_type
        ]

    if actor_id:
        events = [
            event
            for event in events
            if event.actor_id
            == actor_id
        ]

    if created_after:
        events = [
            event
            for event in events
            if event.created_at
            >= created_after
        ]

    if created_before:
        events = [
            event
            for event in events
            if event.created_at
            <= created_before
        ]

    await db.commit()

    return build_paginated_response(
        events,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


@router.get("/events/count")
async def count_audit_events(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "audit:read"
            )
        ),
    ],
    db: DatabaseSession,
    event_type: Annotated[
        str | None,
        Query(
            max_length=100,
            description="Optional event type to count."
        ),
    ] = None,
):
    """Count tenant-scoped audit events, optionally by event type."""
    try:
        if event_type:
            _, total_count = (
                await AuditLogRepository.list_events_by_event_type(
                    db,
                    tenant_id=principal.tenant_id,
                    event_type=event_type,
                    skip=0,
                    limit=1,
                )
            )
        else:
            _, total_count = (
                await AuditLogRepository.list_for_tenant(
                    db,
                    tenant_id=principal.tenant_id,
                    skip=0,
                    limit=1,
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
        "event_type": event_type,
        "count": total_count,
    }


# ---------------------------------------------------------------------------
# Workflow execution audit history
# ---------------------------------------------------------------------------


@router.get(
    "/executions/{execution_id}/events"
)
async def list_execution_audit_events(
    execution_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "audit:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
):
    """List audit events for a tenant workflow execution."""
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


# ---------------------------------------------------------------------------
# Actor audit history
# ---------------------------------------------------------------------------


@router.get(
    "/actors/{actor_id}/events"
)
async def list_actor_audit_events(
    actor_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "audit:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
):
    """List audit events generated by an actor within the tenant."""
    events, total_count = (
        await AuditLogRepository.list_events_by_actor(
            db,
            tenant_id=principal.tenant_id,
            actor_id=actor_id,
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


# ---------------------------------------------------------------------------
# Resource audit history
# ---------------------------------------------------------------------------


@router.get(
    "/resources/{resource_type}/{resource_id}/events"
)
async def list_resource_audit_events(
    resource_type: str,
    resource_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "audit:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
):
    """List audit events for a tenant-scoped resource."""
    normalized_resource_type = (
        resource_type.strip().lower()
    )

    events, total_count = (
        await AuditLogRepository.list_events_by_resource(
            db,
            tenant_id=principal.tenant_id,
            resource_type=normalized_resource_type,
            resource_id=resource_id,
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


# ---------------------------------------------------------------------------
# Audit summaries
# ---------------------------------------------------------------------------


@router.get("/summary")
async def get_audit_summary(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "audit:read"
            )
        ),
    ],
    db: DatabaseSession,
):
    """Return a tenant-level audit-event summary."""
    try:
        total_events = (
            await AuditLogRepository.count_events(
                db,
                tenant_id=principal.tenant_id,
            )
        )

        event_type_counts = (
            await AuditLogRepository.count_events_by_event_type(
                db,
                tenant_id=principal.tenant_id,
            )
        )

        actor_activity = (
            await AuditLogRepository.summarize_activity_by_actor(
                db,
                tenant_id=principal.tenant_id,
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
        "total_events": total_events,
        "events_by_type": event_type_counts,
        "activity_by_actor": actor_activity,
    }


@router.get("/event-types")
async def list_audit_event_types(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "audit:read"
            )
        ),
    ],
    db: DatabaseSession,
):
    """List distinct audit event types recorded for the tenant."""
    event_types = (
        await AuditLogRepository.list_distinct_event_types(
            db,
            tenant_id=principal.tenant_id,
        )
    )

    await db.commit()

    return {
        "tenant_id": str(
            principal.tenant_id
        ),
        "event_types": event_types,
    }