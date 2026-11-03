"""
Provider usage and metering API routes.

These routes expose tenant-scoped AI provider usage summaries, model-level
breakdowns, user-level breakdowns, usage events, cost estimates, quotas, and
usage exports. All data is isolated to the authenticated principal's tenant.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Annotated, Any
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
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
from app.repositories.usage_repository import (
    UsageRepository,
)


logger = logging.getLogger(
    "agentic_bridge.api.usage"
)


router = APIRouter(
    prefix="/usage",
    tags=["Usage & Metering"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class UsageEventResponse(BaseModel):
    """Individual provider usage event."""

    model_config = {
        "from_attributes": True
    }

    id: UUID
    tenant_id: UUID
    user_id: UUID | None = None
    execution_id: UUID | None = None
    provider: str
    model: str
    operation: str | None = None
    prompt_tokens: int = Field(
        ge=0,
    )
    completion_tokens: int = Field(
        ge=0,
    )
    total_tokens: int = Field(
        ge=0,
    )
    estimated_cost: float | None = Field(
        default=None,
        ge=0.0,
    )
    occurred_at: datetime
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class UsageSummaryResponse(BaseModel):
    """Tenant-level usage summary."""

    tenant_id: UUID
    usage_events: int = Field(
        ge=0,
    )
    prompt_tokens: int = Field(
        ge=0,
    )
    completion_tokens: int = Field(
        ge=0,
    )
    total_tokens: int = Field(
        ge=0,
    )
    estimated_cost: float = Field(
        ge=0.0,
    )


class UsageByModelItem(BaseModel):
    """Usage totals for one provider and model."""

    provider: str
    model: str
    usage_events: int = Field(
        ge=0,
    )
    prompt_tokens: int = Field(
        ge=0,
    )
    completion_tokens: int = Field(
        ge=0,
    )
    total_tokens: int = Field(
        ge=0,
    )
    estimated_cost: float = Field(
        ge=0.0,
    )


class UsageByUserItem(BaseModel):
    """Usage totals for one tenant user."""

    user_id: UUID
    usage_events: int = Field(
        ge=0,
    )
    prompt_tokens: int = Field(
        ge=0,
    )
    completion_tokens: int = Field(
        ge=0,
    )
    total_tokens: int = Field(
        ge=0,
    )
    estimated_cost: float = Field(
        ge=0.0,
    )


class UsageQuotaResponse(BaseModel):
    """Tenant usage quota and consumption."""

    tenant_id: UUID
    period_start: datetime
    period_end: datetime
    token_quota: int = Field(
        ge=0,
    )
    tokens_used: int = Field(
        ge=0,
    )
    tokens_remaining: int = Field(
        ge=0,
    )
    utilization_percent: float = Field(
        ge=0.0,
        le=100.0,
    )


# ---------------------------------------------------------------------------
# Usage summary
# ---------------------------------------------------------------------------


@router.get(
    "/summary",
    response_model=UsageSummaryResponse,
    summary="Usage summary",
    description=(
        "Return aggregate token usage and estimated cost for the tenant."
    ),
)
async def get_usage_summary(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "usage:read"
            )
        ),
    ],
    db: DatabaseSession,
    start_date: Annotated[
        datetime | None,
        Query(
            description=(
                "Include usage events occurring on or after this time."
            ),
        ),
    ] = None,
    end_date: Annotated[
        datetime | None,
        Query(
            description=(
                "Include usage events occurring on or before this time."
            ),
        ),
    ] = None,
) -> UsageSummaryResponse:
    """Return tenant-level usage totals."""
    try:
        summary = (
            await UsageRepository.summarize_usage_by_tenant(
                db,
                tenant_id=principal.tenant_id,
                start_date=start_date,
                end_date=end_date,
            )
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    await db.commit()

    return UsageSummaryResponse(
        tenant_id=principal.tenant_id,
        usage_events=summary.get(
            "usage_events",
            0,
        ),
        prompt_tokens=summary.get(
            "prompt_tokens",
            0,
        ),
        completion_tokens=summary.get(
            "completion_tokens",
            0,
        ),
        total_tokens=summary.get(
            "total_tokens",
            0,
        ),
        estimated_cost=summary.get(
            "estimated_cost",
            0.0,
        ),
    )


# ---------------------------------------------------------------------------
# Usage breakdowns
# ---------------------------------------------------------------------------


@router.get(
    "/models",
    response_model=list[UsageByModelItem],
    summary="Usage by model",
    description=(
        "Return tenant usage grouped by AI provider and model."
    ),
)
async def get_usage_by_model(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "usage:read"
            )
        ),
    ],
    db: DatabaseSession,
    start_date: Annotated[
        datetime | None,
        Query(
            description=(
                "Include usage events occurring on or after this time."
            ),
        ),
    ] = None,
    end_date: Annotated[
        datetime | None,
        Query(
            description=(
                "Include usage events occurring on or before this time."
            ),
        ),
    ] = None,
) -> list[UsageByModelItem]:
    """Return tenant usage grouped by provider and model."""
    usage_by_model = (
        await UsageRepository.summarize_usage_by_model(
            db,
            tenant_id=principal.tenant_id,
            start_date=start_date,
            end_date=end_date,
        )
    )

    await db.commit()

    return [
        UsageByModelItem(
            provider=item.get(
                "provider",
                "unknown",
            ),
            model=item.get(
                "model",
                "unknown",
            ),
            usage_events=item.get(
                "usage_events",
                0,
            ),
            prompt_tokens=item.get(
                "prompt_tokens",
                0,
            ),
            completion_tokens=item.get(
                "completion_tokens",
                0,
            ),
            total_tokens=item.get(
                "total_tokens",
                0,
            ),
            estimated_cost=item.get(
                "estimated_cost",
                0.0,
            ),
        )
        for item in usage_by_model
    ]


@router.get(
    "/users",
    response_model=list[UsageByUserItem],
    summary="Usage by user",
    description=(
        "Return tenant usage grouped by individual user."
    ),
)
async def get_usage_by_user(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "usage:read"
            )
        ),
    ],
    db: DatabaseSession,
    start_date: Annotated[
        datetime | None,
        Query(
            description=(
                "Include usage events occurring on or after this time."
            ),
        ),
    ] = None,
    end_date: Annotated[
        datetime | None,
        Query(
            description=(
                "Include usage events occurring on or before this time."
            ),
        ),
    ] = None,
) -> list[UsageByUserItem]:
    """Return tenant usage grouped by user."""
    usage_by_user = (
        await UsageRepository.summarize_usage_by_user(
            db,
            tenant_id=principal.tenant_id,
            start_date=start_date,
            end_date=end_date,
        )
    )

    await db.commit()

    return [
        UsageByUserItem(
            user_id=item["user_id"],
            usage_events=item.get(
                "usage_events",
                0,
            ),
            prompt_tokens=item.get(
                "prompt_tokens",
                0,
            ),
            completion_tokens=item.get(
                "completion_tokens",
                0,
            ),
            total_tokens=item.get(
                "total_tokens",
                0,
            ),
            estimated_cost=item.get(
                "estimated_cost",
                0.0,
            ),
        )
        for item in usage_by_user
    ]


# ---------------------------------------------------------------------------
# Usage events
# ---------------------------------------------------------------------------


@router.get(
    "/events",
    summary="List usage events",
    description=(
        "List tenant-scoped provider usage events with optional filters."
    ),
)
async def list_usage_events(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "usage:read"
            )
        ),
    ],
    db: DatabaseSession,
    pagination: Pagination,
    provider: Annotated[
        str | None,
        Query(
            max_length=100,
            description="Filter by provider name.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        Query(
            max_length=150,
            description="Filter by model name.",
        ),
    ] = None,
    user_id: Annotated[
        UUID | None,
        Query(
            description="Filter by user ID.",
        ),
    ] = None,
    execution_id: Annotated[
        UUID | None,
        Query(
            description="Filter by workflow execution ID.",
        ),
    ] = None,
    start_date: Annotated[
        datetime | None,
        Query(
            description=(
                "Include usage events occurring on or after this time."
            ),
        ),
    ] = None,
    end_date: Annotated[
        datetime | None,
        Query(
            description=(
                "Include usage events occurring on or before this time."
            ),
        ),
    ] = None,
):
    """List tenant usage events."""
    events, total_count = (
        await UsageRepository.list_usage_events(
            db,
            tenant_id=principal.tenant_id,
            skip=pagination.skip,
            limit=pagination.limit,
            provider=provider,
            model=model,
            user_id=user_id,
            execution_id=execution_id,
            start_date=start_date,
            end_date=end_date,
        )
    )

    await db.commit()

    return build_paginated_response(
        events,
        skip=pagination.skip,
        limit=pagination.limit,
        total_count=total_count,
    )


@router.get(
    "/events/{usage_event_id}",
    response_model=UsageEventResponse,
    summary="Get usage event",
    description=(
        "Retrieve a tenant-scoped provider usage event."
    ),
)
async def get_usage_event(
    usage_event_id: UUID,
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "usage:read"
            )
        ),
    ],
    db: DatabaseSession,
) -> UsageEventResponse:
    """Retrieve a single usage event."""
    usage_event = (
        await UsageRepository.get_usage_event(
            db,
            usage_event_id=usage_event_id,
            tenant_id=principal.tenant_id,
        )
    )

    if not usage_event:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Usage event not found.",
        )

    await db.commit()

    return usage_event


# ---------------------------------------------------------------------------
# Quotas and exports
# ---------------------------------------------------------------------------


@router.get(
    "/quota",
    response_model=UsageQuotaResponse,
    summary="Usage quota",
    description=(
        "Return the tenant's current token quota and consumption."
    ),
)
async def get_usage_quota(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "usage:read"
            )
        ),
    ],
    db: DatabaseSession,
) -> UsageQuotaResponse:
    """Return tenant quota consumption for the current period."""
    quota = (
        await UsageRepository.get_usage_quota(
            db,
            tenant_id=principal.tenant_id,
        )
    )

    if not quota:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No usage quota is configured for this tenant.",
        )

    await db.commit()

    tokens_used = quota.get(
        "tokens_used",
        0,
    )

    token_quota = quota.get(
        "token_quota",
        0,
    )

    tokens_remaining = max(
        token_quota - tokens_used,
        0,
    )

    utilization_percent = (
        round(
            (
                tokens_used
                / token_quota
                * 100
            ),
            2,
        )
        if token_quota > 0
        else 0.0
    )

    return UsageQuotaResponse(
        tenant_id=principal.tenant_id,
        period_start=quota[
            "period_start"
        ],
        period_end=quota[
            "period_end"
        ],
        token_quota=token_quota,
        tokens_used=tokens_used,
        tokens_remaining=tokens_remaining,
        utilization_percent=min(
            utilization_percent,
            100.0,
        ),
    )


@router.get(
    "/export",
    summary="Export usage events",
    description=(
        "Export tenant usage events as CSV. This operation is recorded in "
        "the tenant audit trail."
    ),
)
async def export_usage_events(
    principal: Annotated[
        object,
        Depends(
            require_scoped_access(
                "usage:export"
            )
        ),
    ],
    db: DatabaseSession,
    response: Response,
    start_date: Annotated[
        datetime | None,
        Query(
            description=(
                "Include usage events occurring on or after this time."
            ),
        ),
    ] = None,
    end_date: Annotated[
        datetime | None,
        Query(
            description=(
                "Include usage events occurring on or before this time."
            ),
        ),
    ] = None,
) -> Response:
    """Export tenant usage events as CSV."""
    events, _ = (
        await UsageRepository.list_usage_events(
            db,
            tenant_id=principal.tenant_id,
            skip=0,
            limit=10_000,
            start_date=start_date,
            end_date=end_date,
        )
    )

    csv_lines = [
        "usage_event_id,tenant_id,user_id,execution_id,"
        "provider,model,operation,prompt_tokens,"
        "completion_tokens,total_tokens,estimated_cost,occurred_at"
    ]

    for event in events:
        csv_lines.append(
            ",".join(
                [
                    str(event.id),
                    str(event.tenant_id),
                    str(event.user_id or ""),
                    str(event.execution_id or ""),
                    event.provider,
                    event.model,
                    event.operation or "",
                    str(event.prompt_tokens),
                    str(event.completion_tokens),
                    str(event.total_tokens),
                    str(
                        event.estimated_cost
                        if event.estimated_cost
                        is not None
                        else ""
                    ),
                    event.occurred_at.isoformat(),
                ]
            )
        )

    csv_content = "\n".join(
        csv_lines
    )

    await AuditLogRepository.record_event(
        db,
        tenant_id=principal.tenant_id,
        event_type="usage_exported",
        actor_id=principal.user_id,
        resource_type="usage",
        action="export_usage_events",
        outcome="completed",
        metadata={
            "event_count": len(
                events
            ),
            "start_date": (
                start_date.isoformat()
                if start_date
                else None
            ),
            "end_date": (
                end_date.isoformat()
                if end_date
                else None
            ),
        },
    )

    await db.commit()

    logger.info(
        "Usage events exported.",
        extra={
            "tenant_id": str(
                principal.tenant_id
            ),
            "user_id": str(
                principal.user_id
            ),
            "event_count": len(
                events
            ),
        },
    )

    filename = (
        "usage-events-"
        + datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        + ".csv"
    )

    response.headers[
        "Content-Disposition"
    ] = f'attachment; filename="{filename}"'

    response.media_type = (
        "text/csv"
    )

    response.body = csv_content.encode(
        "utf-8"
    )

    return response