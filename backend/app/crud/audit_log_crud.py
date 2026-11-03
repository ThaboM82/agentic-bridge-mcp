"""
CRUD operations for immutable audit log events.

Audit records should capture security-relevant and business-critical events,
such as authentication changes, approval decisions, workflow publication,
document deletion, and administrative actions.

Responsibilities:
- Create immutable audit events.
- Retrieve events by ID, actor, resource, tenant, or event type.
- List events with pagination and time-range filters.
- Map database failures to stable application exceptions.

Non-goals:
- Do not update or delete audit records through normal application flows.
- Do not store passwords, API keys, tokens, raw prompts, or full request bodies.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    DatabaseIntegrityError,
    NotFoundError,
    PersistenceError,
    ValidationError,
)
from app.core.pagination import Page, PaginationParams
from app.crud.base import create_record, get_by_id_or_404
from app.models.audit_log_model import AuditLogModel


logger = logging.getLogger("agentic_bridge.crud.audit_log")

MAX_EVENT_TYPE_LENGTH = 100
MAX_ACTOR_LENGTH = 128
MAX_RESOURCE_TYPE_LENGTH = 100
MAX_RESOURCE_ID_LENGTH = 128
MAX_IP_ADDRESS_LENGTH = 64
MAX_USER_AGENT_LENGTH = 512
MAX_METADATA_JSON_CHARS = 16_000


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _normalize_audit_id(audit_id: str | UUID) -> UUID:
    """Validate and normalize an audit-event ID."""
    if isinstance(audit_id, UUID):
        return audit_id

    if not isinstance(audit_id, str) or not audit_id.strip():
        raise ValidationError("audit_id must be a non-empty string.")

    try:
        return UUID(audit_id.strip())
    except ValueError as exc:
        raise ValidationError(
            "audit_id must be a valid UUID."
        ) from exc


def _validate_required_text(
    value: str,
    field_name: str,
    max_length: int,
) -> str:
    """Validate a required bounded text field."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} is required.")

    normalized = value.strip()

    if len(normalized) > max_length:
        raise ValidationError(
            f"{field_name} may not exceed {max_length} characters."
        )

    return normalized


def _validate_optional_text(
    value: str | None,
    field_name: str,
    max_length: int,
) -> str | None:
    """Validate an optional bounded text field."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string.")

    normalized = value.strip()

    if not normalized:
        return None

    if len(normalized) > max_length:
        raise ValidationError(
            f"{field_name} may not exceed {max_length} characters."
        )

    return normalized


def _validate_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """
    Validate audit metadata.

    Rejects obviously sensitive keys and bounds the serialized payload size.
    """
    if metadata is None:
        return {}

    if not isinstance(metadata, dict):
        raise ValidationError("Audit metadata must be an object.")

    sensitive_keys = {
        "password",
        "secret",
        "secret_key",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "cookie",
        "database_url",
        "connection_string",
    }

    for key in metadata:
        if str(key).lower() in sensitive_keys:
            raise ValidationError(
                f"Sensitive field '{key}' cannot be stored in audit metadata."
            )

    try:
        serialized_length = len(str(metadata))
    except Exception as exc:
        raise ValidationError(
            "Audit metadata must be JSON-serializable."
        ) from exc

    if serialized_length > MAX_METADATA_JSON_CHARS:
        raise ValidationError(
            "Audit metadata exceeds the maximum allowed size."
        )

    return metadata


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _apply_tenant_filter(query, tenant_id: str | None):
    """Apply tenant isolation to an audit query."""
    if tenant_id:
        return query.where(AuditLogModel.tenant_id == tenant_id)

    return query


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


async def create_audit_event(
    db: AsyncSession,
    *,
    event_type: str,
    actor_id: str | None = None,
    actor_type: str = "user",
    tenant_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    action: str | None = None,
    outcome: str = "success",
    ip_address: str | None = None,
    user_agent: str | None = None,
    request_id: str | None = None,
    workflow_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AuditLogModel:
    """
    Create and persist an immutable audit event.

    Args:
        db: Active async database session.
        event_type: Stable event name, such as `workflow_published`.
        actor_id: User, service, or system identifier.
        actor_type: Actor category, such as `user`, `service`, or `system`.
        tenant_id: Owning tenant identifier.
        resource_type: Resource category, such as `document` or `workflow`.
        resource_id: Resource identifier.
        action: Action performed, such as `create`, `update`, or `delete`.
        outcome: Event outcome, such as `success`, `failure`, or `denied`.
        ip_address: Client IP address, when available.
        user_agent: Bounded client user agent.
        request_id: HTTP request correlation ID.
        workflow_id: Related workflow execution ID.
        metadata: Non-sensitive structured event context.

    Returns:
        The persisted audit event.

    Raises:
        ValidationError: If required fields or metadata are invalid.
        DatabaseIntegrityError: If a constraint is violated.
        PersistenceError: If another database failure occurs.
    """
    safe_event_type = _validate_required_text(
        event_type,
        "event_type",
        MAX_EVENT_TYPE_LENGTH,
    )

    safe_actor_id = _validate_optional_text(
        actor_id,
        "actor_id",
        MAX_ACTOR_LENGTH,
    )

    safe_actor_type = _validate_required_text(
        actor_type,
        "actor_type",
        50,
    )

    safe_resource_type = _validate_optional_text(
        resource_type,
        "resource_type",
        MAX_RESOURCE_TYPE_LENGTH,
    )

    safe_resource_id = _validate_optional_text(
        resource_id,
        "resource_id",
        MAX_RESOURCE_ID_LENGTH,
    )

    safe_action = _validate_optional_text(
        action,
        "action",
        100,
    )

    safe_outcome = _validate_required_text(
        outcome,
        "outcome",
        50,
    )

    safe_ip_address = _validate_optional_text(
        ip_address,
        "ip_address",
        MAX_IP_ADDRESS_LENGTH,
    )

    safe_user_agent = _validate_optional_text(
        user_agent,
        "user_agent",
        MAX_USER_AGENT_LENGTH,
    )

    safe_request_id = _validate_optional_text(
        request_id,
        "request_id",
        128,
    )

    safe_workflow_id = _validate_optional_text(
        workflow_id,
        "workflow_id",
        128,
    )

    safe_metadata = _validate_metadata(metadata)

    audit_event = AuditLogModel(
        event_type=safe_event_type,
        actor_id=safe_actor_id,
        actor_type=safe_actor_type,
        tenant_id=tenant_id,
        resource_type=safe_resource_type,
        resource_id=safe_resource_id,
        action=safe_action,
        outcome=safe_outcome,
        ip_address=safe_ip_address,
        user_agent=safe_user_agent,
        request_id=safe_request_id,
        workflow_id=safe_workflow_id,
        metadata=safe_metadata,
        occurred_at=_utc_now(),
    )

    audit_event = await create_record(db, audit_event)

    logger.info(
        "audit_event_created",
        audit_id=str(audit_event.id),
        event_type=audit_event.event_type,
        actor_id=audit_event.actor_id,
        resource_type=audit_event.resource_type,
        resource_id=audit_event.resource_id,
        outcome=audit_event.outcome,
    )

    return audit_event


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


async def get_audit_event(
    db: AsyncSession,
    audit_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> AuditLogModel | None:
    """
    Retrieve an audit event by ID.

    Args:
        db: Active async database session.
        audit_id: Audit-event UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The audit event, or `None` when it does not exist or belongs to
        another tenant.
    """
    audit_uuid = _normalize_audit_id(audit_id)

    query = select(AuditLogModel).where(
        AuditLogModel.id == audit_uuid
    )

    query = _apply_tenant_filter(query, tenant_id)

    result = await db.execute(query)

    return result.scalar_one_or_none()


async def get_audit_event_or_404(
    db: AsyncSession,
    audit_id: str | UUID,
    *,
    tenant_id: str | None = None,
) -> AuditLogModel:
    """
    Retrieve an audit event by ID or raise `NotFoundError`.

    Args:
        db: Active async database session.
        audit_id: Audit-event UUID.
        tenant_id: Optional tenant isolation filter.

    Returns:
        The matching audit event.

    Raises:
        NotFoundError: If the event does not exist or is outside the tenant.
    """
    audit_event = await get_audit_event(
        db,
        audit_id,
        tenant_id=tenant_id,
    )

    if audit_event is None:
        raise NotFoundError(
            "Audit event was not found.",
            details={"audit_id": str(audit_id)},
        )

    return audit_event


async def list_audit_events(
    db: AsyncSession,
    *,
    pagination: PaginationParams,
    tenant_id: str | None = None,
    event_type: str | None = None,
    actor_id: str | None = None,
    actor_type: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    workflow_id: str | None = None,
    request_id: str | None = None,
    outcome: str | None = None,
    occurred_after: datetime | None = None,
    occurred_before: datetime | None = None,
) -> Page[AuditLogModel]:
    """
    Retrieve a paginated, filtered list of audit events.

    Args:
        db: Active async database session.
        pagination: Validated pagination parameters.
        tenant_id: Optional tenant isolation filter.
        event_type: Optional event-type filter.
        actor_id: Optional actor filter.
        actor_type: Optional actor-type filter.
        resource_type: Optional resource-type filter.
        resource_id: Optional resource-ID filter.
        workflow_id: Optional workflow filter.
        request_id: Optional request-correlation filter.
        outcome: Optional outcome filter.
        occurred_after: Optional inclusive lower occurred-at bound.
        occurred_before: Optional inclusive upper occurred-at bound.

    Returns:
        A `Page` containing audit events and pagination metadata.
    """
    filters = []

    if tenant_id:
        filters.append(AuditLogModel.tenant_id == tenant_id)

    if event_type:
        filters.append(
            AuditLogModel.event_type == event_type.strip().lower()
        )

    if actor_id:
        filters.append(AuditLogModel.actor_id == actor_id)

    if actor_type:
        filters.append(AuditLogModel.actor_type == actor_type)

    if resource_type:
        filters.append(AuditLogModel.resource_type == resource_type)

    if resource_id:
        filters.append(AuditLogModel.resource_id == resource_id)

    if workflow_id:
        filters.append(AuditLogModel.workflow_id == workflow_id)

    if request_id:
        filters.append(AuditLogModel.request_id == request_id)

    if outcome:
        filters.append(AuditLogModel.outcome == outcome.strip().lower())

    if occurred_after is not None:
        filters.append(AuditLogModel.occurred_at >= occurred_after)

    if occurred_before is not None:
        filters.append(AuditLogModel.occurred_at <= occurred_before)

    count_query = select(func.count()).select_from(AuditLogModel)

    if filters:
        count_query = count_query.where(and_(*filters))

    count_result = await db.execute(count_query)
    total = int(count_result.scalar_one())

    query = select(AuditLogModel).order_by(
        AuditLogModel.occurred_at.desc(),
        AuditLogModel.id.desc(),
    )

    if filters:
        query = query.where(and_(*filters))

    result = await db.execute(
        query.offset(pagination.skip).limit(pagination.limit)
    )

    audit_events = list(result.scalars().all())

    return Page[AuditLogModel].create(
        items=audit_events,
        total=total,
        skip=pagination.skip,
        limit=pagination.limit,
    )


async def list_audit_events_by_resource(
    db: AsyncSession,
    *,
    resource_type: str,
    resource_id: str,
    tenant_id: str | None = None,
    limit: int = 100,
) -> Sequence[AuditLogModel]:
    """
    Retrieve the audit history for a resource.

    Args:
        db: Active async database session.
        resource_type: Resource category.
        resource_id: Resource identifier.
        tenant_id: Optional tenant isolation filter.
        limit: Maximum number of events to return.

    Returns:
        Audit events for the resource, newest first.
    """
    safe_resource_type = _validate_required_text(
        resource_type,
        "resource_type",
        MAX_RESOURCE_TYPE_LENGTH,
    )

    safe_resource_id = _validate_required_text(
        resource_id,
        "resource_id",
        MAX_RESOURCE_ID_LENGTH,
    )

    query = select(AuditLogModel).where(
        and_(
            AuditLogModel.resource_type == safe_resource_type,
            AuditLogModel.resource_id == safe_resource_id,
        )
    )

    query = _apply_tenant_filter(query, tenant_id)

    query = query.order_by(
        AuditLogModel.occurred_at.desc(),
        AuditLogModel.id.desc(),
    ).limit(max(1, limit))

    result = await db.execute(query)

    return result.scalars().all()


async def list_audit_events_by_actor(
    db: AsyncSession,
    *,
    actor_id: str,
    tenant_id: str | None = None,
    limit: int = 100,
) -> Sequence[AuditLogModel]:
    """
    Retrieve audit events created by an actor.

    Args:
        db: Active async database session.
        actor_id: User, service, or system identifier.
        tenant_id: Optional tenant isolation filter.
        limit: Maximum number of events to return.

    Returns:
        Audit events for the actor, newest first.
    """
    safe_actor_id = _validate_required_text(
        actor_id,
        "actor_id",
        MAX_ACTOR_LENGTH,
    )

    query = select(AuditLogModel).where(
        AuditLogModel.actor_id == safe_actor_id
    )

    query = _apply_tenant_filter(query, tenant_id)

    query = query.order_by(
        AuditLogModel.occurred_at.desc(),
        AuditLogModel.id.desc(),
    ).limit(max(1, limit))

    result = await db.execute(query)

    return result.scalars().all()


async def count_audit_events(
    db: AsyncSession,
    *,
    tenant_id: str | None = None,
    event_type: str | None = None,
    outcome: str | None = None,
) -> int:
    """
    Count audit events matching optional filters.

    Args:
        db: Active async database session.
        tenant_id: Optional tenant isolation filter.
        event_type: Optional event-type filter.
        outcome: Optional outcome filter.

    Returns:
        Total matching audit-event count.
    """
    filters = []

    if tenant_id:
        filters.append(AuditLogModel.tenant_id == tenant_id)

    if event_type:
        filters.append(
            AuditLogModel.event_type == event_type.strip().lower()
        )

    if outcome:
        filters.append(AuditLogModel.outcome == outcome.strip().lower())

    query = select(func.count()).select_from(AuditLogModel)

    if filters:
        query = query.where(and_(*filters))

    result = await db.execute(query)

    return int(result.scalar_one())