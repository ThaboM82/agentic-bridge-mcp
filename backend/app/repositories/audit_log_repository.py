"""
Append-only audit-log repository.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ValidationError
from app.models.audit_log_model import AuditLogModel
from app.repositories.base_repository import BaseRepository


logger = logging.getLogger(
    "agentic_bridge.repositories.audit_log"
)


SENSITIVE_METADATA_KEYS = {
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "credential",
    "cookie",
    "private_key",
    "database_url",
    "connection_string",
    "prompt",
    "completion",
    "messages",
    "embedding",
    "vector",
    "generated_code",
    "signed_url",
    "storage_url",
    "file_url",
    "download_url",
}


class AuditLogRepository(
    BaseRepository[AuditLogModel]
):
    """Tenant-scoped append-only audit event persistence."""

    model = AuditLogModel

    @staticmethod
    def _validate_metadata(
        metadata: Dict[str, Any]
        | None,
    ) -> Dict[str, Any]:
        if metadata is None:
            return {}

        if not isinstance(
            metadata, dict
        ):
            raise ValidationError(
                "metadata must be a dictionary."
            )

        if len(metadata) > 50:
            raise ValidationError(
                "metadata exceeds the maximum supported number of keys."
            )

        def inspect(
            value: Any,
        ) -> None:
            if isinstance(value, dict):
                for key, nested_value in value.items():
                    if (
                        isinstance(key, str)
                        and key.lower()
                        in SENSITIVE_METADATA_KEYS
                    ):
                        raise ValidationError(
                            "Audit metadata must not contain sensitive keys."
                        )

                    inspect(nested_value)

            elif isinstance(
                value, (list, tuple)
            ):
                for item in value:
                    inspect(item)

        inspect(metadata)

        return metadata

    @classmethod
    async def record_event(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        event_type: str,
        actor_id: Optional[
            str | UUID
        ] = None,
        execution_id: Optional[
            str | UUID
        ] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[
            str | UUID
        ] = None,
        action: Optional[str] = None,
        outcome: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> AuditLogModel:
        """Persist an immutable audit event."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        normalized_actor_id = (
            cls.normalize_optional_uuid(
                actor_id,
                "actor_id",
            )
        )
        normalized_execution_id = (
            cls.normalize_optional_uuid(
                execution_id,
                "execution_id",
            )
        )
        normalized_resource_id = (
            cls.normalize_optional_uuid(
                resource_id,
                "resource_id",
            )
        )

        event_type = cls.normalize_text(
            event_type,
            "event_type",
            required=True,
            max_length=100,
        )
        resource_type = cls.normalize_text(
            resource_type,
            "resource_type",
            max_length=100,
        )
        action = cls.normalize_text(
            action,
            "action",
            max_length=100,
        )
        outcome = cls.normalize_text(
            outcome,
            "outcome",
            max_length=50,
        )
        ip_address = cls.normalize_text(
            ip_address,
            "ip_address",
            max_length=64,
        )
        user_agent = cls.normalize_text(
            user_agent,
            "user_agent",
            max_length=500,
        )

        validated_metadata = (
            cls._validate_metadata(
                metadata
            )
        )

        db_record = AuditLogModel(
            tenant_id=str(
                normalized_tenant_id
            ),
            event_type=event_type,
            actor_id=str(
                normalized_actor_id
            )
            if normalized_actor_id
            else None,
            execution_id=str(
                normalized_execution_id
            )
            if normalized_execution_id
            else None,
            resource_type=resource_type,
            resource_id=str(
                normalized_resource_id
            )
            if normalized_resource_id
            else None,
            action=action,
            outcome=outcome,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=validated_metadata,
            occurred_at=datetime.now(
                timezone.utc
            ),
        )

        db.add(db_record)
        await db.flush()
        await db.refresh(db_record)

        logger.info(
            "Recorded audit event.",
            extra={
                "audit_id": str(
                    db_record.id
                ),
                "tenant_id": str(
                    normalized_tenant_id
                ),
                "event_type": event_type,
            },
        )

        return db_record

    @classmethod
    async def get_events_for_execution(
        cls,
        db: AsyncSession,
        execution_id: str | UUID,
        tenant_id: str | UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[
        list[AuditLogModel], int
    ]:
        """List audit events for a tenant-scoped execution."""
        normalized_execution_id = (
            cls.normalize_uuid(
                execution_id,
                "execution_id",
            )
        )
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        normalized_skip, normalized_limit = (
            cls.normalize_pagination(
                skip, limit
            )
        )

        filters = [
            cls.model.tenant_id
            == str(normalized_tenant_id),
            cls.model.execution_id
            == str(
                normalized_execution_id
            ),
        ]

        count_stmt = (
            select(func.count())
            .select_from(cls.model)
            .where(*filters)
        )

        total_result = await db.execute(
            count_stmt
        )
        total_count = (
            total_result.scalar_one()
        )

        list_stmt = (
            select(cls.model)
            .where(*filters)
            .order_by(
                cls.model.occurred_at.desc()
            )
            .offset(normalized_skip)
            .limit(normalized_limit)
        )

        result = await db.execute(
            list_stmt
        )

        return (
            list(result.scalars().all()),
            total_count,
        )

    @classmethod
    async def list_events_by_event_type(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        event_type: str,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[
        list[AuditLogModel], int
    ]:
        """List tenant-scoped audit events by event type."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        event_type = cls.normalize_text(
            event_type,
            "event_type",
            required=True,
            max_length=100,
        )
        normalized_skip, normalized_limit = (
            cls.normalize_pagination(
                skip, limit
            )
        )

        filters = [
            cls.model.tenant_id
            == str(normalized_tenant_id),
            cls.model.event_type
            == event_type,
        ]

        count_stmt = (
            select(func.count())
            .select_from(cls.model)
            .where(*filters)
        )

        total_result = await db.execute(
            count_stmt
        )
        total_count = (
            total_result.scalar_one()
        )

        list_stmt = (
            select(cls.model)
            .where(*filters)
            .order_by(
                cls.model.occurred_at.desc()
            )
            .offset(normalized_skip)
            .limit(normalized_limit)
        )

        result = await db.execute(
            list_stmt
        )

        return (
            list(result.scalars().all()),
            total_count,
        )

    @classmethod
    async def count_events(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        event_type: Optional[str] = None,
    ) -> int:
        """Count tenant-scoped audit events."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )

        stmt = (
            select(func.count())
            .select_from(cls.model)
            .where(
                cls.model.tenant_id
                == str(normalized_tenant_id)
            )
        )

        if event_type:
            normalized_event_type = (
                cls.normalize_text(
                    event_type,
                    "event_type",
                    required=True,
                    max_length=100,
                )
            )

            stmt = stmt.where(
                cls.model.event_type
                == normalized_event_type
            )

        result = await db.execute(stmt)

        return result.scalar_one()