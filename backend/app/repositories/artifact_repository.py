"""
Artifact repository for generated workflow deliverables.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ValidationError
from app.models.artifact_model import ArtifactModel
from app.repositories.base_repository import BaseRepository


logger = logging.getLogger(
    "agentic_bridge.repositories.artifact"
)


class ArtifactRepository(
    BaseRepository[ArtifactModel]
):
    """Tenant-scoped persistence for generated artifacts."""

    model = ArtifactModel

    @classmethod
    async def create_artifact(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        execution_id: str | UUID,
        artifact_type: str,
        storage_key: str,
        file_size_bytes: int,
        checksum: Optional[str] = None,
        mime_type: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> ArtifactModel:
        """Create a generated-artifact record."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        normalized_execution_id = (
            cls.normalize_uuid(
                execution_id,
                "execution_id",
            )
        )

        artifact_type = cls.normalize_text(
            artifact_type,
            "artifact_type",
            required=True,
            max_length=100,
        )
        storage_key = cls.normalize_text(
            storage_key,
            "storage_key",
            required=True,
            max_length=1_000,
        )
        checksum = cls.normalize_text(
            checksum,
            "checksum",
            max_length=256,
        )
        mime_type = cls.normalize_text(
            mime_type,
            "mime_type",
            max_length=150,
        )

        if (
            not isinstance(
                file_size_bytes, int
            )
            or isinstance(
                file_size_bytes, bool
            )
            or file_size_bytes <= 0
        ):
            raise ValidationError(
                "file_size_bytes must be a positive integer."
            )

        if metadata is not None and not isinstance(
            metadata, dict
        ):
            raise ValidationError(
                "metadata must be a dictionary."
            )

        db_record = ArtifactModel(
            tenant_id=str(
                normalized_tenant_id
            ),
            execution_id=str(
                normalized_execution_id
            ),
            artifact_type=artifact_type,
            storage_key=storage_key,
            file_size_bytes=file_size_bytes,
            checksum=checksum,
            mime_type=mime_type,
            metadata=metadata or {},
            status="created",
        )

        db.add(db_record)
        await db.flush()
        await db.refresh(db_record)

        logger.info(
            "Created artifact record.",
            extra={
                "artifact_id": str(
                    db_record.id
                ),
                "tenant_id": str(
                    normalized_tenant_id
                ),
                "execution_id": str(
                    normalized_execution_id
                ),
            },
        )

        return db_record

    @classmethod
    async def list_artifacts_for_execution(
        cls,
        db: AsyncSession,
        execution_id: str | UUID,
        tenant_id: str | UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[
        list[ArtifactModel], int
    ]:
        """List artifacts for a tenant-scoped execution."""
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
                cls.model.created_at.desc()
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
    async def mark_uploaded(
        cls,
        db: AsyncSession,
        artifact_id: str | UUID,
        tenant_id: str | UUID,
        storage_key: str,
    ) -> Optional[ArtifactModel]:
        """Mark an artifact as uploaded to object storage."""
        storage_key = cls.normalize_text(
            storage_key,
            "storage_key",
            required=True,
            max_length=1_000,
        )

        record = await cls.get_by_id(
            db,
            record_id=artifact_id,
            tenant_id=tenant_id,
        )

        if not record:
            return None

        record.storage_key = storage_key
        record.status = "uploaded"
        record.uploaded_at = datetime.now(
            timezone.utc
        )

        await db.flush()
        await db.refresh(record)

        return record

    @classmethod
    async def mark_expired(
        cls,
        db: AsyncSession,
        artifact_id: str | UUID,
        tenant_id: str | UUID,
    ) -> Optional[ArtifactModel]:
        """Mark an artifact as expired."""
        record = await cls.get_by_id(
            db,
            record_id=artifact_id,
            tenant_id=tenant_id,
        )

        if not record:
            return None

        record.status = "expired"
        record.expired_at = datetime.now(
            timezone.utc
        )

        await db.flush()
        await db.refresh(record)

        return record

    @classmethod
    async def delete_artifacts_for_execution(
        cls,
        db: AsyncSession,
        execution_id: str | UUID,
        tenant_id: str | UUID,
    ) -> int:
        """Delete artifact records for a tenant execution."""
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

        stmt = (
            delete(cls.model)
            .where(
                cls.model.execution_id
                == str(
                    normalized_execution_id
                )
            )
            .where(
                cls.model.tenant_id
                == str(normalized_tenant_id)
            )
        )

        result = await db.execute(stmt)
        await db.flush()

        return result.rowcount or 0