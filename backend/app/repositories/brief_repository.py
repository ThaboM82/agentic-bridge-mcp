"""
Brief repository.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ValidationError
from app.models.brief_model import BriefModel
from app.repositories.base_repository import BaseRepository


logger = logging.getLogger(
    "agentic_bridge.repositories.brief"
)


class BriefRepository(
    BaseRepository[BriefModel]
):
    """Tenant-scoped persistence for product briefs."""

    model = BriefModel

    @classmethod
    async def create_brief(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        created_by: str | UUID,
        title: str,
        raw_brief_text: str,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> BriefModel:
        """Create a tenant-scoped brief."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        normalized_created_by = (
            cls.normalize_uuid(
                created_by,
                "created_by",
            )
        )

        title = cls.normalize_text(
            title,
            "title",
            required=True,
            max_length=300,
        )
        raw_brief_text = cls.normalize_text(
            raw_brief_text,
            "raw_brief_text",
            required=True,
            max_length=100_000,
        )

        if metadata is not None and not isinstance(
            metadata, dict
        ):
            raise ValidationError(
                "metadata must be a dictionary."
            )

        db_record = BriefModel(
            tenant_id=str(
                normalized_tenant_id
            ),
            created_by=str(
                normalized_created_by
            ),
            title=title,
            raw_brief_text=raw_brief_text,
            metadata=metadata or {},
            status="active",
        )

        db.add(db_record)
        await db.flush()
        await db.refresh(db_record)

        logger.info(
            "Created brief record.",
            extra={
                "brief_id": str(
                    db_record.id
                ),
                "tenant_id": str(
                    normalized_tenant_id
                ),
            },
        )

        return db_record

    @classmethod
    async def get_brief(
        cls,
        db: AsyncSession,
        brief_id: str | UUID,
        tenant_id: str | UUID,
    ) -> Optional[BriefModel]:
        """Retrieve a tenant-scoped brief."""
        return await cls.get_by_id(
            db,
            record_id=brief_id,
            tenant_id=tenant_id,
        )

    @classmethod
    async def list_briefs(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        skip: int = 0,
        limit: int = 20,
        search_term: Optional[str] = None,
    ) -> tuple[
        list[BriefModel], int
    ]:
        """List tenant-scoped briefs with optional title search."""
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
            == str(normalized_tenant_id)
        ]

        if search_term:
            normalized_search_term = (
                cls.normalize_text(
                    search_term,
                    "search_term",
                    required=True,
                    max_length=200,
                )
            )

            filters.append(
                cls.model.title.ilike(
                    f"%{normalized_search_term}%"
                )
            )

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
    async def update_brief(
        cls,
        db: AsyncSession,
        brief_id: str | UUID,
        tenant_id: str | UUID,
        title: Optional[str] = None,
        raw_brief_text: Optional[
            str
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Optional[BriefModel]:
        """Update a tenant-scoped brief."""
        record = await cls.get_by_id(
            db,
            record_id=brief_id,
            tenant_id=tenant_id,
        )

        if not record:
            return None

        if title is not None:
            record.title = (
                cls.normalize_text(
                    title,
                    "title",
                    required=True,
                    max_length=300,
                )
            )

        if raw_brief_text is not None:
            record.raw_brief_text = (
                cls.normalize_text(
                    raw_brief_text,
                    "raw_brief_text",
                    required=True,
                    max_length=100_000,
                )
            )

        if metadata is not None:
            if not isinstance(
                metadata, dict
            ):
                raise ValidationError(
                    "metadata must be a dictionary."
                )

            record.metadata = metadata

        record.updated_at = datetime.now(
            timezone.utc
        )

        await db.flush()
        await db.refresh(record)

        return record

    @classmethod
    async def archive_brief(
        cls,
        db: AsyncSession,
        brief_id: str | UUID,
        tenant_id: str | UUID,
    ) -> Optional[BriefModel]:
        """Archive a tenant-scoped brief."""
        record = await cls.get_by_id(
            db,
            record_id=brief_id,
            tenant_id=tenant_id,
        )

        if not record:
            return None

        record.status = "archived"
        record.updated_at = datetime.now(
            timezone.utc
        )

        await db.flush()
        await db.refresh(record)

        return record