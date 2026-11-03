"""
Citation repository for retrieved evidence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ValidationError
from app.models.citation_model import CitationModel
from app.repositories.base_repository import BaseRepository


logger = logging.getLogger(
    "agentic_bridge.repositories.citation"
)


class CitationRepository(
    BaseRepository[CitationModel]
):
    """Tenant-scoped persistence for workflow citations."""

    model = CitationModel

    @classmethod
    async def create_citation(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        execution_id: str | UUID,
        document_id: str | UUID,
        embedding_id: str | UUID,
        similarity: float,
        source_chunk: Optional[
            str
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> CitationModel:
        """Create a citation linking generated output to retrieved evidence."""
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
        normalized_document_id = (
            cls.normalize_uuid(
                document_id,
                "document_id",
            )
        )
        normalized_embedding_id = (
            cls.normalize_uuid(
                embedding_id,
                "embedding_id",
            )
        )

        if not isinstance(
            similarity, (int, float)
        ) or isinstance(
            similarity, bool
        ):
            raise ValidationError(
                "similarity must be numeric."
            )

        similarity = float(similarity)

        if not 0.0 <= similarity <= 1.0:
            raise ValidationError(
                "similarity must be between 0.0 and 1.0."
            )

        source_chunk = cls.normalize_text(
            source_chunk,
            "source_chunk",
            max_length=10_000,
        )

        if metadata is not None and not isinstance(
            metadata, dict
        ):
            raise ValidationError(
                "metadata must be a dictionary."
            )

        db_record = CitationModel(
            tenant_id=str(
                normalized_tenant_id
            ),
            execution_id=str(
                normalized_execution_id
            ),
            document_id=str(
                normalized_document_id
            ),
            embedding_id=str(
                normalized_embedding_id
            ),
            similarity=similarity,
            source_chunk=source_chunk,
            metadata=metadata or {},
        )

        db.add(db_record)
        await db.flush()
        await db.refresh(db_record)

        logger.info(
            "Created citation record.",
            extra={
                "citation_id": str(
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
    async def list_citations_for_execution(
        cls,
        db: AsyncSession,
        execution_id: str | UUID,
        tenant_id: str | UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[
        list[CitationModel], int
    ]:
        """List citations for a tenant-scoped execution."""
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
                cls.model.similarity.desc()
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
    async def get_citations_for_document(
        cls,
        db: AsyncSession,
        document_id: str | UUID,
        tenant_id: str | UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[
        list[CitationModel], int
    ]:
        """List citations associated with a tenant document."""
        normalized_document_id = (
            cls.normalize_uuid(
                document_id,
                "document_id",
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
            cls.model.document_id
            == str(
                normalized_document_id
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
    async def delete_citations_for_execution(
        cls,
        db: AsyncSession,
        execution_id: str | UUID,
        tenant_id: str | UUID,
    ) -> int:
        """Delete citations associated with a tenant execution."""
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