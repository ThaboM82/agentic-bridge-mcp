"""
Document repository.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import (
    DocumentStatus,
    DocumentType,
)
from app.core.exceptions import ValidationError
from app.models.document_model import DocumentModel
from app.repositories.base_repository import BaseRepository


logger = logging.getLogger(
    "agentic_bridge.repositories.document"
)


class DocumentRepository(
    BaseRepository[DocumentModel]
):
    """Tenant-scoped persistence for uploaded documents."""

    model = DocumentModel

    @staticmethod
    def _normalize_document_type(
        document_type: str
        | DocumentType
        | None,
    ) -> str | None:
        if document_type is None:
            return None

        if isinstance(
            document_type, DocumentType
        ):
            return document_type.value

        if not isinstance(
            document_type, str
        ):
            raise ValidationError(
                "document_type must be a string or DocumentType."
            )

        normalized_type = (
            document_type.strip().lower()
        )

        try:
            return DocumentType(
                normalized_type
            ).value
        except ValueError as exc:
            raise ValidationError(
                f"Unsupported document type: {document_type!r}."
            ) from exc

    @staticmethod
    def _normalize_document_status(
        document_status: str
        | DocumentStatus
        | None,
    ) -> str | None:
        if document_status is None:
            return None

        if isinstance(
            document_status,
            DocumentStatus,
        ):
            return document_status.value

        if not isinstance(
            document_status, str
        ):
            raise ValidationError(
                "document_status must be a string or DocumentStatus."
            )

        normalized_status = (
            document_status.strip().lower()
        )

        try:
            return DocumentStatus(
                normalized_status
            ).value
        except ValueError as exc:
            raise ValidationError(
                f"Unsupported document status: {document_status!r}."
            ) from exc

    @staticmethod
    def _normalize_tags(
        tags: Sequence[str] | None,
    ) -> list[str]:
        if tags is None:
            return []

        if not isinstance(
            tags, (list, tuple)
        ):
            raise ValidationError(
                "tags must be a list of strings."
            )

        normalized_tags: list[str] = []

        for tag in tags:
            if not isinstance(tag, str):
                raise ValidationError(
                    "tags must contain only strings."
                )

            normalized_tag = (
                tag.strip().lower()
            )

            if (
                normalized_tag
                and normalized_tag
                not in normalized_tags
            ):
                normalized_tags.append(
                    normalized_tag
                )

        if len(normalized_tags) > 20:
            raise ValidationError(
                "documents support at most 20 tags."
            )

        for tag in normalized_tags:
            if len(tag) > 50:
                raise ValidationError(
                    "each document tag must be at most 50 characters."
                )

        return normalized_tags

    @classmethod
    async def create_document(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        uploaded_by: str | UUID,
        title: str,
        storage_key: str,
        mime_type: str,
        file_size_bytes: int,
        document_type: str
        | DocumentType
        | None = None,
        summary: Optional[str] = None,
        tags: Optional[
            Sequence[str]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> DocumentModel:
        """Create a tenant-scoped document record."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        normalized_uploaded_by = (
            cls.normalize_uuid(
                uploaded_by,
                "uploaded_by",
            )
        )

        title = cls.normalize_text(
            title,
            "title",
            required=True,
            max_length=300,
        )
        storage_key = cls.normalize_text(
            storage_key,
            "storage_key",
            required=True,
            max_length=1_000,
        )
        mime_type = cls.normalize_text(
            mime_type,
            "mime_type",
            required=True,
            max_length=150,
        )
        summary = cls.normalize_text(
            summary,
            "summary",
            max_length=5_000,
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

        normalized_type = (
            cls._normalize_document_type(
                document_type
            )
        )
        normalized_tags = (
            cls._normalize_tags(tags)
        )

        if metadata is not None and not isinstance(
            metadata, dict
        ):
            raise ValidationError(
                "metadata must be a dictionary."
            )

        db_record = DocumentModel(
            tenant_id=str(
                normalized_tenant_id
            ),
            uploaded_by=str(
                normalized_uploaded_by
            ),
            title=title,
            storage_key=storage_key,
            mime_type=mime_type,
            file_size_bytes=file_size_bytes,
            document_type=normalized_type,
            summary=summary,
            tags=normalized_tags,
            metadata=metadata or {},
            status=DocumentStatus.UPLOADED.value,
        )

        db.add(db_record)
        await db.flush()
        await db.refresh(db_record)

        logger.info(
            "Created document record.",
            extra={
                "document_id": str(
                    db_record.id
                ),
                "tenant_id": str(
                    normalized_tenant_id
                ),
            },
        )

        return db_record

    @classmethod
    async def get_document(
        cls,
        db: AsyncSession,
        document_id: str | UUID,
        tenant_id: str | UUID,
    ) -> Optional[DocumentModel]:
        """Retrieve a tenant-scoped document."""
        return await cls.get_by_id(
            db,
            record_id=document_id,
            tenant_id=tenant_id,
        )

    @classmethod
    async def list_documents(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        skip: int = 0,
        limit: int = 20,
        status_filter: Optional[
            str | DocumentStatus
        ] = None,
        document_type_filter: Optional[
            str | DocumentType
        ] = None,
        search_term: Optional[str] = None,
    ) -> tuple[
        list[DocumentModel], int
    ]:
        """List tenant-scoped documents with optional filters."""
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

        normalized_status = (
            cls._normalize_document_status(
                status_filter
            )
        )

        if normalized_status:
            filters.append(
                cls.model.status
                == normalized_status
            )

        normalized_type = (
            cls._normalize_document_type(
                document_type_filter
            )
        )

        if normalized_type:
            filters.append(
                cls.model.document_type
                == normalized_type
            )

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
    async def update_document_status(
        cls,
        db: AsyncSession,
        document_id: str | UUID,
        tenant_id: str | UUID,
        status: str | DocumentStatus,
    ) -> Optional[DocumentModel]:
        """Update a document processing status."""
        normalized_status = (
            cls._normalize_document_status(
                status
            )
        )

        if normalized_status is None:
            raise ValidationError(
                "status is required."
            )

        record = await cls.get_by_id(
            db,
            record_id=document_id,
            tenant_id=tenant_id,
        )

        if not record:
            return None

        record.status = normalized_status
        record.updated_at = datetime.now(
            timezone.utc
        )

        await db.flush()
        await db.refresh(record)

        return record

    @classmethod
    async def add_tags(
        cls,
        db: AsyncSession,
        document_id: str | UUID,
        tenant_id: str | UUID,
        tags: Sequence[str],
    ) -> Optional[DocumentModel]:
        """Add normalized, deduplicated tags to a document."""
        record = await cls.get_by_id(
            db,
            record_id=document_id,
            tenant_id=tenant_id,
        )

        if not record:
            return None

        existing_tags = list(
            record.tags or []
        )

        for tag in cls._normalize_tags(
            tags
        ):
            if tag not in existing_tags:
                existing_tags.append(tag)

        if len(existing_tags) > 20:
            raise ValidationError(
                "documents support at most 20 tags."
            )

        record.tags = existing_tags
        record.updated_at = datetime.now(
            timezone.utc
        )

        await db.flush()
        await db.refresh(record)

        return record

    @classmethod
    async def delete_document(
        cls,
        db: AsyncSession,
        document_id: str | UUID,
        tenant_id: str | UUID,
    ) -> bool:
        """Delete a tenant-scoped document record."""
        return await cls.delete_by_id(
            db,
            record_id=document_id,
            tenant_id=tenant_id,
        )