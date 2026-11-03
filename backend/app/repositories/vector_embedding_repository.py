"""
Vector embedding repository with tenant-scoped pgvector search.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ValidationError
from app.models.vector_embedding_model import VectorEmbeddingModel
from app.repositories.base_repository import BaseRepository


logger = logging.getLogger(
    "agentic_bridge.repositories.vector_embedding"
)


class VectorEmbeddingRepository(
    BaseRepository[VectorEmbeddingModel]
):
    """Tenant-scoped persistence and similarity search for embeddings."""

    model = VectorEmbeddingModel

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

        return metadata

    @classmethod
    async def create_embedding(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        document_id: str | UUID,
        embedding: Sequence[float],
        source_chunk: Optional[str] = None,
        chunk_index: Optional[int] = None,
        model_name: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> VectorEmbeddingModel:
        """Create a tenant-scoped embedding record."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        normalized_document_id = (
            cls.normalize_uuid(
                document_id,
                "document_id",
            )
        )
        normalized_embedding = (
            cls.normalize_embedding(
                embedding
            )
        )

        if normalized_embedding is None:
            raise ValidationError(
                "embedding is required."
            )

        source_chunk = cls.normalize_text(
            source_chunk,
            "source_chunk",
            max_length=10_000,
        )

        if chunk_index is not None and (
            not isinstance(chunk_index, int)
            or isinstance(
                chunk_index, bool
            )
            or chunk_index < 0
        ):
            raise ValidationError(
                "chunk_index must be a non-negative integer."
            )

        model_name = cls.normalize_text(
            model_name,
            "model_name",
            max_length=150,
        )

        db_record = VectorEmbeddingModel(
            tenant_id=str(
                normalized_tenant_id
            ),
            document_id=str(
                normalized_document_id
            ),
            embedding=normalized_embedding,
            source_chunk=source_chunk,
            chunk_index=chunk_index,
            model_name=model_name,
            metadata=cls._validate_metadata(
                metadata
            ),
        )

        db.add(db_record)
        await db.flush()
        await db.refresh(db_record)

        logger.info(
            "Created vector embedding record.",
            extra={
                "embedding_id": str(
                    db_record.id
                ),
                "tenant_id": str(
                    normalized_tenant_id
                ),
                "document_id": str(
                    normalized_document_id
                ),
            },
        )

        return db_record

    @classmethod
    async def bulk_create_embeddings(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        document_id: str | UUID,
        embeddings: Sequence[
            Sequence[float]
        ],
        source_chunks: Optional[
            Sequence[Optional[str]]
        ] = None,
        model_name: Optional[str] = None,
    ) -> List[VectorEmbeddingModel]:
        """Create multiple embeddings while preserving chunk order."""
        if not embeddings:
            raise ValidationError(
                "embeddings must not be empty."
            )

        if len(embeddings) > 100:
            raise ValidationError(
                "embedding batches support at most 100 vectors."
        )

        chunks = list(source_chunks or [])

        if chunks and len(chunks) != len(
            embeddings
        ):
            raise ValidationError(
                "source_chunks must match the embedding batch length."
            )

        records: list[
            VectorEmbeddingModel
        ] = []

        for index, embedding in enumerate(
            embeddings
        ):
            records.append(
                await cls.create_embedding(
                    db,
                    tenant_id=tenant_id,
                    document_id=document_id,
                    embedding=embedding,
                    source_chunk=chunks[
                        index
                    ]
                    if chunks
                    else None,
                    chunk_index=index,
                    model_name=model_name,
                )
            )

        return records

    @classmethod
    async def get_embedding(
        cls,
        db: AsyncSession,
        embedding_id: str | UUID,
        tenant_id: str | UUID,
    ) -> Optional[
        VectorEmbeddingModel
    ]:
        """Retrieve a tenant-scoped embedding."""
        return await cls.get_by_id(
            db,
            record_id=embedding_id,
            tenant_id=tenant_id,
        )

    @classmethod
    async def list_embeddings_for_document(
        cls,
        db: AsyncSession,
        document_id: str | UUID,
        tenant_id: str | UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[
        list[
            VectorEmbeddingModel
        ],
        int,
    ]:
        """List embeddings for a tenant-scoped document."""
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
                cls.model.chunk_index.asc()
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
    async def search_similar(
        cls,
        db: AsyncSession,
        tenant_id: str | UUID,
        query_embedding: Sequence[float],
        limit: int = 5,
        document_ids: Optional[
            Sequence[str | UUID]
        ] = None,
        similarity_threshold: Optional[
            float
        ] = None,
    ) -> List[VectorEmbeddingModel]:
        """Run a tenant-scoped cosine similarity search."""
        normalized_tenant_id = (
            cls.normalize_uuid(
                tenant_id,
                "tenant_id",
            )
        )
        normalized_embedding = (
            cls.normalize_embedding(
                query_embedding
            )
        )

        if normalized_embedding is None:
            raise ValidationError(
                "query_embedding is required."
            )

        _, normalized_limit = (
            cls.normalize_pagination(
                0, limit
            )
        )
        normalized_limit = min(
            normalized_limit, 50
        )

        if similarity_threshold is not None and (
            not isinstance(
                similarity_threshold,
                (int, float),
            )
            or isinstance(
                similarity_threshold,
                bool,
            )
            or not 0.0
            <= float(
                similarity_threshold
            )
            <= 1.0
        ):
            raise ValidationError(
                "similarity_threshold must be between 0.0 and 1.0."
            )

        stmt = (
            select(cls.model)
            .where(
                cls.model.tenant_id
                == str(normalized_tenant_id)
            )
            .order_by(
                cls.model.embedding.cosine_distance(
                    normalized_embedding
                )
            )
            .limit(normalized_limit)
        )

        if document_ids:
            normalized_document_ids = [
                str(
                    cls.normalize_uuid(
                        document_id,
                        "document_id",
                    )
                )
                for document_id in document_ids
            ]

            stmt = stmt.where(
                cls.model.document_id.in_(
                    normalized_document_ids
                )
            )

        result = await db.execute(stmt)
        records = list(
            result.scalars().all()
        )

        return records

    @classmethod
    async def delete_embeddings_for_document(
        cls,
        db: AsyncSession,
        document_id: str | UUID,
        tenant_id: str | UUID,
    ) -> int:
        """Delete all embeddings associated with a tenant document."""
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

        stmt = (
            delete(cls.model)
            .where(
                cls.model.document_id
                == str(
                    normalized_document_id
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