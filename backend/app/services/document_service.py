import logging
import uuid
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentChunk
from app.schemas.documents import DocumentCreate
from app.schemas.search import SearchQuery, SearchResultItem
from app.services.embedding_service import embedding_service


logger = logging.getLogger(__name__)


class DocumentServiceError(Exception):
    """Base error for document service failures."""


class EmbeddingError(DocumentServiceError):
    """Raised when an embedding cannot be generated."""


class DocumentPersistenceError(DocumentServiceError):
    """Raised when a document cannot be persisted or queried."""


class DocumentService:
    """
    Creates, stores, and searches document embeddings.

    This service owns document business logic. API routers should not call
    embedding providers or construct database queries directly.
    """

    def __init__(
        self,
        embeddings=embedding_service,
        default_min_similarity: float = 0.0,
    ) -> None:
        self.embeddings = embeddings
        self.default_min_similarity = default_min_similarity

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize whitespace before embedding or storage."""
        return " ".join(text.split())

    @staticmethod
    def _build_metadata(
        payload: DocumentCreate,
    ) -> dict[str, Any]:
        """
        Build stable document metadata.

        Extend this with tenant_id, source_url, document_type, and ACL data
        as AgenticBridge grows.
        """
        return {
            "source": payload.model_extra.get("source", "manual"),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

    async def create_document(
        self,
        db: AsyncSession,
        payload: DocumentCreate,
    ) -> DocumentChunk:
        """
        Generate an embedding and persist a document chunk.

        The database transaction is rolled back if embedding generation or
        persistence fails, preventing a partially written record.
        """
        content = self._normalize_text(payload.content)

        if not content:
            raise ValueError("Document content cannot be empty.")

        try:
            vector = await self.embeddings.generate_embedding(content)
        except Exception as exc:
            logger.exception(
                "Embedding generation failed for document title=%r",
                payload.title,
            )
            raise EmbeddingError(
                "Unable to generate document embedding."
            ) from exc

        document = DocumentChunk(
            title=payload.title.strip(),
            content=content,
            embedding=vector,
            document_metadata=self._build_metadata(payload),
        )

        try:
            db.add(document)
            await db.commit()
            await db.refresh(document)
        except SQLAlchemyError as exc:
            await db.rollback()
            logger.exception(
                "Failed to persist document title=%r", payload.title
            )
            raise DocumentPersistenceError(
                "Unable to store document."
            ) from exc

        logger.info(
            "Created document id=%s title=%r embedding_dimensions=%s",
            document.id,
            document.title,
            len(vector),
        )

        return document

    async def create_documents(
        self,
        db: AsyncSession,
        payloads: Sequence[DocumentCreate],
    ) -> list[DocumentChunk]:
        """
        Ingest multiple documents in one database transaction.

        Embeddings are generated first. If any embedding fails, no documents
        are written.
        """
        if not payloads:
            return []

        normalized_payloads: list[DocumentCreate] = []

        for payload in payloads:
            content = self._normalize_text(payload.content)

            if not content:
                raise ValueError(
                    f"Document content cannot be empty for title={payload.title!r}."
                )

            normalized_payloads.append(
                payload.model_copy(update={"content": content})
            )

        try:
            vectors = await self.embeddings.generate_embeddings(
                [payload.content for payload in normalized_payloads]
            )
        except Exception as exc:
            logger.exception(
                "Batch embedding generation failed for %s documents.",
                len(normalized_payloads),
            )
            raise EmbeddingError(
                "Unable to generate embeddings for document batch."
            ) from exc

        documents = [
            DocumentChunk(
                title=payload.title.strip(),
                content=payload.content,
                embedding=vector,
                document_metadata=self._build_metadata(payload),
            )
            for payload, vector in zip(
                normalized_payloads,
                vectors,
                strict=True,
            )
        ]

        try:
            db.add_all(documents)
            await db.commit()

            for document in documents:
                await db.refresh(document)
        except SQLAlchemyError as exc:
            await db.rollback()
            logger.exception(
                "Failed to persist document batch of size=%s",
                len(documents),
            )
            raise DocumentPersistenceError(
                "Unable to store document batch."
            ) from exc

        logger.info("Created %s document chunks.", len(documents))
        return documents

    async def search_documents(
        self,
        db: AsyncSession,
        payload: SearchQuery,
        min_similarity: float | None = None,
    ) -> list[SearchResultItem]:
        """
        Perform semantic search using pgvector cosine distance.

        Cosine distance ranges from 0 (identical direction) to 2 (opposite
        direction). Similarity is reported as `1 - distance`.
        """
        query = self._normalize_text(payload.query)

        if not query:
            raise ValueError("Search query cannot be empty.")

        threshold = (
            self.default_min_similarity
            if min_similarity is None
            else min_similarity
        )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("min_similarity must be between 0.0 and 1.0.")

        try:
            query_vector = await self.embeddings.generate_embedding(query)
        except Exception as exc:
            logger.exception(
                "Query embedding generation failed query=%r", payload.query
            )
            raise EmbeddingError(
                "Unable to generate search embedding."
            ) from exc

        distance = DocumentChunk.embedding.cosine_distance(query_vector)

        statement = (
            select(DocumentChunk, distance.label("distance"))
            .order_by(distance)
            .limit(payload.limit)
        )

        try:
            result = await db.execute(statement)
            rows = result.all()
        except SQLAlchemyError as exc:
            logger.exception("Semantic search query failed.")
            raise DocumentPersistenceError(
                "Unable to execute semantic search."
            ) from exc

        results: list[SearchResultItem] = []

        for document, row_distance in rows:
            similarity = 1.0 - float(row_distance)

            if similarity < threshold:
                continue

            results.append(
                SearchResultItem(
                    id=document.id,
                    title=document.title,
                    content=document.content,
                    similarity_score=similarity,
                )
            )

        logger.info(
            "Semantic search completed query_length=%s limit=%s results=%s",
            len(query),
            payload.limit,
            len(results),
        )

        return results

    async def get_document(
        self,
        db: AsyncSession,
        document_id: int,
    ) -> DocumentChunk | None:
        """Retrieve one document by ID."""
        statement = select(DocumentChunk).where(DocumentChunk.id == document_id)

        try:
            result = await db.execute(statement)
            return result.scalar_one_or_none()
        except SQLAlchemyError as exc:
            logger.exception(
                "Failed to retrieve document id=%s", document_id
            )
            raise DocumentPersistenceError(
                "Unable to retrieve document."
            ) from exc

    async def list_documents(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 25,
    ) -> list[DocumentChunk]:
        """Return a deterministic, paginated list of documents."""
        if skip < 0:
            raise ValueError("skip must be greater than or equal to 0.")

        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")

        statement = (
            select(DocumentChunk)
            .order_by(DocumentChunk.created_at.desc(), DocumentChunk.id.desc())
            .offset(skip)
            .limit(limit)
        )

        try:
            result = await db.execute(statement)
            return list(result.scalars().all())
        except SQLAlchemyError as exc:
            logger.exception("Failed to list documents.")
            raise DocumentPersistenceError(
                "Unable to list documents."
            ) from exc


document_service = DocumentService()