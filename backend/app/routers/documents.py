from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.database import get_db
from app.models import DocumentChunk
from app.schemas import (
    DocumentCreate,
    DocumentResponse,
    SearchQuery,
    SearchResultItem,
)
from app.services import create_document, search_documents

router = APIRouter(prefix="/documents", tags=["Documents & Vector Search"])


@router.post(
    "/", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED
)
async def ingest_document(
    payload: DocumentCreate, db: AsyncSession = Depends(get_db)
):
    """Ingests a new document, generates its vector embedding, and saves it to PostgreSQL."""
    try:
        document = await create_document(db, payload)
        return document
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}",
        )


@router.post("/search", response_model=List[SearchResultItem])
async def semantic_search(
    payload: SearchQuery, db: AsyncSession = Depends(get_db)
):
    """Performs a semantic vector similarity search against stored document chunks."""
    try:
        results = await search_documents(db, payload)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform semantic search: {str(e)}",
        )


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(
    skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)
):
    """Retrieves a paginated list of all ingested documents."""
    stmt = select(DocumentChunk).offset(skip).limit(limit)
    result = await db.execute(stmt)
    documents = result.scalars().all()
    return documents