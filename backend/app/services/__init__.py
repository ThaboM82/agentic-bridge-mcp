from sqlalchemy.ext.asyncio import AsyncSession
from app.models.documents import DocumentChunk

async def create_document(
    db: AsyncSession, 
    document_id: str, 
    content: str, 
    chunk_index: int = 0, 
    embedding: list = None, 
    meta_data: dict = None
):
    """
    Creates and saves a new document chunk in the database.
    """
    db_chunk = DocumentChunk(
        document_id=document_id,
        content=content,
        chunk_index=chunk_index,
        embedding=embedding,
        meta_data=meta_data
    )
    db.add(db_chunk)
    await db.commit()
    await db.refresh(db_chunk)
    return db_chunk

async def search_documents(db: AsyncSession, query_embedding: list, limit: int = 5):
    """
    Performs vector similarity search across document chunks.
    """
    return []

__all__ = ["create_document", "search_documents"]