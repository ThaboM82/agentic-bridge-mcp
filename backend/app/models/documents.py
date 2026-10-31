from sqlalchemy import Column, Integer, String, Text, DateTime, Index, JSON
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.database import Base

class DocumentChunk(Base):
    """
    ORM model representing processed and embedded document chunks 
    for the AgenticBridge document intelligence audit engine.
    """
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    document_id = Column(String(255), index=True, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    
    # 1536 is standard for OpenAI text-embedding-ada-002 / text-embedding-3-small models
    embedding = Column(Vector(1536), nullable=True)
    
    # Optional metadata dictionary for audit tags, source files, or policy rules
    meta_data = Column(JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    __table_args__ = (
        Index("ix_document_chunks_doc_id_index", "document_id", "chunk_index", unique=True),
    )