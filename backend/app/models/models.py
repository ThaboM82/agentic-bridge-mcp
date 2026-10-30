from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.database import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), index=True, nullable=False)
    content = Column(Text, nullable=False)
    # 1536 dimensions standard for OpenAI embedding models (adjust if using another model)
    embedding = Column(Vector(1536), nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )