import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from app.database import Base

class DocumentEmbedding(Base):
    """
    Enterprise ORM model for storing unstructured document chunks 
    and their high-dimensional vector embeddings for RAG retrieval.
    """
    __tablename__ = "document_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    document_id = Column(String(255), index=True, nullable=False)
    content = Column(Text, nullable=False)                            # Raw chunk text
    
    # Vector embedding dimension (e.g., 1536 for OpenAI text-embedding-3-small or Claude embeddings)
    embedding = Column(Vector(1536), nullable=False)                   
    
    metadata_json = Column(JSONB, nullable=True, default={})         # Source file, author, version tags
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Optional: Create an HNSW or IVFFlat index hint for high-performance vector search at scale
    __table_args__ = (
        Index(
            "ix_document_embeddings_vector_hnsw",
            embedding,
            postgresql_using="hnsw",
            postgresql_ops={"m": 16, "ef_construction": 64, "func": "vector_cosine_ops"}
        ),
    )