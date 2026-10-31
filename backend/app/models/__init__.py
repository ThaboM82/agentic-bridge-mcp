from app.database import Base
from app.models.documents import DocumentChunk  # Updated to plural 'documents'

__all__ = ["Base", "DocumentChunk"]