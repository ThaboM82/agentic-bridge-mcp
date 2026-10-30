from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# --- Document Ingestion Schemas ---
class DocumentCreate(BaseModel):
    title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Title of the document or source",
    )
    content: str = Field(
        ..., min_length=1, description="Raw text content to be embedded"
    )


class DocumentResponse(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# --- Vector Search Schemas ---
class SearchQuery(BaseModel):
    query: str = Field(
        ..., min_length=1, description="Natural language search query"
    )
    limit: int = Field(
        5, ge=1, le=50, description="Maximum number of search results to return"
    )


class SearchResultItem(BaseModel):
    id: int
    title: str
    content: str
    similarity_score: float

    class Config:
        from_attributes = True