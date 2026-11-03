from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentCreate(BaseModel):
    """Request schema for creating a document to embed and index."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Title of the document or source.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Raw text content to be embedded.",
    )


class DocumentResponse(BaseModel):
    """Response schema for a stored document."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    content: str
    created_at: datetime
    updated_at: Optional[datetime] = None


class SearchQuery(BaseModel):
    """Request schema for semantic vector search."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language search query.",
    )
    limit: int = Field(
        5,
        ge=1,
        le=50,
        description="Maximum number of search results to return.",
    )


class SearchResultItem(BaseModel):
    """Single semantic-search result."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    content: str
    similarity_score: float