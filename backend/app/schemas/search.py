from pydantic import BaseModel, Field

from app.schemas.documents import SearchResultItem


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


class SearchResponse(BaseModel):
    """Response schema for semantic vector search."""

    query: str
    results: list[SearchResultItem]
    total_results: int