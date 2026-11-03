"""
WordPress-related request and response schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class WordPressContentSearchRequest(BaseModel):
    """Request for tenant-scoped WordPress content search."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    limit: int = Field(
        default=20,
        ge=1,
        le=50,
    )


class WordPressContentSummary(BaseModel):
    """A bounded, workflow-safe WordPress content summary."""

    wordpress_content_id: str = Field(
        ...,
        max_length=100,
    )

    title: str = Field(
        ...,
        max_length=300,
    )

    status: str = Field(
        ...,
        max_length=50,
    )

    slug: str | None = Field(
        default=None,
        max_length=200,
    )

    excerpt: str | None = Field(
        default=None,
        max_length=1_000,
    )

    updated_at: str | None = Field(
        default=None,
        max_length=50,
    )


class WordPressContentSearchResponse(BaseModel):
    """Tenant-safe WordPress content search response."""

    query: str = Field(
        ...,
        max_length=500,
    )

    results: list[WordPressContentSummary] = Field(
        default_factory=list,
    )

    total_results: int = Field(
        default=0,
        ge=0,
    )

    truncated: bool = False


class WordPressCreateDraftRequest(BaseModel):
    """Request to create a WordPress draft."""

    title: str = Field(
        ...,
        min_length=1,
        max_length=300,
    )

    content: str = Field(
        ...,
        min_length=1,
        max_length=50_000,
    )

    excerpt: str | None = Field(
        default=None,
        max_length=1_000,
    )

    slug: str | None = Field(
        default=None,
        max_length=200,
    )

    tags: list[str] = Field(
        default_factory=list,
        max_length=25,
    )

    category_ids: list[str] = Field(
        default_factory=list,
        max_length=25,
    )


class WordPressCreateDraftResponse(BaseModel):
    """Tenant-safe WordPress draft creation response."""

    wordpress_content_id: str = Field(
        ...,
        max_length=100,
    )

    title: str = Field(
        ...,
        max_length=300,
    )

    status: str = Field(
        ...,
        max_length=50,
    )

    slug: str | None = Field(
        default=None,
        max_length=200,
    )

    created_at: str | None = Field(
        default=None,
        max_length=50,
    )


class WordPressUpdateDraftRequest(BaseModel):
    """Request to update a WordPress draft."""

    wordpress_content_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    title: str | None = Field(
        default=None,
        max_length=300,
    )

    content: str | None = Field(
        default=None,
        max_length=50_000,
    )

    excerpt: str | None = Field(
        default=None,
        max_length=1_000,
    )

    slug: str | None = Field(
        default=None,
        max_length=200,
    )

    tags: list[str] = Field(
        default_factory=list,
        max_length=25,
    )

    category_ids: list[str] = Field(
        default_factory=list,
        max_length=25,
    )


class WordPressUpdateDraftResponse(BaseModel):
    """Tenant-safe WordPress draft update response."""

    wordpress_content_id: str = Field(
        ...,
        max_length=100,
    )

    title: str = Field(
        ...,
        max_length=300,
    )

    status: str = Field(
        ...,
        max_length=50,
    )

    slug: str | None = Field(
        default=None,
        max_length=200,
    )

    updated_at: str | None = Field(
        default=None,
        max_length=50,
    )


class WordPressPostRequest(BaseModel):
    """Request to submit a WordPress draft for publication review."""

    wordpress_content_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )


class WordPressPostResponse(BaseModel):
    """Tenant-safe WordPress publication-request response."""

    wordpress_content_id: str = Field(
        ...,
        max_length=100,
    )

    requested_status: str = Field(
        ...,
        max_length=50,
    )

    approval_required: bool = True

    workflow_execution_id: str | None = Field(
        default=None,
        max_length=100,
    )