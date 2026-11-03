"""
Shared pagination models and helpers.

This module defines a consistent pagination envelope for list endpoints and
provides helpers for offset/limit pagination, page-based pagination, cursor
references, and pagination metadata.
"""

from __future__ import annotations

import math
from typing import Generic, Sequence, TypeVar

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


ItemType = TypeVar(
    "ItemType"
)


# ---------------------------------------------------------------------------
# Pagination metadata
# ---------------------------------------------------------------------------


class PageMeta(BaseModel):
    """Offset/limit pagination metadata."""

    skip: int = Field(
        ge=0,
        description="Number of records skipped.",
    )
    limit: int = Field(
        ge=1,
        description="Maximum number of records returned.",
    )
    total_count: int = Field(
        ge=0,
        description="Total number of matching records.",
    )
    returned_count: int = Field(
        ge=0,
        description="Number of records returned in this page.",
    )
    has_more: bool = Field(
        description="Whether additional records exist.",
    )
    has_previous: bool = Field(
        description="Whether a previous page exists.",
    )
    current_page: int = Field(
        ge=1,
        description="One-based page number.",
    )
    total_pages: int = Field(
        ge=0,
        description="Total number of available pages.",
    )
    next_skip: int | None = Field(
        default=None,
        ge=0,
        description="Skip value for the next page, if available.",
    )
    previous_skip: int | None = Field(
        default=None,
        ge=0,
        description="Skip value for the previous page, if available.",
    )

    @field_validator(
        "total_pages"
    )
    @classmethod
    def validate_total_pages(
        cls,
        value: int,
    ) -> int:
        """Ensure total page count is never negative."""
        return max(
            value,
            0,
        )


# ---------------------------------------------------------------------------
# Paginated response envelope
# ---------------------------------------------------------------------------


class PaginatedResponse(
    BaseModel,
    Generic[ItemType],
):
    """Standard paginated API response envelope."""

    items: list[ItemType]
    meta: PageMeta


# ---------------------------------------------------------------------------
# Cursor metadata
# ---------------------------------------------------------------------------


class CursorPageMeta(BaseModel):
    """Cursor-based pagination metadata."""

    limit: int = Field(
        ge=1,
        description="Maximum number of records returned.",
    )
    has_more: bool = Field(
        description="Whether additional records exist.",
    )
    next_cursor: str | None = Field(
        default=None,
        description="Opaque cursor for the next page.",
    )
    previous_cursor: str | None = Field(
        default=None,
        description="Opaque cursor for the previous page.",
    )


class CursorPaginatedResponse(
    BaseModel,
    Generic[ItemType],
):
    """Cursor-based API response envelope."""

    items: list[ItemType]
    meta: CursorPageMeta


# ---------------------------------------------------------------------------
# Offset/limit helpers
# ---------------------------------------------------------------------------


def calculate_total_pages(
    total_count: int,
    limit: int,
) -> int:
    """Calculate the total number of pages for a result set."""
    if total_count <= 0:
        return 0

    if limit <= 0:
        raise ValueError(
            "Pagination limit must be greater than zero."
        )

    return math.ceil(
        total_count / limit
    )


def calculate_page(
    skip: int,
    limit: int,
) -> int:
    """Calculate a one-based page number from an offset."""
    if skip < 0:
        raise ValueError(
            "Pagination skip must be greater than or equal to zero."
        )

    if limit <= 0:
        raise ValueError(
            "Pagination limit must be greater than zero."
        )

    return (skip // limit) + 1


def build_page_meta(
    *,
    skip: int,
    limit: int,
    total_count: int,
    returned_count: int | None = None,
) -> PageMeta:
    """Build pagination metadata for an offset/limit result page."""
    if skip < 0:
        raise ValueError(
            "Pagination skip must be greater than or equal to zero."
        )

    if limit <= 0:
        raise ValueError(
            "Pagination limit must be greater than zero."
        )

    if total_count < 0:
        raise ValueError(
            "Total count must be greater than or equal to zero."
        )

    item_count = (
        returned_count
        if returned_count is not None
        else 0
    )

    has_more = (
        skip + item_count
        < total_count
    )

    has_previous = skip > 0

    current_page = calculate_page(
        skip=skip,
        limit=limit,
    )

    total_pages = calculate_total_pages(
        total_count=total_count,
        limit=limit,
    )

    next_skip = (
        skip + limit
        if has_more
        else None
    )

    previous_skip = (
        max(skip - limit, 0)
        if has_previous
        else None
    )

    return PageMeta(
        skip=skip,
        limit=limit,
        total_count=total_count,
        returned_count=item_count,
        has_more=has_more,
        has_previous=has_previous,
        current_page=current_page,
        total_pages=total_pages,
        next_skip=next_skip,
        previous_skip=previous_skip,
    )


def build_paginated_response(
    items: Sequence[ItemType],
    *,
    skip: int,
    limit: int,
    total_count: int,
) -> PaginatedResponse[ItemType]:
    """Build a consistent offset/limit pagination envelope."""
    item_list = list(
        items
    )

    return PaginatedResponse[ItemType](
        items=item_list,
        meta=build_page_meta(
            skip=skip,
            limit=limit,
            total_count=total_count,
            returned_count=len(
                item_list
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Cursor helpers
# ---------------------------------------------------------------------------


def build_cursor_page_meta(
    *,
    limit: int,
    has_more: bool,
    next_cursor: str | None = None,
    previous_cursor: str | None = None,
) -> CursorPageMeta:
    """Build metadata for a cursor-paginated result page."""
    if limit <= 0:
        raise ValueError(
            "Pagination limit must be greater than zero."
        )

    return CursorPageMeta(
        limit=limit,
        has_more=has_more,
        next_cursor=next_cursor,
        previous_cursor=previous_cursor,
    )


def build_cursor_paginated_response(
    items: Sequence[ItemType],
    *,
    limit: int,
    has_more: bool,
    next_cursor: str | None = None,
    previous_cursor: str | None = None,
) -> CursorPaginatedResponse[ItemType]:
    """Build a consistent cursor-based pagination envelope."""
    item_list = list(
        items
    )

    return CursorPaginatedResponse[ItemType](
        items=item_list,
        meta=build_cursor_page_meta(
            limit=limit,
            has_more=has_more,
            next_cursor=next_cursor,
            previous_cursor=previous_cursor,
        ),
    )


# ---------------------------------------------------------------------------
# Empty responses
# ---------------------------------------------------------------------------


def build_empty_paginated_response(
    *,
    skip: int = 0,
    limit: int = 25,
) -> PaginatedResponse[ItemType]:
    """Build an empty paginated response."""
    return build_paginated_response(
        items=[],
        skip=skip,
        limit=limit,
        total_count=0,
    )


def build_empty_cursor_paginated_response(
    *,
    limit: int = 25,
) -> CursorPaginatedResponse[ItemType]:
    """Build an empty cursor-paginated response."""
    return build_cursor_paginated_response(
        items=[],
        limit=limit,
        has_more=False,
        next_cursor=None,
        previous_cursor=None,
    )