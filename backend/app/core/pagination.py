"""
Shared pagination contracts and helpers.

Provides offset/limit pagination for list endpoints, reusable FastAPI query
dependencies, response envelopes, and helpers for building pagination links.

For very large or rapidly changing collections, prefer cursor-based pagination;
this module also includes a cursor contract for those cases.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from fastapi import Query
from pydantic import BaseModel, Field, field_validator

from app.core.constants import (
    DEFAULT_PAGE_SIZE,
    DEFAULT_SKIP,
    MAX_PAGE_SIZE,
    MAX_SKIP,
    MIN_PAGE_SIZE,
)


ItemT = TypeVar("ItemT")
CursorT = TypeVar("CursorT", bound=str)


# ---------------------------------------------------------------------------
# Offset pagination models
# ---------------------------------------------------------------------------


class PageMeta(BaseModel):
    """Metadata describing an offset/limit paginated result set."""

    total: int = Field(..., ge=0, description="Total number of matching records.")
    skip: int = Field(..., ge=0, description="Number of records skipped.")
    limit: int = Field(..., ge=1, description="Maximum number of records returned.")
    returned: int = Field(..., ge=0, description="Number of records in this page.")
    has_next: bool = Field(..., description="Whether another page is available.")
    has_previous: bool = Field(..., description="Whether a previous page is available.")
    next_skip: int | None = Field(None, ge=0, description="Skip value for the next page.")
    previous_skip: int | None = Field(None, ge=0, description="Skip value for the previous page.")

    @field_validator("next_skip", "previous_skip", mode="before")
    @classmethod
    def validate_optional_skip(cls, value: Any) -> int | None:
        """Normalize optional skip values."""
        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Skip value must be an integer or null.")

        if value < 0:
            return None

        return value


class Page(BaseModel, Generic[ItemT]):
    """Standard paginated response envelope."""

    items: list[ItemT] = Field(default_factory=list)
    meta: PageMeta

    @classmethod
    def create(
        cls,
        *,
        items: list[ItemT],
        total: int,
        skip: int,
        limit: int,
    ) -> "Page[ItemT]":
        """Create a paginated response from a result page and total count."""
        safe_limit = max(1, limit)
        safe_skip = max(0, skip)
        safe_total = max(0, total)

        has_next = safe_skip + safe_limit < safe_total
        has_previous = safe_skip > 0

        next_skip = safe_skip + safe_limit if has_next else None
        previous_skip = max(0, safe_skip - safe_limit) if has_previous else None

        return cls(
            items=items,
            meta=PageMeta(
                total=safe_total,
                skip=safe_skip,
                limit=safe_limit,
                returned=len(items),
                has_next=has_next,
                has_previous=has_previous,
                next_skip=next_skip,
                previous_skip=previous_skip,
            ),
        )


class PaginationParams:
    """
    Reusable FastAPI dependency for offset/limit pagination.

    Example:
        @router.get("/documents")
        async def list_documents(
            pagination: Annotated[PaginationParams, Depends()],
        ):
            ...
    """

    def __init__(
        self,
        skip: int = Query(
            default=DEFAULT_SKIP,
            ge=0,
            le=MAX_SKIP,
            description="Number of records to skip.",
        ),
        limit: int = Query(
            default=DEFAULT_PAGE_SIZE,
            ge=MIN_PAGE_SIZE,
            le=MAX_PAGE_SIZE,
            description="Maximum number of records to return.",
        ),
    ) -> None:
        self.skip = skip
        self.limit = limit

    @property
    def offset(self) -> int:
        """SQLAlchemy-compatible offset value."""
        return self.skip

    @property
    def page_size(self) -> int:
        """Alias for `limit`."""
        return self.limit

    @property
    def page_number(self) -> int:
        """One-based page number represented by the current skip/limit pair."""
        return (self.skip // self.limit) + 1

    def next_page(self) -> "PaginationParams":
        """Return pagination parameters for the next page."""
        return PaginationParams(
            skip=self.skip + self.limit,
            limit=self.limit,
        )

    def previous_page(self) -> "PaginationParams":
        """Return pagination parameters for the previous page."""
        previous_skip = max(0, self.skip - self.limit)

        return PaginationParams(
            skip=previous_skip,
            limit=self.limit,
        )

    def build_meta(self, total: int, returned_count: int | None = None) -> PageMeta:
        """Build pagination metadata for a query result."""
        safe_total = max(0, total)
        safe_returned = (
            len([]) if returned_count is None else max(0, returned_count)
        )

        has_next = self.skip + self.limit < safe_total
        has_previous = self.skip > 0

        return PageMeta(
            total=safe_total,
            skip=self.skip,
            limit=self.limit,
            returned=safe_returned,
            has_next=has_next,
            has_previous=has_previous,
            next_skip=self.skip + self.limit if has_next else None,
            previous_skip=max(0, self.skip - self.limit) if has_previous else None,
        )


# ---------------------------------------------------------------------------
# Cursor pagination
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Cursor:
    """
    Opaque pagination cursor.

    The cursor encodes a sort value and record ID so results remain stable even
    when records are inserted or deleted between requests.
    """

    sort_value: str
    record_id: str

    def encode(self) -> str:
        """Encode the cursor as a URL-safe opaque string."""
        raw_value = f"{self.sort_value}|{self.record_id}"
        encoded_bytes = base64.urlsafe_b64encode(raw_value.encode("utf-8"))

        return encoded_bytes.decode("ascii").rstrip("=")

    @classmethod
    def decode(cls, cursor: str) -> "Cursor":
        """Decode an opaque cursor string."""
        if not cursor:
            raise ValueError("Cursor cannot be empty.")

        try:
            padding = "=" * (-len(cursor) % 4)
            decoded = base64.urlsafe_b64decode(cursor + padding).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
            raise ValueError("Invalid pagination cursor.") from exc

        if "|" not in decoded:
            raise ValueError("Invalid pagination cursor.")

        sort_value, record_id = decoded.split("|", 1)

        if not sort_value or not record_id:
            raise ValueError("Invalid pagination cursor.")

        return cls(sort_value=sort_value, record_id=record_id)


class CursorPageMeta(BaseModel):
    """Metadata describing a cursor-paginated result set."""

    limit: int = Field(..., ge=1)
    returned: int = Field(..., ge=0)
    has_next: bool
    next_cursor: str | None = None


class CursorPage(BaseModel, Generic[ItemT]):
    """Standard cursor-paginated response envelope."""

    items: list[ItemT] = Field(default_factory=list)
    meta: CursorPageMeta

    @classmethod
    def create(
        cls,
        *,
        items: list[ItemT],
        limit: int,
        next_cursor: str | None,
    ) -> "CursorPage[ItemT]":
        """Create a cursor-paginated response."""
        return cls(
            items=items,
            meta=CursorPageMeta(
                limit=max(1, limit),
                returned=len(items),
                has_next=next_cursor is not None,
                next_cursor=next_cursor,
            ),
        )


class CursorPaginationParams:
    """
    Reusable FastAPI dependency for cursor pagination.

    Example:
        @router.get("/documents/cursor")
        async def list_documents_cursor(
            pagination: Annotated[CursorPaginationParams, Depends()],
        ):
            ...
    """

    def __init__(
        self,
        cursor: str | None = Query(
            default=None,
            max_length=512,
            description="Opaque cursor returned by the previous page.",
        ),
        limit: int = Query(
            default=DEFAULT_PAGE_SIZE,
            ge=MIN_PAGE_SIZE,
            le=MAX_PAGE_SIZE,
            description="Maximum number of records to return.",
        ),
    ) -> None:
        self.cursor = cursor
        self.limit = limit

    @property
    def decoded_cursor(self) -> Cursor | None:
        """Decode the supplied cursor, if present."""
        if not self.cursor:
            return None

        return Cursor.decode(self.cursor)

    def build_meta(self, next_cursor: str | None) -> CursorPageMeta:
        """Build metadata for a cursor-paginated response."""
        return CursorPageMeta(
            limit=self.limit,
            returned=0,
            has_next=next_cursor is not None,
            next_cursor=next_cursor,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_pagination(
    skip: int | None,
    limit: int | None,
) -> tuple[int, int]:
    """
    Normalize pagination inputs to safe database bounds.

    Returns:
        A `(skip, limit)` tuple.
    """
    safe_skip = DEFAULT_SKIP if skip is None else max(0, min(skip, MAX_SKIP))
    safe_limit = (
        DEFAULT_PAGE_SIZE
        if limit is None
        else max(MIN_PAGE_SIZE, min(limit, MAX_PAGE_SIZE))
    )

    return safe_skip, safe_limit


def calculate_total_pages(total: int, limit: int) -> int:
    """Calculate the total number of pages for a result set."""
    safe_total = max(0, total)
    safe_limit = max(1, limit)

    return (safe_total + safe_limit - 1) // safe_limit