"""
Tests for offset-based and cursor-based pagination helpers.

These tests cover page metadata, boundary behavior, normalization, total-page
calculation, cursor encoding and decoding, and cursor-page construction.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.core.pagination import (
    Cursor,
    CursorPage,
    Page,
    PaginationParams,
    calculate_total_pages,
    normalize_pagination,
)


# ---------------------------------------------------------------------------
# Offset pagination
# ---------------------------------------------------------------------------


def test_page_metadata():
    page = Page[str].create(
        items=["a", "b"],
        total=5,
        skip=0,
        limit=2,
    )

    assert page.items == ["a", "b"]
    assert page.meta.total == 5
    assert page.meta.returned == 2
    assert page.meta.skip == 0
    assert page.meta.limit == 2
    assert page.meta.has_next is True
    assert page.meta.has_previous is False
    assert page.meta.next_skip == 2
    assert page.meta.previous_skip is None


def test_page_metadata_for_middle_page():
    page = Page[str].create(
        items=["c", "d"],
        total=5,
        skip=2,
        limit=2,
    )

    assert page.meta.total == 5
    assert page.meta.returned == 2
    assert page.meta.has_next is True
    assert page.meta.has_previous is True
    assert page.meta.next_skip == 4
    assert page.meta.previous_skip == 0


def test_last_page_has_no_next():
    page = Page[str].create(
        items=["e"],
        total=5,
        skip=4,
        limit=2,
    )

    assert page.items == ["e"]
    assert page.meta.total == 5
    assert page.meta.returned == 1
    assert page.meta.has_next is False
    assert page.meta.has_previous is True
    assert page.meta.next_skip is None
    assert page.meta.previous_skip == 2


def test_empty_page_has_no_navigation():
    page = Page[str].create(
        items=[],
        total=0,
        skip=0,
        limit=25,
    )

    assert page.items == []
    assert page.meta.total == 0
    assert page.meta.returned == 0
    assert page.meta.has_next is False
    assert page.meta.has_previous is False
    assert page.meta.next_skip is None
    assert page.meta.previous_skip is None


def test_page_with_fewer_items_than_limit():
    page = Page[str].create(
        items=["a"],
        total=1,
        skip=0,
        limit=10,
    )

    assert page.meta.returned == 1
    assert page.meta.has_next is False
    assert page.meta.has_previous is False
    assert page.meta.next_skip is None


def test_pagination_params_page_number():
    pagination = PaginationParams(
        skip=50,
        limit=25,
    )

    assert pagination.page_number == 3
    assert pagination.offset == 50


def test_pagination_params_first_page():
    pagination = PaginationParams(
        skip=0,
        limit=25,
    )

    assert pagination.page_number == 1
    assert pagination.offset == 0


def test_next_page():
    pagination = PaginationParams(
        skip=0,
        limit=25,
    )

    next_page = pagination.next_page()

    assert next_page.skip == 25
    assert next_page.limit == 25


def test_previous_page_is_non_negative():
    pagination = PaginationParams(
        skip=10,
        limit=25,
    )

    previous = pagination.previous_page()

    assert previous.skip == 0
    assert previous.limit == 25


def test_previous_page_moves_back_one_page():
    pagination = PaginationParams(
        skip=50,
        limit=25,
    )

    previous = pagination.previous_page()

    assert previous.skip == 25
    assert previous.limit == 25


def test_normalize_pagination_bounds():
    assert normalize_pagination(
        -10,
        500,
    ) == (0, 100)

    assert normalize_pagination(
        None,
        None,
    ) == (0, 25)

    assert normalize_pagination(
        150,
        10,
    ) == (150, 10)


def test_normalize_pagination_rejects_invalid_limit():
    with pytest.raises(ValueError):
        normalize_pagination(
            0,
            0,
        )


def test_normalize_pagination_rejects_negative_limit():
    with pytest.raises(ValueError):
        normalize_pagination(
            0,
            -1,
        )


def test_calculate_total_pages():
    assert calculate_total_pages(
        0,
        25,
    ) == 0

    assert calculate_total_pages(
        1,
        25,
    ) == 1

    assert calculate_total_pages(
        25,
        25,
    ) == 1

    assert calculate_total_pages(
        26,
        25,
    ) == 2

    assert calculate_total_pages(
        51,
        25,
    ) == 3


def test_calculate_total_pages_rejects_invalid_limit():
    with pytest.raises(ValueError):
        calculate_total_pages(
            10,
            0,
        )


# ---------------------------------------------------------------------------
# Cursor pagination
# ---------------------------------------------------------------------------


def test_cursor_round_trip():
    cursor = Cursor(
        sort_value="2026-11-01T00:00:00Z",
        record_id="doc_123",
    )

    encoded = cursor.encode()
    decoded = Cursor.decode(encoded)

    assert decoded.sort_value == cursor.sort_value
    assert decoded.record_id == cursor.record_id


def test_cursor_round_trip_with_uuid_record_id():
    record_id = str(uuid4())

    cursor = Cursor(
        sort_value="2026-11-01T00:00:00Z",
        record_id=record_id,
    )

    decoded = Cursor.decode(cursor.encode())

    assert decoded.record_id == record_id


def test_cursor_round_trip_with_numeric_sort_value():
    cursor = Cursor(
        sort_value="42",
        record_id="record_123",
    )

    decoded = Cursor.decode(cursor.encode())

    assert decoded.sort_value == "42"
    assert decoded.record_id == "record_123"


def test_invalid_cursor_is_rejected():
    with pytest.raises(ValueError):
        Cursor.decode("not-a-valid-cursor")


def test_empty_cursor_is_rejected():
    with pytest.raises(ValueError):
        Cursor.decode("")


def test_cursor_missing_record_id_is_rejected():
    cursor = Cursor(
        sort_value="2026-11-01T00:00:00Z",
        record_id="doc_123",
    )

    encoded = cursor.encode()

    tampered_payload = encoded.split(".")[0]

    with pytest.raises(ValueError):
        Cursor.decode(tampered_payload)


def test_cursor_page_creation():
    page = CursorPage[str].create(
        items=["a"],
        limit=10,
        next_cursor="abc",
    )

    assert page.items == ["a"]
    assert page.meta.returned == 1
    assert page.meta.limit == 10
    assert page.meta.has_next is True
    assert page.meta.next_cursor == "abc"


def test_cursor_page_without_next_cursor():
    page = CursorPage[str].create(
        items=["a", "b"],
        limit=10,
        next_cursor=None,
    )

    assert page.items == ["a", "b"]
    assert page.meta.returned == 2
    assert page.meta.has_next is False
    assert page.meta.next_cursor is None


def test_cursor_page_with_empty_items():
    page = CursorPage[str].create(
        items=[],
        limit=10,
        next_cursor=None,
    )

    assert page.items == []
    assert page.meta.returned == 0
    assert page.meta.has_next is False
    assert page.meta.next_cursor is None


def test_cursor_page_with_full_page_and_next_cursor():
    page = CursorPage[str].create(
        items=["a", "b", "c"],
        limit=3,
        next_cursor="next-cursor",
    )

    assert page.meta.returned == 3
    assert page.meta.has_next is True
    assert page.meta.next_cursor == "next-cursor"


def test_cursor_page_with_partial_page_has_no_next():
    page = CursorPage[str].create(
        items=["a", "b"],
        limit=3,
        next_cursor=None,
    )

    assert page.meta.returned == 2
    assert page.meta.has_next is False
    assert page.meta.next_cursor is None


# ---------------------------------------------------------------------------
# Integration-style pagination
# ---------------------------------------------------------------------------


def test_offset_pagination_navigation_sequence():
    total_items = 7
    limit = 3

    first_page = Page[str].create(
        items=["a", "b", "c"],
        total=total_items,
        skip=0,
        limit=limit,
    )

    second_page = Page[str].create(
        items=["d", "e", "f"],
        total=total_items,
        skip=3,
        limit=limit,
    )

    third_page = Page[str].create(
        items=["g"],
        total=total_items,
        skip=6,
        limit=limit,
    )

    assert first_page.meta.has_next is True
    assert first_page.meta.next_skip == 3

    assert second_page.meta.has_next is True
    assert second_page.meta.next_skip == 6

    assert third_page.meta.has_next is False
    assert third_page.meta.next_skip is None


def test_cursor_pagination_navigation_sequence():
    first_page = CursorPage[str].create(
        items=["a", "b"],
        limit=2,
        next_cursor="cursor-2",
    )

    second_page = CursorPage[str].create(
        items=["c"],
        limit=2,
        next_cursor=None,
    )

    assert first_page.meta.has_next is True
    assert first_page.meta.next_cursor == "cursor-2"

    assert second_page.meta.has_next is False
    assert second_page.meta.next_cursor is None
