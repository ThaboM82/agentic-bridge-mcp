"""
Comprehensive tests for CMS MCP tool schemas and validation.

These tests cover content listing, search, creation, publication, content-type
validation, pagination limits, query normalization, tag deduplication, and
fail-closed validation behavior.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from app.core.exceptions import AuthorizationError
from app.mcp_server.schemas import (
    CMSContentCreateRequest,
    CMSContentListRequest,
    CMSContentSearchRequest,
    CMSPublishRequest,
)
from app.mcp_server.tools.cms_tools import (
    _validate_content_type,
)


# ---------------------------------------------------------------------------
# CMS list request
# ---------------------------------------------------------------------------


def test_cms_list_request_defaults():
    request = CMSContentListRequest()

    assert request.page == 1
    assert request.page_size == 20
    assert request.content_status is None
    assert request.content_type is None


def test_cms_list_request_bounds_page_size():
    request = CMSContentListRequest(
        page_size=10_000,
    )

    assert request.page_size == 100


@pytest.mark.parametrize(
    "page_size,expected",
    [
        (1, 1),
        (10, 10),
        (20, 20),
        (50, 50),
        (100, 100),
        (101, 100),
        (1_000, 100),
        (10_000, 100),
    ],
)
def test_cms_list_request_clamps_page_size(
    page_size: int,
    expected: int,
):
    request = CMSContentListRequest(
        page_size=page_size,
    )

    assert request.page_size == expected


@pytest.mark.parametrize(
    "page,expected",
    [
        (1, 1),
        (2, 2),
        (100, 100),
        (0, 1),
        (-1, 1),
        (-100, 1),
    ],
)
def test_cms_list_request_clamps_page(
    page: int,
    expected: int,
):
    request = CMSContentListRequest(
        page=page,
    )

    assert request.page == expected


@pytest.mark.parametrize(
    "content_type",
    [
        "article",
        "documentation",
        "guide",
    ],
)
def test_cms_list_request_accepts_supported_content_types(
    content_type: str,
):
    request = CMSContentListRequest(
        content_type=content_type,
    )

    assert (
        request.content_type
        == content_type
    )


def test_cms_list_request_accepts_content_status():
    request = CMSContentListRequest(
        content_status="published",
    )

    assert (
        request.content_status
        == "published"
    )


# ---------------------------------------------------------------------------
# CMS search request
# ---------------------------------------------------------------------------


def test_cms_search_request_normalizes_query():
    request = CMSContentSearchRequest(
        query=(
            "  accessible dashboard guide  "
        ),
    )

    assert request.query == (
        "accessible dashboard guide"
    )


def test_cms_search_request_rejects_empty_query():
    with pytest.raises(Exception):
        CMSContentSearchRequest(
            query="   "
        )


@pytest.mark.parametrize(
    "invalid_query",
    [
        "",
        "   ",
        None,
    ],
)
def test_cms_search_request_rejects_invalid_queries(
    invalid_query,
):
    with pytest.raises(Exception):
        CMSContentSearchRequest(
            query=invalid_query
        )


def test_cms_search_request_preserves_internal_whitespace():
    request = CMSContentSearchRequest(
        query=(
            "accessible   dashboard   guide"
        ),
    )

    assert request.query == (
        "accessible   dashboard   guide"
    )


def test_cms_search_request_defaults():
    request = CMSContentSearchRequest(
        query="accessibility"
    )

    assert request.query == (
        "accessibility"
    )
    assert request.limit == 10


@pytest.mark.parametrize(
    "limit,expected",
    [
        (1, 1),
        (10, 10),
        (25, 25),
        (50, 50),
        (51, 50),
        (100, 50),
        (1_000, 50),
    ],
)
def test_cms_search_request_clamps_limit(
    limit: int,
    expected: int,
):
    request = CMSContentSearchRequest(
        query="accessibility",
        limit=limit,
    )

    assert request.limit == expected


# ---------------------------------------------------------------------------
# CMS create request
# ---------------------------------------------------------------------------


def test_cms_create_request_normalizes_tags():
    request = CMSContentCreateRequest(
        title="Accessibility Guide",
        content=(
            "Use semantic HTML and keyboard navigation."
        ),
        tags=[
            "Accessibility",
            "accessibility",
            "design",
        ],
    )

    assert request.tags == [
        "Accessibility",
        "design",
    ]


def test_cms_create_request_rejects_empty_content():
    with pytest.raises(Exception):
        CMSContentCreateRequest(
            title="Guide",
            content="   ",
        )


@pytest.mark.parametrize(
    "invalid_content",
    [
        "",
        "   ",
        None,
    ],
)
def test_cms_create_request_rejects_invalid_content(
    invalid_content,
):
    with pytest.raises(Exception):
        CMSContentCreateRequest(
            title="Guide",
            content=invalid_content,
        )


@pytest.mark.parametrize(
    "invalid_title",
    [
        "",
        "   ",
        None,
    ],
)
def test_cms_create_request_rejects_invalid_title(
    invalid_title,
):
    with pytest.raises(Exception):
        CMSContentCreateRequest(
            title=invalid_title,
            content=(
                "Use semantic HTML."
            ),
        )


def test_cms_create_request_preserves_title_and_content():
    title = "Accessibility Guide"
    content = (
        "Use semantic HTML and keyboard navigation."
    )

    request = CMSContentCreateRequest(
        title=title,
        content=content,
    )

    assert request.title == title
    assert request.content == content


def test_cms_create_request_strips_title_and_content():
    request = CMSContentCreateRequest(
        title="  Accessibility Guide  ",
        content=(
            "  Use semantic HTML.  "
        ),
    )

    assert request.title == (
        "Accessibility Guide"
    )
    assert request.content == (
        "Use semantic HTML."
    )


def test_cms_create_request_deduplicates_tags_case_insensitively():
    request = CMSContentCreateRequest(
        title="Accessibility Guide",
        content="Use semantic HTML.",
        tags=[
            "accessibility",
            "ACCESSIBILITY",
            "Accessibility",
            "design",
            "DESIGN",
        ],
    )

    assert request.tags == [
        "accessibility",
        "design",
    ]


def test_cms_create_request_handles_empty_tags():
    request = CMSContentCreateRequest(
        title="Accessibility Guide",
        content="Use semantic HTML.",
        tags=[],
    )

    assert request.tags == []


def test_cms_create_request_handles_none_tags():
    request = CMSContentCreateRequest(
        title="Accessibility Guide",
        content="Use semantic HTML.",
        tags=None,
    )

    assert request.tags == []


# ---------------------------------------------------------------------------
# CMS publish request
# ---------------------------------------------------------------------------


def test_cms_publish_request():
    content_id = uuid4()

    request = CMSPublishRequest(
        content_id=content_id,
        comment="Ready for publication.",
    )

    assert (
        request.content_id == content_id
    )
    assert request.comment == (
        "Ready for publication."
    )


def test_cms_publish_request_strips_comment():
    request = CMSPublishRequest(
        content_id=uuid4(),
        comment=(
            "  Ready for publication.  "
        ),
    )

    assert request.comment == (
        "Ready for publication."
    )


def test_cms_publish_request_allows_empty_comment():
    request = CMSPublishRequest(
        content_id=uuid4(),
        comment="",
    )

    assert request.comment == ""


def test_cms_publish_request_allows_none_comment():
    request = CMSPublishRequest(
        content_id=uuid4(),
        comment=None,
    )

    assert request.comment is None


def test_cms_publish_request_requires_content_id():
    with pytest.raises(Exception):
        CMSPublishRequest(
            comment=(
                "Ready for publication."
            )
        )


@pytest.mark.parametrize(
    "invalid_content_id",
    [
        "",
        "not-a-uuid",
        "123",
        None,
    ],
)
def test_cms_publish_request_rejects_invalid_content_id(
    invalid_content_id,
):
    with pytest.raises(Exception):
        CMSPublishRequest(
            content_id=invalid_content_id,
            comment="Ready for publication.",
        )


# ---------------------------------------------------------------------------
# Content-type validation
# ---------------------------------------------------------------------------


def test_validate_content_type_default():
    assert (
        _validate_content_type(None)
        == "article"
    )


def test_validate_content_type_normalizes_value():
    assert (
        _validate_content_type(
            "  DOCUMENTATION  "
        )
        == "documentation"
    )


def test_validate_content_type_rejects_invalid():
    with pytest.raises(
        AuthorizationError
    ):
        _validate_content_type(123)


@pytest.mark.parametrize(
    "content_type,expected",
    [
        (None, "article"),
        ("article", "article"),
        ("ARTICLE", "article"),
        ("  article  ", "article"),
        ("documentation", "documentation"),
        ("DOCUMENTATION", "documentation"),
        ("  Documentation  ", "documentation"),
        ("guide", "guide"),
        ("GUIDE", "guide"),
        ("  guide  ", "guide"),
    ],
)
def test_validate_content_type_accepts_supported_values(
    content_type,
    expected: str,
):
    assert (
        _validate_content_type(
            content_type
        )
        == expected
    )


@pytest.mark.parametrize(
    "invalid_content_type",
    [
        "",
        "   ",
        "unknown",
        "blog",
        "page",
        "product",
        "news",
        "123",
        [],
        {},
        True,
    ],
)
def test_validate_content_type_rejects_unsupported_values(
    invalid_content_type,
):
    with pytest.raises(
        AuthorizationError
    ):
        _validate_content_type(
            invalid_content_type
        )


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(
        pytest.main(
            [
                __file__,
                "-v",
            ]
        )
    )
