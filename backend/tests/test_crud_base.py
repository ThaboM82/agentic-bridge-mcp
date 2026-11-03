"""
Unit tests for shared CRUD query helpers.

These tests verify identifier normalization, tenant-safe filter composition,
pagination bounds, ordering, and deterministic query construction. They do not
require a live database.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy import (
    asc,
    desc,
    select,
)

from app.core.exceptions import (
    ValidationError,
)
from app.crud.base import (
    apply_filters,
    apply_ordering,
    apply_pagination,
    normalize_uuid,
    normalize_uuid_list,
)
from app.models.documents import (
    DocumentModel,
)


# ---------------------------------------------------------------------------
# UUID normalization
# ---------------------------------------------------------------------------


def test_normalize_uuid_from_string():
    value = (
        "123e4567-e89b-12d3-a456-"
        "426614174000"
    )

    normalized_value = normalize_uuid(
        value,
    )

    assert isinstance(
        normalized_value,
        UUID,
    )
    assert str(normalized_value) == value


def test_normalize_uuid_from_uuid_instance():
    value = uuid4()

    assert normalize_uuid(value) == value


def test_normalize_uuid_strips_whitespace():
    value = uuid4()

    assert normalize_uuid(
        f"  {value}  ",
    ) == value


def test_normalize_uuid_accepts_braces_and_urn():
    value = uuid4()

    assert normalize_uuid(
        f"{{{value}}}",
    ) == value

    assert normalize_uuid(
        f"urn:uuid:{value}",
    ) == value


def test_normalize_invalid_uuid():
    with pytest.raises(
        ValidationError,
    ):
        normalize_uuid(
            "not-a-uuid",
        )


def test_normalize_empty_uuid():
    with pytest.raises(
        ValidationError,
    ):
        normalize_uuid("   ")


def test_normalize_none_uuid():
    with pytest.raises(
        ValidationError,
    ):
        normalize_uuid(None)


def test_normalize_uuid_rejects_non_string_types():
    with pytest.raises(
        ValidationError,
    ):
        normalize_uuid(12345)


def test_normalize_uuid_list_deduplicates_and_preserves_order():
    first_id = uuid4()
    second_id = uuid4()

    normalized_ids = normalize_uuid_list(
        [
            str(first_id),
            first_id,
            f"  {second_id}  ",
            str(second_id),
        ],
    )

    assert normalized_ids == [
        first_id,
        second_id,
    ]


def test_normalize_uuid_list_rejects_empty_input():
    with pytest.raises(
        ValidationError,
    ):
        normalize_uuid_list([])


def test_normalize_uuid_list_rejects_invalid_value():
    with pytest.raises(
        ValidationError,
    ):
        normalize_uuid_list(
            ["not-a-uuid"],
        )


def test_normalize_uuid_list_rejects_none_value():
    with pytest.raises(
        ValidationError,
    ):
        normalize_uuid_list(
            [None],
        )


def test_normalize_uuid_list_accepts_uuid_instance():
    value = uuid4()

    assert normalize_uuid_list(
        [value],
    ) == [value]


# ---------------------------------------------------------------------------
# Filter composition
# ---------------------------------------------------------------------------


def test_apply_filters_single_condition():
    document_id = uuid4()

    query = apply_filters(
        select(DocumentModel),
        DocumentModel.id == document_id,
    )

    assert query is not None
    assert str(document_id) in str(query)


def test_apply_filters_multiple_conditions():
    document_id = uuid4()
    tenant_id = uuid4()

    query = apply_filters(
        select(DocumentModel),
        DocumentModel.id == document_id,
        DocumentModel.tenant_id
        == tenant_id,
    )

    assert query is not None
    assert str(document_id) in str(query)
    assert str(tenant_id) in str(query)


def test_apply_filters_with_no_conditions_returns_original_query():
    base_query = select(
        DocumentModel,
    )

    query = apply_filters(
        base_query,
    )

    assert query is not None


def test_apply_filters_ignores_none_conditions():
    document_id = uuid4()

    query = apply_filters(
        select(DocumentModel),
        DocumentModel.id == document_id,
        None,
    )

    assert query is not None
    assert str(document_id) in str(query)


def test_apply_filters_rejects_invalid_condition_type():
    with pytest.raises(
        ValidationError,
    ):
        apply_filters(
            select(DocumentModel),
            "invalid-condition",
        )


def test_apply_filters_rejects_invalid_query_type():
    with pytest.raises(
        ValidationError,
    ):
        apply_filters(
            "not-a-query",
            DocumentModel.id == uuid4(),
        )


def test_apply_filters_supports_tenant_isolation():
    tenant_id = uuid4()

    query = apply_filters(
        select(DocumentModel),
        DocumentModel.tenant_id
        == tenant_id,
    )

    assert query is not None
    assert str(tenant_id) in str(query)


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_apply_pagination():
    query = apply_pagination(
        select(DocumentModel),
        skip=10,
        limit=25,
    )

    assert query is not None
    assert "LIMIT 25" in str(query)
    assert "OFFSET 10" in str(query)


def test_apply_pagination_defaults():
    query = apply_pagination(
        select(DocumentModel),
    )

    assert query is not None
    assert "LIMIT 50" in str(query)
    assert "OFFSET 0" in str(query)


def test_apply_pagination_zero_skip():
    query = apply_pagination(
        select(DocumentModel),
        skip=0,
        limit=10,
    )

    assert query is not None
    assert "LIMIT 10" in str(query)
    assert "OFFSET 0" in str(query)


def test_apply_pagination_clamps_excessive_limit():
    query = apply_pagination(
        select(DocumentModel),
        skip=0,
        limit=10_000,
    )

    assert query is not None
    assert "LIMIT 100" in str(query)


def test_invalid_pagination():
    with pytest.raises(
        ValidationError,
    ):
        apply_pagination(
            select(DocumentModel),
            skip=-1,
            limit=10,
        )

    with pytest.raises(
        ValidationError,
    ):
        apply_pagination(
            select(DocumentModel),
            skip=0,
            limit=0,
        )


def test_pagination_rejects_non_integer_values():
    with pytest.raises(
        ValidationError,
    ):
        apply_pagination(
            select(DocumentModel),
            skip="10",
            limit=10,
        )

    with pytest.raises(
        ValidationError,
    ):
        apply_pagination(
            select(DocumentModel),
            skip=0,
            limit=10.5,
        )


def test_pagination_rejects_boolean_values():
    with pytest.raises(
        ValidationError,
    ):
        apply_pagination(
            select(DocumentModel),
            skip=True,
            limit=10,
        )

    with pytest.raises(
        ValidationError,
    ):
        apply_pagination(
            select(DocumentModel),
            skip=0,
            limit=False,
        )


def test_pagination_rejects_none_values():
    with pytest.raises(
        ValidationError,
    ):
        apply_pagination(
            select(DocumentModel),
            skip=None,
            limit=10,
        )

    with pytest.raises(
        ValidationError,
    ):
        apply_pagination(
            select(DocumentModel),
            skip=0,
            limit=None,
        )


def test_pagination_rejects_invalid_query_type():
    with pytest.raises(
        ValidationError,
    ):
        apply_pagination(
            "not-a-query",
            skip=0,
            limit=10,
        )


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_apply_ordering_single_column():
    query = apply_ordering(
        select(DocumentModel),
        DocumentModel.created_at,
    )

    assert query is not None
    assert "ORDER BY" in str(query)


def test_apply_ordering_multiple_columns():
    query = apply_ordering(
        select(DocumentModel),
        DocumentModel.created_at,
        DocumentModel.title,
    )

    assert query is not None
    assert "ORDER BY" in str(query)


def test_apply_ordering_with_directions():
    query = apply_ordering(
        select(DocumentModel),
        (
            DocumentModel.created_at,
            desc,
        ),
        (
            DocumentModel.title,
            asc,
        ),
    )

    assert query is not None
    assert "ORDER BY" in str(query)
    assert "DESC" in str(query).upper()
    assert "ASC" in str(query).upper()


def test_apply_ordering_with_no_columns_returns_original_query():
    base_query = select(
        DocumentModel,
    )

    query = apply_ordering(
        base_query,
    )

    assert query is not None


def test_apply_ordering_rejects_invalid_column():
    with pytest.raises(
        ValidationError,
    ):
        apply_ordering(
            select(DocumentModel),
            "created_at",
        )


def test_apply_ordering_rejects_invalid_direction():
    with pytest.raises(
        ValidationError,
    ):
        apply_ordering(
            select(DocumentModel),
            (
                DocumentModel.created_at,
                "invalid_direction",
            ),
        )


def test_apply_ordering_rejects_invalid_query_type():
    with pytest.raises(
        ValidationError,
    ):
        apply_ordering(
            "not-a-query",
            DocumentModel.created_at,
        )


def test_apply_ordering_rejects_none_column():
    with pytest.raises(
        ValidationError,
    ):
        apply_ordering(
            select(DocumentModel),
            None,
        )


# ---------------------------------------------------------------------------
# Combined query helpers
# ---------------------------------------------------------------------------


def test_combined_filters_pagination_and_ordering():
    tenant_id = uuid4()

    query = apply_pagination(
        apply_ordering(
            apply_filters(
                select(
                    DocumentModel,
                ),
                DocumentModel.tenant_id
                == tenant_id,
            ),
            (
                DocumentModel.created_at,
                desc,
            ),
        ),
        skip=0,
        limit=20,
    )

    assert query is not None
    assert str(tenant_id) in str(query)
    assert "ORDER BY" in str(query)
    assert "LIMIT 20" in str(query)
    assert "OFFSET 0" in str(query)


def test_combined_query_is_deterministic():
    tenant_id = uuid4()

    first_query = apply_pagination(
        apply_ordering(
            apply_filters(
                select(
                    DocumentModel,
                ),
                DocumentModel.tenant_id
                == tenant_id,
            ),
            DocumentModel.created_at,
        ),
        skip=0,
        limit=20,
    )

    second_query = apply_pagination(
        apply_ordering(
            apply_filters(
                select(
                    DocumentModel,
                ),
                DocumentModel.tenant_id
                == tenant_id,
            ),
            DocumentModel.created_at,
        ),
        skip=0,
        limit=20,
    )

    assert str(first_query) == str(
        second_query,
    )


def test_combined_query_with_multiple_filters_and_order_columns():
    tenant_id = uuid4()
    document_id = uuid4()

    query = apply_pagination(
        apply_ordering(
            apply_filters(
                select(
                    DocumentModel,
                ),
                DocumentModel.tenant_id
                == tenant_id,
                DocumentModel.id
                == document_id,
            ),
            (
                DocumentModel.created_at,
                desc,
            ),
            DocumentModel.title,
        ),
        skip=5,
        limit=15,
    )

    assert query is not None
    assert str(tenant_id) in str(query)
    assert str(document_id) in str(query)
    assert "ORDER BY" in str(query)
    assert "LIMIT 15" in str(query)
    assert "OFFSET 5" in str(query)


__all__ = []
