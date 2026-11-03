"""
Comprehensive tests for vector embedding CRUD normalization and validation.

These tests cover embedding and document IDs, embedding vectors, source
chunks, metadata, search limits, similarity thresholds, security boundaries,
numeric edge cases, and fail-closed validation behavior.
"""

from __future__ import annotations

import asyncio
import math
from uuid import uuid4

import pytest

from app.core.exceptions import ValidationError
from app.crud.vector_embedding_crud import (
    _normalize_document_id,
    _normalize_embedding_id,
    _normalize_search_limit,
    _normalize_similarity_threshold,
    _validate_embedding_vector,
    _validate_metadata,
    _validate_source_chunk,
)


# ---------------------------------------------------------------------------
# Embedding-ID normalization
# ---------------------------------------------------------------------------


def test_normalize_embedding_id():
    embedding_id = "123e4567-e89b-12d3-a456-426614174000"

    assert _normalize_embedding_id(
        embedding_id
    )


def test_normalize_embedding_id_accepts_uuid_object():
    embedding_id = uuid4()

    assert (
        _normalize_embedding_id(
            embedding_id
        )
        == str(embedding_id)
    )


def test_normalize_embedding_id_strips_whitespace():
    embedding_id = str(uuid4())

    assert (
        _normalize_embedding_id(
            f"  {embedding_id}  "
        )
        == embedding_id
    )


def test_normalize_embedding_id_returns_canonical_string():
    embedding_id = uuid4()

    normalized = _normalize_embedding_id(
        str(embedding_id).upper()
    )

    assert normalized == str(embedding_id)


@pytest.mark.parametrize(
    "invalid_embedding_id",
    [
        "",
        "   ",
        "not-a-uuid",
        "123",
        "123e4567-e89b-12d3-a456",
        "123e4567-e89b-12d3-a456-42661417400",
        "123e4567-e89b-12d3-a456-4266141740000",
        "zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz",
        None,
    ],
)
def test_invalid_embedding_id(
    invalid_embedding_id,
):
    with pytest.raises(ValidationError):
        _normalize_embedding_id(
            invalid_embedding_id
        )


# ---------------------------------------------------------------------------
# Document-ID normalization
# ---------------------------------------------------------------------------


def test_normalize_document_id():
    document_id = "123e4567-e89b-12d3-a456-426614174000"

    assert _normalize_document_id(
        document_id
    )


def test_normalize_document_id_accepts_uuid_object():
    document_id = uuid4()

    assert (
        _normalize_document_id(
            document_id
        )
        == str(document_id)
    )


def test_normalize_document_id_strips_whitespace():
    document_id = str(uuid4())

    assert (
        _normalize_document_id(
            f"  {document_id}  "
        )
        == document_id
    )


def test_normalize_document_id_returns_canonical_string():
    document_id = uuid4()

    normalized = _normalize_document_id(
        str(document_id).upper()
    )

    assert normalized == str(document_id)


@pytest.mark.parametrize(
    "invalid_document_id",
    [
        "",
        "   ",
        "not-a-uuid",
        "123",
        "123e4567-e89b-12d3-a456",
        "123e4567-e89b-12d3-a456-42661417400",
        "123e4567-e89b-12d3-a456-4266141740000",
        "zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz",
        None,
    ],
)
def test_invalid_document_id(
    invalid_document_id,
):
    with pytest.raises(ValidationError):
        _normalize_document_id(
            invalid_document_id
        )


# ---------------------------------------------------------------------------
# Embedding-vector validation
# ---------------------------------------------------------------------------


def test_validate_embedding_vector():
    assert (
        _validate_embedding_vector(
            [0.1, 0.2, 0.3]
        )
        == [0.1, 0.2, 0.3]
    )


def test_validate_embedding_vector_accepts_integers():
    assert (
        _validate_embedding_vector(
            [1, 2, 3]
        )
        == [1.0, 2.0, 3.0]
    )


def test_validate_embedding_vector_accepts_single_value():
    assert (
        _validate_embedding_vector([0.5])
        == [0.5]
    )


def test_validate_embedding_vector_accepts_negative_values():
    vector = [-0.1, -0.2, -0.3]

    assert (
        _validate_embedding_vector(vector)
        == vector
    )


def test_validate_embedding_vector_accepts_zero_values():
    assert (
        _validate_embedding_vector(
            [0.0, 0.0, 0.0]
        )
        == [0.0, 0.0, 0.0]
    )


def test_validate_embedding_vector_preserves_order():
    vector = [0.3, 0.1, 0.2]

    assert (
        _validate_embedding_vector(vector)
        == vector
    )


def test_invalid_embedding_vector_empty():
    with pytest.raises(ValidationError):
        _validate_embedding_vector([])


def test_invalid_embedding_vector_non_numeric():
    with pytest.raises(ValidationError):
        _validate_embedding_vector(
            ["a", "b"]
        )


def test_invalid_embedding_vector_nan():
    with pytest.raises(ValidationError):
        _validate_embedding_vector(
            [float("nan")]
        )


def test_invalid_embedding_vector_infinity():
    with pytest.raises(ValidationError):
        _validate_embedding_vector(
            [float("inf")]
        )


def test_invalid_embedding_vector_negative_infinity():
    with pytest.raises(ValidationError):
        _validate_embedding_vector(
            [float("-inf")]
        )


@pytest.mark.parametrize(
    "invalid_vector",
    [
        None,
        "not-a-list",
        123,
        {},
        [None],
        [True],
        [{"value": 1}],
        [[1, 2], [3, 4]],
        ["0.1", "0.2"],
    ],
)
def test_invalid_embedding_vector_types(
    invalid_vector,
):
    with pytest.raises(ValidationError):
        _validate_embedding_vector(
            invalid_vector
        )


def test_embedding_vector_rejects_oversized_vectors():
    oversized_vector = [
        0.1
    ] * 10_001

    with pytest.raises(ValidationError):
        _validate_embedding_vector(
            oversized_vector
        )


# ---------------------------------------------------------------------------
# Source-chunk validation
# ---------------------------------------------------------------------------


def test_validate_source_chunk():
    assert (
        _validate_source_chunk(
            "  design guidance  "
        )
        == "design guidance"
    )
    assert (
        _validate_source_chunk("   ")
        is None
    )
    assert (
        _validate_source_chunk(None)
        is None
    )


def test_validate_source_chunk_preserves_internal_whitespace():
    assert (
        _validate_source_chunk(
            "design   guidance"
        )
        == "design   guidance"
    )


def test_validate_source_chunk_handles_empty_string():
    assert (
        _validate_source_chunk("")
        is None
    )


@pytest.mark.parametrize(
    "valid_chunk",
    [
        "design guidance",
        "Use semantic HTML.",
        "The dashboard must support keyboard navigation.",
        "Chunk with\nnewlines",
        "Chunk with\ttabs",
    ],
)
def test_validate_source_chunk_accepts_valid_text(
    valid_chunk: str,
):
    assert (
        _validate_source_chunk(
            valid_chunk
        )
        == valid_chunk.strip()
    )


def test_invalid_source_chunk_length():
    with pytest.raises(ValidationError):
        _validate_source_chunk(
            "x" * 200_000
        )


def test_source_chunk_rejects_non_string_values():
    with pytest.raises(ValidationError):
        _validate_source_chunk(123)

    with pytest.raises(ValidationError):
        _validate_source_chunk(
            ["chunk"]
        )

    with pytest.raises(ValidationError):
        _validate_source_chunk(
            {"chunk": "value"}
        )


def test_source_chunk_does_not_leak_oversized_content():
    oversized_chunk = "x" * 200_000

    with pytest.raises(ValidationError):
        _validate_source_chunk(
            oversized_chunk
        )


# ---------------------------------------------------------------------------
# Metadata validation
# ---------------------------------------------------------------------------


def test_metadata_rejects_sensitive_keys():
    with pytest.raises(ValidationError):
        _validate_metadata(
            {"api_key": "secret"}
        )


@pytest.mark.parametrize(
    "sensitive_key",
    [
        "api_key",
        "apikey",
        "API_KEY",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "credential",
        "cookie",
        "private_key",
        "database_url",
        "connection_string",
        "prompt",
        "completion",
        "messages",
        "embedding",
        "vector",
        "generated_code",
        "signed_url",
        "storage_url",
        "file_url",
        "download_url",
    ],
)
def test_metadata_rejects_common_sensitive_keys(
    sensitive_key: str,
):
    with pytest.raises(ValidationError):
        _validate_metadata(
            {
                "safe": "value",
                sensitive_key: "should-not-leak",
            }
        )


def test_metadata_accepts_safe_values():
    metadata = {
        "model": "text-embedding-3-small"
    }

    assert (
        _validate_metadata(metadata)
        == metadata
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"model": "text-embedding-3-small"},
        {"source": "confluence"},
        {"chunk_index": 1},
        {"language": "en"},
        {"tags": ["design", "accessibility"]},
    ],
)
def test_metadata_accepts_safe_metadata(
    metadata: dict,
):
    assert (
        _validate_metadata(metadata)
        == metadata
    )


@pytest.mark.parametrize(
    "invalid_metadata",
    [
        "not-a-dictionary",
        123,
        [],
        None,
    ],
)
def test_metadata_rejects_non_dict_metadata(
    invalid_metadata,
):
    with pytest.raises(ValidationError):
        _validate_metadata(
            invalid_metadata
        )


def test_metadata_rejects_nested_sensitive_values():
    with pytest.raises(ValidationError):
        _validate_metadata(
            {
                "source": {
                    "api_key": "should-not-leak"
                }
            }
        )


def test_metadata_rejects_overly_large_metadata():
    metadata = {
        f"key_{index}": f"value-{index}"
        for index in range(100)
    }

    with pytest.raises(ValidationError):
        _validate_metadata(metadata)


# ---------------------------------------------------------------------------
# Search-limit normalization
# ---------------------------------------------------------------------------


def test_normalize_search_limit():
    assert (
        _normalize_search_limit(None)
        == 5
    )
    assert (
        _normalize_search_limit(500)
        == 50
    )


@pytest.mark.parametrize(
    "limit,expected",
    [
        (None, 5),
        (1, 1),
        (5, 5),
        (10, 10),
        (50, 50),
        (100, 50),
        (500, 50),
    ],
)
def test_normalize_search_limit_bounds_values(
    limit,
    expected: int,
):
    assert (
        _normalize_search_limit(limit)
        == expected
    )


@pytest.mark.parametrize(
    "invalid_limit",
    [
        0,
        -1,
        -100,
        "five",
        1.5,
        [],
        {},
    ],
)
def test_invalid_search_limit(
    invalid_limit,
):
    with pytest.raises(ValidationError):
        _normalize_search_limit(
            invalid_limit
        )


# ---------------------------------------------------------------------------
# Similarity-threshold normalization
# ---------------------------------------------------------------------------


def test_normalize_similarity_threshold():
    assert (
        _normalize_similarity_threshold(
            None
        )
        == 0.0
    )
    assert (
        _normalize_similarity_threshold(
            0.75
        )
        == 0.75
    )


@pytest.mark.parametrize(
    "threshold,expected",
    [
        (None, 0.0),
        (0.0, 0.0),
        (0.25, 0.25),
        (0.5, 0.5),
        (0.75, 0.75),
        (1.0, 1.0),
    ],
)
def test_normalize_similarity_threshold_accepts_valid_values(
    threshold,
    expected: float,
):
    assert (
        _normalize_similarity_threshold(
            threshold
        )
        == expected
    )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        -0.1,
        -1.0,
        1.1,
        1.5,
        100,
        "high",
        [],
        {},
    ],
)
def test_invalid_similarity_threshold(
    invalid_threshold,
):
    with pytest.raises(ValidationError):
        _normalize_similarity_threshold(
            invalid_threshold
        )


# ---------------------------------------------------------------------------
# Cross-field consistency
# ---------------------------------------------------------------------------


def test_embedding_and_document_ids_are_normalized_independently():
    embedding_id = uuid4()
    document_id = uuid4()

    normalized_embedding_id = (
        _normalize_embedding_id(
            embedding_id
        )
    )
    normalized_document_id = (
        _normalize_document_id(
            document_id
        )
    )

    assert (
        normalized_embedding_id
        == str(embedding_id)
    )
    assert (
        normalized_document_id
        == str(document_id)
    )


def test_default_search_limit_and_threshold_are_safe():
    assert (
        _normalize_search_limit(None)
        == 5
    )
    assert (
        _normalize_similarity_threshold(
            None
        )
        == 0.0
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
