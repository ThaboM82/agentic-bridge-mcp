"""
Unit tests for shared application constants.

These tests protect critical security, pagination, upload, workflow, retrieval,
and observability boundaries from accidental regression. They do not require a
database, external services, or application startup.
"""

from __future__ import annotations

import pytest

from app.core.constants import (
    API_V1_PREFIX,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ALLOWED_UPLOAD_EXTENSIONS,
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_INGESTION_BATCH_SIZE,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_PAGE_SIZE,
    DEFAULT_VECTOR_SEARCH_LIMIT,
    DEFAULT_VECTOR_SEARCH_SIMILARITY_THRESHOLD,
    ENVIRONMENT,
    MAX_ACCESS_TOKEN_EXPIRE_MINUTES,
    MAX_CONCURRENT_WORKFLOW_EXECUTIONS,
    MAX_DOCUMENT_BATCH_SIZE,
    MAX_DOCUMENT_TITLE_LENGTH,
    MAX_EMBEDDING_DIMENSIONS,
    MAX_GENERATED_FILE_COUNT,
    MAX_GENERATED_FILE_SIZE_CHARS,
    MAX_INGESTION_BATCH_SIZE,
    MAX_PAGE_SIZE,
    MAX_QUERY_VECTOR_DIMENSIONS,
    MAX_RETRY_ATTEMPTS,
    MAX_SEARCH_QUERY_LENGTH,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_WORKFLOW_EXECUTION_TIMEOUT_SECONDS,
    MAX_WORKFLOW_LOG_ENTRIES,
    MAX_WORKFLOW_REQUIREMENTS,
    MIN_ACCESS_TOKEN_EXPIRE_MINUTES,
    MIN_WORKFLOW_EXECUTION_TIMEOUT_SECONDS,
    REDIS_CACHE_TTL_SECONDS,
    REQUEST_ID_HEADER,
    RETRY_AFTER_HEADER,
    SECURITY_HEADERS,
    WORKFLOW_EXECUTION_TIMEOUT_SECONDS,
)


# ---------------------------------------------------------------------------
# API and HTTP conventions
# ---------------------------------------------------------------------------


def test_api_prefix():
    assert API_V1_PREFIX == "/api/v1"


def test_api_prefix_format():
    assert API_V1_PREFIX.startswith("/")
    assert not API_V1_PREFIX.endswith("/")
    assert " " not in API_V1_PREFIX


def test_required_headers():
    assert REQUEST_ID_HEADER == "X-Request-ID"
    assert RETRY_AFTER_HEADER == "Retry-After"


def test_security_headers_present():
    assert SECURITY_HEADERS["X-Content-Type-Options"] == "nosniff"
    assert SECURITY_HEADERS["X-Frame-Options"] == "DENY"


def test_security_headers_include_hsts():
    assert "Strict-Transport-Security" in SECURITY_HEADERS


def test_security_headers_include_referrer_policy():
    assert "Referrer-Policy" in SECURITY_HEADERS


def test_security_headers_are_immutable():
    with pytest.raises(TypeError):
        SECURITY_HEADERS["X-Frame-Options"] = "SAMEORIGIN"


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_pagination_bounds():
    assert DEFAULT_PAGE_SIZE <= MAX_PAGE_SIZE
    assert MAX_PAGE_SIZE > 0


def test_pagination_values_are_positive_integers():
    assert isinstance(DEFAULT_PAGE_SIZE, int)
    assert isinstance(MAX_PAGE_SIZE, int)
    assert DEFAULT_PAGE_SIZE > 0
    assert MAX_PAGE_SIZE > 0


def test_document_batch_limits():
    assert DEFAULT_INGESTION_BATCH_SIZE > 0
    assert MAX_DOCUMENT_BATCH_SIZE > 0
    assert MAX_INGESTION_BATCH_SIZE >= DEFAULT_INGESTION_BATCH_SIZE


def test_document_title_limit():
    assert MAX_DOCUMENT_TITLE_LENGTH > 0


# ---------------------------------------------------------------------------
# Uploads and documents
# ---------------------------------------------------------------------------


def test_upload_size_limit():
    assert MAX_UPLOAD_SIZE_BYTES > 0


def test_allowed_upload_extensions_are_normalized():
    assert ALLOWED_UPLOAD_EXTENSIONS

    for extension in ALLOWED_UPLOAD_EXTENSIONS:
        assert extension.startswith(".")
        assert extension == extension.lower()
        assert " " not in extension


def test_document_batch_size_bounds():
    assert MAX_DOCUMENT_BATCH_SIZE >= 1
    assert MAX_INGESTION_BATCH_SIZE >= MAX_DOCUMENT_BATCH_SIZE


# ---------------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------------


def test_workflow_timeout_bounds():
    assert (
        MIN_WORKFLOW_EXECUTION_TIMEOUT_SECONDS
        <= WORKFLOW_EXECUTION_TIMEOUT_SECONDS
        <= MAX_WORKFLOW_EXECUTION_TIMEOUT_SECONDS
    )


def test_workflow_concurrency_bounds():
    assert MAX_CONCURRENT_WORKFLOW_EXECUTIONS > 0


def test_workflow_retry_bounds():
    assert MAX_RETRY_ATTEMPTS >= 0


def test_workflow_content_bounds():
    assert MAX_WORKFLOW_REQUIREMENTS > 0
    assert MAX_WORKFLOW_LOG_ENTRIES > 0
    assert MAX_GENERATED_FILE_COUNT > 0
    assert MAX_GENERATED_FILE_SIZE_CHARS > 0


def test_workflow_generated_file_limits_are_ordered():
    assert MAX_GENERATED_FILE_SIZE_CHARS > 0
    assert MAX_GENERATED_FILE_COUNT >= 1


# ---------------------------------------------------------------------------
# Retrieval and embeddings
# ---------------------------------------------------------------------------


def test_vector_search_defaults():
    assert DEFAULT_VECTOR_SEARCH_LIMIT > 0
    assert -1.0 <= DEFAULT_VECTOR_SEARCH_SIMILARITY_THRESHOLD <= 1.0


def test_embedding_dimensions():
    assert DEFAULT_EMBEDDING_DIMENSIONS > 0
    assert MAX_EMBEDDING_DIMENSIONS >= DEFAULT_EMBEDDING_DIMENSIONS


def test_query_vector_dimensions():
    assert MAX_QUERY_VECTOR_DIMENSIONS > 0
    assert MAX_QUERY_VECTOR_DIMENSIONS >= MAX_EMBEDDING_DIMENSIONS


def test_search_query_limit():
    assert MAX_SEARCH_QUERY_LENGTH > 0


# ---------------------------------------------------------------------------
# Authentication and caching
# ---------------------------------------------------------------------------


def test_access_token_expiry_bounds():
    assert (
        MIN_ACCESS_TOKEN_EXPIRE_MINUTES
        <= ACCESS_TOKEN_EXPIRE_MINUTES
        <= MAX_ACCESS_TOKEN_EXPIRE_MINUTES
    )


def test_redis_cache_ttl():
    assert REDIS_CACHE_TTL_SECONDS > 0


# ---------------------------------------------------------------------------
# LLM defaults
# ---------------------------------------------------------------------------


def test_default_llm_temperature():
    assert 0.0 <= DEFAULT_LLM_TEMPERATURE <= 2.0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_environment_is_supported():
    assert ENVIRONMENT in {
        "local",
        "development",
        "test",
        "staging",
        "production",
    }


# ---------------------------------------------------------------------------
# Cross-cutting invariants
# ---------------------------------------------------------------------------


def test_security_headers_do_not_contain_secrets():
    serialized_headers = str(SECURITY_HEADERS)

    forbidden_fragments = (
        "api_key",
        "apikey",
        "secret",
        "password",
        "token",
        "authorization",
        "postgresql://",
        "redis://",
    )

    for fragment in forbidden_fragments:
        assert fragment not in serialized_headers.lower()


def test_constant_names_are_importable():
    # Ensures the public constant surface remains stable for importing modules.
    import app.core.constants as constants

    required_constants = (
        "API_V1_PREFIX",
        "DEFAULT_PAGE_SIZE",
        "MAX_PAGE_SIZE",
        "REQUEST_ID_HEADER",
        "RETRY_AFTER_HEADER",
        "SECURITY_HEADERS",
        "MAX_UPLOAD_SIZE_BYTES",
        "MAX_WORKFLOW_EXECUTION_TIMEOUT_SECONDS",
        "MAX_QUERY_VECTOR_DIMENSIONS",
    )

    for constant_name in required_constants:
        assert hasattr(constants, constant_name)
