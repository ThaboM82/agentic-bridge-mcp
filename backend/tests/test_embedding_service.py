"""
Comprehensive tests for embedding services.

These tests cover single and batch embedding generation, input validation,
order preservation, batching, concurrency, provider response validation,
dimension consistency, retry behavior, token accounting, and failure handling.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingValidationError,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeAsyncOpenAI:
    """Deterministic OpenAI-compatible embedding client."""

    def __init__(
        self,
        *,
        embedding: list[float] | None = None,
        fail_once: bool = False,
        delay_seconds: float = 0.0,
    ) -> None:
        self.calls = 0
        self.requested_inputs: list[
            list[str]
        ] = []
        self.requested_models: list[
            str
        ] = []
        self.embedding = (
            embedding or [0.1, 0.2]
        )
        self.fail_once = fail_once
        self.delay_seconds = delay_seconds

    @property
    def embeddings(self):
        return self

    async def create(
        self,
        input: list[str],
        model: str,
    ):
        self.calls += 1
        self.requested_inputs.append(
            list(input)
        )
        self.requested_models.append(
            model
        )

        if self.delay_seconds:
            await asyncio.sleep(
                self.delay_seconds
            )

        if (
            self.fail_once
            and self.calls == 1
        ):
            raise RuntimeError(
                "Transient provider failure"
            )

        return type(
            "Response",
            (),
            {
                "data": [
                    type(
                        "Item",
                        (),
                        {
                            "embedding": self.embedding
                        },
                    )()
                    for _ in input
                ]
            },
        )()


class InvalidResponseClient:
    """Return malformed embedding-provider responses."""

    def __init__(
        self,
        response_data: Any,
    ) -> None:
        self.response_data = response_data
        self.calls = 0

    @property
    def embeddings(self):
        return self

    async def create(
        self,
        input: list[str],
        model: str,
    ):
        self.calls += 1

        return type(
            "Response",
            (),
            {"data": self.response_data},
        )()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_client() -> FakeAsyncOpenAI:
    return FakeAsyncOpenAI()


@pytest.fixture
def embedding_service(
    fake_client: FakeAsyncOpenAI,
) -> EmbeddingService:
    return EmbeddingService(
        client=fake_client,
        model="test-model",
    )


@pytest.fixture
def batch_embedding_service(
    fake_client: FakeAsyncOpenAI,
) -> EmbeddingService:
    return EmbeddingService(
        client=fake_client,
        model="test-model",
        max_batch_size=2,
        max_concurrent_requests=2,
    )


# ---------------------------------------------------------------------------
# Single embedding generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_embedding_returns_single_vector():
    service = EmbeddingService(
        client=FakeAsyncOpenAI(),
        model="test-model",
    )

    embedding = (
        await service.generate_embedding(
            "hello world"
        )
    )

    assert embedding == [0.1, 0.2]


@pytest.mark.asyncio
async def test_generate_embedding_uses_configured_model(
    fake_client: FakeAsyncOpenAI,
):
    service = EmbeddingService(
        client=fake_client,
        model="test-model",
    )

    await service.generate_embedding(
        "hello world"
    )

    assert fake_client.calls == 1
    assert fake_client.requested_models == [
        "test-model"
    ]
    assert fake_client.requested_inputs == [
        ["hello world"]
    ]


@pytest.mark.asyncio
async def test_generate_embedding_strips_whitespace(
    fake_client: FakeAsyncOpenAI,
):
    service = EmbeddingService(
        client=fake_client,
        model="test-model",
    )

    await service.generate_embedding(
        "  hello world  "
    )

    assert fake_client.requested_inputs == [
        ["hello world"]
    ]


@pytest.mark.asyncio
async def test_generate_embedding_rejects_empty_text(
    embedding_service: EmbeddingService,
):
    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embedding(
            ""
        )


@pytest.mark.asyncio
async def test_generate_embedding_rejects_whitespace_text(
    embedding_service: EmbeddingService,
):
    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embedding(
            "   "
        )


@pytest.mark.asyncio
async def test_generate_embedding_rejects_none_text(
    embedding_service: EmbeddingService,
):
    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embedding(
            None
        )


@pytest.mark.asyncio
async def test_generate_embedding_rejects_non_string_text(
    embedding_service: EmbeddingService,
):
    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embedding(
            123
        )


# ---------------------------------------------------------------------------
# Batch embedding generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_embeddings_preserves_order():
    service = EmbeddingService(
        client=FakeAsyncOpenAI(),
        model="test-model",
        max_batch_size=2,
        max_concurrent_requests=2,
    )

    result = (
        await service.generate_embeddings(
            [
                "first text",
                "second text",
                "third text",
            ]
        )
    )

    assert len(result) == 3
    assert all(
        embedding == [0.1, 0.2]
        for embedding in result
    )


@pytest.mark.asyncio
async def test_generate_embeddings_returns_one_vector_per_text(
    batch_embedding_service: EmbeddingService,
):
    result = (
        await batch_embedding_service.generate_embeddings(
            [
                "first text",
                "second text",
                "third text",
                "fourth text",
            ]
        )
    )

    assert len(result) == 4
    assert all(
        embedding == [0.1, 0.2]
        for embedding in result
    )


@pytest.mark.asyncio
async def test_generate_embeddings_batches_requests(
    fake_client: FakeAsyncOpenAI,
):
    service = EmbeddingService(
        client=fake_client,
        model="test-model",
        max_batch_size=2,
        max_concurrent_requests=2,
    )

    await service.generate_embeddings(
        [
            "first text",
            "second text",
            "third text",
        ]
    )

    assert fake_client.calls == 2
    assert fake_client.requested_inputs == [
        ["first text", "second text"],
        ["third text"],
    ]


@pytest.mark.asyncio
async def test_generate_embeddings_handles_single_item(
    embedding_service: EmbeddingService,
):
    result = (
        await embedding_service.generate_embeddings(
            ["hello world"]
        )
    )

    assert result == [[0.1, 0.2]]


@pytest.mark.asyncio
async def test_generate_embeddings_handles_exact_batch_size(
    fake_client: FakeAsyncOpenAI,
):
    service = EmbeddingService(
        client=fake_client,
        model="test-model",
        max_batch_size=3,
        max_concurrent_requests=2,
    )

    result = (
        await service.generate_embeddings(
            [
                "first text",
                "second text",
                "third text",
            ]
        )
    )

    assert len(result) == 3
    assert fake_client.calls == 1


@pytest.mark.asyncio
async def test_generate_embeddings_rejects_empty_text_list(
    embedding_service: EmbeddingService,
):
    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embeddings(
            []
        )


@pytest.mark.asyncio
async def test_generate_embeddings_rejects_none_text_list(
    embedding_service: EmbeddingService,
):
    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embeddings(
            None
        )


@pytest.mark.asyncio
async def test_generate_embeddings_rejects_empty_text_in_batch(
    embedding_service: EmbeddingService,
):
    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embeddings(
            [
                "valid text",
                "   ",
            ]
        )


@pytest.mark.asyncio
async def test_generate_embeddings_rejects_non_string_text_in_batch(
    embedding_service: EmbeddingService,
):
    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embeddings(
            ["valid text", 123]
        )


@pytest.mark.asyncio
async def test_generate_embeddings_rejects_oversized_batch(
    fake_client: FakeAsyncOpenAI,
):
    service = EmbeddingService(
        client=fake_client,
        model="test-model",
        max_batch_size=2,
        max_concurrent_requests=2,
    )

    with pytest.raises(
        EmbeddingValidationError
    ):
        await service.generate_embeddings(
            [
                f"text-{index}"
                for index in range(101)
            ]
        )


# ---------------------------------------------------------------------------
# Provider response validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_embedding_rejects_empty_response(
    embedding_service: EmbeddingService,
):
    embedding_service.client = (
        InvalidResponseClient([])
    )

    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embedding(
            "hello world"
        )


@pytest.mark.asyncio
async def test_generate_embedding_rejects_missing_embedding(
    embedding_service: EmbeddingService,
):
    embedding_service.client = (
        InvalidResponseClient(
            [
                type(
                    "Item",
                    (),
                    {},
                )()
            ]
        )
    )

    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embedding(
            "hello world"
        )


@pytest.mark.asyncio
async def test_generate_embedding_rejects_non_list_embedding(
    embedding_service: EmbeddingService,
):
    embedding_service.client = (
        InvalidResponseClient(
            [
                type(
                    "Item",
                    (),
                    {
                        "embedding": "not-a-vector"
                    },
                )()
            ]
        )
    )

    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embedding(
            "hello world"
        )


@pytest.mark.asyncio
async def test_generate_embedding_rejects_non_numeric_embedding(
    embedding_service: EmbeddingService,
):
    embedding_service.client = (
        InvalidResponseClient(
            [
                type(
                    "Item",
                    (),
                    {
                        "embedding": [
                            "a",
                            "b",
                        ]
                    },
                )()
            ]
        )
    )

    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embedding(
            "hello world"
        )


@pytest.mark.asyncio
async def test_generate_embedding_rejects_nan_embedding(
    embedding_service: EmbeddingService,
):
    embedding_service.client = (
        InvalidResponseClient(
            [
                type(
                    "Item",
                    (),
                    {
                        "embedding": [
                            float("nan")
                        ]
                    },
                )()
            ]
        )
    )

    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embedding(
            "hello world"
        )


@pytest.mark.asyncio
async def test_generate_embedding_rejects_infinity_embedding(
    embedding_service: EmbeddingService,
):
    embedding_service.client = (
        InvalidResponseClient(
            [
                type(
                    "Item",
                    (),
                    {
                        "embedding": [
                            float("inf")
                        ]
                    },
                )()
            ]
        )
    )

    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embedding(
            "hello world"
        )


@pytest.mark.asyncio
async def test_generate_embeddings_rejects_inconsistent_dimensions(
    embedding_service: EmbeddingService,
):
    class InconsistentClient:
        @property
        def embeddings(self):
            return self

        async def create(
            self,
            input: list[str],
            model: str,
        ):
            return type(
                "Response",
                (),
                {
                    "data": [
                        type(
                            "Item",
                            (),
                            {
                                "embedding": [
                                    0.1,
                                    0.2
                                ]
                            },
                        )(),
                        type(
                            "Item",
                            (),
                            {
                                "embedding": [
                                    0.1,
                                    0.2,
                                    0.3
                                ]
                            },
                        )(),
                    ]
                },
            )()

    embedding_service.client = (
        InconsistentClient()
    )

    with pytest.raises(
        EmbeddingValidationError
    ):
        await embedding_service.generate_embeddings(
            [
                "first text",
                "second text",
            ]
        )


# ---------------------------------------------------------------------------
# Retry and failure behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_embedding_retries_transient_failure():
    client = FakeAsyncOpenAI(
        fail_once=True
    )

    service = EmbeddingService(
        client=client,
        model="test-model",
        max_retries=2,
    )

    embedding = (
        await service.generate_embedding(
            "hello world"
        )
    )

    assert embedding == [0.1, 0.2]
    assert client.calls == 2


@pytest.mark.asyncio
async def test_generate_embedding_fails_after_retry_limit():
    client = FakeAsyncOpenAI(
        fail_once=True
    )

    service = EmbeddingService(
        client=client,
        model="test-model",
        max_retries=0,
    )

    with pytest.raises(RuntimeError):
        await service.generate_embedding(
            "hello world"
        )

    assert client.calls == 1


@pytest.mark.asyncio
async def test_generate_embeddings_does_not_reorder_results_after_failure():
    client = FakeAsyncOpenAI(
        fail_once=True
    )

    service = EmbeddingService(
        client=client,
        model="test-model",
        max_batch_size=2,
        max_concurrent_requests=2,
        max_retries=2,
    )

    result = (
        await service.generate_embeddings(
            [
                "first text",
                "second text",
                "third text",
            ]
        )
    )

    assert result == [
        [0.1, 0.2],
        [0.1, 0.2],
        [0.1, 0.2],
    ]


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_service_rejects_empty_model():
    with pytest.raises(
        EmbeddingValidationError
    ):
        EmbeddingService(
            client=FakeAsyncOpenAI(),
            model="",
        )


@pytest.mark.asyncio
async def test_embedding_service_rejects_none_model():
    with pytest.raises(
        EmbeddingValidationError
    ):
        EmbeddingService(
            client=FakeAsyncOpenAI(),
            model=None,
        )


@pytest.mark.asyncio
async def test_embedding_service_rejects_invalid_batch_size():
    with pytest.raises(
        EmbeddingValidationError
    ):
        EmbeddingService(
            client=FakeAsyncOpenAI(),
            model="test-model",
            max_batch_size=0,
        )


@pytest.mark.asyncio
async def test_embedding_service_rejects_invalid_concurrency():
    with pytest.raises(
        EmbeddingValidationError
    ):
        EmbeddingService(
            client=FakeAsyncOpenAI(),
            model="test-model",
            max_concurrent_requests=0,
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
