import asyncio
import logging
import random
import time
from collections.abc import Sequence
from typing import Any

from openai import AsyncOpenAI

from app.config import settings


logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Base error for embedding generation failures."""


class EmbeddingValidationError(EmbeddingError):
    """Raised when embedding input is invalid."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when the embedding provider fails after retries."""


class EmbeddingRateLimitError(EmbeddingProviderError):
    """Raised when the provider continues to rate-limit requests."""


class EmbeddingService:
    """
    Generates embeddings through a provider abstraction.

    Current provider: OpenAI.

    Production features:
    - Single and batch embedding generation.
    - Configurable batch size and bounded concurrency.
    - Exponential backoff with jitter.
    - Retry-After support for rate limits.
    - Request timeout and retry budget.
    - Structured logging and optional metrics hooks.
    """

    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
        max_retries: int = 4,
        initial_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 30.0,
        request_timeout_seconds: float = 30.0,
        max_batch_size: int = 100,
        max_concurrent_requests: int = 4,
        metrics_hook: Any = None,
    ) -> None:
        self.model = model or settings.embedding_model
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.max_batch_size = max_batch_size
        self.metrics_hook = metrics_hook

        self._semaphore = asyncio.Semaphore(max_concurrent_requests)

        self.client = client or AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=request_timeout_seconds,
            max_retries=0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate one embedding vector."""
        embeddings = await self.generate_embeddings([text])

        if not embeddings:
            raise EmbeddingProviderError(
                "Embedding provider returned no embeddings."
            )

        return embeddings[0]

    async def generate_embeddings(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for many texts.

        Inputs are normalized, validated, split into provider-safe batches,
        and processed with bounded concurrency. The returned embeddings remain
        in the same order as the input texts.
        """
        started_at = time.perf_counter()

        normalized_texts = self._normalize_inputs(texts)
        batches = self._create_batches(normalized_texts)

        results: list[list[float] | None] = [None] * len(normalized_texts)

        async def process_batch(batch_index: int, batch: list[str]) -> None:
            embeddings = await self._generate_batch_with_retry(batch)

            start_index = batch_index * self.max_batch_size

            for offset, embedding in enumerate(embeddings):
                results[start_index + offset] = embedding

        await asyncio.gather(
            *(
                process_batch(batch_index, batch)
                for batch_index, batch in enumerate(batches)
            )
        )

        if any(embedding is None for embedding in results):
            raise EmbeddingProviderError(
                "Embedding provider returned an incomplete result set."
            )

        duration = time.perf_counter() - started_at

        logger.info(
            "Embedding generation completed texts=%s batches=%s model=%s duration_seconds=%.3f",
            len(normalized_texts),
            len(batches),
            self.model,
            duration,
        )

        self._record_metric(
            "embedding_generation_completed",
            {
                "text_count": len(normalized_texts),
                "batch_count": len(batches),
                "model": self.model,
                "duration_seconds": duration,
            },
        )

        return [embedding for embedding in results if embedding is not None]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_inputs(texts: Sequence[str]) -> list[str]:
        if not texts:
            raise EmbeddingValidationError("At least one text is required.")

        normalized: list[str] = []

        for index, text in enumerate(texts):
            if text is None:
                raise EmbeddingValidationError(
                    f"Text at index {index} cannot be None."
                )

            if not isinstance(text, str):
                raise EmbeddingValidationError(
                    f"Text at index {index} must be a string."
                )

            cleaned = " ".join(text.split())

            if not cleaned:
                raise EmbeddingValidationError(
                    f"Text at index {index} cannot be empty."
                )

            normalized.append(cleaned)

        return normalized

    def _create_batches(
        self,
        texts: list[str],
    ) -> list[list[str]]:
        if self.max_batch_size <= 0:
            raise EmbeddingValidationError(
                "max_batch_size must be greater than zero."
            )

        return [
            texts[start : start + self.max_batch_size]
            for start in range(0, len(texts), self.max_batch_size)
        ]

    async def _generate_batch_with_retry(
        self,
        batch: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for one batch with retry and backoff.

        SDK-level retries are disabled to prevent nested retry loops; this
        service owns the retry policy.
        """
        last_error: Exception | None = None
        retry_after: float | None = None

        for attempt in range(1, self.max_retries + 1):
            async with self._semaphore:
                try:
                    started_at = time.perf_counter()

                    response = await self.client.embeddings.create(
                        input=batch,
                        model=self.model,
                    )

                    duration = time.perf_counter() - started_at

                    embeddings = [
                        item.embedding for item in response.data
                    ]

                    if len(embeddings) != len(batch):
                        raise EmbeddingProviderError(
                            "Provider returned a mismatched number of embeddings."
                        )

                    self._record_metric(
                        "embedding_request_succeeded",
                        {
                            "model": self.model,
                            "batch_size": len(batch),
                            "attempt": attempt,
                            "duration_seconds": duration,
                        },
                    )

                    return embeddings

                except Exception as exc:
                    last_error = exc

                    retry_after = self._extract_retry_after(exc)

                    is_rate_limit = self._is_rate_limit_error(exc)
                    is_retryable = self._is_retryable_error(exc)

                    self._record_metric(
                        "embedding_request_failed",
                        {
                            "model": self.model,
                            "batch_size": len(batch),
                            "attempt": attempt,
                            "rate_limited": is_rate_limit,
                        },
                    )

                    if attempt >= self.max_retries:
                        break

                    if not is_retryable:
                        break

                    delay = self._calculate_backoff(
                        attempt=attempt,
                        retry_after=retry_after,
                    )

                    logger.warning(
                        "Embedding request failed; retrying attempt=%s/%s delay_seconds=%.2f model=%s batch_size=%s error=%s",
                        attempt,
                        self.max_retries,
                        delay,
                        self.model,
                        len(batch),
                        exc,
                    )

                    await asyncio.sleep(delay)

        logger.exception(
            "Embedding generation failed after retries model=%s batch_size=%s",
            self.model,
            len(batch),
        )

        if self._is_rate_limit_error(last_error):
            raise EmbeddingRateLimitError(
                "Embedding provider rate limit exceeded."
            ) from last_error

        raise EmbeddingProviderError(
            "Unable to generate embeddings after retries."
        ) from last_error

    @staticmethod
    def _is_rate_limit_error(error: Exception | None) -> bool:
        if error is None:
            return False

        status = getattr(error, "status_code", None)
        return status == 429

    @staticmethod
    def _is_retryable_error(error: Exception | None) -> bool:
        if error is None:
            return False

        status = getattr(error, "status_code", None)

        # Retry transient provider/network failures.
        # Do not retry validation, authentication, permission, or billing errors.
        return status is None or status in {408, 409, 425, 429} or status >= 500

    @staticmethod
    def _extract_retry_after(error: Exception | None) -> float | None:
        if error is None:
            return None

        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)

        if not headers:
            return None

        raw_value = headers.get("Retry-After") or headers.get("retry-after")

        if raw_value is None:
            return None

        try:
            return max(0.0, float(raw_value))
        except (TypeError, ValueError):
            return None

    def _calculate_backoff(
        self,
        attempt: int,
        retry_after: float | None,
    ) -> float:
        """
        Calculate the next retry delay.

        Honors Retry-After when present, then applies exponential backoff
        with jitter to avoid synchronized retries.
        """
        if retry_after is not None:
            base_delay = retry_after
        else:
            base_delay = min(
                self.max_backoff_seconds,
                self.initial_backoff_seconds * (2 ** (attempt - 1)),
            )

        jitter = random.uniform(0.5, 1.5)

        return min(
            self.max_backoff_seconds,
            base_delay * jitter,
        )

    def _record_metric(
        self,
        event_name: str,
        attributes: dict[str, Any],
    ) -> None:
        """
        Optional observability integration.

        Connect this to Prometheus, OpenTelemetry, Datadog, or your internal
        metrics pipeline.
        """
        if self.metrics_hook:
            try:
                self.metrics_hook(event_name, attributes)
            except Exception:
                logger.exception(
                    "Metrics hook failed for event=%s", event_name
                )


embedding_service = EmbeddingService()