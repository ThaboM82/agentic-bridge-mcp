"""
Application configuration for AgenticBridge.

This module centralizes environment-driven configuration using Pydantic
Settings. It validates required values at startup, applies safe defaults for
non-production environments, and never logs secrets or connection strings.

Security rules:
- Never log `SECRET_KEY`, database URLs, provider API keys, OAuth secrets, or
  other credentials.
- Use environment variables or a managed secret manager in production.
- Require explicit production-critical settings; do not silently fall back to
  insecure defaults.
- Keep configuration immutable after application startup.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


logger = logging.getLogger("agentic_bridge.core.config")

EnvironmentName = Literal[
    "local",
    "development",
    "test",
    "staging",
    "production",
]

LogLevelName = Literal[
    "CRITICAL",
    "ERROR",
    "WARNING",
    "INFO",
    "DEBUG",
]


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        frozen=True,
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------

    APP_NAME: str = "AgenticBridge"
    APP_ENV: EnvironmentName = "local"
    APP_VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    DEBUG: bool = False

    LOG_LEVEL: LogLevelName = "INFO"
    LOG_JSON_FORMAT: bool = True

    # -------------------------------------------------------------------------
    # Security
    # -------------------------------------------------------------------------

    SECRET_KEY: str = Field(
        default="",
        description=(
            "Cryptographic signing key. Required outside local/test "
            "environments."
        ),
    )

    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=5,
        le=1_440,
    )

    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        ge=1,
        le=30,
    )

    ALLOWED_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
    )

    TRUSTED_HOSTS: list[str] = Field(
        default_factory=lambda: ["*"],
    )

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------

    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://agentic:agentic@localhost:5432/agentic_bridge",
        description=(
            "Async SQLAlchemy database URL. Required in production."
        ),
    )

    DATABASE_POOL_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    DATABASE_MAX_OVERFLOW: int = Field(
        default=20,
        ge=0,
        le=200,
    )

    DATABASE_POOL_TIMEOUT_SECONDS: int = Field(
        default=30,
        ge=1,
        le=300,
    )

    DATABASE_POOL_RECYCLE_SECONDS: int = Field(
        default=1_800,
        ge=60,
        le=86_400,
    )

    DATABASE_ECHO: bool = False

    # -------------------------------------------------------------------------
    # Redis
    # -------------------------------------------------------------------------

    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description=(
            "Redis connection URL used for caching, rate limiting, and "
            "background job coordination."
        ),
    )

    REDIS_CACHE_TTL_SECONDS: int = Field(
        default=300,
        ge=1,
        le=86_400,
    )

    # -------------------------------------------------------------------------
    # CORS and rate limiting
    # -------------------------------------------------------------------------

    RATE_LIMIT_ENABLED: bool = True

    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=60,
        ge=1,
        le=10_000,
    )

    RATE_LIMIT_BURST: int = Field(
        default=20,
        ge=1,
        le=1_000,
    )

    # -------------------------------------------------------------------------
    # Workflow execution
    # -------------------------------------------------------------------------

    WORKFLOW_EXECUTION_TIMEOUT_SECONDS: int = Field(
        default=900,
        ge=30,
        le=7_200,
    )

    WORKFLOW_MAX_CONCURRENT_EXECUTIONS: int = Field(
        default=10,
        ge=1,
        le=1_000,
    )

    WORKFLOW_CHECKPOINT_BACKEND: Literal[
        "memory",
        "postgres",
    ] = "memory"

    WORKFLOW_CHECKPOINT_DATABASE_URL: str | None = None

    WORKFLOW_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
    )

    WORKFLOW_RETRY_BACKOFF_SECONDS: float = Field(
        default=2.0,
        ge=0.1,
        le=60.0,
    )

    # -------------------------------------------------------------------------
    # LLM providers
    # -------------------------------------------------------------------------

    DEFAULT_LLM_PROVIDER: Literal[
        "openai",
        "anthropic",
        "google",
        "azure_openai",
    ] = "openai"

    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4.1-mini"
    OPENAI_TIMEOUT_SECONDS: int = Field(
        default=60,
        ge=5,
        le=600,
    )

    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5"
    ANTHROPIC_TIMEOUT_SECONDS: int = Field(
        default=60,
        ge=5,
        le=600,
    )

    GOOGLE_API_KEY: str | None = None
    GOOGLE_MODEL: str = "gemini-2.5-pro"
    GOOGLE_TIMEOUT_SECONDS: int = Field(
        default=60,
        ge=5,
        le=600,
    )

    AZURE_OPENAI_API_KEY: str | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_API_VERSION: str = "2024-10-21"
    AZURE_OPENAI_DEPLOYMENT_NAME: str | None = None

    LLM_MAX_OUTPUT_TOKENS: int = Field(
        default=4_096,
        ge=256,
        le=200_000,
    )

    LLM_REQUEST_TIMEOUT_SECONDS: int = Field(
        default=90,
        ge=5,
        le=600,
    )

    # -------------------------------------------------------------------------
    # Embeddings and retrieval
    # -------------------------------------------------------------------------

    EMBEDDING_PROVIDER: Literal[
        "openai",
        "azure_openai",
        "google",
    ] = "openai"

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = Field(
        default=1_536,
        ge=64,
        le=10_000,
    )

    VECTOR_SEARCH_LIMIT: int = Field(
        default=5,
        ge=1,
        le=50,
    )

    VECTOR_SEARCH_SIMILARITY_THRESHOLD: float = Field(
        default=0.2,
        ge=-1.0,
        le=1.0,
    )

    # -------------------------------------------------------------------------
    # File uploads and documents
    # -------------------------------------------------------------------------

    MAX_UPLOAD_SIZE_BYTES: int = Field(
        default=25 * 1024 * 1024,
        ge=1,
        le=500 * 1024 * 1024,
    )

    ALLOWED_UPLOAD_EXTENSIONS: list[str] = Field(
        default_factory=lambda: [
            ".pdf",
            ".docx",
            ".txt",
            ".md",
            ".csv",
            ".json",
        ],
    )

    DOCUMENT_STORAGE_BACKEND: Literal[
        "local",
        "s3",
        "azure_blob",
        "gcs",
    ] = "local"

    LOCAL_UPLOAD_DIRECTORY: str = "./uploads"

    S3_BUCKET_NAME: str | None = None
    S3_REGION: str | None = None

    AZURE_STORAGE_ACCOUNT_NAME: str | None = None
    AZURE_STORAGE_CONTAINER_NAME: str | None = None

    GCS_BUCKET_NAME: str | None = None

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------

    OTEL_ENABLED: bool = False

    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None

    OTEL_SERVICE_NAME: str = "agentic-bridge"

    SENTRY_DSN: str | None = None

    PROMETHEUS_ENABLED: bool = True

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    @field_validator("ALLOWED_ORIGINS", "TRUSTED_HOSTS", mode="before")
    @classmethod
    def parse_comma_separated_list(
        cls,
        value: str | list[str] | None,
    ) -> list[str]:
        """Parse comma-separated environment values into lists."""
        if value is None or value == "":
            return []

        if isinstance(value, str):
            return [
                item.strip()
                for item in value.split(",")
                if item.strip()
            ]

        if isinstance(value, list):
            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        raise ValueError("Expected a comma-separated string or list.")

    @field_validator("ALLOWED_UPLOAD_EXTENSIONS", mode="before")
    @classmethod
    def normalize_upload_extensions(
        cls,
        value: str | list[str] | None,
    ) -> list[str]:
        """Normalize upload extensions to lowercase with leading dots."""
        if value is None or value == "":
            return []

        if isinstance(value, str):
            raw_extensions = [
                item.strip()
                for item in value.split(",")
                if item.strip()
            ]
        elif isinstance(value, list):
            raw_extensions = [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]
        else:
            raise ValueError(
                "Expected a comma-separated string or list of extensions."
            )

        normalized_extensions: list[str] = []

        for extension in raw_extensions:
            normalized_extension = extension.lower()

            if not normalized_extension.startswith("."):
                normalized_extension = f".{normalized_extension}"

            if normalized_extension not in normalized_extensions:
                normalized_extensions.append(normalized_extension)

        return normalized_extensions

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        """Validate the async database URL."""
        normalized_url = value.strip()

        if not normalized_url:
            raise ValueError("DATABASE_URL is required.")

        if not normalized_url.startswith(
            (
                "postgresql+asyncpg://",
                "postgresql://",
            )
        ):
            raise ValueError(
                "DATABASE_URL must use PostgreSQL with the asyncpg driver."
            )

        return normalized_url

    @field_validator("REDIS_URL")
    @classmethod
    def validate_redis_url(cls, value: str) -> str:
        """Validate the Redis connection URL."""
        normalized_url = value.strip()

        if not normalized_url:
            raise ValueError("REDIS_URL is required.")

        if not normalized_url.startswith(
            (
                "redis://",
                "rediss://",
            )
        ):
            raise ValueError(
                "REDIS_URL must use the redis:// or rediss:// scheme."
            )

        return normalized_url

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, value: str) -> str:
        """
        Validate the application signing key.

        A non-empty key is required outside local and test environments.
        """
        normalized_key = value.strip()

        return normalized_key

    @field_validator("WORKFLOW_CHECKPOINT_DATABASE_URL")
    @classmethod
    def validate_checkpoint_database_url(
        cls,
        value: str | None,
    ) -> str | None:
        """Validate the optional workflow checkpoint database URL."""
        if value is None:
            return None

        normalized_url = value.strip()

        if not normalized_url:
            return None

        if not normalized_url.startswith(
            (
                "postgresql+asyncpg://",
                "postgresql://",
            )
        ):
            raise ValueError(
                "WORKFLOW_CHECKPOINT_DATABASE_URL must use PostgreSQL."
            )

        return normalized_url

    # -------------------------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------------------------

    @property
    def is_production(self) -> bool:
        """Return whether the application is running in production."""
        return self.APP_ENV == "production"

    @property
    def is_local_or_test(self) -> bool:
        """Return whether the application is running locally or in tests."""
        return self.APP_ENV in {"local", "test"}

    @property
    def database_url_safe(self) -> str:
        """Return a redacted database URL safe for diagnostics."""
        return "[REDACTED_DATABASE_URL]"

    @property
    def redis_url_safe(self) -> str:
        """Return a redacted Redis URL safe for diagnostics."""
        return "[REDACTED_REDIS_URL]"

    @property
    def workflow_checkpoint_database_url_safe(self) -> str | None:
        """Return a redacted checkpoint database URL for diagnostics."""
        if self.WORKFLOW_CHECKPOINT_DATABASE_URL:
            return "[REDACTED_CHECKPOINT_DATABASE_URL]"

        return None

    def get_llm_api_key(
        self,
        provider: str,
    ) -> str | None:
        """
        Return the configured API key for an LLM provider.

        Args:
            provider: LLM provider name.

        Returns:
            The provider API key or `None`.

        Raises:
            ValueError: If the provider is unsupported.
        """
        provider_keys = {
            "openai": self.OPENAI_API_KEY,
            "anthropic": self.ANTHROPIC_API_KEY,
            "google": self.GOOGLE_API_KEY,
            "azure_openai": self.AZURE_OPENAI_API_KEY,
        }

        if provider not in provider_keys:
            raise ValueError(
                f"Unsupported LLM provider '{provider}'."
            )

        return provider_keys[provider]

    def get_safe_startup_diagnostics(self) -> dict[str, object]:
        """
        Return non-sensitive startup diagnostics.

        Never include credentials, database URLs, API keys, or provider secrets
        in the returned dictionary.
        """
        return {
            "app_name": self.APP_NAME,
            "app_env": self.APP_ENV,
            "app_version": self.APP_VERSION,
            "debug": self.DEBUG,
            "log_level": self.LOG_LEVEL,
            "database_configured": bool(self.DATABASE_URL),
            "redis_configured": bool(self.REDIS_URL),
            "workflow_checkpoint_backend": (
                self.WORKFLOW_CHECKPOINT_BACKEND
            ),
            "workflow_checkpoint_database_configured": bool(
                self.WORKFLOW_CHECKPOINT_DATABASE_URL
            ),
            "default_llm_provider": self.DEFAULT_LLM_PROVIDER,
            "embedding_provider": self.EMBEDDING_PROVIDER,
            "embedding_model": self.EMBEDDING_MODEL,
            "embedding_dimensions": self.EMBEDDING_DIMENSIONS,
            "document_storage_backend": self.DOCUMENT_STORAGE_BACKEND,
            "otel_enabled": self.OTEL_ENABLED,
            "prometheus_enabled": self.PROMETHEUS_ENABLED,
            "rate_limit_enabled": self.RATE_LIMIT_ENABLED,
        }

    def validate_production_configuration(self) -> None:
        """
        Validate production-critical configuration.

        Raises:
            ValueError: If a required production setting is missing or unsafe.
        """
        errors: list[str] = []

        if not self.SECRET_KEY:
            errors.append("SECRET_KEY is required in production.")

        if len(self.SECRET_KEY) < 32:
            errors.append(
                "SECRET_KEY must be at least 32 characters in production."
            )

        if self.DEBUG:
            errors.append("DEBUG must be disabled in production.")

        if "*" in self.TRUSTED_HOSTS:
            errors.append(
                "TRUSTED_HOSTS must not contain '*' in production."
            )

        if not self.WORKFLOW_CHECKPOINT_DATABASE_URL:
            errors.append(
                "WORKFLOW_CHECKPOINT_DATABASE_URL is required when using "
                "PostgreSQL workflow checkpointing."
            )

        if self.DEFAULT_LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            errors.append(
                "OPENAI_API_KEY is required when OpenAI is the default "
                "LLM provider."
            )

        if self.DEFAULT_LLM_PROVIDER == "anthropic" and (
            not self.ANTHROPIC_API_KEY
        ):
            errors.append(
                "ANTHROPIC_API_KEY is required when Anthropic is the default "
                "LLM provider."
            )

        if self.DEFAULT_LLM_PROVIDER == "google" and not self.GOOGLE_API_KEY:
            errors.append(
                "GOOGLE_API_KEY is required when Google is the default "
                "LLM provider."
            )

        if self.DEFAULT_LLM_PROVIDER == "azure_openai" and (
            not self.AZURE_OPENAI_API_KEY
            or not self.AZURE_OPENAI_ENDPOINT
            or not self.AZURE_OPENAI_DEPLOYMENT_NAME
        ):
            errors.append(
                "Azure OpenAI requires AZURE_OPENAI_API_KEY, "
                "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME."
            )

        if self.DOCUMENT_STORAGE_BACKEND == "s3" and (
            not self.S3_BUCKET_NAME or not self.S3_REGION
        ):
            errors.append(
                "S3 document storage requires S3_BUCKET_NAME and S3_REGION."
            )

        if self.DOCUMENT_STORAGE_BACKEND == "azure_blob" and (
            not self.AZURE_STORAGE_ACCOUNT_NAME
            or not self.AZURE_STORAGE_CONTAINER_NAME
        ):
            errors.append(
                "Azure Blob document storage requires "
                "AZURE_STORAGE_ACCOUNT_NAME and "
                "AZURE_STORAGE_CONTAINER_NAME."
            )

        if self.DOCUMENT_STORAGE_BACKEND == "gcs" and (
            not self.GCS_BUCKET_NAME
        ):
            errors.append(
                "GCS document storage requires GCS_BUCKET_NAME."
            )

        if errors:
            raise ValueError(
                "Invalid production configuration: " + "; ".join(errors)
            )


@lru_cache
def get_settings() -> Settings:
    """
    Return cached application settings.

    Returns:
        The validated application settings instance.
    """
    settings = Settings()

    if settings.is_production:
        settings.validate_production_configuration()

    logger.info(
        "application_settings_loaded",
        **settings.get_safe_startup_diagnostics(),
    )

    return settings


settings = get_settings()