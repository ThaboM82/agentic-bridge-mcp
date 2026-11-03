"""
Application configuration.

Values are loaded from environment variables and `.env` for local development.
Never commit real secrets, credentials, API keys, or database URLs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import (
    Field,
    SecretStr,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


Environment = Literal[
    "development",
    "test",
    "staging",
    "production",
]


LogLevel = Literal[
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]


class Settings(BaseSettings):
    """
    Typed application settings.

    All values can be overridden using environment variables or a local `.env`
    file. Secret values use `SecretStr` so they are not accidentally exposed in
    logs, repr output, or error messages.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------

    APP_NAME: str = "AgenticBridge MCP"

    APP_ENV: Environment = "development"

    APP_VERSION: str = "1.0.0"

    DEBUG: bool = False

    API_V1_PREFIX: str = "/api/v1"

    ENABLE_DOCS: bool = True

    ENABLE_METRICS: bool = True

    # -------------------------------------------------------------------------
    # Security
    # -------------------------------------------------------------------------

    SECRET_KEY: SecretStr = Field(
        default=SecretStr(
            "change-me-in-production",
        ),
        description=(
            "Secret used for signing tokens and "
            "sensitive values."
        ),
    )

    JWT_SECRET_KEY: SecretStr = Field(
        default=SecretStr(
            "local-development-secret-key-with-"
            "at-least-32-characters",
        ),
        description=(
            "Secret used to sign JWT access, "
            "refresh, and service tokens."
        ),
    )

    JWT_ALGORITHM: str = "HS256"

    JWT_ISSUER: str = "agentic-bridge-mcp"

    JWT_AUDIENCE: str = "agentic-bridge-mcp"

    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=1,
        le=1_440,
    )

    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=14,
        ge=1,
        le=365,
    )

    JWT_LEEWAY_SECONDS: int = Field(
        default=30,
        ge=0,
        le=300,
    )

    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=1,
        le=1_440,
    )

    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=30,
        ge=1,
        le=365,
    )

    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=5,
        le=1_440,
    )

    MAX_FAILED_LOGIN_ATTEMPTS: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    ACCOUNT_LOCKOUT_MINUTES: int = Field(
        default=15,
        ge=1,
        le=1_440,
    )

    REQUIRE_MFA: bool = False

    ALLOWED_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
        ],
    )

    ALLOWED_HOSTS: list[str] = Field(
        default_factory=lambda: ["*"],
    )

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------

    DATABASE_URL: str | None = None

    DB_ECHO: bool = False

    DB_POOL_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    DB_MAX_OVERFLOW: int = Field(
        default=20,
        ge=0,
        le=200,
    )

    DB_POOL_TIMEOUT_SECONDS: int = Field(
        default=30,
        ge=1,
        le=300,
    )

    DB_POOL_RECYCLE_SECONDS: int = Field(
        default=1_800,
        ge=60,
        le=86_400,
    )

    DB_POOL_PRE_PING: bool = True

    DB_STATEMENT_TIMEOUT_SECONDS: int = Field(
        default=30,
        ge=1,
        le=600,
    )

    # -------------------------------------------------------------------------
    # Redis / cache / queues
    # -------------------------------------------------------------------------

    REDIS_URL: str | None = None

    CACHE_ENABLED: bool = False

    CACHE_DEFAULT_TTL_SECONDS: int = Field(
        default=300,
        ge=1,
        le=86_400,
    )

    TASK_QUEUE_URL: str | None = None

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------

    LOG_LEVEL: LogLevel = "INFO"

    LOG_FORMAT_JSON: bool = True

    LOG_REQUESTS: bool = True

    SENTRY_DSN: str | None = None

    METRICS_ENABLED: bool = True

    # -------------------------------------------------------------------------
    # Rate limiting
    # -------------------------------------------------------------------------

    RATE_LIMIT_ENABLED: bool = True

    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=120,
        ge=1,
        le=100_000,
    )

    RATE_LIMIT_BURST: int = Field(
        default=20,
        ge=1,
        le=10_000,
    )

    # -------------------------------------------------------------------------
    # LLM providers
    # -------------------------------------------------------------------------

    OPENAI_API_KEY: SecretStr | None = None

    ANTHROPIC_API_KEY: SecretStr | None = None

    DEFAULT_LLM_PROVIDER: Literal[
        "openai",
        "anthropic",
    ] = "openai"

    DEFAULT_LLM_MODEL: str = "gpt-4o-mini"

    LLM_TIMEOUT_SECONDS: int = Field(
        default=60,
        ge=1,
        le=600,
    )

    LLM_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
    )

    LLM_MAX_TOKENS: int = Field(
        default=4_096,
        ge=1,
        le=1_000_000,
    )

    # -------------------------------------------------------------------------
    # Embeddings / vector search
    # -------------------------------------------------------------------------

    EMBEDDING_MODEL: str = (
        "text-embedding-3-small"
    )

    EMBEDDING_DIMENSIONS: int = Field(
        default=1_536,
        ge=1,
        le=10_000,
    )

    EMBEDDING_BATCH_SIZE: int = Field(
        default=100,
        ge=1,
        le=1_000,
    )

    VECTOR_SEARCH_TOP_K: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    VECTOR_SEARCH_SCORE_THRESHOLD: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------

    INGESTION_BATCH_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    MAX_UPLOAD_SIZE_MB: int = Field(
        default=20,
        ge=1,
        le=200,
    )

    DOCUMENT_CHUNK_SIZE: int = Field(
        default=1_000,
        ge=100,
        le=10_000,
    )

    DOCUMENT_CHUNK_OVERLAP: int = Field(
        default=150,
        ge=0,
        le=2_000,
    )

    # -------------------------------------------------------------------------
    # Webhooks
    # -------------------------------------------------------------------------

    WEBHOOK_TIMEOUT_SECONDS: int = Field(
        default=10,
        ge=1,
        le=120,
    )

    WEBHOOK_MAX_ATTEMPTS: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    WEBHOOK_RETRY_BASE_DELAY_SECONDS: int = Field(
        default=5,
        ge=1,
        le=3_600,
    )

    # -------------------------------------------------------------------------
    # Retention
    # -------------------------------------------------------------------------

    RETENTION_ENABLED: bool = True

    RETENTION_BATCH_SIZE: int = Field(
        default=1_000,
        ge=1,
        le=100_000,
    )

    RETENTION_MAX_RUNTIME_MINUTES: int = Field(
        default=60,
        ge=1,
        le=1_440,
    )

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------

    @field_validator(
        "ALLOWED_ORIGINS",
        "ALLOWED_HOSTS",
        mode="before",
    )
    @classmethod
    def parse_comma_separated_list(
        cls,
        value: object,
    ) -> list[str] | object:
        """
        Allow comma-separated values in environment variables.
        """
        if isinstance(value, str):
            return [
                item.strip()
                for item in value.split(",")
                if item.strip()
            ]

        return value

    @field_validator(
        "DATABASE_URL",
    )
    @classmethod
    def validate_database_url(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Validate that the configured database URL is PostgreSQL-compatible.
        """
        if value is None:
            return value

        url = value.strip()

        allowed_prefixes = (
            "postgres://",
            "postgresql://",
            "postgresql+asyncpg://",
        )

        if not url.startswith(
            allowed_prefixes,
        ):
            raise ValueError(
                "DATABASE_URL must be a PostgreSQL URL "
                "compatible with asyncpg.",
            )

        return url

    # -------------------------------------------------------------------------
    # Compatibility properties
    # -------------------------------------------------------------------------

    @property
    def environment(self) -> str:
        """
        Lowercase environment name used by application middleware.
        """
        return self.APP_ENV

    @property
    def cors_origins(self) -> list[str]:
        """
        CORS origins used by FastAPI CORSMiddleware.
        """
        return self.ALLOWED_ORIGINS

    @property
    def log_level(self) -> str:
        """
        Logging level used by application bootstrap.
        """
        return self.LOG_LEVEL

    @property
    def is_production(self) -> bool:
        """Return whether the application runs in production."""
        return self.APP_ENV == "production"

    @property
    def is_staging(self) -> bool:
        """Return whether the application runs in staging."""
        return self.APP_ENV == "staging"

    @property
    def is_development(self) -> bool:
        """Return whether the application runs in development."""
        return self.APP_ENV == "development"

    @property
    def is_test(self) -> bool:
        """Return whether the application runs in test mode."""
        return self.APP_ENV == "test"

    @property
    def sqlalchemy_database_uri(self) -> str:
        """
        Return a PostgreSQL URL compatible with SQLAlchemy asyncpg.
        """
        if not self.DATABASE_URL:
            raise ValueError(
                "DATABASE_URL environment variable "
                "is not set in .env",
            )

        url = self.DATABASE_URL.strip()

        if url.startswith(
            "postgres://",
        ):
            return url.replace(
                "postgres://",
                "postgresql+asyncpg://",
                1,
            )

        if (
            url.startswith(
                "postgresql://",
            )
            and "+asyncpg" not in url
        ):
            return url.replace(
                "postgresql://",
                "postgresql+asyncpg://",
                1,
            )

        if url.startswith(
            "postgresql+asyncpg://",
        ):
            return url

        raise ValueError(
            "DATABASE_URL must be a PostgreSQL URL "
            "compatible with asyncpg.",
        )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


settings = get_settings()


APP_VERSION: str = settings.APP_VERSION


__all__ = [
    "APP_VERSION",
    "Environment",
    "LogLevel",
    "Settings",
    "get_settings",
    "settings",
]