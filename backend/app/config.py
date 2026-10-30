import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Enterprise application settings validated via Pydantic BaseSettings.
    Automatically loads variables from environment or .env file.
    """
    APP_NAME: str = "AgenticBridge-MCP"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG_MODE: bool = False
    
    # Database Configuration (PostgreSQL + pgvector)
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "agentic_bridge"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: Optional[str] = None

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        """Constructs asynchronous PostgreSQL connection URI."""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # LLM Provider Configuration
    DEFAULT_LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL_NAME: str = "gpt-4o"
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL_NAME: str = "claude-3-5-sonnet-20241022"
    AWS_REGION: str = "us-east-1"
    GOOGLE_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()