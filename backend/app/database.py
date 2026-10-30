import os
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

logger = logging.getLogger("agentic_bridge.database")

# Asynchronous PostgreSQL connection string using the asyncpg driver
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/agentic_bridge"
)

# Configure the asynchronous SQLAlchemy engine with robust connection pooling
async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Configured async session maker for FastAPI dependency injection
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Declarative base class for all SQLAlchemy ORM models
Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency yielding an asynchronous SQLAlchemy database session
    with automatic transaction commit, rollback on error, and cleanup.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()