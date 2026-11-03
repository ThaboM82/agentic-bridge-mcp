import asyncio
import logging
import sys

from sqlalchemy import text

from app.core.database import (
    async_engine,
    Base,
)
from app.models.workflow_model import WorkflowExecutionModel  # Ensures models are registered in metadata

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("agentic_bridge.init_db")

async def init_database() -> None:
    """
    Asynchronously initializes the PostgreSQL database by enabling the pgvector extension,
    verifying connection health, and creating all registered SQLAlchemy ORM tables.
    """
    logger.info("Starting database initialization sequence...")
    
    try:
        async with async_engine.begin() as conn:
            # 1. Test connection health
            logger.info("Testing database connection health...")
            await conn.execute(text("SELECT 1"))
            logger.info("Database connection verified successfully.")

            # 2. Enable pgvector extension
            logger.info("Enabling PostgreSQL 'vector' (pgvector) extension...")
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            logger.info("pgvector extension enabled successfully.")

            # 3. Create all tables bound to Base.metadata
            logger.info("Creating database tables from SQLAlchemy metadata...")
            await conn.run_sync(Base.metadata.create_all)
            logger.info("All database tables created successfully.")

    except Exception as e:
        logger.error(f"Critical error during database initialization: {str(e)}", exc_info=True)
        raise
    finally:
        await async_engine.dispose()
        logger.info("Database engine connection pool closed.")

if __name__ == "__main__":
    asyncio.run(init_database())
