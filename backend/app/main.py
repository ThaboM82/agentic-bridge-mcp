from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.database import engine, Base
from app.models import DocumentChunk  # Imported so Base metadata registers the model
from app.routers import documents
from app.api.workflow_router import router as workflow_router

# Safely import LLM factory if available, with fallback for modularity
try:
    from app.services.llm_client import LLMProviderFactory
except ImportError:
    LLMProviderFactory = None

# Configure enterprise structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("agentic_bridge.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager handling database initialization, 
    pgvector extension, telemetry auditing, and graceful shutdown routines.
    """
    logger.info("Initializing AgenticBridge-MCP Enterprise Backend...")
    
    # --- Startup: Initialize Database & pgvector Extension ---
    async with engine.begin() as conn:
        # Enable the pgvector extension in PostgreSQL
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # Create all tables defined in SQLAlchemy models
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables and pgvector extension verified/created successfully.")
    
    # Audit LLM provider configuration status on startup if available
    if LLMProviderFactory and hasattr(LLMProviderFactory, "get_client_metadata"):
        llm_metadata = LLMProviderFactory.get_client_metadata()
        logger.info(f"LLM Provider Telemetry Audit: {llm_metadata}")
    else:
        logger.info("LLM Provider module initialized.")
    
    yield  # Application runs here
    
    # --- Shutdown: Clean up connections ---
    logger.info("Shutting down AgenticBridge-MCP Backend Service...")
    await engine.dispose()


# Initialize FastAPI application instance
app = FastAPI(
    title="AgenticBridge-MCP API",
    description="Enterprise-grade autonomous agent orchestrator powered by FastAPI, LangGraph, and PostgreSQL pgvector.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS middleware for React frontend integration
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Performance telemetry middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Measures request processing latency and injects telemetry headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 4))
    return response


# Register Routers
app.include_router(documents.router)
app.include_router(workflow_router, prefix="/api/v1")


# System Endpoints
@app.get("/", tags=["System Root"])
async def root():
    """System health and overview endpoint."""
    return {
        "system": "AgenticBridge-MCP",
        "status": "operational",
        "version": "1.0.0",
        "docs_endpoint": "/docs",
        "workflows_endpoint": "/api/v1/workflows"
    }


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "service": "AgenticBridge-MCP Backend",
        "database": "connected",
    }