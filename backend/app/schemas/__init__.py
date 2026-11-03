"""
Public schema exports for AgenticBridge-MCP.

Prefer explicit submodule imports in application code:

    from app.schemas.documents import DocumentCreate
    from app.schemas.search import SearchQuery
    from app.schemas.ingestion import BatchIngestionRequest

These re-exports exist for convenient public-package access:

    from app.schemas import DocumentCreate, SearchQuery
"""

from app.schemas.documents import (
    DocumentCreate,
    DocumentResponse,
)

from app.schemas.search import (
    SearchQuery,
    SearchResponse,
    SearchResultItem,
)

from app.schemas.ingestion import (
    BatchIngestionRequest,
    IngestionBatchError,
    IngestionDocumentCreate,
    IngestionJobCancelResponse,
    IngestionJobCreatedResponse,
    IngestionJobListResponse,
    IngestionJobResponse,
    IngestionJobResult,
    IngestionJobStatus,
    IngestionProgressEvent,
    IngestionSourceType,
)

from app.schemas.workflow_schema import (
    WorkflowRequestSchema,
    WorkflowResponseSchema,
)


__all__ = [
    # Documents
    "DocumentCreate",
    "DocumentResponse",
    # Search
    "SearchQuery",
    "SearchResponse",
    "SearchResultItem",
    # Ingestion documents
    "IngestionDocumentCreate",
    "IngestionSourceType",
    # Batch ingestion requests
    "BatchIngestionRequest",
    # Ingestion jobs
    "IngestionJobStatus",
    "IngestionJobResponse",
    "IngestionJobCreatedResponse",
    "IngestionJobResult",
    "IngestionJobListResponse",
    "IngestionJobCancelResponse",
    "IngestionBatchError",
    "IngestionProgressEvent",
    # Workflow
    "WorkflowRequestSchema",
    "WorkflowResponseSchema",
]