from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.schemas.documents import DocumentCreate


class IngestionJobStatus(StrEnum):
    """Lifecycle states for an asynchronous ingestion job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = "partially_completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IngestionSourceType(StrEnum):
    """Supported ingestion sources."""

    MANUAL = "manual"
    GOOGLE_DOCS = "google_docs"
    GOOGLE_DRIVE = "google_drive"
    FIGMA = "figma"
    INTERNAL_CMS = "internal_cms"
    WORDPRESS = "wordpress"
    FILE_UPLOAD = "file_upload"


class DocumentMetadata(BaseModel):
    """
    Optional metadata attached to an ingested document.

    Use this for tenant isolation, provenance, access control, and audit data.
    """

    model_config = ConfigDict(extra="forbid")

    source_type: IngestionSourceType = IngestionSourceType.MANUAL
    source_url: HttpUrl | None = Field(
        default=None,
        description="Canonical source URL, if available.",
    )
    source_id: str | None = Field(
        default=None,
        max_length=255,
        description="Provider-specific source identifier, such as a Google Drive file ID.",
    )
    tenant_id: str | None = Field(
        default=None,
        max_length=64,
        description="Tenant or workspace identifier.",
    )
    created_by: str | None = Field(
        default=None,
        max_length=255,
        description="User or service identity that initiated ingestion.",
    )
    tags: list[str] = Field(
        default_factory=list,
        max_length=25,
        description="User- or system-defined classification tags.",
    )
    custom_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Integration-specific metadata.",
    )


class IngestionDocumentCreate(DocumentCreate):
    """
    Document payload for batch ingestion.

    Inherits title and content validation from DocumentCreate and adds
    provenance and access-control metadata.
    """

    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata,
        description="Provenance and governance metadata for this document.",
    )


class BatchIngestionRequest(BaseModel):
    """Request schema for asynchronous batch document ingestion."""

    documents: list[IngestionDocumentCreate] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Documents to embed and store.",
    )

    batch_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description=(
            "Number of documents to send to the embedding provider per request."
        ),
    )

    max_concurrent_batches: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Maximum number of embedding batches processed concurrently.",
    )

    continue_on_error: bool = Field(
        default=True,
        description=(
            "If true, continue processing remaining batches after a batch fails."
        ),
    )

    idempotency_key: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        description=(
            "Optional client-supplied key to prevent duplicate ingestion jobs."
        ),
    )

    callback_url: HttpUrl | None = Field(
        default=None,
        description=(
            "Optional HTTPS webhook called when the ingestion job reaches a "
            "terminal state."
        ),
    )


class IngestionBatchError(BaseModel):
    """Details for a failed ingestion batch."""

    batch_number: int = Field(
        ...,
        ge=1,
        description="One-based batch number that failed.",
    )

    document_indexes: list[int] = Field(
        default_factory=list,
        description="Zero-based indexes of documents affected in the original request.",
    )

    error_code: str = Field(
        default="embedding_or_persistence_error",
        description="Stable machine-readable error code.",
    )

    message: str = Field(
        ...,
        description="Safe, human-readable error description.",
    )

    retryable: bool = Field(
        default=True,
        description="Whether the failure may succeed on retry.",
    )


class IngestionJobResult(BaseModel):
    """Final or in-progress result details for an ingestion job."""

    created_document_ids: list[int] = Field(
        default_factory=list,
        description="Database IDs successfully created by the job.",
    )

    created_document_count: int = Field(
        default=0,
        ge=0,
        description="Number of documents successfully ingested.",
    )

    failed_document_count: int = Field(
        default=0,
        ge=0,
        description="Number of documents that failed ingestion.",
    )

    skipped_document_count: int = Field(
        default=0,
        ge=0,
        description="Number of documents skipped due to deduplication or policy.",
    )

    errors: list[IngestionBatchError] = Field(
        default_factory=list,
        description="Batch-level failures encountered during processing.",
    )

    embedding_model: str | None = Field(
        default=None,
        description="Embedding model used for this job.",
    )

    completed_at: datetime | None = Field(
        default=None,
        description="Time the job reached a terminal state.",
    )


class IngestionJobResponse(BaseModel):
    """
    Public contract for ingestion job progress.

    This model is safe to expose through polling and Server-Sent Events.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Unique ingestion job identifier.",
    )

    status: IngestionJobStatus = Field(
        ...,
        description="Current lifecycle state of the ingestion job.",
    )

    total_documents: int = Field(
        ...,
        ge=0,
        description="Total number of documents submitted.",
    )

    processed_documents: int = Field(
        default=0,
        ge=0,
        description="Number of documents successfully processed.",
    )

    failed_documents: int = Field(
        default=0,
        ge=0,
        description="Number of documents that failed processing.",
    )

    skipped_documents: int = Field(
        default=0,
        ge=0,
        description="Number of documents skipped by deduplication or policy.",
    )

    current_batch: int = Field(
        default=0,
        ge=0,
        description="Batch currently being processed.",
    )

    total_batches: int = Field(
        ...,
        ge=0,
        description="Total number of embedding batches in the job.",
    )

    progress_percent: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Completion percentage based on processed documents.",
    )

    error_message: str | None = Field(
        default=None,
        description="Top-level failure message, if the job failed.",
    )

    result: IngestionJobResult | None = Field(
        default=None,
        description="Detailed ingestion outcome, populated during and after processing.",
    )

    created_at: datetime = Field(
        ...,
        description="Time the ingestion job was created.",
    )

    updated_at: datetime = Field(
        ...,
        description="Time the ingestion job was last updated.",
    )

    @property
    def is_terminal(self) -> bool:
        """Whether the job has reached a final state."""
        return self.status in {
            IngestionJobStatus.COMPLETED,
            IngestionJobStatus.PARTIALLY_COMPLETED,
            IngestionJobStatus.FAILED,
            IngestionJobStatus.CANCELLED,
        }


class IngestionJobCreatedResponse(BaseModel):
    """Response returned immediately after queuing an ingestion job."""

    job_id: str = Field(
        ...,
        description="Unique identifier for the created ingestion job.",
    )

    status: Literal[IngestionJobStatus.QUEUED] = IngestionJobStatus.QUEUED

    total_documents: int = Field(
        ...,
        ge=1,
        description="Number of documents accepted for ingestion.",
    )

    total_batches: int = Field(
        ...,
        ge=1,
        description="Number of embedding batches planned for this job.",
    )

    progress_percent: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    status_endpoint: str = Field(
        ...,
        description="URL for polling ingestion progress.",
    )

    stream_endpoint: str = Field(
        ...,
        description="URL for streaming ingestion progress via SSE.",
    )


class IngestionProgressEvent(BaseModel):
    """Compact event emitted through the ingestion SSE stream."""

    event: Literal[
        "connected",
        "progress",
        "completed",
        "partially_completed",
        "failed",
        "error",
    ]

    job_id: str

    status: IngestionJobStatus | None = None

    processed_documents: int | None = Field(
        default=None,
        ge=0,
    )

    failed_documents: int | None = Field(
        default=None,
        ge=0,
    )

    progress_percent: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    current_batch: int | None = Field(
        default=None,
        ge=0,
    )

    total_batches: int | None = Field(
        default=None,
        ge=0,
    )

    message: str | None = None

    result: IngestionJobResult | None = None


class IngestionJobListResponse(BaseModel):
    """Paginated list of ingestion jobs."""

    jobs: list[IngestionJobResponse]
    total_jobs: int = Field(
        ...,
        ge=0,
    )
    skip: int = Field(
        ...,
        ge=0,
    )
    limit: int = Field(
        ...,
        ge=1,
    )


class IngestionJobCancelResponse(BaseModel):
    """Response for a cancellation request."""

    job_id: str
    status: IngestionJobStatus
    cancelled: bool
    message: str