"""
Shared application enums.

Use these enums for workflow states, ingestion lifecycle, document types,
provider names, approval decisions, interrupt types, recovery actions, and API
error categories. Keeping values in one module prevents inconsistent string
literals across services, models, workflow nodes, and API responses.

Design rules:
- Prefer `StrEnum` so enum values serialize naturally to JSON and databases.
- Keep enum values stable; renaming a value is a breaking API and persistence
  change.
- Add behavior only when it is broadly useful and side-effect free.
- Never store secrets, credentials, raw provider payloads, or customer content
  as enum values.
"""

from __future__ import annotations

from enum import StrEnum, unique


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


@unique
class Environment(StrEnum):
    """Deployment environments."""

    LOCAL = "local"
    DEVELOPMENT = "development"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"

    @property
    def is_production(self) -> bool:
        """Whether the environment is production."""
        return (
            self is Environment.PRODUCTION
        )

    @property
    def is_development(self) -> bool:
        """Whether the environment is local development."""
        return self in {
            Environment.LOCAL,
            Environment.DEVELOPMENT,
        }

    @property
    def is_test(self) -> bool:
        """Whether the environment is used for automated tests."""
        return self is Environment.TEST

    @property
    def requires_strict_security(
        self,
    ) -> bool:
        """Whether the environment requires production-grade safeguards."""
        return self in {
            Environment.STAGING,
            Environment.PRODUCTION,
        }


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------


@unique
class JobStatus(StrEnum):
    """Lifecycle status for asynchronous ingestion and processing jobs."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIALLY_COMPLETED = (
        "partially_completed"
    )
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def is_terminal(self) -> bool:
        """Whether the job has reached a final state."""
        return self in {
            JobStatus.COMPLETED,
            JobStatus.PARTIALLY_COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMED_OUT,
        }

    @property
    def is_successful(self) -> bool:
        """Whether the job completed without failure."""
        return self in {
            JobStatus.COMPLETED,
            JobStatus.PARTIALLY_COMPLETED,
        }

    @property
    def is_retryable(self) -> bool:
        """Whether the job may be retried after a failure."""
        return self in {
            JobStatus.FAILED,
            JobStatus.TIMED_OUT,
        }

    @property
    def is_active(self) -> bool:
        """Whether the job is queued or actively processing."""
        return self in {
            JobStatus.QUEUED,
            JobStatus.PROCESSING,
        }

    @classmethod
    def active_statuses(
        cls,
    ) -> tuple["JobStatus", ...]:
        """Return statuses representing work still in progress."""
        return (
            cls.QUEUED,
            cls.PROCESSING,
        )

    @classmethod
    def terminal_statuses(
        cls,
    ) -> tuple["JobStatus", ...]:
        """Return all final statuses."""
        return (
            cls.COMPLETED,
            cls.PARTIALLY_COMPLETED,
            cls.FAILED,
            cls.CANCELLED,
            cls.TIMED_OUT,
        )


# ---------------------------------------------------------------------------
# Workflow lifecycle
# ---------------------------------------------------------------------------


@unique
class WorkflowStep(StrEnum):
    """Ordered stages in the AgenticBridge generation workflow."""

    INITIALIZED = "initialized"
    RETRIEVAL = "retrieval"
    INGESTION = "ingestion"
    REQUIREMENTS = "requirements"
    DESIGN = "design"
    CODE_GENERATION = "code_generation"
    VALIDATION = "validation"
    REVIEW = "review"
    PUBLISHING = "publishing"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    ERROR = "error"

    @classmethod
    def ordered_steps(
        cls,
    ) -> tuple["WorkflowStep", ...]:
        """Return the normal forward execution order."""
        return (
            cls.INITIALIZED,
            cls.RETRIEVAL,
            cls.INGESTION,
            cls.REQUIREMENTS,
            cls.DESIGN,
            cls.CODE_GENERATION,
            cls.VALIDATION,
            cls.REVIEW,
            cls.PUBLISHING,
            cls.COMPLETED,
        )

    @classmethod
    def retryable_steps(
        cls,
    ) -> tuple["WorkflowStep", ...]:
        """Return workflow steps that may be retried after recovery."""
        return (
            cls.RETRIEVAL,
            cls.INGESTION,
            cls.REQUIREMENTS,
            cls.DESIGN,
            cls.CODE_GENERATION,
            cls.VALIDATION,
            cls.REVIEW,
            cls.PUBLISHING,
        )

    @property
    def is_terminal(self) -> bool:
        """Whether the workflow has reached a final state."""
        return self in {
            WorkflowStep.COMPLETED,
            WorkflowStep.ERROR,
            WorkflowStep.CANCELLED,
        }

    @property
    def is_failed(self) -> bool:
        """Whether the workflow entered an error state."""
        return self is WorkflowStep.ERROR

    @property
    def is_cancelled(self) -> bool:
        """Whether the workflow was cancelled."""
        return (
            self is WorkflowStep.CANCELLED
        )

    @property
    def is_paused(self) -> bool:
        """Whether the workflow is paused."""
        return self is WorkflowStep.PAUSED

    @property
    def requires_human_review(
        self,
    ) -> bool:
        """Whether this step normally requires human review."""
        return self is WorkflowStep.REVIEW


# ---------------------------------------------------------------------------
# Workflow control
# ---------------------------------------------------------------------------


@unique
class WorkflowControlAction(StrEnum):
    """Operator or user actions that affect workflow execution."""

    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    RETRY = "retry"
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    REJECT = "reject"


@unique
class WorkflowTrigger(StrEnum):
    """How a workflow execution was started."""

    API = "api"
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    WEBHOOK = "webhook"
    INTEGRATION = "integration"
    RETRY = "retry"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Document lifecycle
# ---------------------------------------------------------------------------


@unique
class DocumentStatus(StrEnum):
    """Lifecycle status for a source document."""

    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"
    DELETED = "deleted"

    @property
    def is_terminal(self) -> bool:
        """Whether the document has reached a final state."""
        return self in {
            DocumentStatus.READY,
            DocumentStatus.FAILED,
            DocumentStatus.ARCHIVED,
            DocumentStatus.DELETED,
        }


@unique
class DocumentType(StrEnum):
    """Supported logical document categories."""

    REQUIREMENTS = "requirements"
    DESIGN = "design"
    SPECIFICATION = "specification"
    CODE = "code"
    KNOWLEDGE_BASE = "knowledge_base"
    POLICY = "policy"
    CONTRACT = "contract"
    INVOICE = "invoice"
    REPORT = "report"
    OTHER = "other"


@unique
class IngestionSourceType(StrEnum):
    """Origin of ingested content."""

    UPLOAD = "upload"
    URL = "url"
    FIGMA = "figma"
    WORDPRESS = "wordpress"
    CMS = "cms"
    GOOGLE_DRIVE = "google_drive"
    SHAREPOINT = "sharepoint"
    S3 = "s3"
    API = "api"
    WORKFLOW = "workflow"


@unique
class IngestionItemStatus(StrEnum):
    """Status of an individual item in an ingestion job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Embeddings and retrieval
# ---------------------------------------------------------------------------


@unique
class EmbeddingStatus(StrEnum):
    """Lifecycle status for vector embedding generation."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    STALE = "stale"

    @property
    def is_searchable(self) -> bool:
        """Whether an embedding in this state can be retrieved."""
        return (
            self is EmbeddingStatus.READY
        )


@unique
class RetrievalMode(StrEnum):
    """Supported retrieval strategies."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@unique
class RelevanceLabel(StrEnum):
    """Human relevance label for retrieval feedback."""

    HIGHLY_RELEVANT = "highly_relevant"
    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = (
        "partially_relevant"
    )
    NOT_RELEVANT = "not_relevant"


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@unique
class LLMProvider(StrEnum):
    """Supported large-language-model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"


@unique
class EmbeddingProvider(StrEnum):
    """Supported embedding providers."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    BEDROCK = "bedrock"
    LOCAL = "local"


@unique
class ExternalIntegrationProvider(StrEnum):
    """Supported external system integrations."""

    FIGMA = "figma"
    WORDPRESS = "wordpress"
    GITHUB = "github"
    GOOGLE_DRIVE = "google_drive"
    SHAREPOINT = "sharepoint"
    S3 = "s3"
    SLACK = "slack"
    SALESFORCE = "salesforce"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------


@unique
class ApprovalStatus(StrEnum):
    """Lifecycle status for an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        """Whether the approval has reached a final state."""
        return self in {
            ApprovalStatus.APPROVED,
            ApprovalStatus.REJECTED,
            ApprovalStatus.CANCELLED,
            ApprovalStatus.EXPIRED,
        }


@unique
class ApprovalDecision(StrEnum):
    """Decision made by a reviewer or approver."""

    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"
    REJECT = "reject"


@unique
class ApprovalPriority(StrEnum):
    """Priority assigned to an approval request."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


@unique
class ArtifactType(StrEnum):
    """Types of generated or uploaded artifacts."""

    DESIGN = "design"
    CODE = "code"
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    DATA = "data"
    REPORT = "report"
    OTHER = "other"


@unique
class ArtifactStatus(StrEnum):
    """Lifecycle status for an artifact."""

    DRAFT = "draft"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    ARCHIVED = "archived"
    DELETED = "deleted"


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------


@unique
class NotificationStatus(StrEnum):
    """Lifecycle status for an in-app notification."""

    UNREAD = "unread"
    READ = "read"
    ARCHIVED = "archived"


@unique
class NotificationChannel(StrEnum):
    """Supported notification delivery channels."""

    EMAIL = "email"
    IN_APP = "in_app"
    WEBHOOK = "webhook"
    PUSH = "push"
    SMS = "sms"


@unique
class NotificationDeliveryStatus(StrEnum):
    """Lifecycle status for a notification delivery attempt."""

    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether delivery has reached a final state."""
        return self in {
            NotificationDeliveryStatus.DELIVERED,
            NotificationDeliveryStatus.FAILED,
            NotificationDeliveryStatus.CANCELLED,
        }


# ---------------------------------------------------------------------------
# Integrations and synchronization
# ---------------------------------------------------------------------------


@unique
class IntegrationStatus(StrEnum):
    """Lifecycle status for an external integration."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    EXPIRED = "expired"
    DISABLED = "disabled"


@unique
class IntegrationAuthType(StrEnum):
    """Supported integration authentication mechanisms."""

    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    BEARER = "bearer"
    SERVICE_ACCOUNT = "service_account"
    NONE = "none"


@unique
class SyncStatus(StrEnum):
    """Lifecycle status for integration synchronization."""

    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@unique
class SyncType(StrEnum):
    """Type of synchronization operation."""

    FULL = "full"
    INCREMENTAL = "incremental"
    RECONCILIATION = "reconciliation"


# ---------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------


@unique
class WebhookEventStatus(StrEnum):
    """Lifecycle status for an inbound webhook event."""

    RECEIVED = "received"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"
    IGNORED = "ignored"

    @property
    def is_terminal(self) -> bool:
        """Whether the webhook event has reached a final state."""
        return self in {
            WebhookEventStatus.PROCESSED,
            WebhookEventStatus.FAILED,
            WebhookEventStatus.DEAD_LETTERED,
            WebhookEventStatus.IGNORED,
        }


@unique
class WebhookDeliveryStatus(StrEnum):
    """Lifecycle status for an outbound webhook delivery."""

    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEAD_LETTERED = "dead_lettered"

    @property
    def is_terminal(self) -> bool:
        """Whether delivery has reached a final state."""
        return self in {
            WebhookDeliveryStatus.DELIVERED,
            WebhookDeliveryStatus.FAILED,
            WebhookDeliveryStatus.CANCELLED,
            WebhookDeliveryStatus.DEAD_LETTERED,
        }


@unique
class WebhookSignatureAlgorithm(StrEnum):
    """Supported webhook signature algorithms."""

    HMAC_SHA256 = "hmac-sha256"
    HMAC_SHA512 = "hmac-sha512"


# ---------------------------------------------------------------------------
# Usage and retention
# ---------------------------------------------------------------------------


@unique
class UsageType(StrEnum):
    """Metered usage categories."""

    LLM_TOKENS = "llm_tokens"
    EMBEDDING_TOKENS = (
        "embedding_tokens"
    )
    API_CALLS = "api_calls"
    DOCUMENTS = "documents"
    STORAGE_BYTES = "storage_bytes"
    WORKFLOW_EXECUTIONS = (
        "workflow_executions"
    )


@unique
class RetentionAction(StrEnum):
    """Action taken when a retention period expires."""

    DELETE = "delete"
    ARCHIVE = "archive"
    ANONYMIZE = "anonymize"


@unique
class RetentionJobStatus(StrEnum):
    """Lifecycle status for a retention sweep."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Whether the retention job has reached a final state."""
        return self in {
            RetentionJobStatus.COMPLETED,
            RetentionJobStatus.FAILED,
            RetentionJobStatus.CANCELLED,
        }


# ---------------------------------------------------------------------------
# User and tenant lifecycle
# ---------------------------------------------------------------------------


@unique
class UserRole(StrEnum):
    """Tenant-scoped user roles."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


@unique
class UserStatus(StrEnum):
    """Lifecycle status for a user account."""

    ACTIVE = "active"
    INVITED = "invited"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"
    DELETED = "deleted"


@unique
class TenantStatus(StrEnum):
    """Lifecycle status for a tenant workspace."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


@unique
class TenantPlan(StrEnum):
    """Tenant subscription plans."""

    TRIAL = "trial"
    STANDARD = "standard"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@unique
class InvitationStatus(StrEnum):
    """Lifecycle status for a tenant invitation."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REVOKED = "revoked"
    EXPIRED = "expired"


# ---------------------------------------------------------------------------
# Interrupts, recovery, and audit
# ---------------------------------------------------------------------------


@unique
class InterruptType(StrEnum):
    """Types of workflow interruptions."""

    HUMAN_APPROVAL = "human_approval"
    VALIDATION_FAILURE = (
        "validation_failure"
    )
    PROVIDER_FAILURE = "provider_failure"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    MANUAL_PAUSE = "manual_pause"
    SYSTEM_ERROR = "system_error"


@unique
class RecoveryAction(StrEnum):
    """Recovery actions available after an interruption."""

    RESUME = "resume"
    RETRY = "retry"
    ROLLBACK = "rollback"
    SKIP = "skip"
    CANCEL = "cancel"
    ESCALATE = "escalate"


@unique
class AuditEventType(StrEnum):
    """High-level audit event categories."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DOCUMENT = "document"
    WORKFLOW = "workflow"
    APPROVAL = "approval"
    ARTIFACT = "artifact"
    INTEGRATION = "integration"
    WEBHOOK = "webhook"
    NOTIFICATION = "notification"
    USAGE = "usage"
    RETENTION = "retention"
    ADMINISTRATION = "administration"


@unique
class AuditOutcome(StrEnum):
    """Outcome of an audited action."""

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# API error categories
# ---------------------------------------------------------------------------


@unique
class ApiErrorCode(StrEnum):
    """Stable machine-readable API error codes."""

    INTERNAL_ERROR = "internal_error"
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    AUTHENTICATION_ERROR = (
        "authentication_error"
    )
    AUTHORIZATION_ERROR = (
        "authorization_error"
    )
    RATE_LIMIT_EXCEEDED = (
        "rate_limit_exceeded"
    )
    EXTERNAL_PROVIDER_ERROR = (
        "external_provider_error"
    )
    DATABASE_UNAVAILABLE = (
        "database_unavailable"
    )
    SERVICE_UNAVAILABLE = (
        "service_unavailable"
    )


@unique
class ErrorCode(StrEnum):
    """
    Stable machine-readable application error codes.

    This enum is retained for compatibility with modules and tests that import
    `ErrorCode` directly from `app.core.enums`.
    """

    INTERNAL_ERROR = "internal_error"
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND_ERROR = "not_found"
    CONFLICT_ERROR = "conflict"
    AUTHENTICATION_ERROR = (
        "authentication_error"
    )
    AUTHORIZATION_ERROR = (
        "authorization_error"
    )
    RATE_LIMIT_ERROR = (
        "rate_limit_exceeded"
    )
    PROVIDER_ERROR = (
        "external_provider_error"
    )
    DATABASE_ERROR = "database_error"
    DATABASE_INTEGRITY_ERROR = (
        "database_integrity_error"
    )
    DATABASE_UNAVAILABLE = (
        "database_unavailable"
    )
    WORKFLOW_ERROR = "workflow_error"
    INGESTION_ERROR = "ingestion_error"
    EMBEDDING_ERROR = "embedding_error"
    RETRIEVAL_ERROR = "retrieval_error"
    INTEGRATION_ERROR = "integration_error"
    WEBHOOK_ERROR = "webhook_error"


__all__ = [
    "ApiErrorCode",
    "ApprovalDecision",
    "ApprovalPriority",
    "ApprovalStatus",
    "ArtifactStatus",
    "ArtifactType",
    "AuditEventType",
    "AuditOutcome",
    "DocumentStatus",
    "DocumentType",
    "EmbeddingProvider",
    "EmbeddingStatus",
    "Environment",
    "ErrorCode",
    "ExternalIntegrationProvider",
    "IngestionItemStatus",
    "IngestionSourceType",
    "IntegrationAuthType",
    "IntegrationStatus",
    "InterruptType",
    "InvitationStatus",
    "JobStatus",
    "LLMProvider",
    "NotificationChannel",
    "NotificationDeliveryStatus",
    "NotificationStatus",
    "RecoveryAction",
    "RelevanceLabel",
    "RetentionAction",
    "RetentionJobStatus",
    "RetrievalMode",
    "SyncStatus",
    "SyncType",
    "TenantPlan",
    "TenantStatus",
    "UsageType",
    "UserRole",
    "UserStatus",
    "WebhookDeliveryStatus",
    "WebhookEventStatus",
    "WebhookSignatureAlgorithm",
    "WorkflowControlAction",
    "WorkflowStep",
    "WorkflowTrigger",
]