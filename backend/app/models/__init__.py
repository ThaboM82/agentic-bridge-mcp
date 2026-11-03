"""
Database model package for AgenticBridge.

Importing this package registers all SQLAlchemy ORM models with the shared
declarative base.
"""

from app.models.base import (
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)
from app.models.tenant import (
    Tenant,
    TenantInvitation,
    TenantSettings,
)
from app.models.user import (
    User,
    UserSession,
)
from app.models.api_key import (
    ApiKey,
)
from app.models.document import (
    Document,
    DocumentChunk,
    DocumentVersion,
)
from app.models.vector_model import (
    DocumentEmbedding,
    EmbeddingQueryLog,
)
from app.models.ingestion_job import (
    IngestionJob,
    IngestionJobItem,
)
from app.models.brief import (
    Brief,
    BriefTemplate,
    BriefVersion,
)
from app.models.workflow_model import (
    WorkflowApprovalModel,
    WorkflowArtifactModel,
    WorkflowAuditEventModel,
    WorkflowExecutionModel,
    WorkflowStepModel,
)
from app.models.approval import (
    Approval,
    ApprovalDelegation,
    ApprovalPolicy,
)
from app.models.artifact import (
    Artifact,
    ArtifactDownloadToken,
    ArtifactVersion,
)
from app.models.audit import (
    AuditEvent,
    AuditRetentionPolicy,
)
from app.models.usage import (
    UsageEvent,
    UsageQuota,
    UsageRateCard,
)
from app.models.webhook import (
    WebhookDelivery,
    WebhookEndpoint,
    WebhookEvent,
)
from app.models.notification import (
    Notification,
    NotificationDelivery,
    NotificationPreference,
)
from app.models.integration import (
    Integration,
    IntegrationCredential,
    IntegrationSyncState,
)
from app.models.search import (
    RetrievalExperiment,
    SavedSearch,
    SearchFeedback,
)
from app.models.retention import (
    RetentionPolicy,
    RetentionJob,
)

__all__ = [
    "ApiKey",
    "Approval",
    "ApprovalDelegation",
    "ApprovalPolicy",
    "Artifact",
    "ArtifactDownloadToken",
    "ArtifactVersion",
    "AuditEvent",
    "AuditRetentionPolicy",
    "Brief",
    "BriefTemplate",
    "BriefVersion",
    "Document",
    "DocumentChunk",
    "DocumentEmbedding",
    "DocumentVersion",
    "EmbeddingQueryLog",
    "IngestionJob",
    "IngestionJobItem",
    "Integration",
    "IntegrationCredential",
    "IntegrationSyncState",
    "Notification",
    "NotificationDelivery",
    "NotificationPreference",
    "RetentionPolicy",
    "RetentionJob",
    "RetrievalExperiment",
    "SavedSearch",
    "SearchFeedback",
    "Tenant",
    "TenantInvitation",
    "TenantSettings",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
    "UsageEvent",
    "UsageQuota",
    "UsageRateCard",
    "User",
    "UserSession",
    "WebhookDelivery",
    "WebhookEndpoint",
    "WebhookEvent",
    "WorkflowApprovalModel",
    "WorkflowArtifactModel",
    "WorkflowAuditEventModel",
    "WorkflowExecutionModel",
    "WorkflowStepModel",
]