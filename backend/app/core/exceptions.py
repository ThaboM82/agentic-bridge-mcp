"""
Application exception hierarchy.

Services raise domain errors; FastAPI exception handlers translate them into
consistent HTTP responses. Never expose internal stack traces, provider errors,
database errors, credentials, or raw upstream payloads directly to clients.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


JsonDict = dict[str, Any]


class AppError(Exception):
    """
    Base class for all expected application errors.

    Attributes:
        code: Stable machine-readable error code returned to API clients.
        message: Safe, user-facing error message.
        http_status: HTTP status code used by the global exception handler.
        details: Optional non-sensitive structured context.
        retryable: Whether the caller may safely retry the operation.
        headers: Optional HTTP headers to include in the error response.
    """

    code: str = "internal_error"
    message: str = "An unexpected error occurred."
    http_status: int = 500
    details: JsonDict | None = None
    retryable: bool = False
    headers: dict[str, str] | None = None

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        details: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        retryable: bool | None = None,
    ) -> None:
        self.message = message or self.message

        if code:
            self.code = code

        if details is not None:
            self.details = dict(details)

        if headers is not None:
            self.headers = dict(headers)

        if retryable is not None:
            self.retryable = retryable

        super().__init__(
            self.message,
        )

    def to_dict(self) -> JsonDict:
        """Return a safe JSON-serializable representation for API responses."""
        payload: JsonDict = {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
            },
        }

        if self.details:
            payload["error"][
                "details"
            ] = self.details

        return payload


# ---------------------------------------------------------------------------
# Request and resource errors
# ---------------------------------------------------------------------------


class ValidationError(AppError):
    """Raised when user input or a domain invariant fails validation."""

    code = "validation_error"
    message = (
        "The request contains invalid data."
    )
    http_status = 422


class SchemaValidationError(
    ValidationError,
):
    """Raised when structured data does not match an expected schema."""

    code = "schema_validation_error"
    message = (
        "The response could not be validated "
        "against the required schema."
    )


class NotFoundError(AppError):
    """Raised when a requested resource does not exist."""

    code = "not_found"
    message = (
        "The requested resource was not found."
    )
    http_status = 404


class ConflictError(AppError):
    """Raised when an operation conflicts with current resource state."""

    code = "conflict"
    message = (
        "The request conflicts with the "
        "current resource state."
    )
    http_status = 409


class ResourceAlreadyExistsError(
    ConflictError,
):
    """Raised when attempting to create a resource that already exists."""

    code = "resource_already_exists"
    message = (
        "A resource with the provided identifier "
        "already exists."
    )


class OptimisticLockError(
    ConflictError,
):
    """Raised when a concurrent update invalidated the caller's version."""

    code = "optimistic_lock_error"
    message = (
        "The resource was modified by another "
        "request. Please refresh and retry."
    )
    retryable = True


class PayloadTooLargeError(AppError):
    """Raised when an uploaded payload exceeds configured limits."""

    code = "payload_too_large"
    message = (
        "The uploaded file exceeds the allowed "
        "size limit."
    )
    http_status = 413


class UnsupportedMediaTypeError(AppError):
    """Raised when an uploaded file or request body has an unsupported type."""

    code = "unsupported_media_type"
    message = (
        "The provided file type is not supported."
    )
    http_status = 415


# ---------------------------------------------------------------------------
# Authentication and authorization
# ---------------------------------------------------------------------------


class AuthenticationError(AppError):
    """Raised when valid authentication credentials are required."""

    code = "authentication_error"
    message = "Authentication is required."
    http_status = 401
    headers = {
        "WWW-Authenticate": "Bearer",
    }


class InvalidCredentialsError(
    AuthenticationError,
):
    """Raised when supplied credentials are invalid."""

    code = "invalid_credentials"
    message = (
        "The provided credentials are invalid."
    )


class InvalidTokenError(
    AuthenticationError,
):
    """
    Raised when a token is malformed, expired, or fails validation.
    """

    code = "invalid_token"
    message = (
        "The authentication token is invalid."
    )


class TokenExpiredError(
    AuthenticationError,
):
    """Raised when an access or refresh token has expired."""

    code = "token_expired"
    message = (
        "The authentication token has expired."
    )


class AuthorizationError(AppError):
    """Raised when an authenticated caller lacks required permissions."""

    code = "authorization_error"
    message = (
        "You do not have permission to perform "
        "this action."
    )
    http_status = 403


class TenantAccessError(
    AuthorizationError,
):
    """Raised when a caller attempts to access another tenant's resources."""

    code = "tenant_access_denied"
    message = (
        "You do not have access to this tenant's "
        "resources."
    )


class ApprovalRequiredError(AppError):
    """Raised when human approval is required before execution can continue."""

    code = "approval_required"
    message = (
        "Human approval is required before this "
        "action can proceed."
    )
    http_status = 202


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class RateLimitError(AppError):
    """Raised when a caller exceeds an allowed request rate."""

    code = "rate_limit_exceeded"
    message = (
        "Too many requests. Please retry later."
    )
    http_status = 429
    retryable = True


# ---------------------------------------------------------------------------
# External service errors
# ---------------------------------------------------------------------------


class ExternalProviderError(AppError):
    """
    Raised when an upstream provider fails, returns an invalid response, or
    cannot be reached.
    """

    code = "external_provider_error"
    message = (
        "An upstream provider is temporarily "
        "unavailable."
    )
    http_status = 502
    retryable = True


class ProviderUnavailableError(
    ExternalProviderError,
):
    """Raised when an upstream provider cannot be reached."""

    code = "provider_unavailable"
    message = (
        "The upstream provider is temporarily "
        "unavailable."
    )
    http_status = 503


class ProviderTimeoutError(
    ExternalProviderError,
):
    """Raised when an upstream provider does not respond in time."""

    code = "provider_timeout"
    message = (
        "The upstream provider did not respond "
        "in time."
    )
    http_status = 504
    retryable = True


class ProviderResponseError(
    ExternalProviderError,
):
    """Raised when an upstream provider returns an invalid or unexpected response."""

    code = "provider_response_error"
    message = (
        "The upstream provider returned an "
        "invalid response."
    )
    http_status = 502
    retryable = False


class ConfigurationError(AppError):
    """Raised when required configuration is missing or invalid."""

    code = "configuration_error"
    message = (
        "The service is not configured correctly."
    )
    http_status = 500
    retryable = False


class DatabaseNotConfiguredError(
    ConfigurationError,
):
    """Raised when database configuration is missing or invalid."""

    code = "database_not_configured"
    message = (
        "Database configuration is missing or "
        "invalid."
    )


class DatabaseError(AppError):
    """Raised when a database operation fails."""

    code = "database_error"
    message = (
        "The service is temporarily unable to "
        "access its database."
    )
    http_status = 503
    retryable = True


class DatabaseIntegrityError(
    DatabaseError,
):
    """Raised when a database integrity constraint is violated."""

    code = "database_integrity_error"
    message = (
        "The requested change conflicts with "
        "existing data."
    )
    http_status = 409
    retryable = False


class CacheError(AppError):
    """Raised when a cache operation fails."""

    code = "cache_error"
    message = (
        "The service is temporarily unable to "
        "access its cache."
    )
    http_status = 503
    retryable = True


class QueueError(AppError):
    """Raised when a background-job queue operation fails."""

    code = "queue_error"
    message = (
        "The service is temporarily unable to "
        "process background work."
    )
    http_status = 503
    retryable = True


class WorkflowError(AppError):
    """Raised when a workflow cannot be started, resumed, or completed."""

    code = "workflow_error"
    message = (
        "The workflow could not be processed."
    )
    http_status = 500


class WorkflowStateError(
    WorkflowError,
):
    """Raised when a workflow is in an invalid state for the requested action."""

    code = "workflow_state_error"
    message = (
        "The workflow is not in a valid state "
        "for this action."
    )
    http_status = 409


class WorkflowExecutionError(
    WorkflowError,
):
    """Raised when a workflow execution fails."""

    code = "workflow_execution_error"
    message = (
        "The workflow execution failed."
    )
    http_status = 500


class WorkflowNotFoundError(
    NotFoundError,
):
    """Raised when a workflow cannot be found."""

    code = "workflow_not_found"
    message = (
        "The requested workflow was not found."
    )


class IngestionError(AppError):
    """Raised when document ingestion fails."""

    code = "ingestion_error"
    message = (
        "Document ingestion could not be completed."
    )
    http_status = 500


class EmbeddingError(AppError):
    """Raised when embedding generation or vector indexing fails."""

    code = "embedding_error"
    message = (
        "Embedding generation could not be completed."
    )
    http_status = 500
    retryable = True


class RetrievalError(AppError):
    """Raised when semantic or hybrid retrieval fails."""

    code = "retrieval_error"
    message = (
        "Retrieval could not be completed."
    )
    http_status = 500
    retryable = True


class IntegrationError(AppError):
    """Raised when an external integration operation fails."""

    code = "integration_error"
    message = (
        "The external integration could not be "
        "completed."
    )
    http_status = 502
    retryable = True


class WebhookError(AppError):
    """Raised when webhook ingestion or delivery fails."""

    code = "webhook_error"
    message = (
        "The webhook could not be processed."
    )
    http_status = 500


class WebhookSignatureError(
    WebhookError,
):
    """Raised when an inbound webhook signature cannot be verified."""

    code = "webhook_signature_invalid"
    message = (
        "The webhook signature could not be verified."
    )
    http_status = 400
    retryable = False


class NotificationError(AppError):
    """Raised when a notification cannot be created or delivered."""

    code = "notification_error"
    message = (
        "The notification could not be processed."
    )
    http_status = 500
    retryable = True


class UsageLimitError(AppError):
    """Raised when a tenant or user exceeds an enforced usage limit."""

    code = "usage_limit_exceeded"
    message = (
        "The configured usage limit has been reached."
    )
    http_status = 429
    retryable = False


class QuotaExceededError(
    UsageLimitError,
):
    """Raised when a configured quota has been exhausted."""

    code = "quota_exceeded"
    message = (
        "The configured quota has been exhausted."
    )


class RetentionError(AppError):
    """Raised when a retention policy or retention job fails."""

    code = "retention_error"
    message = (
        "The retention operation could not be "
        "completed."
    )
    http_status = 500


class TemporaryUnavailableError(AppError):
    """Raised when a required service is temporarily unavailable."""

    code = "service_unavailable"
    message = (
        "The service is temporarily unavailable."
    )
    http_status = 503
    retryable = True


class NotImplementedFeatureError(AppError):
    """Raised when a requested capability is not implemented."""

    code = "not_implemented"
    message = (
        "This feature is not currently available."
    )
    http_status = 501


__all__ = [
    "AppError",
    "ApprovalRequiredError",
    "AuthenticationError",
    "AuthorizationError",
    "CacheError",
    "ConfigurationError",
    "ConflictError",
    "DatabaseError",
    "DatabaseIntegrityError",
    "DatabaseNotConfiguredError",
    "EmbeddingError",
    "ExternalProviderError",
    "IngestionError",
    "IntegrationError",
    "InvalidCredentialsError",
    "InvalidTokenError",
    "NotFoundError",
    "NotImplementedFeatureError",
    "NotificationError",
    "OptimisticLockError",
    "PayloadTooLargeError",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "QueueError",
    "RateLimitError",
    "RetrievalError",
    "RetentionError",
    "SchemaValidationError",
    "TemporaryUnavailableError",
    "TenantAccessError",
    "TokenExpiredError",
    "UnsupportedMediaTypeError",
    "UsageLimitError",
    "ValidationError",
    "WebhookError",
    "WebhookSignatureError",
    "WorkflowError",
    "WorkflowExecutionError",
    "WorkflowNotFoundError",
    "WorkflowStateError",
]