"""
Signed webhook ingestion routes.

These routes receive HMAC-signed webhook events from CMS platforms, object
storage providers, and external AI/data providers. Payload signatures are
verified before parsing, events are persisted for reliable processing, and
ingestion outcomes are recorded in the tenant audit trail when a tenant can be
resolved.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Request,
    status,
)

from app.api.dependencies import DatabaseSession
from app.core.config import get_settings
from app.core.exceptions import ValidationError
from app.repositories.audit_log_repository import (
    AuditLogRepository,
)
from app.repositories.webhook_repository import (
    WebhookRepository,
)


logger = logging.getLogger(
    "agentic_bridge.api.webhooks",
)


router = APIRouter(
    prefix="/webhooks",
    tags=["Webhooks"],
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGNATURE_HEADER = "X-Signature"
TIMESTAMP_HEADER = "X-Signature-Timestamp"
EVENT_ID_HEADER = "X-Event-ID"
EVENT_TYPE_HEADER = "X-Event-Type"

MAX_WEBHOOK_BODY_BYTES = 1_048_576
DEFAULT_SIGNATURE_TOLERANCE_SECONDS = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_webhook_secret() -> str:
    """Return the configured webhook signing secret."""
    settings = get_settings()

    webhook_secret = getattr(
        settings,
        "WEBHOOK_SECRET",
        "",
    )

    if not webhook_secret:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Webhook processing is not configured."
            ),
        )

    return webhook_secret


def _compute_signature(
    secret: str,
    body: bytes,
    timestamp: str | None = None,
) -> str:
    """
    Compute an HMAC-SHA256 signature for a webhook payload.

    When a timestamp is supplied, it is included in the signed payload as:

        timestamp.body
    """
    signature_payload = body

    if timestamp:
        signature_payload = (
            timestamp.encode("utf-8")
            + b"."
            + body
        )

    return hmac.new(
        secret.encode("utf-8"),
        signature_payload,
        hashlib.sha256,
    ).hexdigest()


def _extract_signature(
    provided_signature: str,
) -> str:
    """Extract the hexadecimal signature from a provider header."""
    normalized_signature = (
        provided_signature.strip()
    )

    if normalized_signature.lower().startswith(
        "sha256=",
    ):
        normalized_signature = (
            normalized_signature.split(
                "=",
                maxsplit=1,
            )[1]
        )

    return normalized_signature


def _verify_timestamp(
    timestamp: str | None,
    tolerance_seconds: int,
) -> None:
    """Reject webhook requests outside the allowed signature window."""
    if not timestamp:
        return

    try:
        webhook_timestamp = int(timestamp)
    except ValueError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Invalid webhook signature timestamp."
            ),
        ) from exc

    current_timestamp = int(
        time.time(),
    )

    if abs(
        current_timestamp - webhook_timestamp
    ) > tolerance_seconds:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail=(
                "Webhook signature has expired."
            ),
        )


def _parse_json_body(
    body: bytes,
) -> dict[str, Any]:
    """Parse and validate a JSON webhook payload."""
    if not body:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="Webhook payload is required.",
        )

    try:
        parsed_payload = json.loads(
            body.decode("utf-8"),
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Webhook payload must be valid JSON."
            ),
        ) from exc

    if not isinstance(
        parsed_payload,
        dict,
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Webhook payload must be a JSON object."
            ),
        )

    return parsed_payload


def _resolve_event_id(
    headers: Any,
    payload: dict[str, Any],
) -> str:
    """Resolve an event ID from headers or payload."""
    event_id = (
        headers.get(EVENT_ID_HEADER)
        or payload.get("event_id")
        or payload.get("id")
    )

    if event_id:
        return str(event_id)

    return str(uuid4())


def _resolve_event_type(
    headers: Any,
    payload: dict[str, Any],
    default_event_type: str,
) -> str:
    """Resolve an event type from headers or payload."""
    event_type = (
        headers.get(EVENT_TYPE_HEADER)
        or payload.get("event_type")
        or payload.get("type")
    )

    if event_type:
        return str(event_type)

    return default_event_type


async def _record_audit_event(
    db: DatabaseSession,
    *,
    event_type: str,
    action: str,
    outcome: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Record a webhook audit event when the repository supports it.

    Audit failures must not prevent webhook acceptance.
    """
    try:
        await AuditLogRepository.create_event(
            db,
            event_type=event_type,
            action=action,
            outcome=outcome,
            event_metadata=metadata or {},
        )

    except Exception:
        logger.warning(
            "Unable to record webhook audit event.",
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Signature verification dependency
# ---------------------------------------------------------------------------


async def verify_webhook_signature(
    request: Request,
    x_signature: Annotated[
        str | None,
        Header(
            alias=SIGNATURE_HEADER,
        ),
    ] = None,
    x_signature_timestamp: Annotated[
        str | None,
        Header(
            alias=TIMESTAMP_HEADER,
        ),
    ] = None,
) -> bytes:
    """Verify an HMAC-signed webhook payload."""
    secret = _get_webhook_secret()

    body = await request.body()

    if len(body) > MAX_WEBHOOK_BODY_BYTES:
        raise HTTPException(
            status_code=(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            ),
            detail=(
                "Webhook payload is too large."
            ),
        )

    if not x_signature:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail=(
                "Webhook signature is required."
            ),
        )

    _verify_timestamp(
        timestamp=x_signature_timestamp,
        tolerance_seconds=(
            DEFAULT_SIGNATURE_TOLERANCE_SECONDS
        ),
    )

    expected_signature = _compute_signature(
        secret=secret,
        body=body,
        timestamp=x_signature_timestamp,
    )

    provided_signature = _extract_signature(
        x_signature,
    )

    if not hmac.compare_digest(
        provided_signature,
        expected_signature,
    ):
        logger.warning(
            "Invalid webhook signature.",
            extra={
                "path": request.url.path,
                "method": request.method,
                "request_id": (
                    request.headers.get(
                        "X-Request-ID",
                    )
                ),
            },
        )

        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail="Invalid webhook signature.",
        )

    return body


# ---------------------------------------------------------------------------
# CMS webhooks
# ---------------------------------------------------------------------------


@router.post(
    "/cms",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Receive CMS webhook",
    description=(
        "Receive a signed CMS event, persist it for processing, and return "
        "an acceptance receipt."
    ),
)
async def receive_cms_webhook(
    db: DatabaseSession,
    request: Request,
    payload: bytes = Depends(
        verify_webhook_signature,
    ),
) -> dict[str, Any]:
    """Receive signed CMS events."""
    parsed_payload = _parse_json_body(
        payload,
    )

    event_id = _resolve_event_id(
        request.headers,
        parsed_payload,
    )

    event_type = _resolve_event_type(
        request.headers,
        parsed_payload,
        "cms_event",
    )

    try:
        webhook_event = (
            await WebhookRepository.create_event(
                db,
                source="cms",
                event_id=event_id,
                event_type=event_type,
                payload=parsed_payload,
                signature_verified=True,
                received_at=datetime.now(
                    timezone.utc,
                ),
            )
        )

    except ValidationError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(exc),
        ) from exc

    await db.commit()

    await _record_audit_event(
        db,
        event_type="webhook_received",
        action="receive_cms_webhook",
        outcome="success",
        metadata={
            "webhook_event_id": str(
                webhook_event.id,
            ),
            "event_id": event_id,
            "event_type": event_type,
        },
    )

    await db.commit()

    logger.info(
        "CMS webhook accepted.",
        extra={
            "webhook_event_id": str(
                webhook_event.id,
            ),
            "event_id": event_id,
            "event_type": event_type,
        },
    )

    return {
        "status": "accepted",
        "event": "cms_webhook",
        "webhook_event_id": str(
            webhook_event.id,
        ),
        "event_id": event_id,
        "event_type": event_type,
    }


# ---------------------------------------------------------------------------
# Storage webhooks
# ---------------------------------------------------------------------------


@router.post(
    "/storage",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Receive storage webhook",
    description=(
        "Receive a signed object-storage event, persist it for processing, "
        "and return an acceptance receipt."
    ),
)
async def receive_storage_webhook(
    db: DatabaseSession,
    request: Request,
    payload: bytes = Depends(
        verify_webhook_signature,
    ),
) -> dict[str, Any]:
    """Receive signed object-storage events."""
    parsed_payload = _parse_json_body(
        payload,
    )

    event_id = _resolve_event_id(
        request.headers,
        parsed_payload,
    )

    event_type = _resolve_event_type(
        request.headers,
        parsed_payload,
        "storage_event",
    )

    try:
        webhook_event = (
            await WebhookRepository.create_event(
                db,
                source="storage",
                event_id=event_id,
                event_type=event_type,
                payload=parsed_payload,
                signature_verified=True,
                received_at=datetime.now(
                    timezone.utc,
                ),
            )
        )

    except ValidationError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(exc),
        ) from exc

    await db.commit()

    await _record_audit_event(
        db,
        event_type="webhook_received",
        action="receive_storage_webhook",
        outcome="success",
        metadata={
            "webhook_event_id": str(
                webhook_event.id,
            ),
            "event_id": event_id,
            "event_type": event_type,
        },
    )

    await db.commit()

    logger.info(
        "Storage webhook accepted.",
        extra={
            "webhook_event_id": str(
                webhook_event.id,
            ),
            "event_id": event_id,
            "event_type": event_type,
        },
    )

    return {
        "status": "accepted",
        "event": "storage_webhook",
        "webhook_event_id": str(
            webhook_event.id,
        ),
        "event_id": event_id,
        "event_type": event_type,
    }


# ---------------------------------------------------------------------------
# Provider webhooks
# ---------------------------------------------------------------------------


@router.post(
    "/providers/{provider_name}",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Receive provider webhook",
    description=(
        "Receive a signed event from an external provider, persist it for "
        "processing, and return an acceptance receipt."
    ),
)
async def receive_provider_webhook(
    provider_name: str,
    db: DatabaseSession,
    request: Request,
    payload: bytes = Depends(
        verify_webhook_signature,
    ),
) -> dict[str, Any]:
    """Receive signed provider events."""
    normalized_provider_name = (
        provider_name.strip().lower()
    )

    if not normalized_provider_name:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail="Provider name is required.",
        )

    parsed_payload = _parse_json_body(
        payload,
    )

    event_id = _resolve_event_id(
        request.headers,
        parsed_payload,
    )

    event_type = _resolve_event_type(
        request.headers,
        parsed_payload,
        "provider_event",
    )

    try:
        webhook_event = (
            await WebhookRepository.create_event(
                db,
                source=normalized_provider_name,
                event_id=event_id,
                event_type=event_type,
                payload=parsed_payload,
                signature_verified=True,
                received_at=datetime.now(
                    timezone.utc,
                ),
            )
        )

    except ValidationError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(exc),
        ) from exc

    await db.commit()

    await _record_audit_event(
        db,
        event_type="webhook_received",
        action="receive_provider_webhook",
        outcome="success",
        metadata={
            "provider": (
                normalized_provider_name
            ),
            "webhook_event_id": str(
                webhook_event.id,
            ),
            "event_id": event_id,
            "event_type": event_type,
        },
    )

    await db.commit()

    logger.info(
        "Provider webhook accepted.",
        extra={
            "provider": normalized_provider_name,
            "webhook_event_id": str(
                webhook_event.id,
            ),
            "event_id": event_id,
            "event_type": event_type,
        },
    )

    return {
        "status": "accepted",
        "provider": normalized_provider_name,
        "webhook_event_id": str(
            webhook_event.id,
        ),
        "event_id": event_id,
        "event_type": event_type,
    }


# ---------------------------------------------------------------------------
# Webhook replay
# ---------------------------------------------------------------------------


@router.post(
    "/events/{event_id}/replay",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Replay webhook event",
    description=(
        "Queue a previously received webhook event for replay. Requires a "
        "valid webhook signature and an event that has not already been "
        "processed."
    ),
)
async def replay_webhook_event(
    event_id: UUID,
    db: DatabaseSession,
    payload: bytes = Depends(
        verify_webhook_signature,
    ),
) -> dict[str, Any]:
    """Queue a webhook event for replay."""
    webhook_event = (
        await WebhookRepository.get_by_event_id(
            db,
            event_id=event_id,
        )
    )

    if not webhook_event:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Webhook event not found.",
        )

    if webhook_event.status == "processed":
        raise HTTPException(
            status_code=(
                status.HTTP_409_CONFLICT
            ),
            detail=(
                "Webhook event has already been "
                "processed."
            ),
        )

    updated_event = (
        await WebhookRepository.mark_replay_requested(
            db,
            webhook_event_id=webhook_event.id,
        )
    )

    await db.commit()

    await _record_audit_event(
        db,
        event_type="webhook_replay_requested",
        action="replay_webhook_event",
        outcome="success",
        metadata={
            "webhook_event_id": str(
                webhook_event.id,
            ),
            "event_id": str(event_id),
        },
    )

    await db.commit()

    logger.info(
        "Webhook event replay requested.",
        extra={
            "webhook_event_id": str(
                webhook_event.id,
            ),
            "event_id": str(event_id),
        },
    )

    return {
        "webhook_event_id": str(
            updated_event.id,
        ),
        "event_id": updated_event.event_id,
        "status": updated_event.status,
        "replay_requested": True,
    }