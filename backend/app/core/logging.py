"""
Structured logging configuration.

In production, logs are emitted as one JSON object per line to stdout so they
can be collected by Docker, Kubernetes, CloudWatch, Datadog, Loki, or
OpenTelemetry-compatible log pipelines.

Design principles:
- Never log secrets, API keys, passwords, raw prompts, or full request bodies.
- Include request, tenant, user, workflow, job, and provider context where relevant.
- Use stable event names rather than free-form sentences.
- Emit JSON in staging and production; use colored console output locally.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.config import settings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


LOG_RECORD_RESERVED_FIELDS: frozenset[
    str
] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    },
)


DEFAULT_NOISY_LOGGERS: tuple[
    str,
    ...,
] = (
    "uvicorn.access",
    "uvicorn.error",
    "sqlalchemy.engine",
    "sqlalchemy.pool",
    "httpx",
    "httpcore",
    "openai",
    "anthropic",
    "asyncio",
)


SENSITIVE_LOG_KEYS: frozenset[
    str
] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "passwd",
        "secret",
        "secret_key",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "client_secret",
        "private_key",
        "database_url",
        "connection_string",
        "cookie",
        "set-cookie",
        "prompt",
        "completion",
        "messages",
        "embedding",
        "vector",
        "generated_code",
        "stack_trace",
        "traceback",
    },
)


SENSITIVE_KEY_FRAGMENTS: tuple[
    str,
    ...,
] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "cookie",
    "signed_url",
    "storage_url",
    "file_url",
    "download_url",
    "prompt",
    "completion",
    "messages",
    "input",
    "output",
    "embedding",
    "vector",
    "generated_code",
    "stack_trace",
    "traceback",
)


REDACTED_VALUE = "[REDACTED]"


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def _is_sensitive_key(
    key: Any,
) -> bool:
    """Return whether a log key appears sensitive."""
    normalized_key = str(key).strip().lower()

    if normalized_key in SENSITIVE_LOG_KEYS:
        return True

    return any(
        fragment in normalized_key
        for fragment in SENSITIVE_KEY_FRAGMENTS
    )


def _redact_mapping(
    mapping: dict[Any, Any],
) -> dict[Any, Any]:
    """
    Recursively redact sensitive values in a mapping.
    """
    redacted: dict[Any, Any] = {}

    for key, value in mapping.items():
        if _is_sensitive_key(key):
            redacted[key] = REDACTED_VALUE

        elif isinstance(value, dict):
            redacted[key] = _redact_mapping(
                value,
            )

        elif isinstance(value, list):
            redacted[key] = [
                _redact_mapping(item)
                if isinstance(item, dict)
                else item
                for item in value
            ]

        else:
            redacted[key] = value

    return redacted


def redact_sensitive_values(
    logger: Any,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """
    Redact sensitive keys from structured log events.

    This is a defense-in-depth control. Services should avoid passing secrets
    into log calls, but this processor prevents accidental leakage if they do.
    """
    return _redact_mapping(
        dict(event_dict),
    )


class SensitiveLogFilter(logging.Filter):
    """
    Standard-library filter that redacts sensitive log-record fields.

    This protects records emitted through the standard logging API, including
    records forwarded from Uvicorn and third-party libraries.
    """

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:
        """Redact sensitive attributes on a log record."""
        for key in list(
            record.__dict__.keys(),
        ):
            if key in LOG_RECORD_RESERVED_FIELDS:
                continue

            if _is_sensitive_key(key):
                setattr(
                    record,
                    key,
                    REDACTED_VALUE,
                )

        return True


sensitive_log_filter = SensitiveLogFilter()


def add_service_context(
    logger: Any,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Attach stable application metadata to every log event."""
    event_dict.setdefault(
        "service",
        settings.APP_NAME,
    )
    event_dict.setdefault(
        "environment",
        settings.APP_ENV,
    )
    event_dict.setdefault(
        "version",
        settings.APP_VERSION,
    )

    return event_dict


def normalize_event_name(
    logger: Any,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """
    Normalize the primary event field.

    Structured log platforms benefit from stable, lowercase, snake_case event
    names such as `workflow_completed` or `database_query_failed`.
    """
    event = event_dict.get("event")

    if isinstance(event, str):
        event_dict["event"] = (
            event.strip()
            .lower()
            .replace(" ", "_")
        )

    return event_dict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _resolve_log_level() -> int:
    """Resolve the configured log level to a standard-library level."""
    level_name = str(
        settings.LOG_LEVEL,
    ).upper()

    level = getattr(
        logging,
        level_name,
        None,
    )

    if not isinstance(level, int):
        return logging.INFO

    return level


def _build_shared_processors() -> list[
    structlog.types.Processor
]:
    """Build processors shared by structlog and third-party log records."""
    return [
        structlog.contextvars.merge_contextvars,
        add_service_context,
        normalize_event_name,
        redact_sensitive_values,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(
            fmt="iso",
            utc=True,
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]


def _build_renderer() -> structlog.types.Processor:
    """Select the appropriate log renderer for the current environment."""
    use_console_renderer = (
        settings.APP_ENV == "development"
        and not settings.LOG_FORMAT_JSON
    )

    if use_console_renderer:
        return (
            structlog.dev.ConsoleRenderer(
                colors=True,
            )
        )

    return (
        structlog.processors.JSONRenderer()
    )


def _build_handler() -> logging.Handler:
    """Build the shared stdout log handler."""
    formatter = (
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=_build_shared_processors(),
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                _build_renderer(),
            ],
        )
    )

    handler = logging.StreamHandler(
        sys.stdout,
    )

    handler.setFormatter(formatter)
    handler.addFilter(sensitive_log_filter)

    return handler


def _configure_uvicorn_loggers(
    handler: logging.Handler,
) -> None:
    """
    Route Uvicorn logs through the application's structured formatter.

    Uvicorn installs its own handlers at startup, so they must be replaced
    after the application logger is configured.
    """
    for logger_name in (
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    ):
        uvicorn_logger = logging.getLogger(
            logger_name,
        )

        uvicorn_logger.handlers.clear()
        uvicorn_logger.addHandler(handler)
        uvicorn_logger.propagate = False


def configure_logging() -> None:
    """
    Configure standard-library and structlog logging.

    Call this once during application startup, before creating routes or
    starting background workers.
    """
    log_level = _resolve_log_level()

    shared_processors = (
        _build_shared_processors()
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    handler = _build_handler()

    root_logger = logging.getLogger()

    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    root_logger.propagate = False

    _configure_uvicorn_loggers(handler)

    # Reduce noisy third-party logs while retaining warnings and errors.
    for logger_name in DEFAULT_NOISY_LOGGERS:
        logging.getLogger(
            logger_name,
        ).setLevel(logging.WARNING)

    # SQLAlchemy SQL statements should only appear when explicitly enabled.
    if settings.DB_ECHO:
        logging.getLogger(
            "sqlalchemy.engine",
        ).setLevel(logging.INFO)

    else:
        logging.getLogger(
            "sqlalchemy.engine",
        ).setLevel(logging.WARNING)

    logger = get_logger(__name__)

    logger.info(
        "logging_configured",
        log_level=logging.getLevelName(
            log_level,
        ),
        json_format=settings.LOG_FORMAT_JSON,
        environment=settings.APP_ENV,
    )


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------


def get_logger(
    name: str,
) -> structlog.stdlib.BoundLogger:
    """
    Return a configured structured logger.

    Args:
        name: Usually `__name__`, for example `app.services.workflow_service`.

    Returns:
        A structlog logger bound to the application's processor chain.
    """
    return structlog.get_logger(name)


def get_module_logger(
    module: str,
) -> structlog.stdlib.BoundLogger:
    """
    Return a logger using a normalized module name.

    Example:
        `get_module_logger("app.services.ingestion")`
    """
    normalized_name = (
        module.strip().removeprefix("app.")
    )

    return get_logger(
        f"agentic_bridge.{normalized_name}",
    )


__all__ = [
    "DEFAULT_NOISY_LOGGERS",
    "LOG_RECORD_RESERVED_FIELDS",
    "REDACTED_VALUE",
    "SENSITIVE_KEY_FRAGMENTS",
    "SENSITIVE_LOG_KEYS",
    "SensitiveLogFilter",
    "add_service_context",
    "configure_logging",
    "get_logger",
    "get_module_logger",
    "normalize_event_name",
    "redact_sensitive_values",
    "sensitive_log_filter",
]