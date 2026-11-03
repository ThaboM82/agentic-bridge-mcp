"""
LangGraph output-validation node.

Validates generated workflow artifacts before human review or publication.
This node performs deterministic, bounded checks and records a structured
validation outcome; it does not execute generated code or make the final
approval decision.

Responsibilities:
- Validate generated file count, paths, sizes, and total output size.
- Perform lightweight Python syntax validation.
- Detect obviously dangerous generated-code patterns.
- Check required design and accessibility context.
- Record validation issues, outcome, score, and quality-gate results.
- Map validation failures to a recoverable workflow error.

Non-goals:
- Do not run tests, static analyzers, security scanners, or generated code.
- Do not approve, reject, or publish workflow artifacts.
- Do not invoke LLMs or embedding providers in this node.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any, Literal

from app.core.constants import (
    MAX_GENERATED_FILE_COUNT,
    MAX_GENERATED_FILE_SIZE_CHARS,
)
from app.core.enums import JobStatus, WorkflowStep
from app.core.exceptions import WorkflowStateError
from app.core.logging_context import bind_workflow_context
from app.graph.state import AgenticState


logger = logging.getLogger("agentic_bridge.graph.nodes.validation")

MAX_TOTAL_GENERATED_CHARS = 1_000_000
MAX_VALIDATION_ISSUES = 100
MAX_ISSUE_MESSAGE_CHARS = 500

DANGEROUS_CODE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\beval\s*\(", re.IGNORECASE),
    re.compile(r"\bexec\s*\(", re.IGNORECASE),
    re.compile(r"\bos\.system\s*\(", re.IGNORECASE),
    re.compile(
        r"\bsubprocess\.(?:run|call|check_call|check_output|Popen)\s*\(",
        re.IGNORECASE,
    ),
    re.compile(r"\bshutil\.rmtree\s*\(", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
)

SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)\b(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?token|"
        r"refresh[_-]?token|password)\s*[=:]\s*[\"'][^\"']{8,}[\"']"
    ),
    re.compile(
        r"(?i)\b(?:sk|pk)_-[A-Za-z0-9_-]{20,}"
    ),
    re.compile(
        r"(?i)postgresql(?:\+asyncpg)?://[^\s\"']+:[^\s\"']+@"
    ),
)

UNSAFE_PATH_PATTERN = re.compile(
    r"(^/|^\.\.?/|(^|/)\.\.(/|$))"
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _add_issue(
    issues: list[dict[str, Any]],
    *,
    severity: Literal["error", "warning", "info"],
    code: str,
    message: str,
    file_path: str | None = None,
) -> None:
    """Append a bounded validation issue."""
    if len(issues) >= MAX_VALIDATION_ISSUES:
        return

    issue: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": truncate_issue_message(message),
    }

    if file_path:
        issue["file_path"] = file_path

    issues.append(issue)


def truncate_issue_message(message: str) -> str:
    """Bound a validation-issue message."""
    normalized_message = message.strip()

    if len(normalized_message) > MAX_ISSUE_MESSAGE_CHARS:
        return normalized_message[:MAX_ISSUE_MESSAGE_CHARS] + "..."

    return normalized_message


def _validate_file_paths(
    generated_files: dict[str, str],
    issues: list[dict[str, Any]],
) -> None:
    """Validate generated file paths."""
    seen_paths: set[str] = set()

    for file_path in generated_files:
        normalized_path = str(file_path).replace("\\", "/")
        normalized_key = normalized_path.lower()

        if not normalized_path:
            _add_issue(
                issues,
                severity="error",
                code="empty_file_path",
                message="Generated file path is empty.",
            )
            continue

        if len(normalized_path) > 255:
            _add_issue(
                issues,
                severity="error",
                code="file_path_too_long",
                message="Generated file path exceeds the allowed length.",
                file_path=normalized_path,
            )

        if UNSAFE_PATH_PATTERN.search(normalized_path):
            _add_issue(
                issues,
                severity="error",
                code="unsafe_file_path",
                message=(
                    "Generated file path must be a safe relative path."
                ),
                file_path=normalized_path,
            )

        if normalized_key in seen_paths:
            _add_issue(
                issues,
                severity="error",
                code="duplicate_file_path",
                message="Duplicate generated file path detected.",
                file_path=normalized_path,
            )

        seen_paths.add(normalized_key)


def _validate_file_contents(
    generated_files: dict[str, str],
    issues: list[dict[str, Any]],
) -> None:
    """Validate individual generated file contents."""
    for file_path, content in generated_files.items():
        normalized_path = str(file_path).replace("\\", "/")

        if not isinstance(content, str):
            _add_issue(
                issues,
                severity="error",
                code="invalid_file_content",
                message="Generated file content must be a string.",
                file_path=normalized_path,
            )
            continue

        if not content.strip():
            _add_issue(
                issues,
                severity="error",
                code="empty_file_content",
                message="Generated file is empty.",
                file_path=normalized_path,
            )
            continue

        if len(content) > MAX_GENERATED_FILE_SIZE_CHARS:
            _add_issue(
                issues,
                severity="error",
                code="file_too_large",
                message=(
                    "Generated file exceeds the maximum allowed size."
                ),
                file_path=normalized_path,
            )

        if normalized_path.lower().endswith(".py"):
            _validate_python_syntax(
                content,
                normalized_path,
                issues,
            )

        _detect_dangerous_patterns(
            content,
            normalized_path,
            issues,
        )

        _detect_potential_secrets(
            content,
            normalized_path,
            issues,
        )


def _validate_python_syntax(
    content: str,
    file_path: str,
    issues: list[dict[str, Any]],
) -> None:
    """Perform lightweight Python syntax validation without executing code."""
    try:
        ast.parse(content)
    except SyntaxError as exc:
        _add_issue(
            issues,
            severity="error",
            code="python_syntax_error",
            message=(
                f"Python syntax error: {exc.msg} "
                f"at line {exc.lineno}."
            ),
            file_path=file_path,
        )
    except (ValueError, RecursionError):
        _add_issue(
            issues,
            severity="error",
            code="python_parse_error",
            message="Python file could not be parsed safely.",
            file_path=file_path,
        )


def _detect_dangerous_patterns(
    content: str,
    file_path: str,
    issues: list[dict[str, Any]],
) -> None:
    """Detect obviously dangerous generated-code patterns."""
    for pattern in DANGEROUS_CODE_PATTERNS:
        if pattern.search(content):
            _add_issue(
                issues,
                severity="warning",
                code="potentially_dangerous_code",
                message=(
                    "Potentially dangerous code pattern detected. "
                    "Human review is required."
                ),
                file_path=file_path,
            )


def _detect_potential_secrets(
    content: str,
    file_path: str,
    issues: list[dict[str, Any]],
) -> None:
    """Detect likely hardcoded credentials or secrets."""
    for pattern in SECRET_PATTERNS:
        if pattern.search(content):
            _add_issue(
                issues,
                severity="error",
                code="potential_hardcoded_secret",
                message=(
                    "Potential hardcoded credential or secret detected. "
                    "Remove it and use environment configuration."
                ),
                file_path=file_path,
            )


def _validate_quality_gates(
    state: AgenticState,
    issues: list[dict[str, Any]],
) -> tuple[bool | None, bool | None, bool | None]:
    """
    Validate required workflow context and quality gates.

    Returns:
        A tuple of:
        `(static_analysis_passed, security_scan_passed, accessibility_check_passed)`

        These values describe only the checks performed by this node. External
        analyzers and scanners should override them in a service or worker.
    """
    design_tokens = state.get("design_tokens") or {}
    accessibility_requirements = (
        state.get("accessibility_requirements") or []
    )

    static_analysis_passed: bool | None = None
    security_scan_passed: bool | None = None
    accessibility_check_passed: bool | None = None

    if not design_tokens:
        _add_issue(
            issues,
            severity="error",
            code="missing_design_context",
            message=(
                "Workflow cannot be validated without design context."
            ),
        )

    if not accessibility_requirements and not design_tokens.get(
        "accessibility"
    ):
        _add_issue(
            issues,
            severity="warning",
            code="missing_accessibility_context",
            message=(
                "No explicit accessibility requirements were supplied."
            ),
        )
        accessibility_check_passed = False
    else:
        accessibility_check_passed = True

    return (
        static_analysis_passed,
        security_scan_passed,
        accessibility_check_passed,
    )


def _calculate_validation_score(
    *,
    generated_file_count: int,
    issue_count: int,
    error_count: int,
    warning_count: int,
) -> float:
    """
    Calculate a simple, bounded validation score.

    This is an internal quality signal, not a compliance certification.
    """
    if generated_file_count == 0:
        return 0.0

    base_score = 1.0

    error_penalty = min(0.8, error_count * 0.2)
    warning_penalty = min(0.2, warning_count * 0.05)

    return round(max(0.0, base_score - error_penalty - warning_penalty), 2)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def validation_node(
    state: AgenticState,
) -> AgenticState:
    """
    Validate generated workflow artifacts before review or publication.

    Args:
        state: Current LangGraph workflow state.

    Returns:
        State updates containing validation status, outcome, issues, score,
        quality-gate results, logs, and error information.

    Raises:
        WorkflowStateError: If required workflow inputs are missing.
    """
    execution_id = str(state.get("execution_id") or "unknown-execution")
    brief_id = str(state.get("brief_id") or "unknown-brief")
    generated_files = state.get("generated_code_files") or {}

    bind_workflow_context(
        execution_id,
        workflow_step=WorkflowStep.VALIDATION,
        tenant_id=state.get("tenant_id"),
        user_id=state.get("user_id"),
    )

    logger.info(
        "validation_started",
        brief_id=brief_id,
        generated_file_count=len(generated_files),
    )

    if not generated_files:
        raise WorkflowStateError(
            "Validation cannot continue without generated code files."
        )

    issues: list[dict[str, Any]] = []

    if len(generated_files) > MAX_GENERATED_FILE_COUNT:
        _add_issue(
            issues,
            severity="error",
            code="too_many_files",
            message=(
                f"Workflow generated {len(generated_files)} files, exceeding "
                f"the limit of {MAX_GENERATED_FILE_COUNT}."
            ),
        )

    total_characters = sum(
        len(content)
        for content in generated_files.values()
        if isinstance(content, str)
    )

    if total_characters > MAX_TOTAL_GENERATED_CHARS:
        _add_issue(
            issues,
            severity="error",
            code="generated_output_too_large",
            message=(
                "Total generated code exceeds the maximum allowed size."
            ),
        )

    _validate_file_paths(generated_files, issues)
    _validate_file_contents(generated_files, issues)

    (
        static_analysis_passed,
        security_scan_passed,
        accessibility_check_passed,
    ) = _validate_quality_gates(state, issues)

    error_count = sum(
        1 for issue in issues if issue.get("severity") == "error"
    )

    warning_count = sum(
        1 for issue in issues if issue.get("severity") == "warning"
    )

    if error_count > 0:
        validation_outcome = "failed"
        validation_status = JobStatus.FAILED
        recoverable_error = True
    elif warning_count > 0:
        validation_outcome = "warning"
        validation_status = JobStatus.COMPLETED
        recoverable_error = False
    else:
        validation_outcome = "passed"
        validation_status = JobStatus.COMPLETED
        recoverable_error = False

    validation_score = _calculate_validation_score(
        generated_file_count=len(generated_files),
        issue_count=len(issues),
        error_count=error_count,
        warning_count=warning_count,
    )

    log_level = "ERROR" if error_count else (
        "WARNING" if warning_count else "INFO"
    )

    log_message = (
        f"Validation completed with outcome '{validation_outcome}'. "
        f"Found {error_count} errors and {warning_count} warnings."
    )

    logger.log(
        logging.ERROR if error_count else (
            logging.WARNING if warning_count else logging.INFO
        ),
        "validation_completed",
        brief_id=brief_id,
        outcome=validation_outcome,
        error_count=error_count,
        warning_count=warning_count,
        validation_score=validation_score,
        generated_file_count=len(generated_files),
    )

    state_updates: AgenticState = {
        "current_step": WorkflowStep.VALIDATION,
        "validation_status": validation_status,
        "validation_outcome": validation_outcome,
        "validation_issues": issues,
        "validation_score": validation_score,
        "static_analysis_passed": static_analysis_passed,
        "security_scan_passed": security_scan_passed,
        "accessibility_check_passed": accessibility_check_passed,
        "validation_completed_at": _utc_now_iso(),
        "execution_logs": [
            f"{log_level}: {log_message}"
        ],
    }

    if error_count > 0:
        state_updates.update(
            {
                "current_error": (
                    "Generated artifacts failed validation."
                ),
                "error_step": WorkflowStep.VALIDATION,
                "error_code": "validation_failed",
                "recoverable_error": recoverable_error,
            }
        )
    else:
        state_updates.update(
            {
                "current_error": None,
                "error_step": None,
                "error_code": None,
            }
        )

    return state_updates


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()