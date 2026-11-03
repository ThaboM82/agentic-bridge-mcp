"""
LangGraph publishing node.

Prepares approved generated code for publication. This node validates artifact
integrity, creates artifact references, and marks the workflow ready for
human approval or final publication.

Security and reliability rules:
- Never execute generated code.
- Never publish unapproved workflows when approval is required.
- Never store full file contents in artifact metadata.
- Use checksums and bounded artifact references for traceability.
- Return only new workflow-state fields from the node.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from app.core.constants import MAX_GENERATED_FILE_COUNT
from app.core.enums import ApprovalStatus, JobStatus, WorkflowStep
from app.core.exceptions import (
    WorkflowStateError,
    WorkflowExecutionError,
)
from app.core.logging_context import bind_workflow_context
from app.graph.state import AgenticState
from app.graph.tools.code_tools import (
    has_blocking_issues,
    validate_generated_files,
)


logger = logging.getLogger("agentic_bridge.graph.nodes.publishing")

MAX_ARTIFACT_METADATA_CHARS = 8_000


def _calculate_checksum(content: str) -> str:
    """Calculate a deterministic SHA-256 checksum for generated content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _build_artifact_metadata(
    generated_files: dict[str, str],
) -> dict[str, Any]:
    """
    Build bounded, non-sensitive artifact metadata.

    Stores file names, sizes, and checksums—not source-code contents.
    """
    files_metadata: list[dict[str, Any]] = []

    for filename, content in generated_files.items():
        files_metadata.append(
            {
                "filename": filename,
                "size_chars": len(content),
                "checksum": _calculate_checksum(content),
            }
        )

    metadata = {
        "file_count": len(generated_files),
        "total_size_chars": sum(
            len(content) for content in generated_files.values()
        ),
        "files": files_metadata,
    }

    serialized_length = len(str(metadata))

    if serialized_length > MAX_ARTIFACT_METADATA_CHARS:
        # Keep the summary if the per-file manifest becomes too large.
        metadata["files"] = []
        metadata["manifest_truncated"] = True

    return metadata


async def publishing_node(
    state: AgenticState,
) -> AgenticState:
    """
    Validate generated artifacts and prepare them for review or publication.

    Args:
        state: Current LangGraph workflow state.

    Returns:
        State updates containing artifact references, review status,
        publishing status, logs, and error information.

    Raises:
        WorkflowStateError: If required workflow inputs are missing.
        WorkflowExecutionError: If artifact validation fails.
    """
    execution_id = str(state.get("execution_id") or "unknown-execution")
    brief_id = str(state.get("brief_id") or "unknown-brief")
    generated_files = state.get("generated_code_files") or {}

    bind_workflow_context(
        execution_id,
        workflow_step=WorkflowStep.PUBLISHING,
    )

    logger.info(
        "publishing_started",
        brief_id=brief_id,
        generated_file_count=len(generated_files),
    )

    if not generated_files:
        raise WorkflowStateError(
            "Publishing cannot continue without generated code files."
        )

    if len(generated_files) > MAX_GENERATED_FILE_COUNT:
        raise WorkflowStateError(
            "Publishing cannot continue because the generated file count "
            "exceeds the allowed limit."
        )

    validation_issues = validate_generated_files(state)

    if has_blocking_issues(validation_issues):
        logger.warning(
            "publishing_validation_failed",
            brief_id=brief_id,
            issue_count=len(validation_issues),
        )

        return {
            "current_step": WorkflowStep.PUBLISHING,
            "publishing_status": JobStatus.FAILED,
            "review_status": "rejected",
            "current_error": (
                "Publishing failed because generated artifacts failed "
                "validation."
            ),
            "error_step": WorkflowStep.PUBLISHING,
            "error_code": "artifact_validation_failed",
            "recoverable_error": True,
            "validation_issues": validation_issues,
            "execution_logs": [
                "ERROR: Generated artifacts failed pre-publication validation."
            ],
        }

    artifact_metadata = _build_artifact_metadata(generated_files)

    artifact_id = f"artifact_{execution_id}"

    artifact_reference = {
        "artifact_id": artifact_id,
        "artifact_type": "code_bundle",
        "uri": None,
        "checksum": _calculate_checksum(
            "".join(
                generated_files[filename]
                for filename in sorted(generated_files)
            )
        ),
        "created_at": _utc_now_iso(),
        "metadata": artifact_metadata,
    }

    approval_required = bool(state.get("approval_required", False))

    if approval_required:
        review_status = "pending_approval"
        approval_status = ApprovalStatus.PENDING
        publishing_status = JobStatus.PROCESSING
        log_message = (
            "SUCCESS: Publishing preparation completed. "
            f"Prepared {len(generated_files)} files for human approval."
        )
    else:
        review_status = "approved"
        approval_status = ApprovalStatus.APPROVED
        publishing_status = JobStatus.COMPLETED
        log_message = (
            "SUCCESS: Publishing preparation completed. "
            f"Prepared {len(generated_files)} files for publication."
        )

    logger.info(
        "publishing_preparation_completed",
        brief_id=brief_id,
        artifact_id=artifact_id,
        generated_file_count=len(generated_files),
        approval_required=approval_required,
    )

    return {
        "current_step": WorkflowStep.PUBLISHING,
        "publishing_status": publishing_status,
        "review_status": review_status,
        "approval_status": approval_status,
        "approval_required": approval_required,
        "artifacts": [artifact_reference],
        "published_artifact_ids": [artifact_id],
        "current_error": None,
        "error_step": None,
        "error_code": None,
        "execution_logs": [log_message],
    }


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()