"""
LangGraph code-generation node.

Generates a bounded set of source-code artifacts from validated requirements
and design context. The node validates model output before placing it into
workflow state.

Security and reliability rules:
- Never execute generated code.
- Never store credentials, secrets, or raw provider responses in state.
- Bound file count and file size.
- Reject unsafe file paths and dangerous code patterns.
- Return only new workflow-state fields from the node.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator

from app.core.constants import (
    MAX_GENERATED_FILE_COUNT,
    MAX_GENERATED_FILE_SIZE_CHARS,
)
from app.core.enums import WorkflowStep
from app.core.exceptions import (
    ExternalProviderError,
    ProviderTimeoutError,
    ValidationError,
    WorkflowStateError,
)
from app.core.logging_context import bind_workflow_context
from app.core.utils import truncate_text
from app.graph.prompts.code_generation import (
    CODE_GENERATION_PROMPT_VERSION,
    build_code_generation_messages,
)
from app.graph.state import AgenticState
from app.services.llm_client import LLMProviderFactory


logger = logging.getLogger("agentic_bridge.graph.nodes.code_generation")

MAX_REQUIREMENTS = 50
MAX_REQUIREMENT_CHARS = 500
MAX_TOTAL_GENERATED_CHARS = 1_000_000
MAX_FILENAME_LENGTH = 255

UNSAFE_PATH_PATTERN = re.compile(
    r"(^/|^\.\.?/|(^|/)\.\.(/|$))"
)

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


class GeneratedCodeFile(BaseModel):
    """A single generated source-code file."""

    filename: str = Field(
        description=(
            "Relative repository path, such as "
            "'frontend/src/components/Dashboard.tsx' or "
            "'backend/app/api/example.py'."
        )
    )
    content: str = Field(
        description="Complete source-code content for the file."
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Validate a safe relative file path."""
        normalized_path = value.strip().replace("\\", "/")

        if not normalized_path:
            raise ValueError("Filename is required.")

        if len(normalized_path) > MAX_FILENAME_LENGTH:
            raise ValueError(
                f"Filename may not exceed {MAX_FILENAME_LENGTH} characters."
            )

        if UNSAFE_PATH_PATTERN.search(normalized_path):
            raise ValueError(
                "Filename must be a safe relative path."
            )

        return normalized_path

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Validate bounded non-empty source content."""
        normalized_content = value.strip()

        if not normalized_content:
            raise ValueError("File content cannot be empty.")

        if len(normalized_content) > MAX_GENERATED_FILE_SIZE_CHARS:
            raise ValueError(
                f"File content may not exceed "
                f"{MAX_GENERATED_FILE_SIZE_CHARS} characters."
            )

        return normalized_content


class MultiFileCodeGenerationSchema(BaseModel):
    """Validated multi-file code-generation output."""

    files: list[GeneratedCodeFile] = Field(
        default_factory=list,
        description="Complete generated source-code files.",
    )
    implementation_notes: str | None = Field(
        default=None,
        description="Brief non-sensitive implementation summary.",
    )

    @field_validator("files")
    @classmethod
    def validate_files(
        cls,
        files: list[GeneratedCodeFile],
    ) -> list[GeneratedCodeFile]:
        """Validate file count, uniqueness, and total output size."""
        if not files:
            raise ValueError("At least one generated file is required.")

        if len(files) > MAX_GENERATED_FILE_COUNT:
            raise ValueError(
                f"Code generation may not produce more than "
                f"{MAX_GENERATED_FILE_COUNT} files."
            )

        seen_paths: set[str] = set()
        total_characters = 0

        for file in files:
            normalized_path = file.filename.lower()

            if normalized_path in seen_paths:
                raise ValueError(
                    f"Duplicate generated file path: {file.filename}"
                )

            seen_paths.add(normalized_path)
            total_characters += len(file.content)

            if total_characters > MAX_TOTAL_GENERATED_CHARS:
                raise ValueError(
                    "Total generated code exceeds the allowed size."
                )

        return files


def _build_requirements_text(
    requirements: list[str],
) -> str:
    """Create a bounded requirements block for the prompt."""
    bounded_requirements = requirements[:MAX_REQUIREMENTS]

    return "\n".join(
        f"- {truncate_text(str(requirement), MAX_REQUIREMENT_CHARS)}"
        for requirement in bounded_requirements
    )


def _build_design_context(
    design_tokens: dict[str, Any] | None,
) -> str:
    """Create a bounded, JSON-like design context block."""
    if not design_tokens:
        return "No design-system context was supplied."

    try:
        import json

        serialized = json.dumps(
            design_tokens,
            default=str,
            ensure_ascii=False,
            indent=2,
        )
    except (TypeError, ValueError):
        serialized = str(design_tokens)

    return truncate_text(serialized, 20_000)


def _contains_dangerous_pattern(content: str) -> bool:
    """Detect obviously dangerous generated-code patterns."""
    return any(
        pattern.search(content)
        for pattern in DANGEROUS_CODE_PATTERNS
    )


async def code_generation_node(
    state: AgenticState,
) -> AgenticState:
    """
    Generate validated multi-file source code.

    Args:
        state: Current LangGraph workflow state.

    Returns:
        State updates containing generated files, artifacts, logs, and
        provider telemetry.

    Raises:
        WorkflowStateError: If required workflow inputs are missing.
        ExternalProviderError: If the language-model provider fails.
    """
    execution_id = str(state.get("execution_id") or "unknown-execution")
    brief_id = str(state.get("brief_id") or "unknown-brief")
    extracted_requirements = state.get("extracted_requirements") or []
    design_tokens = state.get("design_tokens") or {}

    bind_workflow_context(
        execution_id,
        workflow_step=WorkflowStep.CODE_GENERATION,
    )

    logger.info(
        "code_generation_started",
        brief_id=brief_id,
        requirement_count=len(extracted_requirements),
    )

    if not extracted_requirements:
        raise WorkflowStateError(
            "Code generation cannot continue without extracted requirements."
        )

    if not design_tokens:
        raise WorkflowStateError(
            "Code generation cannot continue without design context."
        )

    try:
        llm = LLMProviderFactory.get_llm(temperature=0.2)

        parser = PydanticOutputParser(
            pydantic_object=MultiFileCodeGenerationSchema
        )

        messages = build_code_generation_messages(
            requirements=_build_requirements_text(extracted_requirements),
            design_context=_build_design_context(design_tokens),
            format_instructions=parser.get_format_instructions(),
        )

        logger.debug(
            "code_generation_llm_dispatched",
            brief_id=brief_id,
            requirement_count=len(extracted_requirements),
        )

        response = await llm.ainvoke(messages)

        parsed_code = parser.parse(response.content)

        generated_files: dict[str, str] = {}
        warnings: list[str] = []

        for generated_file in parsed_code.files:
            if _contains_dangerous_pattern(generated_file.content):
                warnings.append(
                    f"Potentially dangerous pattern detected in "
                    f"{generated_file.filename}."
                )

            generated_files[generated_file.filename] = generated_file.content

        if not generated_files:
            raise ValidationError(
                "The language model did not return any usable code files."
            )

        logger.info(
            "code_generation_completed",
            brief_id=brief_id,
            generated_file_count=len(generated_files),
            warning_count=len(warnings),
        )

        state_updates: AgenticState = {
            "current_step": WorkflowStep.CODE_GENERATION,
            "generated_code_files": generated_files,
            "code_generation_completed_at": _utc_now_iso(),
            "execution_logs": [
                (
                    "SUCCESS: Code generation completed. Generated "
                    f"{len(generated_files)} source files."
                )
            ],
            "current_error": None,
        }

        if warnings:
            state_updates["execution_logs"] = [
                *state_updates["execution_logs"],
                *[
                    f"WARNING: {warning}"
                    for warning in warnings
                ],
            ]

        if parsed_code.implementation_notes:
            state_updates["design_rationale"] = truncate_text(
                parsed_code.implementation_notes,
                2_000,
            )

        return state_updates

    except (ProviderTimeoutError, ExternalProviderError):
        logger.exception(
            "code_generation_provider_failure",
            brief_id=brief_id,
        )

        return {
            "current_step": WorkflowStep.CODE_GENERATION,
            "current_error": (
                "Code generation failed because the language-model provider "
                "was unavailable or timed out."
            ),
            "error_step": WorkflowStep.CODE_GENERATION,
            "error_code": "provider_timeout",
            "recoverable_error": True,
            "execution_logs": [
                "ERROR: Code generation failed due to a provider timeout."
            ],
        }

    except ValidationError as exc:
        logger.warning(
            "code_generation_validation_failed",
            brief_id=brief_id,
            error=str(exc),
        )

        return {
            "current_step": WorkflowStep.CODE_GENERATION,
            "current_error": (
                "Code generation produced invalid or unsafe output."
            ),
            "error_step": WorkflowStep.CODE_GENERATION,
            "error_code": "code_generation_validation_error",
            "recoverable_error": True,
            "execution_logs": [
                "ERROR: Code generation output failed validation."
            ],
        }

    except Exception:
        logger.exception(
            "code_generation_unexpected_failure",
            brief_id=brief_id,
        )

        return {
            "current_step": WorkflowStep.CODE_GENERATION,
            "current_error": (
                "Code generation failed due to an unexpected workflow error."
            ),
            "error_step": WorkflowStep.CODE_GENERATION,
            "error_code": "code_generation_error",
            "recoverable_error": False,
            "execution_logs": [
                "ERROR: Code generation failed unexpectedly."
            ],
        }


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()