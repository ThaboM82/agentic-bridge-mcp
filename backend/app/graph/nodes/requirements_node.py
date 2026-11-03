"""
LangGraph requirements-extraction node.

Extracts structured, actionable requirements from the normalized source brief
using the configured language-model provider. Output is strictly validated
before it enters workflow state.

Responsibilities:
- Validate and bound the ingestion brief.
- Build a versioned, sanitized requirements-extraction prompt.
- Invoke the configured language-model provider asynchronously.
- Validate structured model output.
- Record requirements, confidence, prompt version, and provider telemetry.
- Map provider and validation failures to recoverable workflow errors.

Non-goals:
- Do not retrieve vector context, generate code, enforce approval policy,
  publish artifacts, or implement retry/backoff policy in this node.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator

from app.core.constants import MAX_SEARCH_QUERY_LENGTH
from app.core.enums import WorkflowStep
from app.core.exceptions import (
    ExternalProviderError,
    ProviderTimeoutError,
    ValidationError,
    WorkflowStateError,
)
from app.core.logging_context import bind_workflow_context
from app.core.utils import truncate_text
from app.graph.prompts.requirements import (
    REQUIREMENTS_EXTRACTION_PROMPT_VERSION,
    RequirementsExtractionOutputSchema,
    build_requirements_extraction_messages,
    estimate_requirements_prompt_tokens,
    get_requirements_prompt_metadata,
)
from app.graph.state import AgenticState
from app.services.llm_client import LLMProviderFactory


logger = logging.getLogger("agentic_bridge.graph.nodes.requirements")

MAX_REQUIREMENTS = 100
MAX_REQUIREMENT_CHARS = 500
MAX_BRIEF_PROMPT_CHARS = MAX_SEARCH_QUERY_LENGTH
MAX_FORMAT_INSTRUCTIONS_CHARS = 10_000


# ---------------------------------------------------------------------------
# Prompt input helpers
# ---------------------------------------------------------------------------


def _build_brief_text(raw_brief_text: str | None) -> str:
    """
    Validate and bound the source brief text.

    Args:
        raw_brief_text: Normalized brief text produced by the ingestion node.

    Returns:
        Bounded brief text.

    Raises:
        WorkflowStateError: If the brief is missing or empty.
    """
    if not raw_brief_text or not raw_brief_text.strip():
        raise WorkflowStateError(
            "Requirements extraction cannot continue without a source brief."
        )

    return truncate_text(
        raw_brief_text.strip(),
        MAX_BRIEF_PROMPT_CHARS,
    )


def _normalize_requirement(
    requirement: str,
) -> str | None:
    """Normalize and bound a single extracted requirement."""
    if not isinstance(requirement, str):
        return None

    normalized_requirement = requirement.strip()

    if not normalized_requirement:
        return None

    return truncate_text(
        normalized_requirement,
        MAX_REQUIREMENT_CHARS,
    )


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def requirements_node(
    state: AgenticState,
) -> AgenticState:
    """
    Extract validated requirements from the source brief.

    Args:
        state: Current LangGraph workflow state.

    Returns:
        State updates containing extracted requirements, confidence,
        assumptions, prompt metadata, provider telemetry, logs, and error
        information.

    Raises:
        WorkflowStateError: If required workflow inputs are missing.
        ExternalProviderError: If the language-model provider fails.
    """
    execution_id = str(state.get("execution_id") or "unknown-execution")
    brief_id = str(state.get("brief_id") or "unknown-brief")
    raw_brief_text = state.get("raw_brief_text")

    bind_workflow_context(
        execution_id,
        workflow_step=WorkflowStep.REQUIREMENTS,
        tenant_id=state.get("tenant_id"),
        user_id=state.get("user_id"),
    )

    logger.info(
        "requirements_extraction_started",
        brief_id=brief_id,
        brief_length=len(raw_brief_text or ""),
    )

    try:
        bounded_brief = _build_brief_text(raw_brief_text)

        llm = LLMProviderFactory.get_llm(temperature=0.1)

        parser = PydanticOutputParser(
            pydantic_object=RequirementsExtractionOutputSchema
        )

        format_instructions = parser.get_format_instructions()

        messages = build_requirements_extraction_messages(
            brief_text=bounded_brief,
            format_instructions=format_instructions,
        )

        estimated_prompt_tokens = estimate_requirements_prompt_tokens(
            brief_text=bounded_brief,
            format_instructions=format_instructions,
        )

        prompt_metadata = get_requirements_prompt_metadata()

        logger.debug(
            "requirements_extraction_llm_dispatched",
            brief_id=brief_id,
            estimated_prompt_tokens=estimated_prompt_tokens,
            prompt_version=REQUIREMENTS_EXTRACTION_PROMPT_VERSION,
        )

        response = await llm.ainvoke(messages)

        parsed_requirements = parser.parse(response.content)

        normalized_requirements: list[str] = []
        seen_requirements: set[str] = set()

        for requirement in parsed_requirements.requirements:
            normalized_requirement = _normalize_requirement(requirement)

            if normalized_requirement is None:
                continue

            requirement_key = normalized_requirement.lower()

            if requirement_key not in seen_requirements:
                seen_requirements.add(requirement_key)
                normalized_requirements.append(normalized_requirement)

        if not normalized_requirements:
            raise ValidationError(
                "The language model did not return any usable requirements."
            )

        if len(normalized_requirements) > MAX_REQUIREMENTS:
            normalized_requirements = normalized_requirements[
                :MAX_REQUIREMENTS
            ]

        logger.info(
            "requirements_extraction_completed",
            brief_id=brief_id,
            requirement_count=len(normalized_requirements),
            confidence=parsed_requirements.confidence,
            prompt_version=REQUIREMENTS_EXTRACTION_PROMPT_VERSION,
        )

        return {
            "current_step": WorkflowStep.REQUIREMENTS,
            "extracted_requirements": normalized_requirements,
            "requirement_confidence": parsed_requirements.confidence,
            "requirement_metadata": {
                "assumptions": parsed_requirements.assumptions,
                "missing_information": parsed_requirements.missing_information,
                "prompt_version": REQUIREMENTS_EXTRACTION_PROMPT_VERSION,
                "prompt_name": prompt_metadata["prompt_name"],
            },
            "requirements_finalized": True,
            "design_prompt_version": None,
            "provider_name": state.get("provider_name"),
            "provider_model": state.get("provider_model"),
            "provider_token_usage": {
                "estimated_prompt_tokens": estimated_prompt_tokens,
            },
            "execution_logs": [
                (
                    "SUCCESS: Requirements extraction completed. Extracted "
                    f"{len(normalized_requirements)} actionable requirements."
                )
            ],
            "current_error": None,
            "error_step": None,
            "error_code": None,
        }

    except (ProviderTimeoutError, ExternalProviderError):
        logger.exception(
            "requirements_extraction_provider_failure",
            brief_id=brief_id,
        )

        return {
            "current_step": WorkflowStep.REQUIREMENTS,
            "current_error": (
                "Requirements extraction failed because the language-model "
                "provider was unavailable or timed out."
            ),
            "error_step": WorkflowStep.REQUIREMENTS,
            "error_code": "provider_timeout",
            "recoverable_error": True,
            "execution_logs": [
                "ERROR: Requirements extraction failed due to a provider timeout."
            ],
        }

    except ValidationError as exc:
        logger.warning(
            "requirements_extraction_validation_failed",
            brief_id=brief_id,
            error=str(exc),
        )

        return {
            "current_step": WorkflowStep.REQUIREMENTS,
            "current_error": (
                "Requirements extraction produced invalid or incomplete output."
            ),
            "error_step": WorkflowStep.REQUIREMENTS,
            "error_code": "requirements_validation_error",
            "recoverable_error": True,
            "execution_logs": [
                "ERROR: Requirements extraction output failed validation."
            ],
        }

    except Exception:
        logger.exception(
            "requirements_extraction_unexpected_failure",
            brief_id=brief_id,
        )

        return {
            "current_step": WorkflowStep.REQUIREMENTS,
            "current_error": (
                "Requirements extraction failed due to an unexpected "
                "workflow error."
            ),
            "error_step": WorkflowStep.REQUIREMENTS,
            "error_code": "requirements_extraction_error",
            "recoverable_error": False,
            "execution_logs": [
                "ERROR: Requirements extraction failed unexpectedly."
            ],
        }