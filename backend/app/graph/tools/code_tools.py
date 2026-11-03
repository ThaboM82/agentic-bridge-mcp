"""
Generated-code safety and quality tools.

These helpers perform lightweight, deterministic checks. They do not replace
static analysis, security scanning, tests, or human review.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

from app.core.exceptions import ValidationError
from app.graph.state import AgenticState


logger = logging.getLogger("agentic_bridge.graph.tools.code")

MAX_GENERATED_FILES = 100
MAX_FILE_CONTENT_CHARS = 250_000
MAX_TOTAL_CONTENT_CHARS = 1_000_000

DANGEROUS_PATTERNS = [
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bos\.system\s*\(",
    r"\bsubprocess\.(?:run|call|Popen)\s*\(",
    r"\bshutil\.rmtree\s*\(",
    r"\brm\s+-rf\b",
]

DANGEROUS_PATTERN_COMPILED = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in DANGEROUS_PATTERNS
]


def validate_generated_files(
    state: AgenticState,
    *,
    max_files: int = MAX_GENERATED_FILES,
) -> list[dict[str, Any]]:
    """
    Validate generated code files in workflow state.

    Returns:
        A list of validation issues. An empty list means no issues were found.
    """
    generated_files = state.get("generated_code_files", {})
    issues: list[dict[str, Any]] = []

    if not generated_files:
        issues.append(
            {
                "severity": "error",
                "code": "no_generated_files",
                "message": "Workflow did not generate any code files.",
            }
        )

        return issues

    if len(generated_files) > max_files:
        issues.append(
            {
                "severity": "error",
                "code": "too_many_files",
                "message": (
                    f"Workflow generated {len(generated_files)} files, "
                    f"exceeding the limit of {max_files}."
                ),
            }
        )

    total_characters = 0

    for file_path, content in generated_files.items():
        normalized_path = str(file_path).replace("\\", "/")

        if normalized_path.startswith("/") or ".." in normalized_path.split("/"):
            issues.append(
                {
                    "severity": "error",
                    "code": "unsafe_file_path",
                    "message": f"Generated file path is unsafe: {file_path}",
                }
            )

        if len(content) > MAX_FILE_CONTENT_CHARS:
            issues.append(
                {
                    "severity": "error",
                    "code": "file_too_large",
                    "message": f"Generated file exceeds size limit: {file_path}",
                }
            )

        total_characters += len(content)

        if normalized_path.endswith(".py"):
            try:
                ast.parse(content)
            except SyntaxError as exc:
                issues.append(
                    {
                        "severity": "error",
                        "code": "python_syntax_error",
                        "message": (
                            f"Python syntax error in {file_path}: "
                            f"{exc.msg}"
                        ),
                    }
                )

        for pattern in DANGEROUS_PATTERN_COMPILED:
            if pattern.search(content):
                issues.append(
                    {
                        "severity": "warning",
                        "code": "potentially_dangerous_code",
                        "message": (
                            f"Potentially dangerous pattern detected in "
                            f"{file_path}."
                        ),
                    }
                )

    if total_characters > MAX_TOTAL_CONTENT_CHARS:
        issues.append(
            {
                "severity": "error",
                "code": "generated_output_too_large",
                "message": "Total generated code exceeds the allowed size.",
            }
        )

    logger.info(
        "generated_code_validated",
        file_count=len(generated_files),
        issue_count=len(issues),
    )

    return issues


def has_blocking_issues(issues: list[dict[str, Any]]) -> bool:
    """Determine whether validation issues include blocking errors."""
    return any(issue.get("severity") == "error" for issue in issues)