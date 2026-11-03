"""
Figma-related request and response schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FigmaDesignContextRequest(BaseModel):
    """Request for workflow-safe Figma design context."""

    figma_file_key: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    query: str | None = Field(
        default=None,
        max_length=500,
    )

    limit: int = Field(
        default=20,
        ge=1,
        le=50,
    )


class FigmaDesignContextItem(BaseModel):
    """A bounded, workflow-safe Figma design-context item."""

    node_id: str = Field(
        ...,
        max_length=100,
    )

    name: str = Field(
        ...,
        max_length=500,
    )

    node_type: str = Field(
        ...,
        max_length=100,
    )

    summary: str | None = Field(
        default=None,
        max_length=2_000,
    )

    design_intent: str | None = Field(
        default=None,
        max_length=2_000,
    )

    component_name: str | None = Field(
        default=None,
        max_length=500,
    )

    width: float | None = None

    height: float | None = None


class FigmaDesignContextResponse(BaseModel):
    """Tenant-safe Figma design-context response."""

    figma_file_key: str = Field(
        ...,
        max_length=100,
    )

    items: list[FigmaDesignContextItem] = Field(
        default_factory=list,
    )

    truncated: bool = False


class FigmaFrameRequest(BaseModel):
    """Request for a bounded Figma frame summary."""

    figma_file_key: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    figma_node_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )


class FigmaFrameResponse(BaseModel):
    """Tenant-safe Figma frame summary."""

    figma_file_key: str = Field(
        ...,
        max_length=100,
    )

    node_id: str = Field(
        ...,
        max_length=100,
    )

    name: str = Field(
        ...,
        max_length=500,
    )

    node_type: str = Field(
        ...,
        max_length=100,
    )

    summary: str | None = Field(
        default=None,
        max_length=2_000,
    )

    design_intent: str | None = Field(
        default=None,
        max_length=2_000,
    )

    component_name: str | None = Field(
        default=None,
        max_length=500,
    )

    width: float | None = None

    height: float | None = None