from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.mcp_server.auth import MCPPrincipal
from app.mcp_server.tools.figma_tools import (
    register_figma_tools,
)


@pytest.fixture
def principal():
    return MCPPrincipal(
        user_id=str(uuid4()),
        tenant_id=str(uuid4()),
        role="designer",
        scopes=frozenset({"figma:read"}),
    )


@pytest.fixture
def figma_service():
    with patch(
        "app.services.figma_service.FigmaService"
    ) as mock_service_class:
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        yield mock_service


def test_register_figma_tools():
    from app.mcp_server.server import mcp

    register_figma_tools(mcp)

    tool_names = [
        tool.name
        for tool in mcp._tool_manager.list_tools()
    ]

    assert "get_figma_design_context" in tool_names
    assert "get_figma_frame_summary" in tool_names
