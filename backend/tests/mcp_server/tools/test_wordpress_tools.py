from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.mcp_server.auth import MCPPrincipal
from app.mcp_server.tools.wordpress_tools import (
    register_wordpress_tools,
)


@pytest.fixture
def principal():
    return MCPPrincipal(
        user_id=str(uuid4()),
        tenant_id=str(uuid4()),
        role="content_editor",
        scopes=frozenset(
            {
                "wordpress:read",
                "wordpress:write",
                "wordpress:publish",
            }
        ),
    )


@pytest.fixture
def wordpress_service():
    with patch(
        "app.services.wordpress_service.WordPressService"
    ) as mock_service_class:
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        yield mock_service


def test_register_wordpress_tools():
    from app.mcp_server.server import mcp

    register_wordpress_tools(mcp)

    tool_names = [
        tool.name
        for tool in mcp._tool_manager.list_tools()
    ]

    assert "search_wordpress_content" in tool_names
    assert "create_wordpress_draft" in tool_names
    assert "update_wordpress_draft" in tool_names
    assert "request_wordpress_publication" in tool_names
