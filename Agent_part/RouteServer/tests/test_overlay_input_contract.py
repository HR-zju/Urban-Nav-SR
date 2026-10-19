"""Protect the current default MCP registration boundary."""
from __future__ import annotations

import asyncio
import sys
import unittest


class DefaultMCPRegistrationTest(unittest.TestCase):
    def test_default_server_does_not_register_legacy_visual_prompt(self) -> None:
        """Loading the entrypoint registers current tools without legacy policy code."""
        from mcp_server.server import mcp

        names = {tool.name for tool in asyncio.run(mcp.list_tools())}
        self.assertEqual(len(names), 37)
        self.assertTrue({"geocode_address", "create_route_session", "analyze_image"} <= names)
        self.assertNotIn("run_zero_shot_visual_prompt_agent", names)
        for name in sys.modules:
            self.assertFalse(name.startswith(("mcp_server.brain", "mcp_server.som")))
        self.assertNotIn("mcp_server.services.visual_prompt_service", sys.modules)
