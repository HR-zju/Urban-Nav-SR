"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.legacy.reasoning_v2.views.
"""
from mcp_server.data.legacy.reasoning_v2.views import *  # noqa: F403

