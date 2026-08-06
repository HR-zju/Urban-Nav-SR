"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.stage_ii.views.
"""
from mcp_server.data.stage_ii.views import *  # noqa: F403

