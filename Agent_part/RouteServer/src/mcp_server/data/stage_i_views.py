"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.stage_i.views.
"""
from mcp_server.data.stage_i.views import *  # noqa: F403

