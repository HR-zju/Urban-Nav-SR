"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.stage_ii.sources.
"""
from mcp_server.data.stage_ii.sources import *  # noqa: F403

