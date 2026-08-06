"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.stage_ii.legacy_direct.
"""
from mcp_server.data.stage_ii.legacy_direct import *  # noqa: F403

