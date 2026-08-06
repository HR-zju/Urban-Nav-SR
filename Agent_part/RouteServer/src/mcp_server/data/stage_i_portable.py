"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.stage_i.portable.
"""
from mcp_server.data.stage_i.portable import *  # noqa: F403

