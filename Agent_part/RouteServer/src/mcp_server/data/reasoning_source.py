"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.legacy.reasoning_v2.source.
"""
from mcp_server.data.legacy.reasoning_v2.source import *  # noqa: F403

