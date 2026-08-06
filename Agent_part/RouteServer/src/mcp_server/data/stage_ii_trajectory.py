"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.stage_ii.trajectory.
"""
from mcp_server.data.stage_ii.trajectory import *  # noqa: F403

