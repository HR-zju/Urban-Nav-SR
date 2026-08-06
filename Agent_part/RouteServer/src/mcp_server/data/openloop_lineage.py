"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.openloop.lineage.
"""
from mcp_server.data.openloop.lineage import *  # noqa: F403

