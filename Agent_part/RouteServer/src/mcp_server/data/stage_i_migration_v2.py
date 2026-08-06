"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.stage_i.migration_v2.
"""
from mcp_server.data.stage_i.migration_v2 import *  # noqa: F403

