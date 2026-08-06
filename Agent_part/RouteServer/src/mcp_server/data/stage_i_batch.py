"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.stage_i.batch.
"""
from mcp_server.data.stage_i.batch import *  # noqa: F403

