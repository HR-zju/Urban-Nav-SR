"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.stage_i.provenance.
"""
from mcp_server.data.stage_i.provenance import *  # noqa: F403

