"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.canonical.reasoning_v4_source.
"""
from mcp_server.data.canonical.reasoning_v4_source import *  # noqa: F403
from mcp_server.data.canonical.reasoning_v4_source import _bev_route_evidence_v4  # noqa: F401

