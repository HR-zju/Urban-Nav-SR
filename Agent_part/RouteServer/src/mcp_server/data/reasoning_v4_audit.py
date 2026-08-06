"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.canonical.reasoning_v4_audit.
"""
from mcp_server.data.canonical.reasoning_v4_audit import *  # noqa: F403
from mcp_server.data.canonical.reasoning_v4_audit import audit_reasoning_v4_pilot  # noqa: F401
