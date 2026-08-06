"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.vqa.audit.
"""
from mcp_server.data.vqa.audit import *  # noqa: F403

