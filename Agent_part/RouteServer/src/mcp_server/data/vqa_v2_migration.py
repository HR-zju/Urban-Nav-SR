"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.vqa.migration.
"""
from mcp_server.data.vqa.migration import *  # noqa: F403

