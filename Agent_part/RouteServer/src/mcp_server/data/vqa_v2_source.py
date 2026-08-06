"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.vqa.source.
"""
from mcp_server.data.vqa.source import *  # noqa: F403

