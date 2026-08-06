"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.vqa.views.
"""
from mcp_server.data.vqa.views import *  # noqa: F403

