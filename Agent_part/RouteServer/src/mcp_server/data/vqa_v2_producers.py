"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.vqa.producers.
"""
from mcp_server.data.vqa.producers import *  # noqa: F403

