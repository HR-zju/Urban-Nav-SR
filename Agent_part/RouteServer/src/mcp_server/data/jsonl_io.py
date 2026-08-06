"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.io.jsonl.
"""
from mcp_server.data.io.jsonl import *  # noqa: F403

