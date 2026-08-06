"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.vqa.review.
"""
from mcp_server.data.vqa.review import *  # noqa: F403

