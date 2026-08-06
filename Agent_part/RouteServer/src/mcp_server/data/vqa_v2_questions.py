"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.vqa.questions.
"""
from mcp_server.data.vqa.questions import *  # noqa: F403

