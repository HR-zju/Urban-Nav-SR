"""Compatibility shim for the relocated data module.

New code should import mcp_server.data.openloop.benchmark.
"""
from mcp_server.data.openloop.benchmark import *  # noqa: F403

