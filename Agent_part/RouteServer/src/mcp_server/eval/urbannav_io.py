"""Compatibility import for the relocated evaluation module.

New code should import `mcp_server.eval.urbannav.urbannav_io`.
"""
from mcp_server.eval.urbannav.urbannav_io import *  # noqa: F403
from mcp_server.eval.urbannav.urbannav_io import iter_json_records  # noqa: F401

