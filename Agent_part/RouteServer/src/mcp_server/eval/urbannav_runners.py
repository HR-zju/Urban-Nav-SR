"""Compatibility import for the relocated evaluation module.

New code should import `mcp_server.eval.urbannav.urbannav_runners`.
"""
from mcp_server.eval.urbannav.urbannav_runners import *  # noqa: F403
from mcp_server.eval.urbannav.urbannav_runners import (  # noqa: F401
    _best_origin_matched_route,
    _effective_instruction,
)

