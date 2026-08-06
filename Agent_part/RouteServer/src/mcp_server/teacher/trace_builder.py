"""Compatibility import for the moved agent-tools teacher wrapper.

New code should import from ``mcp_server.teacher.agent_tools``. This module remains
for existing tests and downstream callers during the staged directory migration.
"""

from mcp_server.teacher.agent_tools.trace_builder import *  # noqa: F403
from mcp_server.teacher.agent_tools.trace_builder import (
    RouteServerAgentTeacher,
    RouteServerTeacherConfig,
)

__all__ = ["RouteServerAgentTeacher", "RouteServerTeacherConfig"]
