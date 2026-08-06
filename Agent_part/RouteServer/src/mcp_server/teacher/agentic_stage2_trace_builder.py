"""Compatibility import for the moved agent-tools teacher builder.

New code should import from ``mcp_server.teacher.agent_tools``. This module remains
for existing tests and downstream callers during the staged directory migration.
"""

from mcp_server.teacher.agent_tools.agentic_stage2_trace_builder import *  # noqa: F403
from mcp_server.teacher.agent_tools.agentic_stage2_trace_builder import (
    AgenticStage2TraceBuilder,
    AgenticStage2TraceBuilderConfig,
)

__all__ = ["AgenticStage2TraceBuilder", "AgenticStage2TraceBuilderConfig"]
