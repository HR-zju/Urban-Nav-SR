"""RouteServer adapter for the standalone UrbanNav AgentTools package.

This module is deliberately not imported by the AgentTools core. RouteServer
owns its tool registry and supplies the explicit definitions at this boundary.
"""

from __future__ import annotations

from typing import Any

from urbannav_agent_tools.export import trace_to_training_row
from urbannav_agent_tools.schemas import AgentStage2Case, AgentTrace

from mcp_server.agent.schemas import (
    AgentStage2Case as RouteServerStage2Case,
    AgentTrace as RouteServerTrace,
)
from mcp_server.agent.tool_schema import get_tool_definitions


def export_route_trace_to_training_row(
    trace: RouteServerTrace,
    *,
    case: RouteServerStage2Case | None = None,
) -> dict[str, Any]:
    """Convert a RouteServer trace through the standalone AgentTools core.

    Args:
        trace: RouteServer-owned trace instance.
        case: Optional RouteServer case used for image and supervision data.

    Returns:
        Generic ``messages/images/tools`` training row.
    """

    package_trace = AgentTrace.from_dict(trace.to_dict())
    package_case = (
        AgentStage2Case.from_dict(case.to_dict()) if case is not None else None
    )
    return trace_to_training_row(
        package_trace,
        tool_definitions=get_tool_definitions(),
        case=package_case,
    )
