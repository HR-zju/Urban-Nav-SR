"""Stable core operations shared by MCP tools, HTTP APIs, and evaluators."""

import importlib
from typing import Any

from .map_ops import geocode, get_route, reverse_geocode, search_category, search_poi
from .schemas import CoreToolResult

_ROUTE_SESSION_EXPORTS = {
    "add_via_point",
    "create_route_session",
    "get_route_session_snapshot",
    "list_route_sessions",
    "remove_via_point",
    "replace_destination",
    "replan_route_session_from_origin",
}

__all__ = [
    "CoreToolResult",
    "add_via_point",
    "create_route_session",
    "geocode",
    "get_route",
    "get_route_session_snapshot",
    "list_route_sessions",
    "remove_via_point",
    "replace_destination",
    "replan_route_session_from_origin",
    "reverse_geocode",
    "search_category",
    "search_poi",
]


def __getattr__(name: str) -> Any:
    """Lazy-load route-session ops so map-only evaluators avoid provider deps."""

    if name in _ROUTE_SESSION_EXPORTS:
        module = importlib.import_module("mcp_server.core.route_session_ops")
        return getattr(module, name)
    raise AttributeError(name)
