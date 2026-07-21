"""Closed-loop UrbanNav runtime exports with optional dependency isolation.

Offline data tools import :mod:`mcp_server.runtime.system1_contract` but do not
need LangGraph.  Lazy package exports keep that lightweight path usable while
preserving the public runtime API for full RouteServer environments.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from mcp_server.runtime.system1_contract import (
    SYSTEM1_COORDINATE_FRAME_CURRENT_EGO_FLU,
    SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1,
    canonical_trajectory_policy,
    canonicalize_trajectory_ego_t,
)


_LAZY_EXPORTS = {
    "EpisodeRuntime": ("mcp_server.runtime.graph", "EpisodeRuntime"),
    "EpisodeRuntimeConfig": ("mcp_server.runtime.graph", "EpisodeRuntimeConfig"),
    "build_episode_graph": ("mcp_server.runtime.graph", "build_episode_graph"),
    "EventReduction": ("mcp_server.runtime.events", "EventReduction"),
    "ExternalEventStore": ("mcp_server.runtime.events", "ExternalEventStore"),
    "InMemoryExternalEventStore": ("mcp_server.runtime.events", "InMemoryExternalEventStore"),
    "reduce_external_event": ("mcp_server.runtime.events", "reduce_external_event"),
    "EnvironmentExecutionManager": ("mcp_server.runtime.execution", "EnvironmentExecutionManager"),
    "ExecutionFeedback": ("mcp_server.runtime.execution", "ExecutionFeedback"),
    "HaltFeedback": ("mcp_server.runtime.execution", "HaltFeedback"),
    "ExecutionLease": ("mcp_server.runtime.execution", "ExecutionLease"),
    "bind_system1_execution_context": ("mcp_server.runtime.execution", "bind_system1_execution_context"),
    "validate_system1_freshness": ("mcp_server.runtime.execution", "validate_system1_freshness"),
    "EnvironmentStepResult": ("mcp_server.runtime.interfaces", "EnvironmentStepResult"),
    "ObservationResult": ("mcp_server.runtime.interfaces", "ObservationResult"),
    "SafetyResult": ("mcp_server.runtime.interfaces", "SafetyResult"),
    "System1Output": ("mcp_server.runtime.interfaces", "System1Output"),
    "System2Action": ("mcp_server.runtime.interfaces", "System2Action"),
    "System2ActionType": ("mcp_server.runtime.interfaces", "System2ActionType"),
    "TerminalDecision": ("mcp_server.runtime.interfaces", "TerminalDecision"),
    "TerminalOutcome": ("mcp_server.runtime.interfaces", "TerminalOutcome"),
    "TerminalValidationResult": ("mcp_server.runtime.interfaces", "TerminalValidationResult"),
    "EpisodeRuntimeEventService": ("mcp_server.runtime.service", "EpisodeRuntimeEventService"),
    "LiveToolExecutorAdapter": ("mcp_server.runtime.tool_adapters", "LiveToolExecutorAdapter"),
    "UrbanNavReplayToolExecutorAdapter": (
        "mcp_server.runtime.tool_adapters",
        "UrbanNavReplayToolExecutorAdapter",
    ),
    "bind_tool_call_context": ("mcp_server.runtime.tool_adapters", "bind_tool_call_context"),
    "validate_tool_call_context": ("mcp_server.runtime.tool_adapters", "validate_tool_call_context"),
    "RuntimeTerminalValidator": ("mcp_server.runtime.terminal", "RuntimeTerminalValidator"),
}


def __getattr__(name: str) -> Any:
    """Load heavy runtime symbols only when callers request them."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "System1Output",
    "SYSTEM1_COORDINATE_FRAME_CURRENT_EGO_FLU",
    "SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1",
    "canonical_trajectory_policy",
    "canonicalize_trajectory_ego_t",
    *sorted(_LAZY_EXPORTS),
]
