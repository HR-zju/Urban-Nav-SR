"""Versioned runtime contracts shared by RouteServer tools and episodes."""

from mcp_server.contracts.episode import (
    EPISODE_TRACE_SCHEMA_VERSION,
    EpisodeEvent,
    EpisodeEventType,
    EpisodeStatus,
    EpisodeTraceV2,
)
from mcp_server.contracts.tool import (
    TOOL_CONTRACT_SCHEMA_VERSION,
    ToolCallV2,
    ToolResponseV2,
)
from mcp_server.contracts.trace_adapter import (
    agent_trace_to_episode_trace,
    attach_episode_trace_v2,
    merge_agent_traces_to_episode,
)

__all__ = [
    "EPISODE_TRACE_SCHEMA_VERSION",
    "EpisodeEvent",
    "EpisodeEventType",
    "EpisodeStatus",
    "EpisodeTraceV2",
    "TOOL_CONTRACT_SCHEMA_VERSION",
    "ToolCallV2",
    "ToolResponseV2",
    "agent_trace_to_episode_trace",
    "attach_episode_trace_v2",
    "merge_agent_traces_to_episode",
]
