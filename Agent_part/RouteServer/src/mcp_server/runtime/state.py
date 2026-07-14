"""Serializable LangGraph state for hierarchical navigation episodes."""

from __future__ import annotations

from typing import Any, TypedDict


class EpisodeGraphState(TypedDict, total=False):
    """Checkpoint-safe state exchanged by episode graph nodes."""

    episode_id: str
    session_id: str | None
    tick_id: int
    initial_context: dict[str, Any]
    observation: dict[str, Any]
    observation_info: dict[str, Any]
    history: list[dict[str, Any]]
    episode_trace: dict[str, Any]
    system2_action: dict[str, Any]
    last_tool_response: dict[str, Any]
    pending_user_request: dict[str, Any]
    user_response: Any
    system1_output: dict[str, Any]
    safety_result: dict[str, Any]
    environment_result: dict[str, Any]
    tool_call_count: int
    safety_retry_count: int
    done: bool
    terminal_status: str
    stop_reason: str
