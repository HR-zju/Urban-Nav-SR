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
    observation_id: str
    observation_info: dict[str, Any]
    history: list[dict[str, Any]]
    episode_trace: dict[str, Any]
    task_state: dict[str, Any]
    dialogue_state: dict[str, Any]
    control_state: dict[str, Any]
    latest_external_event: dict[str, Any]
    event_route: str
    decision_id: str
    system2_action: dict[str, Any]
    last_tool_response: dict[str, Any]
    pending_user_request: dict[str, Any]
    user_response: Any
    system1_output: dict[str, Any]
    executable_system1_output: dict[str, Any]
    safety_result: dict[str, Any]
    execution_lease: dict[str, Any]
    execution_interrupted: bool
    environment_result: dict[str, Any]
    tool_call_count: int
    total_tool_call_count: int
    safety_retry_count: int
    terminal_rejection_count: int
    terminal_rejected: bool
    terminal_validation: dict[str, Any]
    done: bool
    terminal_status: str | None
    stop_reason: str | None
