"""Adapters from legacy Agent traces to the canonical episode event stream."""

from __future__ import annotations

from typing import Any, Iterable

from mcp_server.contracts.episode import EpisodeEventType, EpisodeStatus, EpisodeTraceV2


def agent_trace_to_episode_trace(
    trace: Any,
    *,
    episode_id: str | None = None,
    session_id: str | None = None,
    tick_id: int = 0,
    source: str = "agent_trace_adapter",
) -> EpisodeTraceV2:
    """Convert one legacy ``AgentTrace`` into ``EpisodeTraceV2``.

    Args:
        trace: Legacy trace exposing ``case_id``, ``turns`` and metadata fields.
        episode_id: Optional parent episode id.
        session_id: Optional route-session id.
        tick_id: Zero-based parent tick.
        source: Adapter source written into events.

    Returns:
        Canonical event-sourced episode trace.
    """

    resolved_episode_id = str(episode_id or getattr(trace, "case_id", "") or "episode:unknown")
    episode = EpisodeTraceV2(
        episode_id=resolved_episode_id,
        session_id=session_id,
        metadata={
            "legacy_schema_version": getattr(trace, "schema_version", None),
            "task_type": getattr(trace, "task_type", None),
            "tools": list(getattr(trace, "tools", None) or []),
            "artifacts": list(getattr(trace, "artifacts", None) or []),
            "supervision": dict(getattr(trace, "supervision", None) or {}),
            "model_run": dict(getattr(trace, "model_run", None) or {}),
        },
    )
    episode.append(
        EpisodeEventType.EPISODE_STARTED,
        tick_id=tick_id,
        source=source,
        payload={"case_id": getattr(trace, "case_id", None)},
    )
    for turn_index, turn in enumerate(getattr(trace, "turns", None) or []):
        role = str(getattr(turn, "role", "") or "")
        content = getattr(turn, "content", None)
        tool = getattr(turn, "tool", None)
        turn_metadata = dict(getattr(turn, "metadata", None) or {})
        event_type = _event_type_for_turn(role, turn_metadata)
        episode.append(
            event_type,
            tick_id=tick_id,
            source=source,
            payload={
                "role": role,
                "content": content,
                "tool": tool,
                "turn_index": turn_index,
                "turn_metadata": turn_metadata,
            },
        )
        if event_type == EpisodeEventType.TOOL_RESPONSE:
            response_payload = content if isinstance(content, dict) else {}
            result = response_payload.get("result") if isinstance(response_payload.get("result"), dict) else {}
            state_delta = result.get("state_delta") or {}
            if state_delta:
                episode.append(
                    EpisodeEventType.ROUTE_STATE_DELTA,
                    tick_id=tick_id,
                    source=source,
                    payload={"state_delta": state_delta, "call_id": response_payload.get("call_id")},
                )

    trace_metadata = dict(getattr(trace, "metadata", None) or {})
    if _trace_waits_for_user(trace_metadata):
        episode.status = EpisodeStatus.WAITING_FOR_USER
        episode.stop_reason = "user_interaction"
        episode.append(
            EpisodeEventType.USER_INTERRUPT,
            tick_id=tick_id,
            source=source,
            payload={"dialogue_state": trace_metadata.get("dialogue_state") or {}},
        )
    else:
        episode.finish(
            status=EpisodeStatus.SUCCEEDED,
            stop_reason="legacy_trace_completed",
            tick_id=tick_id,
            source=source,
        )
    return episode


def attach_episode_trace_v2(
    trace: Any,
    *,
    episode_id: str | None = None,
    session_id: str | None = None,
    tick_id: int = 0,
    source: str,
) -> EpisodeTraceV2:
    """Attach a canonical episode view to legacy trace metadata.

    Args:
        trace: Mutable legacy ``AgentTrace``.
        episode_id: Optional parent episode id.
        session_id: Optional route-session id.
        tick_id: Parent tick id.
        source: Adapter source.

    Returns:
        Attached ``EpisodeTraceV2``.
    """

    episode = agent_trace_to_episode_trace(
        trace,
        episode_id=episode_id,
        session_id=session_id,
        tick_id=tick_id,
        source=source,
    )
    trace.metadata["episode_trace_v2"] = episode.model_dump(mode="json")
    return episode


def merge_agent_traces_to_episode(
    traces: Iterable[Any],
    *,
    episode_id: str,
    session_id: str | None,
    stop_reason: str,
    source: str,
) -> EpisodeTraceV2:
    """Merge ordered single-tick traces into one session episode.

    Args:
        traces: Ordered legacy traces.
        episode_id: Parent episode id.
        session_id: Optional route-session id.
        stop_reason: Session terminal or interrupt reason.
        source: Adapter source.

    Returns:
        One causally ordered multi-tick trace.
    """

    merged = EpisodeTraceV2(episode_id=episode_id, session_id=session_id)
    merged.append(EpisodeEventType.EPISODE_STARTED, tick_id=0, source=source, payload={})
    last_tick = 0
    waits_for_user = False
    for tick_id, trace in enumerate(traces):
        last_tick = tick_id
        converted = agent_trace_to_episode_trace(
            trace,
            episode_id=episode_id,
            session_id=session_id,
            tick_id=tick_id,
            source=source,
        )
        waits_for_user = waits_for_user or converted.status == EpisodeStatus.WAITING_FOR_USER
        for event in converted.events:
            if event.event_type in {EpisodeEventType.EPISODE_STARTED, EpisodeEventType.EPISODE_FINISHED}:
                continue
            merged.append(
                event.event_type,
                tick_id=event.tick_id,
                source=event.source,
                payload=event.payload,
                metadata=event.metadata,
            )
    if waits_for_user:
        merged.status = EpisodeStatus.WAITING_FOR_USER
        merged.stop_reason = stop_reason
    else:
        terminal_status = EpisodeStatus.TRUNCATED if stop_reason == "max_ticks" else EpisodeStatus.SUCCEEDED
        merged.finish(status=terminal_status, stop_reason=stop_reason, tick_id=last_tick, source=source)
    return merged


def _event_type_for_turn(role: str, metadata: dict[str, Any]) -> EpisodeEventType:
    """Map one legacy message role to a canonical event type."""

    if role == "system":
        return EpisodeEventType.SYSTEM_MESSAGE
    if role == "assistant":
        return EpisodeEventType.MODEL_OUTPUT
    if role == "tool_call":
        return EpisodeEventType.TOOL_CALL
    if role == "tool_response":
        return EpisodeEventType.TOOL_RESPONSE
    if metadata.get("stage2_step") == "trajectory_readout_prompt":
        return EpisodeEventType.READOUT_REQUESTED
    return EpisodeEventType.OBSERVATION_RECEIVED


def _trace_waits_for_user(metadata: dict[str, Any]) -> bool:
    """Return whether legacy runtime metadata represents an HITL interrupt."""

    if metadata.get("runtime_state") == "WAITING_FOR_USER":
        return True
    dialogue_state = metadata.get("dialogue_state")
    return isinstance(dialogue_state, dict) and bool(dialogue_state.get("pending_user_interaction"))
