"""Externally-driven frozen episode service for ms-swift rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcp_server.agent.tool_parser import parse_agent_action, parse_system2_action_text
from mcp_server.contracts.task import ExternalUserEvent
from mcp_server.eval.episode.schema import FrozenEpisodeCase
from mcp_server.runtime.service import EpisodeEventSubmitResponse, EpisodeResetResponse, EpisodeStepResponse


@dataclass
class _ActiveFrozenEpisode:
    """Mutable cursor over one immutable case.

    Args:
        case: Frozen source case.
        transition_index: Next transition index.
    """

    case: FrozenEpisodeCase
    transition_index: int = 0
    external_events: list[ExternalUserEvent] = field(default_factory=list)


class FrozenExternalEpisodeService:
    """Serve frozen action/observation transitions to ``UrbanNavEnvClient``.

    Args:
        cases: Frozen cases keyed internally by ``episode_id``.

    This service performs no live tool, map, simulator or LLM calls. Each
    assistant action advances exactly one immutable transition after optional
    action-type and tool-op checks.
    """

    def __init__(self, cases: list[FrozenEpisodeCase]) -> None:
        self.cases = {case.episode_id: case for case in cases}
        self.active: dict[str, _ActiveFrozenEpisode] = {}

    async def reset(self, payload: dict[str, Any]) -> EpisodeResetResponse:
        """Reset one frozen case selected by stable id.

        Args:
            payload: Request containing ``episode.case_id`` or ``episode.episode_id``.

        Returns:
            Strict reset response.

        Raises:
            KeyError: If the case id is not present in the frozen manifest.
        """

        episode_payload = payload.get("episode") if isinstance(payload.get("episode"), dict) else payload
        episode_id = str(episode_payload.get("episode_id") or episode_payload.get("case_id") or "")
        if episode_id not in self.cases:
            raise KeyError(f"unknown frozen episode: {episode_id}")
        case = self.cases[episode_id]
        self.active[episode_id] = _ActiveFrozenEpisode(case=case)
        return EpisodeResetResponse(
            episode_id=episode_id,
            observation=dict(case.initial_observation),
            replay_key=case.initial_replay_key,
            info={"instruction": case.instruction, "case_metadata": case.metadata},
        )

    async def step(self, episode_id: str, payload: dict[str, Any]) -> EpisodeStepResponse:
        """Validate one assistant action and advance the frozen cursor.

        Args:
            episode_id: Active episode id.
            payload: Request containing the latest assistant action.

        Returns:
            Strict frozen transition response.

        Raises:
            KeyError: If the episode was not reset.
            RuntimeError: If stepped after termination.
            ValueError: If action type or operation violates the frozen case.
        """

        active = self.active.get(episode_id)
        if active is None:
            raise KeyError(f"inactive frozen episode: {episode_id}")
        if active.transition_index >= len(active.case.transitions):
            raise RuntimeError("frozen episode stepped after terminal transition")
        transition = active.case.transitions[active.transition_index]
        action_payload = payload.get("action") if isinstance(payload.get("action"), dict) else {}
        action = parse_agent_action(str(action_payload.get("content") or ""))
        if action.action_type == "invalid":
            raise ValueError("invalid external agent action: " + ",".join(action.errors))
        if transition.expected_action_type:
            if transition.expected_action_type in {"tool_call", "finish"}:
                actual_action_type = action.action_type
            else:
                semantic_action, semantic_errors = parse_system2_action_text(
                    str(action_payload.get("content") or "")
                )
                if semantic_action is None:
                    raise ValueError("invalid System2 semantics: " + ",".join(semantic_errors))
                actual_action_type = semantic_action.action_type.value
            if actual_action_type != transition.expected_action_type:
                raise ValueError(
                    f"unexpected action type: expected={transition.expected_action_type}, actual={actual_action_type}"
                )
        actual_op = action.tool_call.op if action.tool_call is not None else None
        if transition.expected_tool_op and actual_op != transition.expected_tool_op:
            raise ValueError(f"unexpected tool op: expected={transition.expected_tool_op}, actual={actual_op}")
        active.transition_index += 1
        info = dict(transition.info)
        reward_components = info.pop("reward_components", {})
        return EpisodeStepResponse(
            episode_id=episode_id,
            observation=dict(transition.observation),
            reward=transition.reward,
            done=transition.done,
            replay_key=transition.replay_key,
            reward_components=dict(reward_components),
            info=info,
            stop_reason=str(info.get("stop_reason")) if info.get("stop_reason") is not None else None,
        )

    async def submit_event(self, episode_id: str, payload: dict[str, Any]) -> EpisodeEventSubmitResponse:
        """Record one deterministic external event for frozen multi-turn replay.

        Args:
            episode_id: Active frozen episode id.
            payload: Event mapping with or without an explicit episode id.

        Returns:
            Idempotent event-ingress response.

        Raises:
            KeyError: If the frozen episode is inactive.
        """

        active = self.active.get(episode_id)
        if active is None:
            raise KeyError(f"inactive frozen episode: {episode_id}")
        value = dict(payload.get("event") or payload)
        value.setdefault("episode_id", episode_id)
        event = ExternalUserEvent.model_validate(value)
        if event.episode_id != episode_id:
            raise ValueError(
                f"event episode_id mismatch: path={episode_id}, payload={event.episode_id}"
            )
        duplicate = any(item.event_id == event.event_id for item in active.external_events)
        if not duplicate:
            active.external_events.append(event)
        return EpisodeEventSubmitResponse(
            episode_id=episode_id,
            event=event,
            accepted=not duplicate,
            duplicate=duplicate,
            info={"queued_event_count": len(active.external_events)},
        )
