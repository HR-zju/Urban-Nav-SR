"""External event storage and deterministic navigation-task reduction."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from mcp_server.contracts.task import (
    ControlState,
    ControlStatus,
    DialogueState,
    DialogueTurn,
    ExternalUserEvent,
    ExternalUserEventType,
    TaskState,
    TaskStatus,
)


class ExternalEventStore(Protocol):
    """Persistent-queue boundary for events arriving outside LangGraph."""

    async def append(self, event: ExternalUserEvent) -> bool:
        """Append an event and return whether it was new."""

    async def claim_next(self, episode_id: str) -> ExternalUserEvent | None:
        """Claim the next unprocessed event for one episode."""

    async def acknowledge(self, event_id: str) -> None:
        """Mark a claimed event as durably processed."""

    async def release(self, event_id: str) -> None:
        """Release a failed claim so the event can be retried."""

    async def wait_for_pending(self, episode_id: str) -> None:
        """Wait until one unprocessed event is available."""


class InMemoryExternalEventStore:
    """Deterministic in-memory event store for tests and single-process smoke.

    Production deployments should implement the same protocol with a
    transactional database. This implementation intentionally has no hidden
    fallback to a process-external service.
    """

    def __init__(self) -> None:
        self._events: dict[str, list[ExternalUserEvent]] = defaultdict(list)
        self._known_ids: set[str] = set()
        self._claimed_ids: set[str] = set()
        self._acknowledged_ids: set[str] = set()
        self._condition = asyncio.Condition()

    async def append(self, event: ExternalUserEvent) -> bool:
        """Append one event exactly once.

        Args:
            event: Validated event.

        Returns:
            ``True`` for a new event and ``False`` for a duplicate id.
        """

        value = ExternalUserEvent.model_validate(event)
        async with self._condition:
            if value.event_id in self._known_ids:
                return False
            self._known_ids.add(value.event_id)
            self._events[value.episode_id].append(value)
            self._condition.notify_all()
        return True

    async def claim_next(self, episode_id: str) -> ExternalUserEvent | None:
        """Claim the first pending event without removing its audit record.

        Args:
            episode_id: Episode queue to inspect.

        Returns:
            Claimed event or ``None`` when the queue has no pending item.
        """

        async with self._condition:
            for event in self._events.get(episode_id, []):
                if event.event_id not in self._claimed_ids and event.event_id not in self._acknowledged_ids:
                    self._claimed_ids.add(event.event_id)
                    return event
        return None

    async def acknowledge(self, event_id: str) -> None:
        """Acknowledge one claimed event.

        Args:
            event_id: Claimed event id.

        Returns:
            None.
        """

        async with self._condition:
            self._claimed_ids.discard(event_id)
            self._acknowledged_ids.add(event_id)

    async def release(self, event_id: str) -> None:
        """Release a claimed event after reducer/controller failure.

        Args:
            event_id: Claimed event id.

        Returns:
            None.
        """

        async with self._condition:
            self._claimed_ids.discard(event_id)
            self._condition.notify_all()

    async def wait_for_pending(self, episode_id: str) -> None:
        """Wait for an event without claiming it.

        Args:
            episode_id: Episode queue to observe.

        Returns:
            None after at least one pending event becomes available.
        """

        async with self._condition:
            await self._condition.wait_for(
                lambda: any(
                    event.event_id not in self._claimed_ids and event.event_id not in self._acknowledged_ids
                    for event in self._events.get(episode_id, [])
                )
            )


class EventReduction(BaseModel):
    """Deterministic result of applying one external event.

    Args:
        task_state: Updated task state.
        dialogue_state: Updated dialogue state.
        control_state: Updated controller state.
        accepted: Whether revision/correlation checks passed.
        requires_system2: Whether semantic reasoning is now required.
        requires_preempt: Whether the current action lease is invalid.
        terminal: Whether the event cancels the episode.
        reason: Stable reducer reason code.
    """

    model_config = ConfigDict(extra="forbid")

    task_state: TaskState
    dialogue_state: DialogueState
    control_state: ControlState
    accepted: bool
    requires_system2: bool = False
    requires_preempt: bool = False
    terminal: bool = False
    reason: str


def reduce_external_event(
    event: ExternalUserEvent,
    *,
    task_state: TaskState,
    dialogue_state: DialogueState,
    control_state: ControlState,
    max_dialogue_turns: int = 32,
) -> EventReduction:
    """Apply one event with revision and interaction-correlation checks.

    Args:
        event: Claimed external event.
        task_state: Current task state.
        dialogue_state: Current dialogue state.
        control_state: Current controller state.
        max_dialogue_turns: Bounded verbatim dialogue window.

    Returns:
        Pure deterministic reduction result.
    """

    task = task_state.model_copy(deep=True)
    dialogue = dialogue_state.model_copy(deep=True)
    control = control_state.model_copy(deep=True)
    if event.expected_task_revision is not None and event.expected_task_revision != task.task_revision:
        return EventReduction(
            task_state=task,
            dialogue_state=dialogue,
            control_state=control,
            accepted=False,
            reason="task_revision_conflict",
        )

    task.last_event_id = event.event_id
    event_type = event.event_type
    if event_type in {ExternalUserEventType.PAUSE, ExternalUserEventType.EMERGENCY_STOP}:
        task.status = TaskStatus.PAUSED
        control.control_epoch += 1
        control.status = ControlStatus.PAUSED
        control.last_preempt_reason = event_type.value
        control.active_action_id = None
        return EventReduction(
            task_state=task,
            dialogue_state=dialogue,
            control_state=control,
            accepted=True,
            requires_preempt=True,
            reason=event_type.value,
        )
    if event_type == ExternalUserEventType.CANCEL:
        task.status = TaskStatus.CANCELLED
        control.control_epoch += 1
        control.status = ControlStatus.STOPPED
        control.last_preempt_reason = "cancel"
        control.active_action_id = None
        return EventReduction(
            task_state=task,
            dialogue_state=dialogue,
            control_state=control,
            accepted=True,
            requires_preempt=True,
            terminal=True,
            reason="cancel",
        )
    if event_type == ExternalUserEventType.RESUME:
        if task.status in {TaskStatus.CANCELLED, TaskStatus.COMPLETED, TaskStatus.FAILED}:
            return EventReduction(
                task_state=task,
                dialogue_state=dialogue,
                control_state=control,
                accepted=False,
                reason=f"{task.status.value}_task_cannot_resume",
            )
        task.status = TaskStatus.ACTIVE
        control.control_epoch += 1
        control.status = ControlStatus.IDLE
        control.last_preempt_reason = None
        return EventReduction(
            task_state=task,
            dialogue_state=dialogue,
            control_state=control,
            accepted=True,
            requires_system2=True,
            reason="resume",
        )

    # 只有显式关联 pending interaction 的回答才能解锁 HITL，防止过期回答误改当前任务。
    if event_type == ExternalUserEventType.USER_RESPONSE:
        pending = dialogue.pending_interaction
        if pending is not None and event.interaction_id == pending.interaction_id:
            dialogue.recent_turns.append(
                DialogueTurn(
                    role="user",
                    content=event.content,
                    event_id=event.event_id,
                    interaction_id=pending.interaction_id,
                )
            )
            dialogue.pending_interaction = None
            dialogue.recent_turns = dialogue.recent_turns[-max_dialogue_turns:]
            return EventReduction(
                task_state=task,
                dialogue_state=dialogue,
                control_state=control,
                accepted=True,
                requires_system2=True,
                reason="interaction_resolved",
            )
        event_type = ExternalUserEventType.NEW_INSTRUCTION

    if event_type in {ExternalUserEventType.NEW_INSTRUCTION, ExternalUserEventType.MODIFY_TASK}:
        task.task_revision += 1
        task.status = TaskStatus.ACTIVE
        # 新任务语义使旧问题失效，不应在后续回合继续等待旧 interaction。
        dialogue.pending_interaction = None
        if event_type == ExternalUserEventType.NEW_INSTRUCTION:
            task.active_instruction = _event_text(event.content)
        elif isinstance(event.content, dict):
            if event.content.get("active_instruction") is not None:
                task.active_instruction = str(event.content["active_instruction"])
            if isinstance(event.content.get("goal_spec"), dict):
                task.goal_spec = dict(event.content["goal_spec"])
            if isinstance(event.content.get("constraints"), dict):
                task.constraints.update(event.content["constraints"])
        dialogue.recent_turns.append(
            DialogueTurn(role="user", content=event.content, event_id=event.event_id)
        )
        dialogue.recent_turns = dialogue.recent_turns[-max_dialogue_turns:]
        control.control_epoch += 1
        control.status = ControlStatus.IDLE
        control.active_action_id = None
        control.last_preempt_reason = event_type.value
        return EventReduction(
            task_state=task,
            dialogue_state=dialogue,
            control_state=control,
            accepted=True,
            requires_system2=True,
            requires_preempt=True,
            reason=event_type.value,
        )

    return EventReduction(
        task_state=task,
        dialogue_state=dialogue,
        control_state=control,
        accepted=False,
        reason="unsupported_external_event",
    )


def _event_text(content: Any) -> str:
    """Return the canonical instruction text from event content."""

    if isinstance(content, dict):
        return str(content.get("instruction") or content.get("text") or content)
    return str(content or "")
