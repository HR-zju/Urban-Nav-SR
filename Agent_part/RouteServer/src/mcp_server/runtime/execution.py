"""Revision-aware receding-horizon execution and preemption boundaries."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from mcp_server.contracts.task import ControlState, ControlStatus, TaskState
from mcp_server.runtime.interfaces import EnvironmentStepResult, EpisodeEnvironment, System1Output


class ExecutionLease(BaseModel):
    """Version-bound permission to execute one System1 chunk.

    Args:
        episode_id: Parent episode.
        action_id: Stable dispatched action id.
        task_revision: Task revision used by System1.
        route_revision: Route revision used by System1.
        control_epoch: Controller generation used by System1.
        observation_id: Observation used to predict the chunk.
        execute_steps: Maximum prefix length dispatched to the environment.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    action_id: str
    task_revision: int = Field(ge=0)
    route_revision: int = Field(ge=0)
    control_epoch: int = Field(ge=0)
    observation_id: str
    execute_steps: int = Field(ge=1)


class ExecutionFeedback(BaseModel):
    """Environment completion or deterministic preemption result.

    Args:
        lease: Corresponding action lease.
        result: Environment transition when execution completed.
        preempted: Whether execution was cancelled before completion.
        reason: Completion or preemption reason.
    """

    model_config = ConfigDict(extra="forbid")

    lease: ExecutionLease
    result: EnvironmentStepResult | None = None
    preempted: bool = False
    reason: str


class HaltFeedback(BaseModel):
    """Idempotent deterministic-stop acknowledgement.

    Args:
        episode_id: Halted episode.
        reason: Stable terminal stop reason.
        had_active_action: Whether halt also cancelled an active lease.
        action_id: Cancelled action id when one existed.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    reason: str
    had_active_action: bool
    action_id: str | None = None


class ExecutionManager(Protocol):
    """Controller boundary used by LangGraph without embedding robot IO."""

    async def dispatch(self, output: System1Output, state: dict[str, Any]) -> ExecutionLease:
        """Dispatch a version-bound short action prefix."""

    async def wait(self, lease: ExecutionLease) -> ExecutionFeedback:
        """Wait for controller feedback for one lease."""

    async def preempt(self, episode_id: str, *, reason: str) -> ExecutionFeedback | None:
        """Cancel and stop the current short-horizon action, if one exists.

        Args:
            episode_id: Episode whose active lease is invalidated.
            reason: Stable preemption reason.

        Returns:
            Preemption feedback or ``None`` when no action is active.
        """

    async def halt(self, episode_id: str, *, reason: str) -> HaltFeedback:
        """Issue one idempotent terminal physical-stop command.

        Args:
            episode_id: Episode entering a terminal state.
            reason: Stable terminal reason.

        Returns:
            Deterministic halt acknowledgement.
        """


@dataclass
class _ExecutionRecord:
    """One action task and its single completion/preemption future."""

    lease: ExecutionLease
    task: asyncio.Task[None]
    feedback: asyncio.Future[ExecutionFeedback]


class EnvironmentExecutionManager:
    """Async receding-horizon adapter around the existing environment protocol.

    Args:
        environment: Existing live, simulator or frozen environment.
        execute_steps: Number of trajectory waypoints dispatched per tick.
    """

    def __init__(self, environment: EpisodeEnvironment, *, execute_steps: int = 2) -> None:
        if execute_steps < 1:
            raise ValueError("execute_steps must be >= 1")
        self.environment = environment
        self.execute_steps = execute_steps
        self._active: dict[str, _ExecutionRecord] = {}
        self._feedback_by_action: dict[str, asyncio.Future[ExecutionFeedback]] = {}
        self._halted: dict[str, HaltFeedback] = {}
        self._halt_locks: dict[str, asyncio.Lock] = {}

    async def dispatch(self, output: System1Output, state: dict[str, Any]) -> ExecutionLease:
        """Dispatch only the first ``M`` points of a fresh System1 trajectory.

        Args:
            output: Revision-bound cumulative FLU trajectory.
            state: Current graph state.

        Returns:
            Active execution lease.

        Raises:
            RuntimeError: If another action is already active.
            ValueError: If the output is stale.
        """

        episode_id = str(state["episode_id"])
        if episode_id in self._halted:
            raise RuntimeError(f"episode controller is terminally halted: {episode_id}")
        if episode_id in self._active:
            raise RuntimeError(f"episode already has an active action: {episode_id}")
        task = TaskState.model_validate(state["task_state"])
        control = ControlState.model_validate(state["control_state"])
        observation_id = str(state.get("observation_id") or "")
        valid, reason = validate_system1_freshness(
            output,
            task_state=task,
            control_state=control,
            observation_id=observation_id,
        )
        if not valid:
            raise ValueError(f"stale System1 output: {reason}")
        execute_steps = min(self.execute_steps, len(output.trajectory_ego_t))
        action_id = str(output.action_id or f"action:{uuid4().hex}")
        lease = ExecutionLease(
            episode_id=episode_id,
            action_id=action_id,
            task_revision=task.task_revision,
            route_revision=task.route_revision,
            control_epoch=control.control_epoch,
            observation_id=observation_id,
            execute_steps=execute_steps,
        )
        short_output = output.model_copy(deep=True)
        short_output.action_id = action_id
        short_output.trajectory_ego_t = short_output.trajectory_ego_t[:execute_steps]
        if short_output.action_mask is not None:
            short_output.action_mask = short_output.action_mask[:execute_steps]
        # feedback future 在 active lease 被抢占删除后仍保留，消除 preempt-before-wait 竞态。
        feedback = asyncio.get_running_loop().create_future()
        task_handle = asyncio.create_task(
            self._run_environment_step(
                lease=lease,
                output=short_output,
                state=dict(state),
                feedback=feedback,
            )
        )
        self._active[episode_id] = _ExecutionRecord(lease=lease, task=task_handle, feedback=feedback)
        self._feedback_by_action[action_id] = feedback
        return lease

    async def wait(self, lease: ExecutionLease) -> ExecutionFeedback:
        """Wait for the environment result and release the active lease.

        Args:
            lease: Active execution lease.

        Returns:
            Completed environment feedback.

        Raises:
            KeyError: If the lease is not active.
        """

        feedback = self._feedback_by_action.get(lease.action_id)
        if feedback is None:
            raise KeyError(f"unknown execution lease: {lease.action_id}")
        try:
            return await asyncio.shield(feedback)
        finally:
            self._feedback_by_action.pop(lease.action_id, None)
            active = self._active.get(lease.episode_id)
            if active is not None and active.lease.action_id == lease.action_id:
                self._active.pop(lease.episode_id, None)
                if not active.task.done():
                    active.task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await active.task

    async def preempt(self, episode_id: str, *, reason: str) -> ExecutionFeedback | None:
        """Cancel the active environment task and invoke a live preempt hook.

        Args:
            episode_id: Episode whose action must stop.
            reason: Stable preemption reason.

        Returns:
            Preemption feedback, or ``None`` when no action was active.
        """

        active = self._active.pop(episode_id, None)
        if active is None:
            return None
        active.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await active.task
        value = self.environment.preempt(reason=reason)
        if asyncio.iscoroutine(value):
            await value
        feedback = ExecutionFeedback(lease=active.lease, preempted=True, reason=reason)
        if not active.feedback.done():
            active.feedback.set_result(feedback)
        return feedback

    async def halt(self, episode_id: str, *, reason: str) -> HaltFeedback:
        """Cancel any active lease and issue one deterministic physical stop.

        Args:
            episode_id: Episode entering a terminal state.
            reason: Stable terminal reason sent to the controller.

        Returns:
            Cached halt acknowledgement. Repeated calls for the same episode do
            not invoke the physical stop hook again.
        """

        lock = self._halt_locks.setdefault(episode_id, asyncio.Lock())
        async with lock:
            cached = self._halted.get(episode_id)
            if cached is not None:
                return cached
            active = self._active.pop(episode_id, None)
            if active is not None:
                active.task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await active.task
            # 终态停车不能依赖 active lease 是否存在；真实底盘必须显式确认 halt。
            value = self.environment.halt(reason=reason)
            if asyncio.iscoroutine(value):
                await value
            if active is not None and not active.feedback.done():
                active.feedback.set_result(
                    ExecutionFeedback(lease=active.lease, preempted=True, reason=reason)
                )
            feedback = HaltFeedback(
                episode_id=episode_id,
                reason=reason,
                had_active_action=active is not None,
                action_id=active.lease.action_id if active is not None else None,
            )
            self._halted[episode_id] = feedback
            return feedback

    async def _run_environment_step(
        self,
        *,
        lease: ExecutionLease,
        output: System1Output,
        state: dict[str, Any],
        feedback: asyncio.Future[ExecutionFeedback],
    ) -> None:
        """Resolve exactly one feedback future from an environment transition."""

        try:
            result = EnvironmentStepResult.model_validate(await self.environment.step(output, state))
        except asyncio.CancelledError:
            # preempt() 负责在真实 stop hook 完成后写入 preempted feedback。
            return
        except Exception as exc:
            if not feedback.done():
                feedback.set_exception(exc)
        else:
            if not feedback.done():
                feedback.set_result(ExecutionFeedback(lease=lease, result=result, reason="completed"))


def bind_system1_execution_context(output: System1Output, state: dict[str, Any]) -> System1Output:
    """Bind an untrusted model output to the current runtime revisions.

    Args:
        output: Parsed System1 trajectory.
        state: Current graph state.

    Returns:
        Copy carrying explicit task, route, control and observation versions.
    """

    task = TaskState.model_validate(state["task_state"])
    control = ControlState.model_validate(state["control_state"])
    bound = output.model_copy(deep=True)
    bound.task_revision = task.task_revision
    bound.route_revision = task.route_revision
    bound.control_epoch = control.control_epoch
    bound.observation_id = str(state.get("observation_id") or "")
    bound.action_id = bound.action_id or f"action:{uuid4().hex}"
    return bound


def validate_system1_freshness(
    output: System1Output,
    *,
    task_state: TaskState,
    control_state: ControlState,
    observation_id: str,
) -> tuple[bool, str]:
    r"""Validate the action lease invariant before every dispatch.

    The executable indicator is

    \[
    \chi(a)=[r_a=r_t]\land[\rho_a=\rho_t]\land[e_a=e_t]\land[o_a=o_t].
    \]

    Args:
        output: Candidate System1 output.
        task_state: Current task revisions and status.
        control_state: Current controller generation and status.
        observation_id: Latest observation id.

    Returns:
        ``(fresh, reason)`` where reason lists failed invariants.
    """

    checks = {
        "task_revision": output.task_revision == task_state.task_revision,
        "route_revision": output.route_revision == task_state.route_revision,
        "control_epoch": output.control_epoch == control_state.control_epoch,
        "observation_id": output.observation_id == observation_id,
        "task_active": task_state.status.value == "active",
        "control_not_paused": control_state.status not in {ControlStatus.PAUSED, ControlStatus.STOPPED},
    }
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, ",".join(failed) if failed else "fresh"
