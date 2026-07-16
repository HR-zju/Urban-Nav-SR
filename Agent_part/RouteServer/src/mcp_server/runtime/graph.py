"""Event-driven LangGraph runtime for closed-loop UrbanNav episodes."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field

from mcp_server.contracts.episode import EpisodeEventType, EpisodeStatus, EpisodeTraceV2
from mcp_server.contracts.task import (
    ControlState,
    ControlStatus,
    DialogueState,
    DialogueTurn,
    ExternalUserEvent,
    ExternalUserEventType,
    PendingInteraction,
    TaskState,
    TaskStatus,
    initial_task_state,
)
from mcp_server.contracts.tool import ToolResponseV2
from mcp_server.runtime.events import (
    ExternalEventStore,
    InMemoryExternalEventStore,
    reduce_external_event,
)
from mcp_server.runtime.execution import (
    EnvironmentExecutionManager,
    ExecutionLease,
    ExecutionManager,
    bind_system1_execution_context,
)
from mcp_server.runtime.interfaces import (
    EpisodeEnvironment,
    EpisodeToolExecutor,
    ObservationResult,
    SafetyResult,
    SafetyValidator,
    System1Output,
    System1Policy,
    System2Action,
    System2ActionType,
    System2Policy,
    TerminalValidationResult,
    TerminalValidator,
)
from mcp_server.runtime.state import EpisodeGraphState
from mcp_server.runtime.terminal import RuntimeTerminalValidator
from mcp_server.runtime.tool_adapters import bind_tool_call_context


class EpisodeRuntimeConfig(BaseModel):
    """Budgets and execution policy for one closed-loop episode.

    Args:
        max_ticks: Maximum completed environment transitions.
        max_tool_calls: Maximum consecutive tool calls before a physical step.
        max_total_tool_calls: Episode-wide diagnostic safety bound.
        max_safety_retries: Maximum rejected chunks before failure.
        max_terminal_rejections: Maximum rejected completion requests before
            terminating as a policy-contract failure.
        execute_steps: Receding-horizon prefix dispatched from each trajectory.
        max_dialogue_turns: Verbatim dialogue turns kept in graph state.
        source: Runtime source written into episode events.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_ticks: int = Field(default=32, ge=1)
    max_tool_calls: int = Field(default=8, ge=0)
    max_total_tool_calls: int = Field(default=128, ge=0)
    max_safety_retries: int = Field(default=2, ge=0)
    max_terminal_rejections: int = Field(default=2, ge=0)
    execute_steps: int = Field(default=2, ge=1)
    max_dialogue_turns: int = Field(default=32, ge=1)
    source: str = "langgraph_episode_runtime"


#定义回合
def build_episode_graph(
    *,
    system2_policy: System2Policy,
    system1_policy: System1Policy,
    tool_executor: EpisodeToolExecutor,
    safety_validator: SafetyValidator,
    environment: EpisodeEnvironment,
    config: EpisodeRuntimeConfig | None = None,
    checkpointer: Any | None = None,
    external_event_store: ExternalEventStore | None = None,
    execution_manager: ExecutionManager | None = None,
    terminal_validator: TerminalValidator | None = None,
):
    """Build a revision-aware, event-driven navigation graph.

    Args:
        system2_policy: Slow policy selecting tools, interaction or navigation.
        system1_policy: Fast policy producing cumulative FLU trajectories.
        tool_executor: Registry-conformant live or replay executor.
        safety_validator: Deterministic pre-execution validator.
        environment: Live, simulator or frozen replay environment.
        config: Runtime budgets and receding-horizon settings.
        checkpointer: LangGraph checkpointer used for resumable waits.
        external_event_store: Out-of-band user event queue.
        execution_manager: Controller adapter supporting dispatch and preemption.
        terminal_validator: Deterministic completion/abort validator.

    Returns:
        Compiled LangGraph state machine carrying its event/controller adapters.
    """

    runtime_config = config or EpisodeRuntimeConfig()
    event_store = external_event_store or InMemoryExternalEventStore()
    controller = execution_manager or EnvironmentExecutionManager(
        environment,
        execute_steps=runtime_config.execute_steps,
    )
    terminal_guard = terminal_validator or RuntimeTerminalValidator()

    async def observe(state: EpisodeGraphState) -> dict[str, Any]:
        """Reset or expose the latest environment observation."""

        trace = _trace(state)
        if not state.get("observation"):
            result = ObservationResult.model_validate(
                await _await(environment.reset(dict(state.get("initial_context") or {})))
            )
            observation = result.observation
            info = result.info
            replay_key = result.replay_key
            for event_payload in result.external_events:
                await event_store.append(ExternalUserEvent.model_validate(event_payload))
        else:
            observation = dict(state["observation"])
            info = dict(state.get("observation_info") or {})
            replay_key = info.get("replay_key")
        tick_id = int(state.get("tick_id", 0))
        observation_id = str(info.get("observation_id") or replay_key or f"observation:{state['episode_id']}:{tick_id}")
        trace.append(
            EpisodeEventType.OBSERVATION_RECEIVED,
            tick_id=tick_id,
            source=runtime_config.source,
            payload={
                "observation": observation,
                "observation_id": observation_id,
                "info": info,
                "replay_key": replay_key,
            },
        )
        return {
            "observation": observation,
            "observation_id": observation_id,
            "observation_info": info,
            "episode_trace": trace.model_dump(mode="json"),
        }

    async def system2(state: EpisodeGraphState) -> dict[str, Any]:
        """Run one slow-policy decision against the current revisioned state."""

        trace = _trace(state)
        action = System2Action.model_validate(await _await(system2_policy.decide(dict(state))))
        decision_id = _next_decision_id(trace, state)
        trace.status = EpisodeStatus.RUNNING
        trace.stop_reason = None
        trace.append(
            EpisodeEventType.MODEL_OUTPUT,
            tick_id=int(state.get("tick_id", 0)),
            source="system2_policy",
            payload={"decision_id": decision_id, "action": action.model_dump(mode="json")},
        )
        history = list(state.get("history") or [])
        history.append({"decision_id": decision_id, "role": "assistant", "action": action.model_dump(mode="json")})
        return {
            "decision_id": decision_id,
            "system2_action": action.model_dump(mode="json"),
            "pending_user_request": action.user_request or {},
            "latest_external_event": {},
            "user_response": None,
            "history": history[-runtime_config.max_dialogue_turns * 2 :],
            "episode_trace": trace.model_dump(mode="json"),
        }

    async def execute_tool(state: EpisodeGraphState) -> dict[str, Any]:
        """Execute one validated tool call with revision/idempotency context."""

        trace = _trace(state)
        action = System2Action.model_validate(state["system2_action"])
        assert action.tool_call is not None
        count = int(state.get("tool_call_count", 0))
        total_count = int(state.get("total_tool_call_count", 0))
        if count >= runtime_config.max_tool_calls or total_count >= runtime_config.max_total_tool_calls:
            return {
                "done": True,
                "terminal_status": EpisodeStatus.TRUNCATED.value,
                "stop_reason": "max_tool_calls",
                "episode_trace": trace.model_dump(mode="json"),
            }
        call = bind_tool_call_context(action.tool_call, dict(state))
        decision_id = str(state.get("decision_id") or "")
        trace.append(
            EpisodeEventType.TOOL_CALL,
            tick_id=int(state.get("tick_id", 0)),
            source="tool_executor",
            payload={"decision_id": decision_id, "call": call.model_dump(mode="json")},
        )
        raw_response = await _await(tool_executor.execute(call, dict(state)))
        response = raw_response if isinstance(raw_response, ToolResponseV2) else ToolResponseV2.model_validate(raw_response)
        trace.append(
            EpisodeEventType.TOOL_RESPONSE,
            tick_id=int(state.get("tick_id", 0)),
            source="tool_executor",
            payload={"decision_id": decision_id, "response": response.model_dump(mode="json")},
        )
        task = TaskState.model_validate(state["task_state"])
        if response.state_revision is not None:
            try:
                task.route_revision = int(response.state_revision)
            except (TypeError, ValueError):
                pass
        if response.state_delta:
            trace.append(
                EpisodeEventType.ROUTE_STATE_DELTA,
                tick_id=int(state.get("tick_id", 0)),
                source="tool_executor",
                payload={
                    "decision_id": decision_id,
                    "state_delta": response.state_delta,
                    "state_revision": response.state_revision,
                },
            )
        result = response.result if isinstance(response.result, dict) else {}
        pending = (
            result.get("dialogue_state", {}).get("pending_user_interaction")
            if isinstance(result.get("dialogue_state"), dict)
            else None
        )
        return {
            "task_state": task.model_dump(mode="json"),
            "last_tool_response": response.model_dump(mode="json"),
            "pending_user_request": dict(pending or {}),
            "tool_call_count": count + 1,
            "total_tool_call_count": total_count + 1,
            "episode_trace": trace.model_dump(mode="json"),
        }

    async def prepare_interaction(state: EpisodeGraphState) -> dict[str, Any]:
        """Persist the waiting state before entering LangGraph ``interrupt``."""

        trace = _trace(state)
        request = dict(state.get("pending_user_request") or {})
        interaction_id = str(
            request.get("interaction_id")
            or f"interaction:{state['episode_id']}:{state.get('tick_id', 0)}:{len(trace.events)}"
        )
        request["interaction_id"] = interaction_id
        dialogue = DialogueState.model_validate(state.get("dialogue_state") or {})
        dialogue.pending_interaction = PendingInteraction(
            interaction_id=interaction_id,
            request=request,
            motion_policy=str(request.get("motion_policy_while_waiting") or "stop"),
        )
        dialogue.recent_turns.append(
            DialogueTurn(role="assistant", content=request, interaction_id=interaction_id)
        )
        dialogue.recent_turns = dialogue.recent_turns[-runtime_config.max_dialogue_turns :]
        tick_id = int(state.get("tick_id", 0))
        trace.status = EpisodeStatus.WAITING_FOR_USER
        trace.stop_reason = "waiting_for_user"
        trace.append(
            EpisodeEventType.INTERACTION_REQUESTED,
            tick_id=tick_id,
            source="hitl",
            payload={"decision_id": state.get("decision_id"), "interaction": request},
        )
        trace.append(EpisodeEventType.USER_INTERRUPT, tick_id=tick_id, source="hitl", payload={"request": request})
        return {
            "dialogue_state": dialogue.model_dump(mode="json"),
            "pending_user_request": request,
            "episode_trace": trace.model_dump(mode="json"),
        }

    async def await_interaction(state: EpisodeGraphState) -> dict[str, Any]:
        """Suspend without side effects, then enqueue the correlated response."""

        request = dict(state.get("pending_user_request") or {})
        response = interrupt({"episode_id": state["episode_id"], "request": request})
        event = _resume_payload_to_event(
            response,
            episode_id=state["episode_id"],
            default_type=ExternalUserEventType.USER_RESPONSE,
            interaction_id=str(request.get("interaction_id") or "") or None,
        )
        await event_store.append(event)
        return {}

    async def await_external_resume(state: EpisodeGraphState) -> dict[str, Any]:
        """Suspend a deterministically paused task until a resume/task event arrives."""

        response = interrupt(
            {
                "episode_id": state["episode_id"],
                "reason": "task_paused",
                "task_state": state.get("task_state"),
            }
        )
        event = _resume_payload_to_event(
            response,
            episode_id=state["episode_id"],
            default_type=ExternalUserEventType.RESUME,
        )
        await event_store.append(event)
        return {}

    async def system1(state: EpisodeGraphState) -> dict[str, Any]:
        """Predict and bind a continuous chunk to current revisions."""

        trace = _trace(state)
        output = System1Output.model_validate(await _await(system1_policy.predict(dict(state))))
        output.action_id = output.action_id or f"action:{state.get('decision_id') or _next_decision_id(trace, state)}"
        output = bind_system1_execution_context(output, dict(state))
        trace.append(
            EpisodeEventType.SYSTEM1_OUTPUT,
            tick_id=int(state.get("tick_id", 0)),
            source="system1_policy",
            payload={"decision_id": state.get("decision_id"), "output": output.model_dump(mode="json")},
        )
        return {
            "system1_output": output.model_dump(mode="json"),
            "executable_system1_output": output.model_dump(mode="json"),
            "episode_trace": trace.model_dump(mode="json"),
        }

    async def safety(state: EpisodeGraphState) -> dict[str, Any]:
        """Validate a chunk and materialize an optional replacement trajectory."""

        trace = _trace(state)
        output = System1Output.model_validate(state["system1_output"])
        result = SafetyResult.model_validate(await _await(safety_validator.validate(output, dict(state))))
        retries = int(state.get("safety_retry_count", 0)) + (0 if result.accepted else 1)
        executable = output if result.accepted else None
        if result.intervention:
            candidate = result.intervention.get("system1_output", result.intervention)
            if isinstance(candidate, dict) and candidate.get("trajectory_ego_t"):
                executable = bind_system1_execution_context(System1Output.model_validate(candidate), dict(state))
                trace.append(
                    EpisodeEventType.SAFETY_INTERVENTION,
                    tick_id=int(state.get("tick_id", 0)),
                    source="safety_validator",
                    payload={
                        "decision_id": state.get("decision_id"),
                        "reason": result.reason,
                        "output": executable.model_dump(mode="json"),
                    },
                )
        trace.append(
            EpisodeEventType.SAFETY_VALIDATION,
            tick_id=int(state.get("tick_id", 0)),
            source="safety_validator",
            payload={"decision_id": state.get("decision_id"), "result": result.model_dump(mode="json"), "retry_count": retries},
        )
        update: dict[str, Any] = {
            "safety_result": result.model_dump(mode="json"),
            "safety_retry_count": retries,
            "executable_system1_output": executable.model_dump(mode="json") if executable is not None else {},
            "episode_trace": trace.model_dump(mode="json"),
        }
        if executable is None and retries > runtime_config.max_safety_retries:
            update.update(done=True, terminal_status=EpisodeStatus.FAILED.value, stop_reason="safety_retry_exhausted")
        return update

    async def dispatch_action(state: EpisodeGraphState) -> dict[str, Any]:
        """Dispatch a fresh, safety-approved short trajectory prefix."""

        trace = _trace(state)
        output = System1Output.model_validate(state["executable_system1_output"])
        lease = await controller.dispatch(output, dict(state))
        control = ControlState.model_validate(state["control_state"])
        control.active_action_id = lease.action_id
        control.status = ControlStatus.EXECUTING
        trace.append(
            EpisodeEventType.ACTION_DISPATCHED,
            tick_id=int(state.get("tick_id", 0)),
            source="execution_manager",
            payload={"decision_id": state.get("decision_id"), "lease": lease.model_dump(mode="json")},
        )
        return {
            "control_state": control.model_dump(mode="json"),
            "execution_lease": lease.model_dump(mode="json"),
            "execution_interrupted": False,
            "episode_trace": trace.model_dump(mode="json"),
        }

    async def await_execution(state: EpisodeGraphState) -> dict[str, Any]:
        """Race controller feedback against an out-of-band user event."""

        trace = _trace(state)
        lease = ExecutionLease.model_validate(state["execution_lease"])
        feedback_task = asyncio.create_task(controller.wait(lease))
        event_task = asyncio.create_task(event_store.wait_for_pending(state["episode_id"]))
        completed, _ = await asyncio.wait({feedback_task, event_task}, return_when=asyncio.FIRST_COMPLETED)
        if event_task in completed:
            feedback = await controller.preempt(state["episode_id"], reason="external_event")
            feedback_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, KeyError):
                await feedback_task
            control = ControlState.model_validate(state["control_state"])
            control.status = ControlStatus.IDLE
            control.active_action_id = None
            trace.append(
                EpisodeEventType.ACTION_PREEMPTED,
                tick_id=int(state.get("tick_id", 0)),
                source="execution_manager",
                payload={
                    "decision_id": state.get("decision_id"),
                    "lease": lease.model_dump(mode="json"),
                    "feedback": feedback.model_dump(mode="json") if feedback is not None else None,
                },
            )
            return {
                "control_state": control.model_dump(mode="json"),
                "execution_interrupted": True,
                "episode_trace": trace.model_dump(mode="json"),
            }

        event_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await event_task
        feedback = await feedback_task
        if feedback.preempted:
            control = ControlState.model_validate(state["control_state"])
            control.status = ControlStatus.IDLE
            control.active_action_id = None
            trace.append(
                EpisodeEventType.ACTION_PREEMPTED,
                tick_id=int(state.get("tick_id", 0)),
                source="execution_manager",
                payload={
                    "decision_id": state.get("decision_id"),
                    "lease": lease.model_dump(mode="json"),
                    "feedback": feedback.model_dump(mode="json"),
                },
            )
            return {
                "control_state": control.model_dump(mode="json"),
                "execution_interrupted": True,
                "episode_trace": trace.model_dump(mode="json"),
            }
        if feedback.result is None:
            raise RuntimeError("completed execution feedback is missing environment result")
        result = feedback.result
        for event_payload in result.external_events:
            await event_store.append(ExternalUserEvent.model_validate(event_payload))
        tick_id = int(state.get("tick_id", 0))
        trace.append(
            EpisodeEventType.CONTROLLER_FEEDBACK,
            tick_id=tick_id,
            source="execution_manager",
            payload={"decision_id": state.get("decision_id"), "feedback": feedback.model_dump(mode="json")},
        )
        trace.append(
            EpisodeEventType.ENVIRONMENT_STEP,
            tick_id=tick_id,
            source="environment",
            payload={"decision_id": state.get("decision_id"), "result": result.model_dump(mode="json")},
        )
        next_tick = tick_id + 1
        control = ControlState.model_validate(state["control_state"])
        control.status = ControlStatus.IDLE
        control.active_action_id = None
        update: dict[str, Any] = {
            "tick_id": next_tick,
            "observation": result.observation,
            "observation_id": "",
            "observation_info": {**result.info, "replay_key": result.replay_key},
            "environment_result": result.model_dump(mode="json"),
            "control_state": control.model_dump(mode="json"),
            "execution_interrupted": False,
            "safety_retry_count": 0,
            "tool_call_count": 0,
            "episode_trace": trace.model_dump(mode="json"),
        }
        if result.done:
            update.update(
                done=True,
                terminal_status=(EpisodeStatus.SUCCEEDED.value if result.info.get("arrival_success") else EpisodeStatus.FAILED.value),
                stop_reason=str(result.info.get("stop_reason") or ("arrival" if result.info.get("arrival_success") else "environment_done")),
            )
        elif next_tick >= runtime_config.max_ticks:
            update.update(done=True, terminal_status=EpisodeStatus.TRUNCATED.value, stop_reason="max_ticks")
        return update

    async def terminal_control(state: EpisodeGraphState) -> dict[str, Any]:
        """Validate model terminal intent and deterministically stop execution.

        A model-declared ``complete_task`` is only accepted when the injected
        terminal validator confirms arrival from authoritative runtime state.
        ``abort_task`` is a typed failed outcome and never invokes System1.
        Environment, budget, and external-cancel terminal states skip model
        validation but still pass through the same idempotent physical halt.
        """

        trace = _trace(state)
        tick_id = int(state.get("tick_id", 0))
        action = System2Action.model_validate(state["system2_action"]) if state.get("system2_action") else None
        terminal_status = state.get("terminal_status")
        stop_reason = state.get("stop_reason")
        validation_payload = dict(state.get("terminal_validation") or {})
        rejection_count = int(state.get("terminal_rejection_count", 0))

        if terminal_status is None:
            if action is None or action.action_type not in {
                System2ActionType.COMPLETE_TASK,
                System2ActionType.ABORT_TASK,
            }:
                raise RuntimeError("terminal_control requires a terminal status or terminal System2 action")
            validation = TerminalValidationResult.model_validate(
                await _await(terminal_guard.validate(action, dict(state)))
            )
            if validation.action_type != action.action_type:
                raise RuntimeError(
                    "terminal validator action mismatch: "
                    f"requested={action.action_type.value}, validated={validation.action_type.value}"
                )
            validation_payload = validation.model_dump(mode="json")
            trace.append(
                EpisodeEventType.TERMINAL_VALIDATION,
                tick_id=tick_id,
                source="terminal_validator",
                payload={
                    "decision_id": state.get("decision_id"),
                    "request": action.model_dump(mode="json"),
                    "result": validation_payload,
                },
            )
            if not validation.accepted:
                rejection_count += 1
                if rejection_count <= runtime_config.max_terminal_rejections:
                    return {
                        "done": False,
                        "terminal_rejected": True,
                        "terminal_validation": validation_payload,
                        "terminal_rejection_count": rejection_count,
                        "terminal_status": None,
                        "stop_reason": None,
                        "episode_trace": trace.model_dump(mode="json"),
                    }
                terminal_status = EpisodeStatus.FAILED.value
                stop_reason = "terminal_validation_retry_exhausted"
            elif action.action_type == System2ActionType.COMPLETE_TASK:
                terminal_status = EpisodeStatus.SUCCEEDED.value
                stop_reason = validation.reason_code
            else:
                terminal_status = EpisodeStatus.FAILED.value
                stop_reason = validation.reason_code

        if stop_reason is None:
            raise RuntimeError("terminal_control requires a stable stop_reason")
        halt_feedback = await controller.halt(state["episode_id"], reason=str(stop_reason))
        control = ControlState.model_validate(state["control_state"])
        if control.status != ControlStatus.STOPPED:
            control.control_epoch += 1
        control.status = ControlStatus.STOPPED
        control.active_action_id = None
        control.last_preempt_reason = str(stop_reason)
        trace.append(
            EpisodeEventType.CONTROL_HALTED,
            tick_id=tick_id,
            source="execution_manager",
            payload={
                "decision_id": state.get("decision_id"),
                "feedback": halt_feedback.model_dump(mode="json"),
            },
        )
        return {
            "control_state": control.model_dump(mode="json"),
            "done": True,
            "terminal_rejected": False,
            "terminal_validation": validation_payload,
            "terminal_rejection_count": rejection_count,
            "terminal_status": str(terminal_status),
            "stop_reason": str(stop_reason),
            "episode_trace": trace.model_dump(mode="json"),
        }

    async def finish(state: EpisodeGraphState) -> dict[str, Any]:
        """Finalize task state and the canonical episode trace."""

        trace = _trace(state)
        if state.get("terminal_status") is None or state.get("stop_reason") is None:
            raise RuntimeError("finish requires terminal_control to set status and stop_reason")
        status = EpisodeStatus(str(state["terminal_status"]))
        reason = str(state["stop_reason"])
        task = TaskState.model_validate(state["task_state"])
        control = ControlState.model_validate(state["control_state"])
        if control.status != ControlStatus.STOPPED:
            raise RuntimeError("finish requires deterministic terminal halt acknowledgement")
        if status == EpisodeStatus.SUCCEEDED:
            task.status = TaskStatus.COMPLETED
        elif status == EpisodeStatus.CANCELLED:
            task.status = TaskStatus.CANCELLED
        else:
            task.status = TaskStatus.FAILED
        if not trace.events or trace.events[-1].event_type != EpisodeEventType.EPISODE_FINISHED:
            trace.finish(status=status, stop_reason=reason, tick_id=int(state.get("tick_id", 0)), source=runtime_config.source)
        return {
            "task_state": task.model_dump(mode="json"),
            "done": True,
            "stop_reason": reason,
            "terminal_status": status.value,
            "episode_trace": trace.model_dump(mode="json"),
        }

    def event_gate(continuation: str) -> Callable[[EpisodeGraphState], Any]:
        """Create one reusable event gate for a high-latency boundary."""

        async def ingest(state: EpisodeGraphState) -> dict[str, Any]:
            # 工具预算、取消等终态不得因空事件队列重新进入 System2。
            if state.get("done"):
                return {"event_route": "terminal"}
            event = await event_store.claim_next(state["episode_id"])
            if event is None:
                return {"event_route": "continue"}
            trace = _trace(state)
            tick_id = int(state.get("tick_id", 0))
            trace.append(
                EpisodeEventType.EXTERNAL_EVENT_RECEIVED,
                tick_id=tick_id,
                source="external_event_store",
                payload={"event": event.model_dump(mode="json")},
            )
            try:
                reduction = reduce_external_event(
                    event,
                    task_state=TaskState.model_validate(state["task_state"]),
                    dialogue_state=DialogueState.model_validate(state.get("dialogue_state") or {}),
                    control_state=ControlState.model_validate(state.get("control_state") or {}),
                    max_dialogue_turns=runtime_config.max_dialogue_turns,
                )
                if reduction.requires_preempt:
                    if reduction.terminal:
                        await controller.halt(state["episode_id"], reason=reduction.reason)
                    else:
                        feedback = await controller.preempt(state["episode_id"], reason=reduction.reason)
                        if feedback is not None:
                            trace.append(
                                EpisodeEventType.ACTION_PREEMPTED,
                                tick_id=tick_id,
                                source="external_event_store",
                                payload={"event_id": event.event_id, "feedback": feedback.model_dump(mode="json")},
                            )
            except Exception:
                await event_store.release(event.event_id)
                raise
            event_type = EpisodeEventType.TASK_STATE_UPDATED if reduction.accepted else EpisodeEventType.EXTERNAL_EVENT_REJECTED
            trace.append(
                event_type,
                tick_id=tick_id,
                source="external_event_reducer",
                payload={
                    "event_id": event.event_id,
                    "reason": reduction.reason,
                    "task_state": reduction.task_state.model_dump(mode="json"),
                    "dialogue_state": reduction.dialogue_state.model_dump(mode="json"),
                    "control_state": reduction.control_state.model_dump(mode="json"),
                },
            )
            if reduction.reason == "task_revision_conflict":
                trace.append(
                    EpisodeEventType.REVISION_CONFLICT,
                    tick_id=tick_id,
                    source="external_event_reducer",
                    payload={"event_id": event.event_id, "expected_task_revision": event.expected_task_revision},
                )
            user_response: Any = None
            if reduction.reason == "interaction_resolved":
                user_response = event.content
                trace.append(
                    EpisodeEventType.INTERACTION_RESOLVED,
                    tick_id=tick_id,
                    source="hitl",
                    payload={"event_id": event.event_id, "interaction_id": event.interaction_id},
                )
                trace.append(
                    EpisodeEventType.USER_RESUME,
                    tick_id=tick_id,
                    source="hitl",
                    payload={"response": event.content, "interaction_id": event.interaction_id},
                )
            await event_store.acknowledge(event.event_id)
            if reduction.terminal:
                route = "terminal"
            elif reduction.task_state.status == TaskStatus.PAUSED:
                trace.status = EpisodeStatus.WAITING_FOR_USER
                trace.stop_reason = reduction.reason
                route = "wait"
            elif reduction.requires_system2:
                trace.status = EpisodeStatus.RUNNING
                trace.stop_reason = None
                route = "system2"
            else:
                route = "continue"
            return {
                "task_state": reduction.task_state.model_dump(mode="json"),
                "dialogue_state": reduction.dialogue_state.model_dump(mode="json"),
                "control_state": reduction.control_state.model_dump(mode="json"),
                "latest_external_event": event.model_dump(mode="json"),
                "user_response": user_response,
                "event_route": route,
                "done": reduction.terminal,
                "terminal_status": EpisodeStatus.CANCELLED.value if reduction.terminal else state.get("terminal_status"),
                "stop_reason": reduction.reason if reduction.terminal else state.get("stop_reason"),
                "tool_call_count": 0 if reduction.requires_system2 else int(state.get("tool_call_count", 0)),
                "episode_trace": trace.model_dump(mode="json"),
            }

        ingest.__name__ = f"event_gate_before_{continuation}"
        return ingest

    graph = StateGraph(EpisodeGraphState)
    graph.add_node("observe", observe)
    graph.add_node("system2", system2)
    graph.add_node("tool", execute_tool)
    graph.add_node("prepare_hitl", prepare_interaction)
    graph.add_node("await_hitl", await_interaction)
    graph.add_node("external_wait", await_external_resume)
    graph.add_node("system1", system1)
    graph.add_node("safety", safety)
    graph.add_node("dispatch", dispatch_action)
    graph.add_node("await_execution", await_execution)
    graph.add_node("terminal_control", terminal_control)
    graph.add_node("finish", finish)

    gate_destinations = {
        "event_before_system2": "system2",
        "event_before_tool": "tool",
        "event_before_system1": "system1",
        "event_before_safety": "safety",
        "event_before_dispatch": "dispatch",
        "event_before_terminal": "terminal_control",
    }
    for name, continuation in gate_destinations.items():
        graph.add_node(name, event_gate(continuation))
        graph.add_conditional_edges(
            name,
            _route_event_gate,
            {
                "continue": continuation,
                "system2": "system2",
                "wait": "external_wait",
                "terminal": "terminal_control",
            },
        )

    graph.add_edge(START, "observe")
    graph.add_edge("observe", "event_before_system2")
    graph.add_conditional_edges(
        "system2",
        _route_system2,
        {
            "tool": "event_before_tool",
            "hitl": "prepare_hitl",
            "system1": "event_before_system1",
            "terminal": "event_before_terminal",
        },
    )
    graph.add_edge("tool", "event_before_system2")
    graph.add_edge("prepare_hitl", "await_hitl")
    graph.add_edge("await_hitl", "event_before_system2")
    graph.add_edge("external_wait", "event_before_system2")
    graph.add_edge("system1", "event_before_safety")
    graph.add_conditional_edges(
        "safety",
        _route_after_safety,
        {"dispatch": "event_before_dispatch", "system2": "event_before_system2", "terminal": "terminal_control"},
    )
    graph.add_edge("dispatch", "await_execution")
    graph.add_conditional_edges(
        "await_execution",
        _route_after_execution,
        {"event": "event_before_system2", "observe": "observe", "terminal": "terminal_control"},
    )
    graph.add_conditional_edges(
        "terminal_control",
        _route_after_terminal_control,
        {"system2": "event_before_system2", "finish": "finish"},
    )
    graph.add_edge("finish", END)
    compiled = graph.compile(checkpointer=checkpointer)
    # Runtime wrapper 通过显式属性访问同一个 store/controller，避免平行全局状态。
    setattr(compiled, "urbannav_event_store", event_store)
    setattr(compiled, "urbannav_execution_manager", controller)
    return compiled


class EpisodeRuntime:
    """Start, resume and externally interrupt one compiled episode graph.

    Args:
        graph: Compiled graph returned by ``build_episode_graph``.
    """

    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.event_store: ExternalEventStore = getattr(graph, "urbannav_event_store")
        self.execution_manager: ExecutionManager = getattr(graph, "urbannav_execution_manager")

    async def run(
        self,
        *,
        episode_id: str,
        initial_context: dict[str, Any],
        session_id: str | None = None,
        thread_id: str | None = None,
    ) -> EpisodeGraphState:
        """Start one event-driven episode.

        Args:
            episode_id: Stable episode id.
            initial_context: Environment reset and initial task context.
            session_id: Optional route-session id.
            thread_id: LangGraph checkpoint thread id.

        Returns:
            Current graph state, which may be interrupted or terminal.
        """

        task = initial_task_state(episode_id=episode_id, initial_context=initial_context, route_session_id=session_id)
        dialogue = DialogueState()
        control = ControlState()
        trace = EpisodeTraceV2(episode_id=episode_id, session_id=session_id)
        # 初始 revision 写入 trace，让冻结评测能独立审计每个 action lease 是否过期。
        trace.append(
            EpisodeEventType.EPISODE_STARTED,
            tick_id=0,
            source="langgraph_episode_runtime",
            payload={
                "task_state": task.model_dump(mode="json"),
                "dialogue_state": dialogue.model_dump(mode="json"),
                "control_state": control.model_dump(mode="json"),
            },
        )
        state: EpisodeGraphState = {
            "episode_id": episode_id,
            "session_id": session_id,
            "tick_id": 0,
            "initial_context": dict(initial_context),
            "history": [],
            "task_state": task.model_dump(mode="json"),
            "dialogue_state": dialogue.model_dump(mode="json"),
            "control_state": control.model_dump(mode="json"),
            "episode_trace": trace.model_dump(mode="json"),
            "tool_call_count": 0,
            "total_tool_call_count": 0,
            "safety_retry_count": 0,
            "terminal_rejection_count": 0,
            "done": False,
        }
        invoke_kwargs = _checkpoint_invoke_kwargs(self.graph)
        return await self.graph.ainvoke(
            state,
            config=_thread_config(thread_id or episode_id),
            **invoke_kwargs,
        )

    async def resume(self, *, thread_id: str, user_response: Any) -> EpisodeGraphState:
        """Resume either a pending question or an externally paused task.

        Args:
            thread_id: Original checkpoint thread id.
            user_response: Human response or serialized external event.

        Returns:
            Resumed graph state.
        """

        invoke_kwargs = _checkpoint_invoke_kwargs(self.graph)
        return await self.graph.ainvoke(
            Command(resume=user_response),
            config=_thread_config(thread_id),
            **invoke_kwargs,
        )

    async def submit_event(self, event: ExternalUserEvent) -> bool:
        """Append an event and immediately preempt for safety-critical types.

        Args:
            event: External event targeting this runtime.

        Returns:
            Whether the event id was newly appended.
        """

        value = ExternalUserEvent.model_validate(event)
        inserted = await self.event_store.append(value)
        if inserted and value.event_type == ExternalUserEventType.CANCEL:
            await self.execution_manager.halt(value.episode_id, reason=value.event_type.value)
        elif inserted and value.event_type in {
            ExternalUserEventType.PAUSE,
            ExternalUserEventType.EMERGENCY_STOP,
            ExternalUserEventType.NEW_INSTRUCTION,
            ExternalUserEventType.MODIFY_TASK,
        }:
            await self.execution_manager.preempt(value.episode_id, reason=value.event_type.value)
        return inserted

    async def resume_with_event(self, *, thread_id: str, event: ExternalUserEvent) -> EpisodeGraphState:
        """Submit an event and resume a graph currently waiting in ``interrupt``.

        Args:
            thread_id: Checkpoint thread id.
            event: Resume, task update or correlated user response.

        Returns:
            Resumed graph state.
        """

        value = ExternalUserEvent.model_validate(event)
        await self.submit_event(value)
        return await self.resume(thread_id=thread_id, user_response=value.model_dump(mode="json"))


def _trace(state: EpisodeGraphState) -> EpisodeTraceV2:
    """Load the Pydantic episode trace from checkpoint-safe graph state."""

    return EpisodeTraceV2.model_validate(state["episode_trace"])


async def _await(value: Any) -> Any:
    """Await protocol results while permitting deterministic synchronous fakes."""

    return await value if inspect.isawaitable(value) else value


async def _route_system2(state: EpisodeGraphState) -> str:
    """Route one typed System2 action."""

    action = System2Action.model_validate(state["system2_action"])
    return {
        System2ActionType.TOOL_CALL: "tool",
        System2ActionType.ASK_USER: "hitl",
        System2ActionType.NAVIGATE: "system1",
        System2ActionType.COMPLETE_TASK: "terminal",
        System2ActionType.ABORT_TASK: "terminal",
    }[action.action_type]


async def _route_event_gate(state: EpisodeGraphState) -> str:
    """Route the result of one reusable external-event gate."""

    return str(state.get("event_route") or "continue")


async def _route_after_safety(state: EpisodeGraphState) -> str:
    """Dispatch accepted/replacement chunks or return to System2 recovery."""

    if state.get("done"):
        return "terminal"
    return "dispatch" if state.get("executable_system1_output") else "system2"


async def _route_after_execution(state: EpisodeGraphState) -> str:
    """Handle preemption, terminal feedback or the next observation."""

    if state.get("execution_interrupted"):
        return "event"
    return "terminal" if state.get("done") else "observe"


async def _route_after_terminal_control(state: EpisodeGraphState) -> str:
    """Retry a rejected completion request or finalize an accepted terminal state."""

    return "finish" if state.get("done") else "system2"


def _resume_payload_to_event(
    payload: Any,
    *,
    episode_id: str,
    default_type: ExternalUserEventType,
    interaction_id: str | None = None,
) -> ExternalUserEvent:
    """Normalize LangGraph resume payloads into the external-event contract."""

    if isinstance(payload, dict) and payload.get("event_type"):
        value = dict(payload)
        value.setdefault("episode_id", episode_id)
        return ExternalUserEvent.model_validate(value)
    return ExternalUserEvent(
        episode_id=episode_id,
        event_type=default_type,
        content=payload,
        interaction_id=interaction_id,
    )


def _next_decision_id(trace: EpisodeTraceV2, state: EpisodeGraphState) -> str:
    """Return a deterministic per-episode decision id for replay digests."""

    count = sum(event.event_type == EpisodeEventType.MODEL_OUTPUT for event in trace.events)
    return f"decision:{state['episode_id']}:{count}"


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    """Build LangGraph checkpoint configuration for one episode thread."""

    return {"configurable": {"thread_id": thread_id}}


def _checkpoint_invoke_kwargs(graph: Any) -> dict[str, str]:
    """Require synchronous durability only when a checkpointer is configured."""

    return {"durability": "sync"} if getattr(graph, "checkpointer", None) is not None else {}
