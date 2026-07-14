"""LangGraph Plan-and-Execute runtime for closed-loop UrbanNav episodes."""

from __future__ import annotations

import inspect
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field

from mcp_server.contracts.episode import EpisodeEventType, EpisodeStatus, EpisodeTraceV2
from mcp_server.contracts.tool import ToolCallV2, ToolResponseV2
from mcp_server.runtime.interfaces import (
    EnvironmentStepResult,
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
)
from mcp_server.runtime.state import EpisodeGraphState


class EpisodeRuntimeConfig(BaseModel):
    """Budgets and source labels for closed-loop graph execution.

    Args:
        max_ticks: Maximum environment transitions.
        max_tool_calls: Maximum tool calls across the episode.
        max_safety_retries: Maximum rejected chunks before failure.
        source: Runtime source written into episode events.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_ticks: int = Field(default=32, ge=1)
    max_tool_calls: int = Field(default=8, ge=0)
    max_safety_retries: int = Field(default=2, ge=0)
    source: str = "langgraph_episode_runtime"


def build_episode_graph(
    *,
    system2_policy: System2Policy,
    system1_policy: System1Policy,
    tool_executor: EpisodeToolExecutor,
    safety_validator: SafetyValidator,
    environment: EpisodeEnvironment,
    config: EpisodeRuntimeConfig | None = None,
    checkpointer: Any | None = None,
):
    """Build the hierarchical closed-loop navigation graph.

    Args:
        system2_policy: Slow policy selecting tools, interaction or navigation.
        system1_policy: Fast policy producing ego-frame continuous chunks.
        tool_executor: Registry-conformant live or replay executor.
        safety_validator: Deterministic pre-execution validator.
        environment: Live, simulator or frozen replay environment.
        config: Runtime budgets.
        checkpointer: LangGraph checkpointer used for HITL persistence.

    Returns:
        Compiled LangGraph state machine.
    """

    runtime_config = config or EpisodeRuntimeConfig()

    async def observe(state: EpisodeGraphState) -> dict[str, Any]:
        trace = _trace(state)
        if not state.get("observation"):
            result = ObservationResult.model_validate(await _await(environment.reset(dict(state.get("initial_context") or {}))))
            observation = result.observation
            info = result.info
            replay_key = result.replay_key
        else:
            observation = dict(state["observation"])
            info = dict(state.get("observation_info") or {})
            replay_key = info.get("replay_key")
        trace.append(
            EpisodeEventType.OBSERVATION_RECEIVED,
            tick_id=int(state.get("tick_id", 0)),
            source=runtime_config.source,
            payload={"observation": observation, "info": info, "replay_key": replay_key},
        )
        return {"observation": observation, "observation_info": info, "episode_trace": trace.model_dump(mode="json")}

    async def system2(state: EpisodeGraphState) -> dict[str, Any]:
        trace = _trace(state)
        action = System2Action.model_validate(await _await(system2_policy.decide(dict(state))))
        trace.append(
            EpisodeEventType.MODEL_OUTPUT,
            tick_id=int(state.get("tick_id", 0)),
            source="system2_policy",
            payload={"action": action.model_dump(mode="json")},
        )
        pending = action.user_request or {}
        return {
            "system2_action": action.model_dump(mode="json"),
            "pending_user_request": pending,
            "episode_trace": trace.model_dump(mode="json"),
        }

    async def execute_tool(state: EpisodeGraphState) -> dict[str, Any]:
        trace = _trace(state)
        action = System2Action.model_validate(state["system2_action"])
        assert action.tool_call is not None
        count = int(state.get("tool_call_count", 0))
        if count >= runtime_config.max_tool_calls:
            return {
                "done": True,
                "terminal_status": EpisodeStatus.TRUNCATED.value,
                "stop_reason": "max_tool_calls",
                "episode_trace": trace.model_dump(mode="json"),
            }
        trace.append(
            EpisodeEventType.TOOL_CALL,
            tick_id=int(state.get("tick_id", 0)),
            source="tool_executor",
            payload={"call": action.tool_call.model_dump(mode="json")},
        )
        raw_response = await _await(tool_executor.execute(action.tool_call, dict(state)))
        response = raw_response if isinstance(raw_response, ToolResponseV2) else ToolResponseV2.model_validate(raw_response)
        trace.append(
            EpisodeEventType.TOOL_RESPONSE,
            tick_id=int(state.get("tick_id", 0)),
            source="tool_executor",
            payload={"response": response.model_dump(mode="json")},
        )
        if response.state_delta:
            trace.append(
                EpisodeEventType.ROUTE_STATE_DELTA,
                tick_id=int(state.get("tick_id", 0)),
                source="tool_executor",
                payload={"state_delta": response.state_delta, "state_revision": response.state_revision},
            )
        result = response.result if isinstance(response.result, dict) else {}
        pending = result.get("dialogue_state", {}).get("pending_user_interaction") if isinstance(result.get("dialogue_state"), dict) else None
        return {
            "last_tool_response": response.model_dump(mode="json"),
            "pending_user_request": dict(pending or {}),
            "tool_call_count": count + 1,
            "episode_trace": trace.model_dump(mode="json"),
        }

    async def user_interaction(state: EpisodeGraphState) -> dict[str, Any]:
        request = dict(state.get("pending_user_request") or {})
        # interrupt 前不执行外部副作用；恢复时节点从头重放也不会重复工具调用。
        user_response = interrupt({"episode_id": state["episode_id"], "request": request})
        trace = _trace(state)
        tick_id = int(state.get("tick_id", 0))
        trace.append(EpisodeEventType.USER_INTERRUPT, tick_id=tick_id, source="hitl", payload={"request": request})
        trace.append(EpisodeEventType.USER_RESUME, tick_id=tick_id, source="hitl", payload={"response": user_response})
        return {
            "user_response": user_response,
            "pending_user_request": {},
            "episode_trace": trace.model_dump(mode="json"),
        }

    async def system1(state: EpisodeGraphState) -> dict[str, Any]:
        trace = _trace(state)
        output = System1Output.model_validate(await _await(system1_policy.predict(dict(state))))
        trace.append(
            EpisodeEventType.SYSTEM1_OUTPUT,
            tick_id=int(state.get("tick_id", 0)),
            source="system1_policy",
            payload={"output": output.model_dump(mode="json")},
        )
        return {"system1_output": output.model_dump(mode="json"), "episode_trace": trace.model_dump(mode="json")}

    async def safety(state: EpisodeGraphState) -> dict[str, Any]:
        trace = _trace(state)
        output = System1Output.model_validate(state["system1_output"])
        result = SafetyResult.model_validate(await _await(safety_validator.validate(output, dict(state))))
        retries = int(state.get("safety_retry_count", 0)) + (0 if result.accepted else 1)
        trace.append(
            EpisodeEventType.SAFETY_VALIDATION,
            tick_id=int(state.get("tick_id", 0)),
            source="safety_validator",
            payload={"result": result.model_dump(mode="json"), "retry_count": retries},
        )
        update: dict[str, Any] = {
            "safety_result": result.model_dump(mode="json"),
            "safety_retry_count": retries,
            "episode_trace": trace.model_dump(mode="json"),
        }
        if not result.accepted and retries > runtime_config.max_safety_retries:
            update.update(done=True, terminal_status=EpisodeStatus.FAILED.value, stop_reason="safety_retry_exhausted")
        return update

    async def environment_step(state: EpisodeGraphState) -> dict[str, Any]:
        trace = _trace(state)
        output = System1Output.model_validate(state["system1_output"])
        result = EnvironmentStepResult.model_validate(await _await(environment.step(output, dict(state))))
        tick_id = int(state.get("tick_id", 0))
        trace.append(
            EpisodeEventType.ENVIRONMENT_STEP,
            tick_id=tick_id,
            source="environment",
            payload={"result": result.model_dump(mode="json")},
        )
        next_tick = tick_id + 1
        update: dict[str, Any] = {
            "tick_id": next_tick,
            "observation": result.observation,
            "observation_info": {**result.info, "replay_key": result.replay_key},
            "environment_result": result.model_dump(mode="json"),
            "safety_retry_count": 0,
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

    async def finish(state: EpisodeGraphState) -> dict[str, Any]:
        trace = _trace(state)
        action = System2Action.model_validate(state["system2_action"]) if state.get("system2_action") else None
        status = EpisodeStatus(str(state.get("terminal_status") or EpisodeStatus.SUCCEEDED.value))
        reason = str(state.get("stop_reason") or ("system2_stop" if action and action.action_type == System2ActionType.STOP else "system2_finish"))
        if not trace.events or trace.events[-1].event_type != EpisodeEventType.EPISODE_FINISHED:
            trace.finish(status=status, stop_reason=reason, tick_id=int(state.get("tick_id", 0)), source=runtime_config.source)
        return {"done": True, "stop_reason": reason, "terminal_status": status.value, "episode_trace": trace.model_dump(mode="json")}

    graph = StateGraph(EpisodeGraphState)
    graph.add_node("observe", observe)
    graph.add_node("system2", system2)
    graph.add_node("tool", execute_tool)
    graph.add_node("hitl", user_interaction)
    graph.add_node("system1", system1)
    graph.add_node("safety", safety)
    graph.add_node("environment_step", environment_step)
    graph.add_node("finish", finish)
    graph.add_edge(START, "observe")
    graph.add_edge("observe", "system2")
    graph.add_conditional_edges("system2", _route_system2, {
        "tool": "tool", "hitl": "hitl", "system1": "system1", "finish": "finish",
    })
    graph.add_conditional_edges("tool", _route_after_tool, {"system2": "system2", "hitl": "hitl", "finish": "finish"})
    graph.add_edge("hitl", "system2")
    graph.add_edge("system1", "safety")
    graph.add_conditional_edges("safety", _route_after_safety, {"environment_step": "environment_step", "system2": "system2", "finish": "finish"})
    graph.add_conditional_edges("environment_step", _route_after_environment, {"observe": "observe", "finish": "finish"})
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


class EpisodeRuntime:
    """Convenience wrapper for starting and resuming a compiled episode graph.

    Args:
        graph: Compiled graph returned by ``build_episode_graph``.
    """

    def __init__(self, graph: Any) -> None:
        self.graph = graph

    async def run(
        self,
        *,
        episode_id: str,
        initial_context: dict[str, Any],
        session_id: str | None = None,
        thread_id: str | None = None,
    ) -> EpisodeGraphState:
        """Start one episode.

        Args:
            episode_id: Stable episode id.
            initial_context: Environment reset input.
            session_id: Optional route-session id.
            thread_id: LangGraph checkpoint thread id.

        Returns:
            Current graph state, which may be interrupted or terminal.
        """

        trace = EpisodeTraceV2(episode_id=episode_id, session_id=session_id)
        trace.append(EpisodeEventType.EPISODE_STARTED, tick_id=0, source="langgraph_episode_runtime", payload={})
        state: EpisodeGraphState = {
            "episode_id": episode_id,
            "session_id": session_id,
            "tick_id": 0,
            "initial_context": dict(initial_context),
            "history": [],
            "episode_trace": trace.model_dump(mode="json"),
            "tool_call_count": 0,
            "safety_retry_count": 0,
            "done": False,
        }
        # HITL 的恢复边界必须在 run 返回前落入 checkpointer；否则调用方紧接着
        # resume 时可能读到尚未完成的异步 checkpoint。
        invoke_kwargs = _checkpoint_invoke_kwargs(self.graph)
        return await self.graph.ainvoke(
            state,
            config=_thread_config(thread_id or episode_id),
            **invoke_kwargs,
        )

    async def resume(self, *, thread_id: str, user_response: Any) -> EpisodeGraphState:
        """Resume an episode paused by ``interrupt()``.

        Args:
            thread_id: Original checkpoint thread id.
            user_response: Human response delivered to the interrupted node.

        Returns:
            Resumed graph state.
        """

        invoke_kwargs = _checkpoint_invoke_kwargs(self.graph)
        return await self.graph.ainvoke(
            Command(resume=user_response),
            config=_thread_config(thread_id),
            **invoke_kwargs,
        )


def _trace(state: EpisodeGraphState) -> EpisodeTraceV2:
    """Load the Pydantic episode trace from checkpoint-safe graph state."""

    return EpisodeTraceV2.model_validate(state["episode_trace"])


async def _await(value: Any) -> Any:
    """Await protocol results while permitting deterministic synchronous fakes."""

    return await value if inspect.isawaitable(value) else value


async def _route_system2(state: EpisodeGraphState) -> str:
    """Route a typed System2 action to the corresponding graph node."""

    action = System2Action.model_validate(state["system2_action"])
    return {
        System2ActionType.TOOL_CALL: "tool",
        System2ActionType.ASK_USER: "hitl",
        System2ActionType.NAVIGATE: "system1",
        System2ActionType.FINISH: "finish",
        System2ActionType.STOP: "finish",
    }[action.action_type]


async def _route_after_tool(state: EpisodeGraphState) -> str:
    """Route tool results to HITL, another System2 turn or termination."""

    if state.get("done"):
        return "finish"
    return "hitl" if state.get("pending_user_request") else "system2"


async def _route_after_safety(state: EpisodeGraphState) -> str:
    """Execute accepted chunks and return rejected chunks to System2 recovery."""

    if state.get("done"):
        return "finish"
    result = SafetyResult.model_validate(state["safety_result"])
    return "environment_step" if result.accepted else "system2"


async def _route_after_environment(state: EpisodeGraphState) -> str:
    """Continue with the next observation or terminate the episode."""

    return "finish" if state.get("done") else "observe"


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    """Build LangGraph checkpoint configuration for one episode thread."""

    return {"configurable": {"thread_id": thread_id}}


def _checkpoint_invoke_kwargs(graph: Any) -> dict[str, str]:
    """Require synchronous durability only when a checkpointer is configured."""

    return {"durability": "sync"} if getattr(graph, "checkpointer", None) is not None else {}
