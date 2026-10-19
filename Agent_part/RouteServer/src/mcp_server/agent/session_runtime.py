"""Session-level runtime orchestration for SideWalk Agent ticks.

This module deliberately stays above the single-tick Stage2 runtime.  It does
not build route context, change prompts, or execute tools directly; it only
orders multiple ``AgentStage2Case`` ticks, records validations, and stops on
auditable session conditions.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from mcp_server.agent.schemas import AgentStage2Case, AgentTrace
from mcp_server.agent.tool_parser import parse_finish_text
from mcp_server.agent.tool_schema import USER_INTERACTION_TOOL_NAME
from mcp_server.eval.agent_tools.stage2_validator import Stage2ValidationResult, validate_stage2_trace


SESSION_TRACE_SCHEMA_VERSION = "sidewalk_agent_session_trace_v1"


class SessionCaseRunner(Protocol):
    """Protocol for one-tick runners used by ``RouteSessionRuntime``.

    Implementations may expose only ``run_case`` or additionally expose
    ``run_case_with_report``.  ``OnlineAgentRunner`` uses the richer method,
    while offline baselines can stay on the simpler ``AgentRunner`` interface.
    """

    async def run_case(self, case: AgentStage2Case) -> AgentTrace:
        """Run one Stage2 case.

        Args:
            case: The current navigation tick encoded as a Stage2 case.

        Returns:
            Replayable ``AgentTrace`` for the tick.
        """


@dataclass(frozen=True)
class SessionRuntimeConfig:
    """Configuration for ``RouteSessionRuntime``.

    Args:
        session_id: Stable id written into the session report.
        max_ticks: Maximum number of Stage2 ticks to run before stopping.
        stop_on_user_interaction: Stop when the tick requests user input.
        stop_on_invalid_trace: Stop when Stage2 trace validation fails.
        source: Runtime name written into session metadata and tick traces.
    """

    session_id: str = "route-session-runtime"
    max_ticks: int = 8
    stop_on_user_interaction: bool = True
    stop_on_invalid_trace: bool = True
    source: str = "route_session_runtime"


@dataclass(frozen=True)
class SessionTickResult:
    """Result for one session tick.

    Args:
        tick_index: Zero-based tick index within the session.
        case_id: Source ``AgentStage2Case.case_id``.
        trace: Replayable single-tick Stage2 trace.
        finish: Parsed final ``<finish>`` payload, if present.
        trace_validation: Validator result for the trace.
        row_validation: Optional exported-row validator result from a rich runner.
        route_action: Parsed ``finish.route_action`` value.
        needs_user_interaction: Whether this tick should pause for the user.
        ms_swift_row: Optional ms-swift replay row from a rich runner.
    """

    tick_index: int
    case_id: str
    trace: AgentTrace
    finish: dict[str, Any] | None
    trace_validation: Stage2ValidationResult
    row_validation: Stage2ValidationResult | None = None
    route_action: str | None = None
    needs_user_interaction: bool = False
    ms_swift_row: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable tick report.

        Returns:
            Dictionary containing trace, validation, finish, and routing flags.
        """

        agent_step_result = self.trace.to_agent_step_result(
            validation=self.trace_validation.to_dict(),
        ).to_dict()
        return {
            "tick_index": self.tick_index,
            "case_id": self.case_id,
            "route_action": self.route_action,
            "needs_user_interaction": self.needs_user_interaction,
            "finish": self.finish,
            "trace_validation": self.trace_validation.to_dict(),
            "row_validation": self.row_validation.to_dict() if self.row_validation else None,
            "trace": self.trace.to_dict(),
            "ms_swift_row": self.ms_swift_row,
            "agent_step_result": agent_step_result,
        }


@dataclass(frozen=True)
class RouteSessionRunResult:
    """Aggregated result for one session runtime invocation.

    Args:
        session_id: Stable session id from ``SessionRuntimeConfig``.
        ticks: Ordered tick results.
        stopped_reason: One of ``completed_all_ticks``, ``max_ticks``,
            ``user_interaction``, or ``invalid_trace``.
        metadata: Additional runtime metadata.
        schema_version: Session report schema version.
    """

    session_id: str
    ticks: list[SessionTickResult]
    stopped_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SESSION_TRACE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable session report.

        Returns:
            Dictionary suitable for online logging or offline replay indexing.
        """

        from mcp_server.contracts.trace_adapter import merge_agent_traces_to_episode

        episode_trace = merge_agent_traces_to_episode(
            self.traces,
            episode_id=self.session_id,
            session_id=self.session_id,
            stop_reason=self.stopped_reason,
            source=str(self.metadata.get("source") or "route_session_runtime"),
        )
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "stopped_reason": self.stopped_reason,
            "tick_count": len(self.ticks),
            "ticks": [tick.to_dict() for tick in self.ticks],
            "metadata": dict(self.metadata),
            "episode_trace_v2": episode_trace.model_dump(mode="json"),
        }

    @property
    def traces(self) -> list[AgentTrace]:
        """Return single-tick traces contained in this session.

        Returns:
            Ordered ``AgentTrace`` objects, directly exportable to Stage2 rows.
        """

        return [tick.trace for tick in self.ticks]


# Public compatibility name for the session-level replay contract.  The
# underlying dataclass name stays unchanged to avoid breaking existing imports.
SessionTrace = RouteSessionRunResult


class RouteSessionRuntime:
    """Run multiple Stage2 ticks as one auditable navigation session.

    Args:
        runner: Single-tick runner, usually ``OnlineAgentRunner`` for deployed
            inference or ``RuleRunner`` for deterministic tests.
        config: Session runtime configuration.

    Raises:
        ValueError: If ``max_ticks`` is less than one.
    """

    def __init__(self, *, runner: SessionCaseRunner, config: SessionRuntimeConfig | None = None) -> None:
        """Create a session runtime.

        Args:
            runner: Object that can execute one ``AgentStage2Case``.
            config: Optional runtime configuration.

        Raises:
            ValueError: If ``config.max_ticks`` is less than one.
        """

        self.runner = runner
        self.config = config or SessionRuntimeConfig()
        if self.config.max_ticks < 1:
            raise ValueError("SessionRuntimeConfig.max_ticks must be >= 1")

    async def run_tick(self, case: AgentStage2Case, *, tick_index: int) -> SessionTickResult:
        """Run and validate one session tick.

        Args:
            case: Current tick encoded as a canonical Stage2 case.
            tick_index: Zero-based index in the parent session.

        Returns:
            ``SessionTickResult`` with parsed finish and validation fields.
        """

        run_result = await self._run_case(case)
        trace = _extract_trace(run_result)
        trace.metadata.setdefault("session_runtime", {})
        trace.metadata["session_runtime"].update(
            {
                "session_id": self.config.session_id,
                "tick_index": tick_index,
                "source": self.config.source,
            }
        )
        # 单 tick runner 可能已附加 episode view；这里用 session/tick id
        # 重新生成，保证多个 tick 能合并为同一条因果事件流。
        from mcp_server.contracts.trace_adapter import attach_episode_trace_v2

        attach_episode_trace_v2(
            trace,
            episode_id=self.config.session_id,
            session_id=self.config.session_id,
            tick_id=tick_index,
            source=self.config.source,
        )
        trace_validation = _extract_trace_validation(run_result) or validate_stage2_trace(trace, case)
        row_validation = _extract_row_validation(run_result)
        finish = _extract_finish(trace)
        route_action = _extract_route_action(finish)
        needs_user_interaction = _needs_user_interaction(trace, finish)
        return SessionTickResult(
            tick_index=tick_index,
            case_id=case.case_id,
            trace=trace,
            finish=finish,
            trace_validation=trace_validation,
            row_validation=row_validation,
            route_action=route_action,
            needs_user_interaction=needs_user_interaction,
            ms_swift_row=_extract_ms_swift_row(run_result),
        )

    async def run_until_stop_condition(self, cases: Iterable[AgentStage2Case]) -> RouteSessionRunResult:
        """Run ordered ticks until the configured session stop condition.

        Args:
            cases: Ordered Stage2 cases representing successive navigation
                observations for the same route session.

        Returns:
            Aggregated session report.  Each tick remains a normal Stage2 trace
            so it can be exported independently for ms-swift training.
        """

        ticks: list[SessionTickResult] = []
        stopped_reason = "completed_all_ticks"
        for tick_index, case in enumerate(cases):
            if tick_index >= self.config.max_ticks:
                stopped_reason = "max_ticks"
                break
            tick = await self.run_tick(case, tick_index=tick_index)
            ticks.append(tick)
            if self.config.stop_on_invalid_trace and not tick.trace_validation.valid:
                stopped_reason = "invalid_trace"
                break
            if self.config.stop_on_user_interaction and tick.needs_user_interaction:
                stopped_reason = "user_interaction"
                break
        return RouteSessionRunResult(
            session_id=self.config.session_id,
            ticks=ticks,
            stopped_reason=stopped_reason,
            metadata={
                "source": self.config.source,
                "max_ticks": self.config.max_ticks,
                "stop_on_user_interaction": self.config.stop_on_user_interaction,
                "stop_on_invalid_trace": self.config.stop_on_invalid_trace,
            },
        )

    async def run_cases(self, cases: Iterable[AgentStage2Case]) -> RouteSessionRunResult:
        """Alias for ``run_until_stop_condition``.

        Args:
            cases: Ordered Stage2 cases for one route session.

        Returns:
            Aggregated session report.
        """

        return await self.run_until_stop_condition(cases)

    async def _run_case(self, case: AgentStage2Case) -> Any:
        if hasattr(self.runner, "run_case_with_report"):
            result = self.runner.run_case_with_report(case)  # type: ignore[attr-defined]
        else:
            result = self.runner.run_case(case)
        if inspect.isawaitable(result):
            return await result
        return result


def _extract_trace(run_result: Any) -> AgentTrace:
    if isinstance(run_result, AgentTrace):
        return run_result
    trace = getattr(run_result, "trace", None)
    if isinstance(trace, AgentTrace):
        return trace
    if isinstance(run_result, dict):
        payload = run_result.get("trace")
        if isinstance(payload, AgentTrace):
            return payload
        if isinstance(payload, dict):
            return AgentTrace.from_dict(payload)
    raise TypeError("session runner must return AgentTrace or an object with .trace")


def _extract_trace_validation(run_result: Any) -> Stage2ValidationResult | None:
    validation = getattr(run_result, "trace_validation", None)
    return validation if isinstance(validation, Stage2ValidationResult) else None


def _extract_row_validation(run_result: Any) -> Stage2ValidationResult | None:
    validation = getattr(run_result, "row_validation", None)
    return validation if isinstance(validation, Stage2ValidationResult) else None


def _extract_ms_swift_row(run_result: Any) -> dict[str, Any] | None:
    row = getattr(run_result, "ms_swift_row", None)
    return dict(row) if isinstance(row, dict) else None


def _extract_finish(trace: AgentTrace) -> dict[str, Any] | None:
    for turn in reversed(trace.turns):
        if turn.role != "assistant" or not isinstance(turn.content, str):
            continue
        finish, errors = parse_finish_text(turn.content)
        if finish is not None and not errors:
            return finish
    return None


def _extract_route_action(finish: dict[str, Any] | None) -> str | None:
    if not isinstance(finish, dict):
        return None
    value = finish.get("route_action")
    return str(value) if value is not None else None


def _needs_user_interaction(trace: AgentTrace, finish: dict[str, Any] | None) -> bool:
    runtime_state = trace.metadata.get("runtime_state") if isinstance(trace.metadata, dict) else None
    if runtime_state == "WAITING_FOR_USER":
        return True
    dialogue_state = trace.metadata.get("dialogue_state") if isinstance(trace.metadata, dict) else None
    if isinstance(dialogue_state, dict) and dialogue_state.get("pending_user_interaction"):
        return True
    if isinstance(finish, dict):
        if finish.get("route_action") == "request_user_input":
            return True
        if finish.get("tool") == USER_INTERACTION_TOOL_NAME:
            return True
    for turn in trace.turns:
        payload = _content_dict(turn.content)
        if turn.role == "tool_call" and _tool_name(payload) == USER_INTERACTION_TOOL_NAME:
            return True
        if turn.role == "tool_response":
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
            if payload.get("tool") == USER_INTERACTION_TOOL_NAME or result.get("await_user") is True:
                return True
    return False


def _content_dict(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {}


def _tool_name(payload: dict[str, Any]) -> str | None:
    value = payload.get("tool") or payload.get("name")
    return str(value) if value is not None else None
