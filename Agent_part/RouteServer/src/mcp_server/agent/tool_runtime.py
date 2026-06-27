r"""Single-tool online runtime for SideWalk Agentic Stage2.

Stage2 cold-start training uses a fixed causal order:

``user -> assistant think -> optional tool_call -> optional tool_response
-> user [TRAJ_i] -> assistant final``.

The trajectory head reads hidden states at ``[TRAJ_i]`` tokens.  For an
autoregressive LM, hidden state \(h_t\) only attends to tokens at positions
\(\le t\).  Therefore runtime must execute the tool before appending the
trajectory readout prompt.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from mcp_server.agent.schemas import AgentStage2Case, AgentTrace, ToolCall, ToolResponse
from mcp_server.agent.stage2 import (
    build_final_assistant_text,
    final_navigation_user_content,
    initial_user_content,
)
from mcp_server.agent.tool_parser import parse_agent_action, parse_tool_call_text
from mcp_server.agent.tool_schema import get_tool_definitions


_ACTION_BLOCK_RE = re.compile(r"<(tool_call|finish)>(.*?)</\1>", re.DOTALL)
_TOOL_DECISION_RE = re.compile(r"<ToolDecision>(.*?)</ToolDecision>", re.DOTALL)


@dataclass(frozen=True)
class ToolBudget:
    """Runtime budget for Stage2 tool execution.

    Args:
        max_tool_calls: Maximum number of tool calls that runtime may execute
            from the first assistant turn.  Stage2 should keep this at ``1``.
        allowed_ops_by_tool: Optional per-tool operation allowlist.  Empty
            mapping means every schema-valid op is allowed.  Example:
            ``{"map_tool": ("search_poi", "create_route_session")}``.
    """

    max_tool_calls: int = 1
    allowed_ops_by_tool: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the budget into stable JSON-compatible metadata.

        Returns:
            Dict with ``max_tool_calls`` and sorted operation allowlists.
        """

        return {
            "max_tool_calls": int(self.max_tool_calls),
            "allowed_ops_by_tool": {
                tool: sorted(str(op) for op in ops)
                for tool, ops in sorted(self.allowed_ops_by_tool.items())
            },
        }

    def denial_for(self, call: ToolCall, *, parsed_tool_call_count: int) -> dict[str, Any] | None:
        """Return a structured denial error if the call violates the budget.

        Args:
            call: First parsed tool call from the model output.
            parsed_tool_call_count: Number of ``<tool_call>`` blocks found in
                the first assistant output.

        Returns:
            ``None`` if runtime may execute ``call``; otherwise a
            ``ToolResponse.error`` payload.
        """

        if self.max_tool_calls <= 0:
            return {
                "code": "tool_budget_exceeded",
                "message": "Runtime budget does not allow tool execution.",
                "max_tool_calls": self.max_tool_calls,
                "parsed_tool_call_count": parsed_tool_call_count,
            }
        allowed_ops = self.allowed_ops_by_tool.get(call.tool)
        if allowed_ops is not None and call.op not in set(allowed_ops):
            return {
                "code": "tool_op_not_allowed",
                "message": f"{call.tool}.{call.op} is not allowed by runtime budget.",
                "tool": call.tool,
                "op": call.op,
                "allowed_ops": sorted(str(op) for op in allowed_ops),
            }
        return None


class RuntimeTraceLogger:
    """Persist single-case runtime logs and traces for manual review.

    Args:
        artifact_dir: Directory where runtime JSON artifacts are written.  The
            logger creates one subdirectory per case id.

    The logger does not change the Stage2 schema.  It only appends review
    artifacts to ``trace.artifacts`` and stores the full runtime log under
    ``trace.metadata["runtime_log"]``.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)

    def persist(self, trace: AgentTrace, runtime_log: dict[str, Any]) -> list[dict[str, Any]]:
        """Write runtime log and trace JSON files.

        Args:
            trace: Fully assembled ``AgentTrace``.
            runtime_log: Structured runtime audit log.

        Returns:
            Artifact metadata entries for the written JSON files.
        """

        case_dir = self.artifact_dir / _safe_case_id(trace.case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        runtime_log_path = case_dir / "runtime_log.json"
        trace_path = case_dir / "agent_trace.json"
        artifacts = [
            {"type": "runtime_log", "path": str(runtime_log_path)},
            {"type": "runtime_trace", "path": str(trace_path)},
        ]
        runtime_log = dict(runtime_log)
        runtime_log["artifacts"] = list(runtime_log.get("artifacts") or []) + artifacts
        trace.artifacts.extend(_new_artifacts(trace.artifacts, artifacts))
        trace.metadata["runtime_log"] = runtime_log
        runtime_log_path.write_text(
            json.dumps(runtime_log, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        trace_path.write_text(
            json.dumps(trace.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return artifacts


@dataclass(frozen=True)
class AgentToolRuntimeConfig:
    """Configuration for ``AgentToolRuntime``.

    Args:
        source: Value written to ``trace.model_run.runner`` and final
            ``<finish>.teacher_source`` when runtime synthesizes the final
            assistant message for smoke tests.
        compute_metrics: Whether to attach protocol metrics to the returned
            trace.
        budget: Tool execution budget and per-op allowlist.
        review_artifact_dir: Optional directory for runtime log and trace JSON
            artifacts.  When set, ``RuntimeTraceLogger`` writes review files
            for every case.
    """

    source: str = "agent_tool_runtime"
    compute_metrics: bool = True
    budget: ToolBudget = field(default_factory=ToolBudget)
    review_artifact_dir: str | Path | None = None


class AgentToolRuntime:
    """Execute one Stage2 agent decision with at most one tool call.

    Args:
        executor: Optional ``ToolExecutor``-compatible object with an async
            ``execute(call, case=case)`` method.  When absent, a tool call is
            converted into a structured ``missing_tool_executor`` response so
            the trace remains replayable.
        config: Runtime configuration.

    This class intentionally does not implement a SpotAgent-style multi-round
    loop.  Stage2 only needs to verify the cold-start skill: decide whether one
    tool is needed, consume the tool result, then place the navigation readout
    tokens after that result.
    """

    def __init__(
        self,
        *,
        executor: Any | None = None,
        config: AgentToolRuntimeConfig | None = None,
        logger: RuntimeTraceLogger | None = None,
    ) -> None:
        self.executor = executor
        self.config = config or AgentToolRuntimeConfig()
        self.logger = logger
        if self.logger is None and self.config.review_artifact_dir is not None:
            self.logger = RuntimeTraceLogger(self.config.review_artifact_dir)

    async def run_case(
        self,
        case: AgentStage2Case,
        *,
        first_assistant_output: str,
        final_assistant_output: str | None = None,
        final_assistant_generator: Callable[[AgentTrace], str | Awaitable[str]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        phase1_response: str | None = None,
    ) -> AgentTrace:
        """Run one Stage2 case through the single-tool runtime.

        Args:
            case: Stage2 input case containing observation and supervision.
            first_assistant_output: Model output before runtime tool execution.
                It should contain ``<think><ToolDecision>...`` and may contain
                one sibling ``<tool_call>{...}</tool_call>`` block.  The
                executable tool call is normalized into an independent
                ``tool_call`` role.
            final_assistant_output: Optional model output after the runtime has
                appended ``tool_response`` and final ``[TRAJ_i]`` readout
                prompt.  When omitted, RouteServer synthesizes a deterministic
                final CoT/``<finish>`` for smoke tests.
            final_assistant_generator: Optional async/sync callback invoked
                after the runtime has appended the final readout user turn.
                The callback receives the prefix ``AgentTrace`` and should
                return the model-generated final assistant text.  It is used by
                ``OnlineAgentRunner`` for the second real model call.
            tools: Optional student-visible tool definitions.  Defaults to
                ``case.tools`` or the built-in RouteServer Stage2 tools.
            phase1_response: Optional Phase1 CoT response used when
                synthesizing the final assistant output.

        Returns:
            ``AgentTrace`` with normalized Stage2 roles.  Tool execution
            failures are represented as ``tool_response.success=false`` instead
            of exceptions.
        """

        # 从标准 Stage2 case 初始化 trace，后续所有 runtime 动作都写成可回放 role。
        trace = AgentTrace.from_case(
            case,
            model_run={
                "runner": self.config.source,
                "runtime": "single_tool",
                "budget": self.config.budget.to_dict(),
            },
        )
        trace.tools = list(tools or case.tools or get_tool_definitions())
        trace.add_turn("user", initial_user_content(case), metadata={"stage2_step": "initial_request"})

        # 第一轮 assistant 可以包含思考和工具请求；runtime 只解析/执行第一个工具。
        call, runtime_errors = self._parse_first_tool_call(first_assistant_output)
        parsed_tool_call_count = _count_tool_calls(first_assistant_output)
        if parsed_tool_call_count > self.config.budget.max_tool_calls:
            runtime_errors.append(
                f"extra_tool_calls_ignored:{parsed_tool_call_count - self.config.budget.max_tool_calls}"
            )
        decision, decision_errors = _parse_tool_decision(first_assistant_output)
        runtime_errors.extend(decision_errors)
        # 可执行 <tool_call> 不保留在 assistant content 中，避免把 action JSON 当普通 CoT。
        assistant_think = _strip_action_blocks(first_assistant_output).strip()
        trace.add_turn(
            "assistant",
            assistant_think,
            metadata={
                "stage2_step": "tool_decision_think",
                "runtime_parse": {
                    "tool_call": call.to_dict() if call is not None else None,
                    "tool_decision": decision,
                    "errors": list(runtime_errors),
                },
            },
        )

        response = None
        if call is not None:
            # tool_call / tool_response 独立成 role，和 ms-swift agent-template 的 loss mask 对齐。
            trace.add_turn("tool_call", call.to_ms_swift_dict(), tool=call.tool, metadata={"stage2_step": "tool_call"})
            response = await self._execute_tool(call, case, parsed_tool_call_count=parsed_tool_call_count)
            trace.add_turn(
                "tool_response",
                response.to_dict(),
                tool=call.tool,
                metadata={"stage2_step": "tool_response"},
            )
            trace.artifacts.extend(_new_artifacts(trace.artifacts, response.artifacts))

        # [TRAJ_i] readout 必须在 tool_response 后，trajectory head 的 hidden state 才能看到工具结果。
        trace.add_turn(
            "user",
            final_navigation_user_content(tool_used=call is not None),
            metadata={"stage2_step": "trajectory_readout_prompt"},
        )
        if final_assistant_output is not None:
            final_text = final_assistant_output.strip()
        elif final_assistant_generator is not None:
            # 真实在线评测在这里触发第二次模型调用，输入已经包含 tool_response 和 [TRAJ_i]。
            generated = final_assistant_generator(trace)
            if inspect.isawaitable(generated):
                generated = await generated
            final_text = str(generated).strip()
        else:
            final_text = build_final_assistant_text(
                case,
                call=call,
                response=response,
                source=self.config.source,
                phase1_response=phase1_response,
            )
        trace.add_turn(
            "assistant",
            final_text,
            metadata={
                "stage2_step": "final_navigation",
                "parsed_action": parse_agent_action(final_text).to_dict(),
            },
        )

        if runtime_errors:
            trace.metadata["runtime_errors"] = list(runtime_errors)
        if self.config.compute_metrics:
            # runtime 内部只附加协议指标，不在这里做训练/优化。
            from mcp_server.eval.metrics import compute_trace_metrics

            metric_result = compute_trace_metrics(trace, case)
            trace.metrics = metric_result.metrics
            if metric_result.errors:
                trace.metadata["metric_errors"] = metric_result.errors
        self._attach_runtime_log(trace, runtime_errors=runtime_errors)
        return trace

    def _parse_first_tool_call(self, first_assistant_output: str) -> tuple[ToolCall | None, list[str]]:
        """Parse a valid first-turn tool call without executing invalid JSON.

        Args:
            first_assistant_output: Raw assistant text from the model.

        Returns:
            ``(ToolCall | None, errors)``.  ``missing_tool_call`` is treated as
            a valid no-tool decision and is not returned as an error.  Other
            parser/schema errors are preserved in trace metadata.
        """

        call, errors = parse_tool_call_text(first_assistant_output)
        if errors == ["missing_tool_call"]:
            return None, []
        if errors:
            return None, errors
        return call, []

    async def _execute_tool(
        self,
        call: ToolCall,
        case: AgentStage2Case,
        *,
        parsed_tool_call_count: int,
    ) -> ToolResponse:
        """Execute one parsed tool call through the configured executor.

        Args:
            call: Parsed and schema-valid tool call.
            case: Stage2 case providing observation context.
            parsed_tool_call_count: Number of tool calls parsed from the first
                assistant turn.  It is recorded when a budget denial happens.

        Returns:
            Structured ``ToolResponse``.  Missing executor is represented as a
            failed response so the trace can still be validated and exported for
            failure-recovery supervision.
        """

        # budget / allowlist 拒绝也写成 failed tool_response，方便训练失败恢复策略。
        budget_denial = self.config.budget.denial_for(call, parsed_tool_call_count=parsed_tool_call_count)
        if budget_denial is not None:
            return ToolResponse(tool=call.tool, success=False, error=budget_denial, call_id=call.call_id)
        if self.executor is None:
            return ToolResponse(
                tool=call.tool,
                success=False,
                error={
                    "code": "missing_tool_executor",
                    "message": "AgentToolRuntime was created without a ToolExecutor.",
                },
                call_id=call.call_id,
            )
        return await self.executor.execute(call, case=case)

    def _attach_runtime_log(self, trace: AgentTrace, *, runtime_errors: list[str]) -> None:
        """Attach runtime audit metadata and optionally persist review files.

        Args:
            trace: Fully assembled trace.
            runtime_errors: Parser/budget warnings produced while building the
                trace.

        Returns:
            None.  The method mutates ``trace.metadata`` and, when configured,
            ``trace.artifacts``.
        """

        runtime_log = _runtime_log_from_trace(
            trace,
            budget=self.config.budget,
            runtime_errors=runtime_errors,
        )
        if self.logger is not None:
            self.logger.persist(trace, runtime_log)
        else:
            trace.metadata["runtime_log"] = runtime_log


def _strip_action_blocks(text: str) -> str:
    """Remove executable action blocks from the assistant reasoning turn."""

    return _ACTION_BLOCK_RE.sub("", text)


def _parse_tool_decision(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse the first ``<ToolDecision>`` payload for runtime audit metadata."""

    match = _TOOL_DECISION_RE.search(text)
    if not match:
        return None, ["missing_tool_decision"]
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError as exc:
        return None, [f"tool_decision_json_decode_error:{exc.msg}"]
    if not isinstance(payload, dict):
        return None, ["tool_decision_json_not_object"]
    return payload, []


def _count_tool_calls(text: str) -> int:
    """Count executable ``<tool_call>`` blocks in one assistant output."""

    return sum(1 for match in _ACTION_BLOCK_RE.finditer(text) if match.group(1) == "tool_call")


def _runtime_log_from_trace(
    trace: AgentTrace,
    *,
    budget: ToolBudget,
    runtime_errors: list[str],
) -> dict[str, Any]:
    """Build a structured runtime log from a normalized Stage2 trace."""

    # 记录完整 conversation 与工具 I/O，便于人工复盘 SpotAgent-style online trace。
    conversation = [turn.to_dict() for turn in trace.turns]
    tool_calls = [turn.to_dict() for turn in trace.turns if turn.role == "tool_call"]
    tool_responses = [turn.to_dict() for turn in trace.turns if turn.role == "tool_response"]
    response_errors = []
    for turn in tool_responses:
        content = turn.get("content") if isinstance(turn, dict) else None
        if isinstance(content, dict) and content.get("success") is False:
            response_errors.append(content.get("error") or {"code": "tool_failed"})
    metric_errors = trace.metadata.get("metric_errors") if isinstance(trace.metadata, dict) else None
    errors: list[Any] = list(runtime_errors)
    if metric_errors:
        errors.extend(metric_errors if isinstance(metric_errors, list) else [metric_errors])
    errors.extend(response_errors)
    return {
        "case_id": trace.case_id,
        "runtime": "single_tool",
        "budget": budget.to_dict(),
        "conversation": conversation,
        "tool_calls": tool_calls,
        "tool_responses": tool_responses,
        "errors": errors,
        "artifacts": list(trace.artifacts),
        "metrics": dict(trace.metrics),
    }


def _safe_case_id(case_id: str) -> str:
    """Return a filesystem-safe case id for artifact paths."""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_id or "case")).strip("._")
    return safe or "case"


def _new_artifacts(existing: list[dict[str, Any]], additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return artifact additions not already present by ``(type, path)``."""

    seen = {
        (str(item.get("type")), str(item.get("path") or item.get("source_path") or ""))
        for item in existing
        if isinstance(item, dict)
    }
    result: list[dict[str, Any]] = []
    for item in additions:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("type")), str(item.get("path") or item.get("source_path") or ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(item))
    return result
