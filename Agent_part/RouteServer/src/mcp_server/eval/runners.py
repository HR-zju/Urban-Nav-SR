"""Agent runners used by offline evaluation.

第一版 runner 不训练模型，只把不同来源的策略统一成 ``AgentTrace``：
rule baseline、RouteServer teacher、后续外部 LVLM、本项目 E2E checkpoint 都走同一接口。
"""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Awaitable, Callable, Iterable

from mcp_server.agent.schemas import AgentEvalCase, AgentTrace, ToolCall
from mcp_server.agent.tool_parser import parse_agent_action
from mcp_server.agent.tool_schema import get_tool_definitions
from mcp_server.eval.metrics import compute_trace_metrics

if TYPE_CHECKING:
    from mcp_server.agent.tool_executor import ToolExecutor


class AgentRunner(ABC):
    """Base class for anything that maps ``AgentEvalCase`` to ``AgentTrace``."""

    @abstractmethod
    async def run_case(self, case: AgentEvalCase) -> AgentTrace:
        """Run one case and return a replayable trace."""


class RuleRunner(AgentRunner):
    """Small deterministic baseline for protocol smoke tests.

    输入:
        ``AgentEvalCase``。

    输出:
        包含 assistant ``<tool_call>`` 和可选 ``<tool_response>/<finish>`` 的
        ``AgentTrace``。规则很简单，目的只是验证协议和 evaluator 不代表最终策略。
    """

    def __init__(self, *, executor: "ToolExecutor | None" = None) -> None:
        self.executor = executor

    async def run_case(self, case: AgentEvalCase) -> AgentTrace:
        trace = AgentTrace.from_case(case, model_run={"runner": "rule"})
        trace.tools = case.tools or get_tool_definitions()
        trace.add_turn("user", "".join("<image>" for _ in case.images) + case.user_request)

        call = self._select_tool_call(case)
        assistant_text = (
            '<reasoning_summary>{"source":"rule_runner","decision":"tool_call"}</reasoning_summary>\n'
            f"<tool_call>{json.dumps(call.to_dict(), ensure_ascii=False)}</tool_call>"
        )
        parsed = parse_agent_action(assistant_text)
        trace.add_turn("assistant", assistant_text, metadata={"parsed_action": parsed.to_dict()})

        if self.executor is not None:
            response = await self.executor.execute(call, case=case)
            trace.add_turn("tool_response", response.to_dict(), tool=call.tool)
            finish_payload = {
                "route_action": "use_tool_result",
                "tool": call.tool,
                "op": call.op,
                "success": response.success,
            }
        else:
            finish_payload = {
                "route_action": "pending_tool_execution",
                "tool": call.tool,
                "op": call.op,
            }
        finish_text = f"<finish>{json.dumps(finish_payload, ensure_ascii=False)}</finish>"
        trace.add_turn("assistant", finish_text, metadata={"parsed_action": parse_agent_action(finish_text).to_dict()})
        metric_result = compute_trace_metrics(trace, case)
        trace.metrics = metric_result.metrics
        if metric_result.errors:
            trace.metadata["metric_errors"] = metric_result.errors
        return trace

    def _select_tool_call(self, case: AgentEvalCase) -> ToolCall:
        request = case.user_request.lower()
        location = case.observation.get("current_location") or case.observation.get("origin") or ""
        if any(keyword in request for keyword in ["厕所", "洗手间", "restroom", "toilet"]):
            return ToolCall(tool="map_tool", arguments={"op": "search_poi", "query": "restroom", "location": location})
        if any(keyword in request for keyword in ["咖啡", "坐", "休息", "coffee", "bench", "rest"]):
            return ToolCall(tool="map_tool", arguments={"op": "search_poi", "query": "rest stop", "location": location})
        if "?" in request or "哪里" in request or "which" in request:
            return ToolCall(
                tool="user_interaction_tool",
                arguments={"op": "request_clarification", "question": case.user_request, "choices": []},
            )
        destination = case.ground_truth.get("destination") or case.observation.get("destination")
        origin = case.observation.get("origin") or case.observation.get("current_location")
        if origin and destination:
            return ToolCall(
                tool="map_tool",
                arguments={"op": "create_route_session", "origin": origin, "destination": destination, "mode": "foot"},
            )
        return ToolCall(
            tool="user_interaction_tool",
            arguments={"op": "request_clarification", "question": "请确认目的地或导航偏好。", "choices": []},
        )


class RouteServerTeacherRunner(AgentRunner):
    """Adapter for existing RouteServer teacher functions.

    teacher callable:
        输入 ``AgentEvalCase``；输出可以是 ``AgentTrace``、assistant 文本、或 trace dict。
        这里保持兼容旧用法。

    default teacher:
        如果没有传 teacher callable，则使用 ``RouteServerAgentTeacher``，把旧
        IntentParser / GoalDecomposer / POI recommendation / visual prompt teacher
        包装成统一 ``AgentTrace``。
    """

    def __init__(
        self,
        teacher: Callable[[AgentEvalCase], AgentTrace | dict | str | Awaitable[AgentTrace | dict | str]] | None = None,
        *,
        app_context: object | None = None,
        use_llm: bool = True,
        execute_tools: bool = False,
        enable_visual_prompt: bool = False,
    ) -> None:
        self.teacher = teacher
        self.app_context = app_context
        self.use_llm = bool(use_llm)
        self.execute_tools = bool(execute_tools)
        self.enable_visual_prompt = bool(enable_visual_prompt)

    async def run_case(self, case: AgentEvalCase) -> AgentTrace:
        if self.teacher is None:
            # Lazy import，避免只使用 eval.cases/metrics 时强依赖 map provider 或视觉环境。
            from mcp_server.teacher.trace_builder import RouteServerAgentTeacher, RouteServerTeacherConfig

            teacher = RouteServerAgentTeacher(
                app_context=self.app_context,
                config=RouteServerTeacherConfig(
                    use_llm=self.use_llm,
                    execute_tools=self.execute_tools,
                    enable_visual_prompt=self.enable_visual_prompt,
                ),
            )
            return await teacher.run_case(case)

        result = self.teacher(case)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, AgentTrace):
            trace = result
        elif isinstance(result, dict):
            trace = AgentTrace.from_dict(result)
        else:
            trace = AgentTrace.from_case(case, model_run={"runner": "routeserver_teacher"})
            trace.tools = case.tools or get_tool_definitions()
            trace.add_turn("user", "".join("<image>" for _ in case.images) + case.user_request)
            trace.add_turn("assistant", str(result), metadata={"parsed_action": parse_agent_action(str(result)).to_dict()})
        metric_result = compute_trace_metrics(trace, case)
        trace.metrics = metric_result.metrics
        if metric_result.errors:
            trace.metadata["metric_errors"] = metric_result.errors
        return trace


async def run_cases(runner: AgentRunner, cases: Iterable[AgentEvalCase]) -> list[AgentTrace]:
    traces: list[AgentTrace] = []
    for case in cases:
        traces.append(await runner.run_case(case))
    return traces
