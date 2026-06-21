"""Metrics for Agent traces.

第一版指标只评估可解析性、工具选择、参数有效性和 route update 是否成功。
它不依赖外部 LVLM 也不评估自然语言质量。
"""

from __future__ import annotations

from typing import Any

from mcp_server.agent.schemas import AgentEvalCase, AgentMetricResult, AgentTrace, ToolResponse
from mcp_server.agent.tool_parser import ParsedAgentAction, parse_agent_action
from mcp_server.agent.tool_schema import validate_tool_call


def _assistant_actions(trace: AgentTrace) -> tuple[list[ParsedAgentAction], list[str]]:
    actions: list[ParsedAgentAction] = []
    errors: list[str] = []
    for turn in trace.turns:
        if turn.role != "assistant" or not isinstance(turn.content, str):
            continue
        parsed = parse_agent_action(turn.content)
        actions.append(parsed)
        errors.extend(parsed.errors)
    return actions, errors


def _tool_responses(trace: AgentTrace) -> list[ToolResponse]:
    responses: list[ToolResponse] = []
    for turn in trace.turns:
        if turn.role != "tool_response":
            continue
        content = turn.content
        if isinstance(content, dict):
            responses.append(
                ToolResponse(
                    tool=str(content.get("tool") or turn.tool or ""),
                    success=bool(content.get("success")),
                    result=content.get("result") or {},
                    error=content.get("error"),
                    artifacts=list(content.get("artifacts") or []),
                    call_id=content.get("call_id"),
                )
            )
    return responses


def _expected_ops(case: AgentEvalCase | None, trace: AgentTrace) -> list[str]:
    payload: dict[str, Any] = {}
    if case is not None:
        payload.update(case.ground_truth)
    payload.update(trace.supervision)
    ops = payload.get("expected_tool_ops") or payload.get("tool_ops")
    if not ops:
        return []
    if isinstance(ops, list):
        return [str(item.get("op") if isinstance(item, dict) else item) for item in ops]
    return [str(ops)]


def compute_trace_metrics(trace: AgentTrace, case: AgentEvalCase | None = None) -> AgentMetricResult:
    """Compute first-pass Agent metrics.

    输出字段:
        ``parse_success``: assistant action block 是否可解析。
        ``tool_name_accuracy``: 工具名是否在允许集合内，若有 GT 则比较序列。
        ``tool_op_accuracy``: 有 ``expected_tool_ops`` 时比较 op 序列。
        ``argument_validity``: tool-call schema 参数是否有效。
        ``finish_validity``: 是否输出合法 ``<finish>{...}</finish>``。
        ``route_update_success``: route-session 类 tool_response 是否成功更新。
    """

    actions, parse_errors = _assistant_actions(trace)
    tool_calls = [action.tool_call for action in actions if action.tool_call is not None]
    finishes = [action.finish for action in actions if action.finish is not None]

    valid_calls = []
    validation_errors: list[str] = []
    for call in tool_calls:
        is_valid, errors = validate_tool_call(call)
        valid_calls.append(is_valid)
        validation_errors.extend(errors)

    expected_ops = _expected_ops(case, trace)
    called_ops = [str(call.op) for call in tool_calls if call.op]
    if expected_ops:
        tool_op_accuracy = 1.0 if called_ops == expected_ops else 0.0
    else:
        tool_op_accuracy = None

    expected_names = []
    if case is not None:
        expected_names = [str(item) for item in case.ground_truth.get("expected_tool_names") or []]
    called_names = [call.tool for call in tool_calls]
    if expected_names:
        tool_name_accuracy = 1.0 if called_names == expected_names else 0.0
    elif tool_calls:
        tool_name_accuracy = 1.0 if all(validate_tool_call(call)[0] for call in tool_calls) else 0.0
    else:
        tool_name_accuracy = None

    responses = _tool_responses(trace)
    route_update_success = None
    for response in responses:
        result = response.result if isinstance(response.result, dict) else {}
        if "session" in result or "route_plan" in result or "session_changed" in result:
            route_update_success = 1.0 if response.success else 0.0
            break

    metrics: dict[str, float | None] = {
        "parse_success": 1.0 if actions and not parse_errors else 0.0,
        "tool_name_accuracy": tool_name_accuracy,
        "tool_op_accuracy": tool_op_accuracy,
        "argument_validity": (sum(1.0 for item in valid_calls if item) / len(valid_calls)) if valid_calls else None,
        "finish_validity": 1.0 if finishes else 0.0,
        "route_update_success": route_update_success,
    }
    return AgentMetricResult(metrics=metrics, errors=parse_errors + validation_errors)
