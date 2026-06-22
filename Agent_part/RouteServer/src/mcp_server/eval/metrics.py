"""Metrics for final Agentic Stage2 traces.

指标只评估协议可解析性、工具选择、参数有效性、最终 CoT/finish 和
``tool_response -> [TRAJ_*]`` 因果顺序。它不依赖外部 LVLM 也不评估自然语言质量。
"""

from __future__ import annotations

import json
from typing import Any

from mcp_server.agent.schemas import AgentMetricResult, AgentStage2Case, AgentTrace, ToolCall, ToolResponse
from mcp_server.agent.stage2 import NAVIGATION_READOUT_TOKENS
from mcp_server.agent.tool_parser import parse_finish_text, parse_tool_call_text
from mcp_server.agent.tool_schema import validate_tool_call


def _json_dict(content: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if isinstance(content, dict):
        return dict(content), []
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            return None, [f"json_decode_error:{exc.msg}"]
        if isinstance(payload, dict):
            return payload, []
        return None, ["json_not_object"]
    return None, [f"unsupported_content_type:{type(content).__name__}"]


def _tool_calls(trace: AgentTrace) -> tuple[list[ToolCall], list[str]]:
    calls: list[ToolCall] = []
    errors: list[str] = []
    for turn in trace.turns:
        if turn.role == "tool_call":
            payload, payload_errors = _json_dict(turn.content)
            errors.extend(payload_errors)
            if payload is not None:
                calls.append(ToolCall.from_dict(payload))
            continue
        if turn.role == "assistant" and isinstance(turn.content, str) and "<tool_call>" in turn.content:
            call, parse_errors = parse_tool_call_text(turn.content)
            errors.extend(parse_errors)
            if call is not None:
                calls.append(call)
    return calls, errors


def _finishes(trace: AgentTrace) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Extract all finish texts from the trace.
    Args:
        trace: The AgentTrace to extract
    Returns:
        finishes: [dict[str, Any]]: A list of finish texts extracted from the trace.
        errors: [str]: A list of errors encountered during extraction.
    
    """
    finishes: list[dict[str, Any]] = []
    errors: list[str] = []
    for turn in trace.turns:
        if turn.role != "assistant" or not isinstance(turn.content, str) or "<finish>" not in turn.content:
            continue
        finish, finish_errors = parse_finish_text(turn.content)
        errors.extend(finish_errors)
        if finish is not None:
            finishes.append(finish)
    return finishes, errors


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


def _role_order_valid(trace: AgentTrace) -> bool:
    initial_user = _first_index(trace, "user", start=0)
    think = _first_index(trace, "assistant", start=(initial_user + 1 if initial_user is not None else 0))
    tool_call = _first_index(trace, "tool_call", start=(think + 1 if think is not None else 0))
    tool_response = _first_index(trace, "tool_response", start=(tool_call + 1 if tool_call is not None else 0))
    readout_user = _first_readout_user_index(trace, start=(tool_response + 1 if tool_response is not None else 0))
    final_assistant = _first_final_assistant_index(trace, start=(readout_user + 1 if readout_user is not None else 0))
    return all(index is not None for index in (initial_user, think, tool_call, tool_response, readout_user, final_assistant))


def _trajectory_readout_order_valid(trace: AgentTrace) -> bool:
    last_tool_response = None
    for index, turn in enumerate(trace.turns):
        if turn.role == "tool_response":
            last_tool_response = index
    if last_tool_response is None:
        return False
    readout_user = _first_readout_user_index(trace, start=last_tool_response + 1)
    return readout_user is not None


def _first_index(trace: AgentTrace, role: str, *, start: int) -> int | None:
    for index, turn in enumerate(trace.turns[start:], start=start):
        if turn.role == role:
            return index
    return None


def _first_readout_user_index(trace: AgentTrace, *, start: int) -> int | None:
    for index, turn in enumerate(trace.turns[start:], start=start):
        if turn.role == "user" and isinstance(turn.content, str) and _contains_readout_tokens(turn.content):
            return index
    return None


def _first_final_assistant_index(trace: AgentTrace, *, start: int) -> int | None:
    for index, turn in enumerate(trace.turns[start:], start=start):
        if turn.role == "assistant" and isinstance(turn.content, str) and "<finish>" in turn.content:
            return index
    return None


def _contains_readout_tokens(text: str) -> bool:
    return all(token in text for token in NAVIGATION_READOUT_TOKENS)


def _final_cot_valid(trace: AgentTrace) -> bool:
    required_tags = ("<Scene>", "<FPVScene>", "<BEVContext>", "<CrossView>", "<Plan>", "<ToolResultUse>", "<finish>")
    final_text = ""
    for turn in reversed(trace.turns):
        if turn.role == "assistant" and isinstance(turn.content, str) and "<finish>" in turn.content:
            final_text = turn.content
            break
    return bool(final_text) and all(tag in final_text for tag in required_tags)


def _finish_trajectory_consistent(
    finishes: list[dict[str, Any]],
    case: AgentStage2Case | None,
    trace: AgentTrace,
) -> float | None:
    payload: dict[str, Any] = {}
    if case is not None:
        payload.update(case.ground_truth)
    payload.update(trace.supervision)
    expected = payload.get("pixel_trajectory")
    if not expected:
        return None
    if not finishes:
        return 0.0
    actual = finishes[-1].get("trajectory")
    return 1.0 if actual == expected else 0.0


def _expected_ops(case: AgentStage2Case | None, trace: AgentTrace) -> list[str]:
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


def compute_trace_metrics(trace: AgentTrace, case: AgentStage2Case | None = None) -> AgentMetricResult:
    """Compute final Stage2 Agent metrics.

    输出字段:
        ``parse_success``: ``tool_call`` role 和最终 ``<finish>`` 是否可解析。
        ``tool_name_accuracy``: 工具名是否在允许集合内，若有 GT 则比较序列。
        ``tool_op_accuracy``: 有 ``expected_tool_ops`` 时比较 op 序列。
        ``argument_validity``: tool-call schema 参数是否有效。
        ``finish_validity``: 是否输出合法 ``<finish>{...}</finish>``。
        ``stage2_role_order_valid``: 是否满足工具交互先于导航 readout 的顺序。
        ``trajectory_readout_order_valid``: ``[TRAJ_*]`` 是否位于 tool_response 之后。
        ``final_cot_validity``: 最终 assistant 是否包含 Phase1 tags 和 ``ToolResultUse``。
        ``route_update_success``: route-session 类 tool_response 是否成功更新。
    """

    tool_calls, tool_parse_errors = _tool_calls(trace)
    finishes, finish_parse_errors = _finishes(trace)
    parse_errors = tool_parse_errors + finish_parse_errors

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
        "parse_success": 1.0 if tool_calls and finishes and not parse_errors else 0.0,
        "tool_name_accuracy": tool_name_accuracy,
        "tool_op_accuracy": tool_op_accuracy,
        "argument_validity": (sum(1.0 for item in valid_calls if item) / len(valid_calls)) if valid_calls else None,
        "finish_validity": 1.0 if finishes else 0.0,
        "stage2_role_order_valid": 1.0 if _role_order_valid(trace) else 0.0,
        "trajectory_readout_order_valid": 1.0 if _trajectory_readout_order_valid(trace) else 0.0,
        "final_cot_validity": 1.0 if _final_cot_valid(trace) else 0.0,
        "finish_trajectory_consistency": _finish_trajectory_consistent(finishes, case, trace),
        "route_update_success": route_update_success,
    }
    return AgentMetricResult(metrics=metrics, errors=parse_errors + validation_errors)
