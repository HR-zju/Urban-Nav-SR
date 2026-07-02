"""Metrics for final Agentic Stage2 traces.

指标只评估协议可解析性、工具选择、参数有效性、最终 CoT/finish 和
``tool_response -> [TRAJ_*]`` 因果顺序。它不依赖外部 LVLM 也不评估自然语言质量。
"""

from __future__ import annotations

import json
import re
from typing import Any

from mcp_server.agent.finish_schema import finish_trajectory_matches
from mcp_server.agent.schemas import AgentMetricResult, AgentStage2Case, AgentTrace, ToolCall, ToolResponse
from mcp_server.agent.stage2 import NAVIGATION_READOUT_TOKENS
from mcp_server.agent.tool_parser import parse_finish_text, parse_tool_call_text
from mcp_server.agent.tool_schema import validate_tool_call


_TAG_RE_CACHE: dict[str, re.Pattern[str]] = {}


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
            # Stage2 正式格式：独立 tool_call role。
            payload, payload_errors = _json_dict(turn.content)
            errors.extend(payload_errors)
            if payload is not None:
                calls.append(ToolCall.from_dict(payload))
            continue
        if turn.role == "assistant" and isinstance(turn.content, str) and "<tool_call>" in turn.content:
            # 兼容/审计旧 XML 格式，但 validator 会要求最终数据使用独立 role。
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


def _tag_json_payload(text: str, tag: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse one JSON payload from an XML-like Stage2 tag."""

    pattern = _TAG_RE_CACHE.get(tag)
    if pattern is None:
        pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL)
        _TAG_RE_CACHE[tag] = pattern
    match = pattern.search(text)
    if not match:
        return None, [f"missing_tag:{tag}"]
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError as exc:
        return None, [f"{tag}_json_decode_error:{exc.msg}"]
    if not isinstance(payload, dict):
        return None, [f"{tag}_json_not_object"]
    return payload, []


def _tool_decisions(trace: AgentTrace) -> tuple[list[dict[str, Any]], list[str]]:
    decisions: list[dict[str, Any]] = []
    errors: list[str] = []
    for turn in trace.turns:
        if turn.role != "assistant" or not isinstance(turn.content, str) or "<ToolDecision>" not in turn.content:
            continue
        payload, payload_errors = _tag_json_payload(turn.content, "ToolDecision")
        errors.extend(payload_errors)
        if payload is not None:
            decisions.append(payload)
    return decisions, errors


def _tool_responses(trace: AgentTrace) -> list[ToolResponse]:
    responses: list[ToolResponse] = []
    for turn in trace.turns:
        if turn.role != "tool_response":
            continue
        content = turn.content
        if isinstance(content, str):
            content, _ = _json_dict(content)
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
    # 只校验相对顺序，不要求 index 固定，便于兼容可选 system turn。
    initial_user = _first_index(trace, "user", start=0)
    think = _first_index(trace, "assistant", start=(initial_user + 1 if initial_user is not None else 0))
    tool_call = _first_index(trace, "tool_call", start=(think + 1 if think is not None else 0))
    if tool_call is not None:
        tool_response = _first_index(trace, "tool_response", start=tool_call + 1)
        readout_user = _first_readout_user_index(trace, start=(tool_response + 1 if tool_response is not None else 0))
        final_assistant = _first_final_assistant_index(trace, start=(readout_user + 1 if readout_user is not None else 0))
        return all(index is not None for index in (initial_user, think, tool_call, tool_response, readout_user, final_assistant))
    tool_response = _first_index(trace, "tool_response", start=(think + 1 if think is not None else 0))
    if tool_response is not None:
        return False
    readout_user = _first_readout_user_index(trace, start=(think + 1 if think is not None else 0))
    final_assistant = _first_final_assistant_index(trace, start=(readout_user + 1 if readout_user is not None else 0))
    return all(index is not None for index in (initial_user, think, readout_user, final_assistant))


def _trajectory_readout_order_valid(trace: AgentTrace) -> bool:
    last_tool_response = None
    for index, turn in enumerate(trace.turns):
        if turn.role == "tool_response":
            last_tool_response = index
    if last_tool_response is not None:
        # tool 路径中 readout 必须在最后一个 tool_response 之后。
        readout_user = _first_readout_user_index(trace, start=last_tool_response + 1)
        return readout_user is not None
    first_think = _first_index(trace, "assistant", start=0)
    readout_user = _first_readout_user_index(trace, start=(first_think + 1 if first_think is not None else 0))
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
    required_tags = ("<Scene>", "<FPVScene>", "<BEVContext>", "<CrossView>", "<Plan>", "<finish>")
    final_text = ""
    for turn in reversed(trace.turns):
        if turn.role == "assistant" and isinstance(turn.content, str) and "<finish>" in turn.content:
            final_text = turn.content
            break
    if not final_text or not all(tag in final_text for tag in required_tags):
        return False
    has_tool_call = any(turn.role == "tool_call" for turn in trace.turns)
    return "<ToolResultUse>" in final_text if has_tool_call else True


def _tool_result_use_valid(trace: AgentTrace) -> bool:
    final_text = ""
    for turn in reversed(trace.turns):
        if turn.role == "assistant" and isinstance(turn.content, str) and "<finish>" in turn.content:
            final_text = turn.content
            break
    has_tool_call = any(turn.role == "tool_call" for turn in trace.turns)
    if has_tool_call:
        # tool 路径要求模型在 final CoT 中总结“如何使用工具结果”。
        payload, errors = _tag_json_payload(final_text, "ToolResultUse")
        return payload is not None and not errors and payload.get("tool_used") is True
    if "<ToolResultUse>" not in final_text:
        return True
    payload, errors = _tag_json_payload(final_text, "ToolResultUse")
    return payload is not None and not errors and payload.get("tool_used") is False


def _tool_decision_consistent(trace: AgentTrace, decisions: list[dict[str, Any]]) -> bool:
    if not decisions:
        return False
    should_call = bool(decisions[0].get("should_call_tool"))
    has_tool_call = any(turn.role == "tool_call" for turn in trace.turns)
    return should_call == has_tool_call


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
    # Text finish 与顶层 pixel_trajectory 等价时，LM target 和 trajectory head target 不冲突。
    return 1.0 if finish_trajectory_matches(finishes[-1], expected) else 0.0


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


def _expected_no_tool(case: AgentStage2Case | None, trace: AgentTrace) -> bool:
    """Return True when supervision explicitly says no tool should be called."""

    payloads: list[dict[str, Any]] = []
    if case is not None:
        payloads.extend([case.observation, case.ground_truth, case.metadata])
    request = trace.request if isinstance(trace.request, dict) else {}
    payloads.extend([request.get("observation") or {}, request.get("ground_truth") or {}, request.get("metadata") or {}])
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if payload.get("force_no_tool") is True:
            return True
        for key in ("tool_policy", "stage2_tool_policy"):
            value = str(payload.get(key) or "").strip().lower()
            if value in {"no_tool", "none", "skip_tool", "local_only"}:
                return True
    return False


def compute_trace_metrics(trace: AgentTrace, case: AgentStage2Case | None = None) -> AgentMetricResult:
    """Compute final Stage2 Agent metrics.

    输出字段:
        ``parse_success``: ``<ToolDecision>``、可选 ``tool_call`` 和最终
            ``<finish>`` 是否可解析且 decision/role 一致。
        ``tool_name_accuracy``: 工具名是否在允许集合内，若有 GT 则比较序列。
        ``tool_op_accuracy``: 有 ``expected_tool_ops`` 时比较 op 序列。
        ``argument_validity``: tool-call schema 参数是否有效。
        ``finish_validity``: 是否输出合法 ``<finish>{...}</finish>``。
        ``stage2_role_order_valid``: 是否满足 tool/no-tool 因果顺序。
        ``trajectory_readout_order_valid``: ``[TRAJ_*]`` 在 tool 路径中是否位于
            ``tool_response`` 之后，在 no-tool 路径中是否位于 ``<ToolDecision>`` 后。
        ``final_cot_validity``: 最终 assistant 是否包含 Phase1 tags 和 ``<finish>``；
            tool 路径还要求 ``<ToolResultUse>``。
        ``tool_result_use_validity``: ``<ToolResultUse>`` 是否只表达工具结果使用结论。
        ``tool_decision_consistency``: ``should_call_tool`` 是否与实际 role 序列一致。
        ``unnecessary_tool_call_rate``: 显式 no-tool 样本是否错误调用工具。
        ``route_update_success``: route-session 类 tool_response 是否成功更新。
    """

    tool_calls, tool_parse_errors = _tool_calls(trace)
    finishes, finish_parse_errors = _finishes(trace)
    tool_decisions, decision_parse_errors = _tool_decisions(trace)
    parse_errors = tool_parse_errors + finish_parse_errors + decision_parse_errors

    valid_calls = []
    validation_errors: list[str] = []
    for call in tool_calls:
        is_valid, errors = validate_tool_call(call)
        valid_calls.append(is_valid)
        validation_errors.extend(errors)

    expected_ops = _expected_ops(case, trace)
    expected_no_tool = _expected_no_tool(case, trace)
    called_ops = [str(call.op) for call in tool_calls if call.op]
    if expected_no_tool:
        # no-tool replay 的正确行为是完全不调用工具。
        tool_op_accuracy = 1.0 if not called_ops else 0.0
    elif expected_ops:
        tool_op_accuracy = 1.0 if called_ops == expected_ops else 0.0
    else:
        tool_op_accuracy = None

    expected_names = []
    if case is not None:
        expected_names = [str(item) for item in case.ground_truth.get("expected_tool_names") or []]
    called_names = [call.tool for call in tool_calls]
    if expected_no_tool:
        tool_name_accuracy = 1.0 if not called_names else 0.0
    elif expected_names:
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
            # route-session 类工具只看是否成功返回更新结果，不在 metrics 中重算路线质量。
            route_update_success = 1.0 if response.success else 0.0
            break

    tool_decision_consistency = 1.0 if _tool_decision_consistent(trace, tool_decisions) else 0.0
    # None 表示该样本没有对应 GT 或不适用，聚合时不会被当作 0。
    metrics: dict[str, float | None] = {
        "parse_success": 1.0 if finishes and not parse_errors and tool_decision_consistency == 1.0 else 0.0,
        "tool_name_accuracy": tool_name_accuracy,
        "tool_op_accuracy": tool_op_accuracy,
        "argument_validity": (sum(1.0 for item in valid_calls if item) / len(valid_calls)) if valid_calls else None,
        "finish_validity": 1.0 if finishes else 0.0,
        "stage2_role_order_valid": 1.0 if _role_order_valid(trace) else 0.0,
        "trajectory_readout_order_valid": 1.0 if _trajectory_readout_order_valid(trace) else 0.0,
        "final_cot_validity": 1.0 if _final_cot_valid(trace) else 0.0,
        "tool_result_use_validity": 1.0 if _tool_result_use_valid(trace) else 0.0,
        "tool_decision_consistency": tool_decision_consistency,
        "unnecessary_tool_call_rate": (1.0 if tool_calls else 0.0) if expected_no_tool else None,
        "finish_trajectory_consistency": _finish_trajectory_consistent(finishes, case, trace),
        "route_update_success": route_update_success,
    }
    return AgentMetricResult(metrics=metrics, errors=parse_errors + validation_errors)
