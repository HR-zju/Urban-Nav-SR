"""Parser for model-emitted ``<tool_call>`` and ``<finish>`` blocks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from mcp_server.agent.schemas import ToolCall
from mcp_server.agent.tool_schema import MAP_TOOL_NAME, MAP_TOOL_OPS, normalize_map_tool_arguments, validate_tool_call
from mcp_server.contracts.tool import ToolCallV2
from mcp_server.runtime.interfaces import System2Action, System2ActionType


_ACTION_BLOCK_RE = re.compile(r"<(tool_call|finish)>(.*?)</\1>", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<(think|reasoning_summary)>(.*?)</\1>", re.DOTALL)
_SYSTEM1_READOUT_RE = re.compile(r"<readout>.*?</readout>|\[(?:SYS1|TRAJ)_\d+\]", re.DOTALL)


@dataclass(frozen=True)
class ParsedAgentAction:
    """Parsed assistant action.

    action_type:
        ``tool_call`` / ``finish`` / ``invalid``。
    tool_call:
        当 action_type 为 ``tool_call`` 时有效。
    finish:
        当 action_type 为 ``finish`` 时有效，通常是 route action JSON。
    errors:
        解析或 schema 校验错误。
    """

    action_type: str
    tool_call: ToolCall | None = None
    finish: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "finish": self.finish,
            "errors": list(self.errors),
        }


def _json_object_from_text(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    raw_text = text.strip()
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        payload, repaired = _json_object_with_trailing_brace_repair(raw_text)
        if payload is None:
            return None, [f"json_decode_error:{exc.msg}"]
        if repaired:
            return payload, []
    if not isinstance(payload, dict):
        return None, ["json_not_object"]
    return payload, []


def _json_object_with_trailing_brace_repair(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Parse a JSON object when the only trailing garbage is extra ``}``.

    Some small LLMs emit ``<tool_call>{"name": ...}}}</tool_call>``.  This is a
    bracket-counting error around an otherwise complete object, not a missing
    tool decision.  The repair is intentionally narrow: if the decoder leaves
    anything other than whitespace and right braces, parsing still fails.
    """

    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(text)
    except json.JSONDecodeError:
        return None, False
    if not isinstance(payload, dict):
        return None, False
    trailing = text[end:].strip()
    if trailing and set(trailing) <= {"}"}:
        return payload, True
    return None, False


def _normalize_tool_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize typed tool names into the stable RouteServer tool schema.

    Args:
        payload: Parsed JSON object from a ``<tool_call>`` block.

    Returns:
        Payload compatible with ``ToolCall.from_dict``.
    """

    tool_name = str(payload.get("tool") or payload.get("name") or "").strip()
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {
            key: value
            for key, value in payload.items()
            if key not in {"tool", "name", "arguments", "call_id", "id"}
        }
    if tool_name in MAP_TOOL_OPS:
        normalized_args = normalize_map_tool_arguments(dict(arguments))
        normalized_args.setdefault("op", tool_name)
        return {
            "tool": MAP_TOOL_NAME,
            "arguments": normalized_args,
            "call_id": payload.get("call_id") or payload.get("id"),
        }
    if tool_name == MAP_TOOL_NAME:
        return {
            "tool": MAP_TOOL_NAME,
            "arguments": normalize_map_tool_arguments(dict(arguments)),
            "call_id": payload.get("call_id") or payload.get("id"),
        }
    return payload


def _action_inside_reasoning(text: str, action_span: tuple[int, int]) -> bool:
    # 关键约束：tool_call/finish 与 reasoning_summary 同级，不允许藏在 think 中。
    for match in _THINK_BLOCK_RE.finditer(text):
        if match.start() <= action_span[0] and action_span[1] <= match.end():
            return True
    return False


def parse_tool_call_text(text: str) -> tuple[ToolCall | None, list[str]]:
    """Parse the first ``<tool_call>`` block.

    输入:
        assistant 原始文本。

    输出:
        ``(ToolCall | None, errors)``。兼容 ``{"tool":..., "op":...}`` 和
        ``{"tool":..., "arguments": {...}}`` 两种 JSON 形状。
    """

    for match in _ACTION_BLOCK_RE.finditer(text):
        if match.group(1) != "tool_call":
            continue
        errors: list[str] = []
        if _action_inside_reasoning(text, match.span()):
            errors.append("tool_call_inside_reasoning")
        payload, json_errors = _json_object_from_text(match.group(2))
        errors.extend(json_errors)
        if payload is None:
            return None, errors
        call = ToolCall.from_dict(_normalize_tool_payload(payload))
        _, schema_errors = validate_tool_call(call)
        errors.extend(schema_errors)
        return call, errors
    return None, ["missing_tool_call"]


def parse_finish_text(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse the first ``<finish>`` block from assistant output."""

    for match in _ACTION_BLOCK_RE.finditer(text):
        if match.group(1) != "finish":
            continue
        errors: list[str] = []
        if _action_inside_reasoning(text, match.span()):
            errors.append("finish_inside_reasoning")
        payload, json_errors = _json_object_from_text(match.group(2))
        errors.extend(json_errors)
        return payload, errors
    return None, ["missing_finish"]


def parse_agent_action(text: str) -> ParsedAgentAction:
    """Parse one assistant message into a tool call or final answer.

    解析优先级:
        如果同时出现 ``<tool_call>`` 和 ``<finish>``，选择最早出现的 action。
        这对应在线 agent 的单步决策：一次 assistant turn 要么调用工具，要么结束。
    """

    matches = list(_ACTION_BLOCK_RE.finditer(text))
    if not matches:
        return ParsedAgentAction(action_type="invalid", errors=["missing_action_block"])
    first = matches[0]
    if first.group(1) == "tool_call":
        call, errors = parse_tool_call_text(text)
        return ParsedAgentAction(
            action_type="tool_call" if call and not errors else "invalid",
            tool_call=call,
            errors=errors,
        )
    finish, errors = parse_finish_text(text)
    return ParsedAgentAction(
        action_type="finish" if finish is not None and not errors else "invalid",
        finish=finish,
        errors=errors,
    )


def parse_system2_action_text(
    text: str,
    *,
    require_system1_readout: bool = False,
) -> tuple[System2Action | None, list[str]]:
    """Parse assistant syntax and resolve its canonical System2 semantics.

    Args:
        text: Assistant text containing one ``<tool_call>`` or ``<finish>``.
        require_system1_readout: Whether canonical ``navigate`` must carry a
            v2 ``[SYS1_i]``/``[TRAJ_i]`` readout suffix.

    Returns:
        ``(action, errors)``. ``<finish>`` remains a syntax envelope; its
        ``decision_type`` and payload determine runtime semantics.
    """

    parsed = parse_agent_action(text)
    if parsed.action_type == "invalid":
        return None, list(parsed.errors)
    if parsed.tool_call is not None:
        call = parsed.tool_call
        return (
            System2Action(
                action_type=System2ActionType.TOOL_CALL,
                tool_call=ToolCallV2(tool=call.tool, arguments=dict(call.arguments), call_id=call.call_id),
            ),
            [],
        )

    finish = dict(parsed.finish or {})
    decision_type, legacy_decision = _finish_decision_type(finish, text=text)
    has_readout = _SYSTEM1_READOUT_RE.search(text) is not None
    if decision_type != System2ActionType.NAVIGATE.value and has_readout:
        return None, ["non_navigation_action_has_system1_readout"]
    if decision_type == System2ActionType.NAVIGATE.value and require_system1_readout and not has_readout:
        return None, ["navigate_action_missing_system1_readout"]

    try:
        if decision_type == System2ActionType.ASK_USER.value:
            request = finish.get("user_request")
            if not isinstance(request, dict):
                request = {
                    key: finish[key]
                    for key in ("question", "choices", "motion_policy_while_waiting")
                    if finish.get(key) is not None
                }
            action = System2Action(
                action_type=System2ActionType.ASK_USER,
                user_request=request or None,
                finish=finish,
            )
        elif decision_type == System2ActionType.NAVIGATE.value:
            action = System2Action(action_type=System2ActionType.NAVIGATE, finish=finish)
        else:
            # legacy finish/stop 交给 System2Action 的单一兼容入口补齐迁移 metadata/reason。
            action = System2Action(
                action_type=legacy_decision or decision_type,
                finish=finish,
            )
    except ValueError as exc:
        return None, [f"system2_action_invalid:{exc}"]
    return action, []


def _finish_decision_type(finish: dict[str, Any], *, text: str) -> tuple[str, str | None]:
    """Resolve canonical decision type plus an optional legacy action value.

    Args:
        finish: Parsed ``<finish>`` JSON object.
        text: Full assistant text used to disambiguate legacy stop-hold rows.

    Returns:
        ``(canonical, legacy_value)``. The legacy value is retained only so the
        central ``System2Action`` adapter can attach auditable migration data.
    """

    raw = str(finish.get("decision_type") or "").strip()
    aliases = {
        "request_user_input": System2ActionType.ASK_USER.value,
        "finish": System2ActionType.COMPLETE_TASK.value,
        "stop": System2ActionType.ABORT_TASK.value,
    }
    if raw:
        canonical = aliases.get(raw, raw)
        return canonical, raw if raw in {"finish", "stop"} else None

    route_action = str(finish.get("route_action") or "").strip()
    if route_action in {"request_user_input", "ask_user", "ask-user", "clarify"}:
        return System2ActionType.ASK_USER.value, None
    if route_action in {"finish", "complete_task"}:
        return System2ActionType.COMPLETE_TASK.value, "finish" if route_action == "finish" else None
    if route_action in {"stop", "abort_task"}:
        is_hold = str(finish.get("trajectory_type") or "") == "stop_hold" or bool(
            _SYSTEM1_READOUT_RE.search(text)
        )
        if is_hold:
            return System2ActionType.NAVIGATE.value, None
        return System2ActionType.ABORT_TASK.value, "stop" if route_action == "stop" else None
    # 历史 Stage2/Phase1 的 finish 没有 decision_type，但轨迹型 finish 均表示继续导航。
    return System2ActionType.NAVIGATE.value, None
