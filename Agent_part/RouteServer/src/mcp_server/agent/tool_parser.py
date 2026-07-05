"""Parser for model-emitted ``<tool_call>`` and ``<finish>`` blocks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from mcp_server.agent.schemas import ToolCall
from mcp_server.agent.tool_schema import MAP_TOOL_NAME, MAP_TOOL_OPS, normalize_map_tool_arguments, validate_tool_call


_ACTION_BLOCK_RE = re.compile(r"<(tool_call|finish)>(.*?)</\1>", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<(think|reasoning_summary)>(.*?)</\1>", re.DOTALL)


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
