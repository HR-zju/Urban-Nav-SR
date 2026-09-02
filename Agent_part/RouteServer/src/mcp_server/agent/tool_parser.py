"""RouteServer adapter for the standalone AgentTools action parser."""

from __future__ import annotations

import re
from typing import Any

from urbannav_agent_tools.parser import (
    ParsedAgentAction,
    parse_agent_action as _parse_agent_action,
    parse_finish_text,
    parse_tool_call_text as _parse_tool_call_text,
)
from mcp_server.agent.reasoning_text import parse_reasoning_target_v2
from mcp_server.agent.schemas import ToolCall
from mcp_server.agent.tool_schema import (
    MAP_TOOL_NAME,
    MAP_TOOL_OPS,
    normalize_map_tool_arguments,
    validate_tool_call,
)
from mcp_server.contracts.tool import ToolCallV2
from mcp_server.runtime.interfaces import System2Action, System2ActionType


_SYSTEM1_READOUT_RE = re.compile(r"<readout>.*?</readout>|\[(?:SYS1|TRAJ)_\d+\]", re.DOTALL)


def _normalize_tool_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize RouteServer typed tool names into the map-tool schema."""

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


def parse_tool_call_text(text: str) -> tuple[ToolCall | None, list[str]]:
    """Parse and validate one RouteServer tool call.

    Args:
        text: Raw assistant text.

    Returns:
        A RouteServer-compatible call and explicit syntax or registry errors.
    """

    return _parse_tool_call_text(
        text,
        normalize_payload=_normalize_tool_payload,
        validate_call=validate_tool_call,
        call_factory=ToolCall.from_dict,
    )


def parse_agent_action(text: str) -> ParsedAgentAction:
    """Parse and validate the first RouteServer assistant action.

    Args:
        text: Raw assistant text.

    Returns:
        The parsed action and explicit syntax or registry errors.
    """

    return _parse_agent_action(
        text,
        normalize_payload=_normalize_tool_payload,
        validate_call=validate_tool_call,
        call_factory=ToolCall.from_dict,
    )


def parse_system2_action_text(
    text: str,
    *,
    require_system1_readout: bool = False,
) -> tuple[System2Action | None, list[str]]:
    """Resolve assistant syntax into the canonical RouteServer action.

    Args:
        text: Assistant text containing one ``<tool_call>`` or ``<finish>``.
        require_system1_readout: Require readout tokens on navigation finishes.

    Returns:
        A canonical action and explicit validation errors.
    """

    parsed = parse_agent_action(text)
    if parsed.action_type == "invalid":
        return None, list(parsed.errors)
    if parsed.tool_call is not None:
        call = parsed.tool_call
        return (
            System2Action(
                action_type=System2ActionType.TOOL_CALL,
                tool_call=ToolCallV2(
                    tool=call.tool,
                    arguments=dict(call.arguments),
                    call_id=call.call_id,
                ),
            ),
            [],
        )

    finish = dict(parsed.finish or {})
    if "<TaskGrounding>" in text:
        # V2 final turn 必须整体验证，不能只因 finish JSON 可解析就进入 runtime。
        reasoning = parse_reasoning_target_v2(text)
        if not reasoning.valid or reasoning.target is None:
            return None, [f"reasoning_v2_invalid:{item}" for item in reasoning.errors]
        finish = reasoning.target.finish.model_dump(mode="json", exclude_none=True)
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
    """Resolve canonical decision type and retain legacy action metadata.

    Args:
        finish: Parsed ``<finish>`` payload.
        text: Full assistant text used to identify legacy stop-hold rows.

    Returns:
        Canonical decision type and an optional legacy action value.
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
