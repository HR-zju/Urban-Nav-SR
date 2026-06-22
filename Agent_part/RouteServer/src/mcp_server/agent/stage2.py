"""Helpers for the final SideWalk Agentic Stage2 protocol.

本模块只处理“协议拼装”，不调用 LLM、地图服务或图像模型。核心数学/因果约束是
自回归 Transformer 的 attention mask 第 t 个 token 只能看到它之前的 token。
因此用于 trajectory head 的 ``[TRAJ_i]`` readout token 必须放在 ``tool_response``
之后，否则读出的 hidden state 无法包含工具结果，轨迹监督会错位。
"""

from __future__ import annotations

import json
import re
from typing import Any

from mcp_server.agent.schemas import AgentStage2Case, ToolCall, ToolResponse


TRAJECTORY_TOKEN_COUNT = 16
NAVIGATION_READOUT_TOKENS = tuple(f"[TRAJ_{index}]" for index in range(TRAJECTORY_TOKEN_COUNT))
FINAL_NAVIGATION_DECISION_PROMPT = (
    "Produce the final sidewalk navigation decision using the tool result, FPV, BEV, and route state.\n\n"
    + " ".join(NAVIGATION_READOUT_TOKENS)
)

_FINISH_RE = re.compile(r"<finish>(.*?)</finish>", re.DOTALL)


def json_dumps(payload: Any) -> str:
    """Serialize protocol JSON deterministically.

    输入:
        任意 JSON 可序列化对象。

    输出:
        无多余空格、key 稳定排序的 UTF-8 JSON 字符串，适合放入 XML-like tag。
    """

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def initial_user_content(case: AgentStage2Case) -> str:
    """Build the first user turn.

    输入:
        ``AgentStage2Case``，其中 ``observation.images`` 决定 ``<image>`` 数量。

    输出:
        ``<image>...`` 前缀 + 用户导航请求。
    """

    return "".join("<image>" for _ in case.images) + case.user_request


def final_navigation_user_content() -> str:
    """Return the fixed final readout prompt containing all trajectory query tokens."""

    return FINAL_NAVIGATION_DECISION_PROMPT


def build_agent_think_text(*, state: dict[str, Any], decision: dict[str, Any]) -> str:
    """Build the assistant reasoning turn before tool execution.

    输入:
        ``state`` 是当前导航状态摘要；``decision`` 是是否调用工具、调用哪个工具的
        结构化教师决策。

    输出:
        assistant content。该文本算 LM loss 但不包含可执行 ``tool_call`` JSON。
    """

    return (
        "<think>\n"
        f"<AgentState>{json_dumps(state)}</AgentState>\n"
        f"<ToolDecision>{json_dumps(decision)}</ToolDecision>\n"
        "</think>\n"
    )


def build_agent_state(case: AgentStage2Case, *, intent: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the compact ``<AgentState>`` payload for the first assistant turn."""

    observation = case.observation
    intent = intent or {}
    return {
        "case_id": case.case_id,
        "route_session_id": observation.get("route_session_id") or observation.get("session_id"),
        "current_location": observation.get("current_location") or observation.get("origin"),
        "mode": observation.get("mode") or intent.get("mode") or "foot",
        "user_intent": intent.get("search_keywords") or intent.get("destination_kind") or case.user_request,
        "available_context": _available_context(case),
    }


def build_tool_decision(call: ToolCall, *, reason: str) -> dict[str, Any]:
    """Build the compact ``<ToolDecision>`` payload aligned with the next tool_call role."""

    return {
        "should_call_tool": True,
        "tool": call.tool,
        "op": call.op,
        "reason": reason,
    }


def synthetic_tool_response(
    call: ToolCall,
    case: AgentStage2Case,
    *,
    source: str,
    recommendation: dict[str, Any] | None = None,
    visual_prompt: dict[str, Any] | None = None,
) -> ToolResponse:
    """Create a deterministic offline tool response for review/cold-start bootstrapping.

    输入:
        ``call`` 是学生可见工具调用；``case`` 提供当前观测和监督；可选
        ``recommendation`` / ``visual_prompt`` 只作为 teacher internal context 摘要。

    输出:
        ``ToolResponse``，可直接写入 trace 的 ``tool_response`` role。

    说明:
        真实 RouteServer executor 接入前 Stage2 仍必须提供 tool_response 保证
        后续 ``[TRAJ_i]`` token 的 hidden state 能看到“工具结果”这一类上下文。
    """

    args = dict(call.arguments)
    op = str(args.get("op") or "")
    if call.tool == "map_tool":
        result = _synthetic_map_result(op=op, args=args, case=case, recommendation=recommendation)
    elif call.tool == "image_tool":
        result = {
            "op": op,
            "image": args.get("image") or args.get("image_path") or "fpv",
            "bbox": args.get("bbox"),
            "visual_prompt": _compact_tool_context(visual_prompt),
            "source": source,
        }
    elif call.tool == "user_interaction_tool":
        result = {
            "op": op,
            "await_user": True,
            "question": args.get("question"),
            "choices": args.get("choices") or [],
            "proposal": args.get("proposal") or {},
            "source": source,
        }
    else:
        result = {"op": op, "source": source}
    return ToolResponse(tool=call.tool, success=True, result=result, call_id=call.call_id)


def build_final_assistant_text(
    case: AgentStage2Case,
    *,
    call: ToolCall,
    response: ToolResponse,
    source: str,
    phase1_response: str | None = None,
) -> str:
    """Build the final assistant CoT and ``<finish>`` target.

    输入:
        Stage2 case、已执行或合成的工具调用/返回，以及可选 Phase1 CoT response。

    输出:
        最终 assistant content。若 case 中已有 Phase1 CoT 会插入
        ``<ToolResultUse>`` 并补齐 ``<finish>`` 的 tool fields 否则生成可 review
        的 deterministic fallback CoT。
    """

    tool_use = _tool_result_use(call=call, response=response)
    finish_extra = _finish_extra(case=case, call=call, response=response, source=source)
    base_response = (phase1_response or _find_phase1_response(case)).strip()
    if base_response:
        text = _insert_tool_result_use(base_response, tool_use)
        return _augment_finish(text, finish_extra)

    scene = {
        "summary": "offline Stage2 teacher sample",
        "request": case.user_request,
        "tool_context_available": True,
    }
    fpv_scene = {
        "navigable_area": "use visible sidewalk and obstacle cues from FPV",
        "risk": "unknown_without_manual_scene_label",
    }
    bev_context = {
        "route_session_id": case.observation.get("route_session_id") or case.observation.get("session_id"),
        "current_location": case.observation.get("current_location") or case.observation.get("origin"),
        "mode": case.observation.get("mode") or "foot",
    }
    cross_view = {
        "alignment": "use FPV appearance with BEV route context",
        "tool_result": "available_before_trajectory_readout",
    }
    plan = {
        "route_action": finish_extra["route_action"],
        "tool": call.tool,
        "op": call.op,
        "next_step": "produce local trajectory from FPV, BEV, and tool result",
    }
    finish = dict(finish_extra)
    return (
        "<think>\n"
        f"<Scene>{json_dumps(scene)}</Scene>\n"
        f"<FPVScene>{json_dumps(fpv_scene)}</FPVScene>\n"
        f"<BEVContext>{json_dumps(bev_context)}</BEVContext>\n"
        f"<CrossView>{json_dumps(cross_view)}</CrossView>\n"
        f"<Plan>{json_dumps(plan)}</Plan>\n"
        f"<ToolResultUse>{json_dumps(tool_use)}</ToolResultUse>\n"
        "</think>\n"
        f"<finish>{json_dumps(finish)}</finish>"
    )


def _available_context(case: AgentStage2Case) -> list[str]:
    context = ["user_request"]
    if case.images:
        context.append("images")
    for key in ("current_location", "origin", "destination", "route_session_id", "route_polyline"):
        if case.observation.get(key) is not None:
            context.append(key)
    for key in ("pixel_trajectory", "trajectory_fpv_t", "trajectory_ego_t", "semantic_segmentation_path"):
        if case.ground_truth.get(key) is not None:
            context.append(key)
    return context


def _synthetic_map_result(
    *,
    op: str,
    args: dict[str, Any],
    case: AgentStage2Case,
    recommendation: dict[str, Any] | None,
) -> dict[str, Any]:
    recommendation = recommendation or {}
    if op == "search_poi":
        chosen = recommendation.get("chosen_destination") if isinstance(recommendation, dict) else None
        candidates = recommendation.get("candidates") if isinstance(recommendation, dict) else None
        if not candidates and isinstance(chosen, dict) and chosen:
            candidates = [chosen]
        if not candidates:
            candidates = [
                {
                    "name": args.get("query") or "candidate_poi",
                    "location": args.get("location") or case.observation.get("current_location"),
                    "source": "synthetic_teacher",
                }
            ]
        return {
            "op": op,
            "query": args.get("query"),
            "location": args.get("location"),
            "candidates": candidates,
            "source": "synthetic_teacher",
        }
    if op == "create_route_session":
        session_id = args.get("session_id") or case.observation.get("route_session_id") or f"{case.case_id}:route"
        return {
            "op": op,
            "session": {
                "session_id": session_id,
                "origin": args.get("origin"),
                "destination": args.get("destination"),
                "mode": args.get("mode") or "foot",
            },
            "route_plan": {"status": "available", "source": "synthetic_teacher"},
        }
    if op in {"add_via_and_replan", "remove_via_and_replan", "replace_destination", "replan_from_current_pose"}:
        return {
            "op": op,
            "session_changed": True,
            "session_id": args.get("session_id") or case.observation.get("route_session_id"),
            "route_plan": {"status": "updated", "source": "synthetic_teacher"},
        }
    return {"op": op, "source": "synthetic_teacher"}


def _compact_tool_context(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        key: payload.get(key)
        for key in ("status", "provider", "chosen_destination", "route_summary", "message")
        if payload.get(key) is not None
    }


def _tool_result_use(*, call: ToolCall, response: ToolResponse) -> dict[str, Any]:
    return {
        "tool_used": True,
        "tool": call.tool,
        "op": call.op,
        "success": response.success,
        "used_for": "final_navigation_decision",
    }


def _finish_extra(*, case: AgentStage2Case, call: ToolCall, response: ToolResponse, source: str) -> dict[str, Any]:
    finish = {
        "route_action": _route_action(call),
        "tool_used": True,
        "tool": call.tool,
        "op": call.op,
        "route_session_id": _route_session_id(case, call, response),
        "trajectory": _trajectory(case),
        "intent": case.ground_truth.get("intent") or case.observation.get("intent") or "follow_sidewalk_route",
        "confidence_level": case.ground_truth.get("confidence_level") or "medium",
        "trajectory_type": case.ground_truth.get("trajectory_type") or "local_navigation",
        "trajectory_validity": case.ground_truth.get("trajectory_validity") or "valid",
        "teacher_source": source,
    }
    return {key: value for key, value in finish.items() if value is not None}


def _route_action(call: ToolCall) -> str:
    op = call.op or ""
    if call.tool == "user_interaction_tool":
        return "request_user_input"
    if call.tool == "image_tool":
        return "inspect_visual_context"
    if op == "create_route_session":
        return "create_route"
    if op in {"add_via_and_replan", "remove_via_and_replan", "replace_destination", "replan_from_current_pose"}:
        return "update_route"
    if op in {"search_poi", "search_category", "geocode", "reverse_geocode"}:
        return "use_map_context"
    return "continue_navigation"


def _route_session_id(case: AgentStage2Case, call: ToolCall, response: ToolResponse) -> Any:
    if call.arguments.get("session_id") is not None:
        return call.arguments.get("session_id")
    result = response.result if isinstance(response.result, dict) else {}
    session = result.get("session") if isinstance(result.get("session"), dict) else {}
    return session.get("session_id") or result.get("session_id") or case.observation.get("route_session_id")


def _trajectory(case: AgentStage2Case) -> list[list[float]]:
    trajectory = case.ground_truth.get("pixel_trajectory") or case.ground_truth.get("trajectory")
    if isinstance(trajectory, list) and trajectory:
        return trajectory
    return [[0.5, round(0.9 - index * 0.03, 4)] for index in range(TRAJECTORY_TOKEN_COUNT)]


def _find_phase1_response(case: AgentStage2Case) -> str:
    for payload in (case.ground_truth, case.metadata, case.observation):
        for key in ("phase1_response", "phase1_answer", "cot_response", "assistant_response", "response"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(value, str) and value.strip():
                return value
    for message in reversed(case.dialogue_history):
        if isinstance(message, dict) and message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


def _insert_tool_result_use(text: str, tool_use: dict[str, Any]) -> str:
    block = f"<ToolResultUse>{json_dumps(tool_use)}</ToolResultUse>"
    if "<ToolResultUse>" in text:
        return text
    if "</think>" in text:
        return text.replace("</think>", f"{block}\n</think>", 1)
    return f"<think>\n{block}\n</think>\n{text}"


def _augment_finish(text: str, extra: dict[str, Any]) -> str:
    match = _FINISH_RE.search(text)
    if not match:
        return f"{text.rstrip()}\n<finish>{json_dumps(extra)}</finish>"
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return text
    if not isinstance(payload, dict):
        return text
    for key, value in extra.items():
        payload.setdefault(key, value)
    return text[: match.start(1)] + json_dumps(payload) + text[match.end(1) :]
