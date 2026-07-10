"""Helpers for the final SideWalk Agentic Stage2 protocol.

本模块只处理“协议拼装”，不调用 LLM、地图服务或图像模型。核心数学/因果约束是
自回归 Transformer 的 attention mask 第 t 个 token 只能看到它之前的 token。
因此用于 trajectory / System1 head 的 readout token 必须放在可见上下文之后，
否则读出的 hidden state 无法包含工具结果，轨迹监督会错位。v1 使用
``user [TRAJ_i] -> assistant final``；v2 使用
``assistant <finish> -> <readout>[SYS1_i]</readout>``，让 System1 latent 还能
看到最终导航语义决策。
"""

from __future__ import annotations

import json
import re
from typing import Any

from mcp_server.agent.finish_schema import (
    build_finish_payload,
    canonicalize_finish_trajectory,
    parse_finish_trajectory_norm,
)
from mcp_server.agent.schemas import AgentStage2Case, ToolCall, ToolResponse
from mcp_server.agent.tool_schema import IMAGE_TOOL_NAME, MAP_TOOL_NAME, USER_INTERACTION_TOOL_NAME


TRAJECTORY_TOKEN_COUNT = 16
NAVIGATION_READOUT_TOKENS = tuple(f"[TRAJ_{index}]" for index in range(TRAJECTORY_TOKEN_COUNT))
SYSTEM1_READOUT_TOKENS = tuple(f"[SYS1_{index}]" for index in range(TRAJECTORY_TOKEN_COUNT))
READOUT_PLACEMENT_USER_PROMPT_V1 = "user_prompt_v1"
READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2 = "assistant_suffix_v2"
READOUT_PLACEMENTS = {
    READOUT_PLACEMENT_USER_PROMPT_V1,
    READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
}
TOOL_DECISION_REASON_CODE_NEED_MAP_CONTEXT = "need_map_context"
TOOL_DECISION_REASON_CODE_NEED_VISUAL_CROP = "need_visual_crop"
TOOL_DECISION_REASON_CODE_NEED_USER_CLARIFICATION = "need_user_clarification"
TOOL_DECISION_REASON_CODE_LOCAL_CONTEXT_SUFFICIENT = "local_context_sufficient"
TOOL_DECISION_REASON_CODES = {
    TOOL_DECISION_REASON_CODE_NEED_MAP_CONTEXT,
    TOOL_DECISION_REASON_CODE_NEED_VISUAL_CROP,
    TOOL_DECISION_REASON_CODE_NEED_USER_CLARIFICATION,
    TOOL_DECISION_REASON_CODE_LOCAL_CONTEXT_SUFFICIENT,
}
FINAL_NAVIGATION_DECISION_PROMPT = (
    "Produce the final sidewalk navigation decision using FPV, BEV, and route state.\n\n"
    + " ".join(NAVIGATION_READOUT_TOKENS)
)
FINAL_NAVIGATION_DECISION_WITH_TOOL_PROMPT = (
    "Produce the final sidewalk navigation decision using the tool result, FPV, BEV, and route state.\n\n"
    + " ".join(NAVIGATION_READOUT_TOKENS)
)
FINAL_NAVIGATION_DECISION_TEXT_PROMPT = (
    "Produce the final sidewalk navigation decision using FPV, BEV, and route state."
)
FINAL_NAVIGATION_DECISION_WITH_TOOL_TEXT_PROMPT = (
    "Produce the final sidewalk navigation decision using the tool result, FPV, BEV, and route state."
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


def final_navigation_user_content(
    *,
    tool_used: bool = True,
    readout_placement: str = READOUT_PLACEMENT_USER_PROMPT_V1,
) -> str:
    """Return the final readout prompt containing all trajectory query tokens.

    Args:
        tool_used: True 时提示模型把已经出现的 ``tool_response`` 纳入最终导航；
            False 时生成 no-tool replay 的最终导航 prompt。
        readout_placement: ``user_prompt_v1`` 时返回带 ``[TRAJ_i]`` 的旧
            readout user prompt；``assistant_suffix_v2`` 时只返回文本触发
            prompt，readout token 由最终 assistant 的 ``<readout>`` 承载。

    Returns:
        最终导航 user turn 文本。v1 包含 ``[TRAJ_i]``；v2 不包含 readout
        token，避免 System1 readout 早于 ``<finish>``。

    Raises:
        ValueError: ``readout_placement`` 不是已知取值时抛出。
    """

    if readout_placement not in READOUT_PLACEMENTS:
        raise ValueError(f"Unknown readout placement: {readout_placement}")
    if readout_placement == READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2:
        return FINAL_NAVIGATION_DECISION_WITH_TOOL_TEXT_PROMPT if tool_used else FINAL_NAVIGATION_DECISION_TEXT_PROMPT
    return FINAL_NAVIGATION_DECISION_WITH_TOOL_PROMPT if tool_used else FINAL_NAVIGATION_DECISION_PROMPT


def system1_readout_block(*, readout_placement: str = READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2) -> str:
    """Build the System1 assistant-suffix readout block.

    Args:
        readout_placement: Placement mode. Only ``assistant_suffix_v2`` emits
            a non-empty block.

    Returns:
        ``<readout>[SYS1_0] ... [SYS1_N]</readout>`` for v2, otherwise an
        empty string. These tokens are latent anchors and must be masked out of
        assistant LM loss by the ms-swift template/trainer.

    Raises:
        ValueError: ``readout_placement`` is unknown.
    """

    if readout_placement not in READOUT_PLACEMENTS:
        raise ValueError(f"Unknown readout placement: {readout_placement}")
    if readout_placement != READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2:
        return ""
    return "<readout>" + " ".join(SYSTEM1_READOUT_TOKENS) + "</readout>"


def append_system1_readout(
    text: str,
    *,
    readout_placement: str = READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
) -> str:
    """Append a v2 System1 readout block to final assistant text.

    Args:
        text: Final assistant text containing ``<finish>``.
        readout_placement: Placement mode. ``user_prompt_v1`` leaves ``text``
            unchanged.

    Returns:
        Assistant text with one trailing ``<readout>`` block in v2 mode.

    Raises:
        ValueError: ``readout_placement`` is unknown.
    """

    block = system1_readout_block(readout_placement=readout_placement)
    if not block:
        return text
    if "<readout>" in text:
        return text
    return f"{text.rstrip()}\n{block}"


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


def infer_tool_decision_reason_code(call: ToolCall | None) -> str:
    """Infer a compact ``<ToolDecision>.reason_code`` from the selected tool.

    Args:
        call: Tool call selected by teacher/runtime. ``None`` represents a
            no-tool replay where local context is already sufficient.

    Returns:
        One value from ``TOOL_DECISION_REASON_CODES``. This field is an audit
        label for tool-decision quality; it is not part of executable tool
        arguments and historical traces may omit it.
    """

    if call is None:
        return TOOL_DECISION_REASON_CODE_LOCAL_CONTEXT_SUFFICIENT
    if call.tool == MAP_TOOL_NAME:
        return TOOL_DECISION_REASON_CODE_NEED_MAP_CONTEXT
    if call.tool == IMAGE_TOOL_NAME:
        return TOOL_DECISION_REASON_CODE_NEED_VISUAL_CROP
    if call.tool == USER_INTERACTION_TOOL_NAME:
        return TOOL_DECISION_REASON_CODE_NEED_USER_CLARIFICATION
    return TOOL_DECISION_REASON_CODE_NEED_MAP_CONTEXT


def build_tool_decision(
    call: ToolCall | None,
    *,
    reason: str,
    reason_code: str | None = None,
) -> dict[str, Any]:
    """Build the compact ``<ToolDecision>`` payload.

    Args:
        call: 下一步要执行的工具调用。None 表示教师标注 no-tool replay。
        reason: 教师给出的决策理由，作为 LM 监督的一部分。
        reason_code: 可选离散原因码。为空时按工具类型推断；旧标注可完全缺省。

    Returns:
        ``<ToolDecision>`` JSON。tool 路径中它与下一条 ``tool_call`` role 对齐；
        no-tool 路径中 ``should_call_tool=false``，后续直接进入导航 readout。
    """

    decision = {
        "should_call_tool": call is not None,
        "reason_code": reason_code or infer_tool_decision_reason_code(call),
        "reason": reason,
    }
    if call is not None:
        decision["tool"] = call.tool
        decision["op"] = call.op
    return decision


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
        requires_user = bool(args.get("requires_user_response", op not in {"answer_query", "notify_issue"}))
        motion_policy = str(
            args.get("motion_policy_while_waiting")
            or ("stop" if requires_user else "continue_previous_safe_policy")
        )
        interaction_id = args.get("interaction_id") or call.call_id or f"ui:{op or 'interaction'}"
        pending = None
        if requires_user:
            pending = {
                "interaction_id": interaction_id,
                "op": op,
                "message": args.get("message") or args.get("question") or "",
                "options": args.get("options") or args.get("choices") or [],
                "recommended_option_id": args.get("recommended_option_id"),
                "motion_policy_while_waiting": motion_policy,
            }
        result = {
            "op": op,
            "await_user": requires_user,
            "interaction_id": interaction_id,
            "message": args.get("message") or args.get("question") or "",
            "question": args.get("question"),
            "choices": args.get("choices") or [],
            "options": args.get("options") or [],
            "recommended_option_id": args.get("recommended_option_id"),
            "requires_user_response": requires_user,
            "interaction_mode": str(args.get("interaction_mode") or ("blocking" if requires_user else "non_blocking")),
            "motion_policy_while_waiting": motion_policy,
            "runtime_state": "WAITING_FOR_USER" if requires_user else "RUNNING",
            "dialogue_state": {"pending_user_interaction": pending} if pending is not None else {},
            "evidence_refs": args.get("evidence_refs") or [],
            "proposal": args.get("proposal") or {},
            "source": source,
        }
    else:
        result = {"op": op, "source": source}
    return ToolResponse(tool=call.tool, success=True, result=result, call_id=call.call_id)


def build_final_assistant_text(
    case: AgentStage2Case,
    *,
    call: ToolCall | None,
    response: ToolResponse | None,
    source: str,
    phase1_response: str | None = None,
    readout_placement: str = READOUT_PLACEMENT_USER_PROMPT_V1,
) -> str:
    """Build the final assistant CoT and ``<finish>`` target.

    输入:
        Stage2 case、可选工具调用/返回，以及可选 Phase1 CoT response。
        ``readout_placement`` 控制是否把 v2 System1 readout 追加到最终
        assistant 后缀。

    输出:
        最终 assistant content。若 case 中已有 Phase1 CoT 会插入
        tool 路径会插入 ``<ToolResultUse>`` 并补齐 ``<finish>`` 的 tool fields。
        no-tool 路径不写 ``<ToolResultUse>``，避免把不存在的环境返回变成 LM target。
    """

    tool_use = _tool_result_use(call=call, response=response) if call is not None and response is not None else None
    finish_extra = _finish_extra(case=case, call=call, response=response, source=source)
    base_response = (phase1_response or _find_phase1_response(case)).strip()
    if base_response:
        text = _insert_tool_result_use(base_response, tool_use) if tool_use is not None else base_response
        return append_system1_readout(
            _augment_finish(text, finish_extra),
            readout_placement=readout_placement,
        )

    scene = {
        "summary": "offline Stage2 teacher sample",
        "request": case.user_request,
        "tool_context_available": tool_use is not None,
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
        "tool_result": "available_before_trajectory_readout" if tool_use is not None else "not_used",
    }
    plan = {
        "route_action": finish_extra["route_action"],
        "next_step": (
            "produce local trajectory from FPV, BEV, and tool result"
            if tool_use is not None
            else "produce local trajectory from FPV, BEV, and route state"
        ),
    }
    if call is not None:
        plan["tool"] = call.tool
        plan["op"] = call.op
    if tool_use is not None:
        bev_context["tool_result_summary"] = tool_use.get("result_summary")
        cross_view["tool_grounding"] = tool_use.get("navigation_effect")
        plan["tool_result_use"] = tool_use.get("used_for")
    finish = dict(finish_extra)
    tool_result_block = f"<ToolResultUse>{json_dumps(tool_use)}</ToolResultUse>\n" if tool_use is not None else ""
    return append_system1_readout(
        "<think>\n"
        f"<Scene>{json_dumps(scene)}</Scene>\n"
        f"<FPVScene>{json_dumps(fpv_scene)}</FPVScene>\n"
        f"<BEVContext>{json_dumps(bev_context)}</BEVContext>\n"
        f"<CrossView>{json_dumps(cross_view)}</CrossView>\n"
        f"<Plan>{json_dumps(plan)}</Plan>\n"
        f"{tool_result_block}"
        "</think>\n"
        f"<finish>{json_dumps(finish)}</finish>",
        readout_placement=readout_placement,
    )


def _available_context(case: AgentStage2Case) -> list[str]:
    """Return only runtime-observable context for ``<AgentState>``.

    ``ground_truth.pixel_trajectory`` / ``trajectory_fpv_t`` /
    ``trajectory_ego_t`` are teacher labels for loss computation.  They are
    exported as top-level ms-swift fields, but must not be advertised as
    available agent context in the assistant target; otherwise Stage2 would
    teach the model a data-leaking state description.
    """

    context = ["user_request"]
    if case.images:
        context.append("images")
    for key in ("current_location", "origin", "destination", "route_session_id", "route_polyline"):
        if case.observation.get(key) is not None:
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
    if op == "get_route_context":
        return {
            "op": op,
            "route_id": args.get("route_id") or args.get("session_id") or case.observation.get("route_id"),
            "context_assets": [
                {
                    "asset_id": f"{case.case_id}:route_instruction_text",
                    "asset_type": "route_instruction_text",
                    "modality": "text",
                    "visibility": "model_visible",
                    "payload": {"text": "Use the available route context for the next local navigation decision."},
                }
            ],
            "source": "synthetic_teacher",
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
    """Summarize how the final assistant should use one tool response.

    The raw ``tool_response`` remains environment context and is not copied into
    assistant LM targets. This summary is the teacher-visible conclusion:
    \\[
    u = f(c, r)
    \\]
    where ``c`` is the tool call and ``r`` is the deterministic tool response.
    """

    payload = {
        "tool_used": True,
        "tool": call.tool,
        "op": call.op,
        "success": response.success,
        "used_for": "final_navigation_decision",
        "result_summary": _tool_result_summary(call=call, response=response),
        "selected_result": _selected_tool_result(call=call, response=response),
        "route_update": _route_update_from_tool(call=call, response=response),
        "navigation_effect": _navigation_effect_from_tool(call=call, response=response),
    }
    if not response.success:
        payload["failure_recovery"] = _failure_recovery_from_tool(call=call, response=response)
    return {key: value for key, value in payload.items() if value is not None}


def _finish_extra(
    *,
    case: AgentStage2Case,
    call: ToolCall | None,
    response: ToolResponse | None,
    source: str,
) -> dict[str, Any]:
    finish_base = {
        "route_action": _route_action(call),
        "tool_used": call is not None,
        "tool": call.tool if call is not None else None,
        "op": call.op if call is not None else None,
        "route_session_id": _route_session_id(case, call, response),
        "intent": case.ground_truth.get("intent") or case.observation.get("intent") or "follow_sidewalk_route",
        "confidence_level": case.ground_truth.get("confidence_level") or "medium",
        "trajectory_type": case.ground_truth.get("trajectory_type") or "local_navigation",
        "trajectory_validity": case.ground_truth.get("trajectory_validity") or "valid",
        "teacher_source": source,
    }
    return build_finish_payload(
        {key: value for key, value in finish_base.items() if value is not None},
        trajectory_norm=_trajectory(case),
    )


def _tool_result_summary(*, call: ToolCall, response: ToolResponse) -> str:
    """Return a short deterministic result summary for ``<ToolResultUse>``."""

    result = response.result if isinstance(response.result, dict) else {}
    if not response.success:
        error = response.error if isinstance(response.error, dict) else {}
        message = error.get("message") or error.get("error") or result.get("message") or "tool_failed"
        return f"{call.tool}.{call.op} failed: {message}"
    if call.tool == "map_tool":
        if call.op in {"search_poi", "search_category"}:
            candidates = _candidate_list(result)
            return f"map search returned {len(candidates)} candidate(s)"
        if call.op == "create_route_session":
            session_id = _nested_get(result, ("session", "session_id")) or result.get("session_id")
            return f"route session created: {session_id}" if session_id else "route session created"
        if call.op in {"add_via_and_replan", "remove_via_and_replan", "replace_destination", "replan_from_current_pose"}:
            return "route session updated"
        if call.op == "get_route_context":
            assets = result.get("context_assets") if isinstance(result.get("context_assets"), list) else []
            return f"route context returned {len(assets)} asset(s)"
        return f"map operation succeeded: {call.op}"
    if call.tool == "image_tool":
        return f"image operation succeeded: {call.op}"
    if call.tool == "user_interaction_tool":
        return "user interaction request prepared"
    return f"tool operation succeeded: {call.tool}.{call.op}"


def _selected_tool_result(*, call: ToolCall, response: ToolResponse) -> str | None:
    """Return the selected POI/route/image artifact in compact text form."""

    result = response.result if isinstance(response.result, dict) else {}
    if not response.success:
        return None
    if call.tool == "map_tool" and call.op in {"search_poi", "search_category"}:
        candidates = _candidate_list(result)
        if not candidates:
            return None
        selected = candidates[0]
        if isinstance(selected, dict):
            name = selected.get("name") or selected.get("title") or selected.get("id") or "candidate"
            location = selected.get("location") or selected.get("coordinate") or selected.get("coordinates")
            distance = selected.get("distance_m") or selected.get("distance")
            parts = [str(name)]
            if location is not None:
                parts.append(f"location={location}")
            if distance is not None:
                parts.append(f"distance_m={distance}")
            return "; ".join(parts)
        return str(selected)
    if call.tool == "map_tool" and call.op in {"create_route_session", "get_route_session"}:
        session_id = _nested_get(result, ("session", "session_id")) or result.get("session_id")
        return str(session_id) if session_id is not None else None
    if call.tool == "map_tool" and call.op == "get_route_context":
        assets = result.get("context_assets") if isinstance(result.get("context_assets"), list) else []
        asset_types = [str(asset.get("asset_type")) for asset in assets if isinstance(asset, dict)]
        return ", ".join(asset_types) if asset_types else None
    if call.tool == "image_tool":
        artifact = result.get("image") or result.get("image_path") or result.get("op")
        return str(artifact) if artifact is not None else None
    return None


def _route_update_from_tool(*, call: ToolCall, response: ToolResponse) -> str:
    """Classify whether the tool changed global route state."""

    if not response.success:
        return "none"
    if call.tool != "map_tool":
        return "none"
    if call.op == "create_route_session":
        return "create_route_session"
    if call.op in {"add_via_and_replan", "remove_via_and_replan", "replace_destination", "replan_from_current_pose"}:
        return str(call.op)
    return "none"


def _navigation_effect_from_tool(*, call: ToolCall, response: ToolResponse) -> str:
    """Describe how the tool result should influence final local navigation."""

    if not response.success:
        return "tool_failed_use_fpv_bev_and_existing_route_context"
    if call.tool == "map_tool":
        if call.op in {"search_poi", "search_category"}:
            return "ground destination or nearby map context before local trajectory generation"
        if call.op == "get_route_context":
            return "condition local trajectory on returned route context assets"
        if call.op == "create_route_session":
            return "condition local trajectory on the newly created route session"
        if call.op in {"add_via_and_replan", "remove_via_and_replan", "replace_destination", "replan_from_current_pose"}:
            return "condition local trajectory on the updated route session"
        return "use map context as a constraint on local navigation"
    if call.tool == "image_tool":
        return "use focused visual evidence for FPV-BEV alignment and obstacle reasoning"
    if call.tool == "user_interaction_tool":
        return "defer or constrain final route decision according to requested clarification"
    return "use external context before final navigation"


def _failure_recovery_from_tool(*, call: ToolCall, response: ToolResponse) -> str:
    """Return the deterministic fallback policy for failed tool execution."""

    del response
    if call.tool == "map_tool":
        return "fallback_to_existing_route_state_and_visible_fpv_bev_context"
    if call.tool == "image_tool":
        return "fallback_to_uncropped_fpv_bev_context"
    if call.tool == "user_interaction_tool":
        return "request_user_input_in_finish_or_keep_low_confidence"
    return "fallback_to_available_context"


def _candidate_list(result: dict[str, Any]) -> list[Any]:
    candidates = result.get("candidates") or result.get("results") or result.get("pois") or []
    return candidates if isinstance(candidates, list) else []


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _route_action(call: ToolCall | None) -> str:
    if call is None:
        return "continue_navigation"
    op = call.op or ""
    if call.tool == "user_interaction_tool":
        return "request_user_input"
    if call.tool == "image_tool":
        return "inspect_visual_context"
    if op == "create_route_session":
        return "create_route"
    if op in {"add_via_and_replan", "remove_via_and_replan", "replace_destination", "replan_from_current_pose"}:
        return "update_route"
    if op in {"search_poi", "search_category", "geocode", "reverse_geocode", "get_route_context"}:
        return "use_map_context"
    return "continue_navigation"


def _route_session_id(case: AgentStage2Case, call: ToolCall | None, response: ToolResponse | None) -> Any:
    if call is not None and call.arguments.get("session_id") is not None:
        return call.arguments.get("session_id")
    result = response.result if response is not None and isinstance(response.result, dict) else {}
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
    if extra.get("trajectory_grid1000") is not None:
        payload = canonicalize_finish_trajectory(payload, trajectory_norm=parse_finish_trajectory_norm(extra) or [])
    for key, value in extra.items():
        if key in {
            "trajectory_grid1000",
            "trajectory_coord_frame",
            "route_action",
            "tool_used",
            "tool",
            "op",
            "route_session_id",
            "teacher_source",
        }:
            payload[key] = value
        else:
            payload.setdefault(key, value)
    return text[: match.start(1)] + json_dumps(payload) + text[match.end(1) :]
