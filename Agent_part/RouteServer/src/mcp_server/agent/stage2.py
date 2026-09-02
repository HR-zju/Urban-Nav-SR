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

from urbannav_agent_tools.message_codec import (
    NAVIGATION_READOUT_TOKENS,
    READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
    READOUT_PLACEMENT_USER_PROMPT_V1,
    READOUT_PLACEMENTS,
    SYSTEM1_READOUT_TOKENS,
    append_system1_readout,
    build_agent_think_text,
    final_navigation_user_content,
    initial_user_content as _initial_user_content,
    system1_readout_block,
)
from mcp_server.agent.reasoning_text import parse_or_adapt_reasoning_text, render_reasoning_target_v2
from mcp_server.agent.finish_schema import (
    build_finish_payload,
    canonicalize_finish_trajectory,
    parse_finish_trajectory_norm,
)
from mcp_server.agent.schemas import AgentStage2Case, ToolCall, ToolResponse
from mcp_server.agent.tool_schema import IMAGE_TOOL_NAME, MAP_TOOL_NAME, USER_INTERACTION_TOOL_NAME
from mcp_server.contracts.reasoning import (
    CrossModalAlignmentV2,
    DecisionV2,
    EvidenceGroundingV2,
    FinishV2,
    GroundedFactV2,
    ReasoningTargetV2,
    ReasoningTraceV2,
    TaskGroundingV2,
    UserRequestV2,
)


TRAJECTORY_TOKEN_COUNT = len(NAVIGATION_READOUT_TOKENS)
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
    """Build the first user turn through the standalone message codec.

    Args:
        case: RouteServer Stage2 case containing ordered images and request.

    Returns:
        Provider-neutral initial user content.
    """

    return _initial_user_content(case.images, case.user_request)


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
    readout_placement: str = READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
) -> str:
    """Build the canonical V2 assistant CoT and ``<finish>`` target.

    输入:
        Stage2 case、可选工具调用/返回，以及可选 Phase1 CoT response。
        ``readout_placement`` 控制是否把 v2 System1 readout 追加到最终
        assistant 后缀。

    输出:
        ``ReasoningTraceV2`` 五段 CoT、V2 finish，以及 navigate-only readout。
        旧 Phase1 CoT 只作为输入，经 adapter 转成 V2 后再写出；连续轨迹不进入
        assistant target。tool 路径写 ``<ToolResultUse>``，no-tool 路径不写。

    Raises:
        ValueError: Phase1 target 无法解析或适配为 V2 contract。
    """

    tool_use = _tool_result_use(call=call, response=response) if call is not None and response is not None else None
    base_response = (phase1_response or _find_phase1_response(case)).strip()
    if base_response:
        parsed = parse_or_adapt_reasoning_text(base_response)
        if not parsed.valid or parsed.target is None:
            raise ValueError(f"Phase1 reasoning target cannot be adapted to V2: {parsed.errors}")
        target = _stage2_reasoning_target(
            parsed.target,
            case=case,
            call=call,
            response=response,
        )
    else:
        target = _fallback_reasoning_target(case=case, call=call, response=response)

    text = render_reasoning_target_v2(target, tool_result_use=tool_use)
    return append_system1_readout(text, readout_placement=readout_placement)


def stage2_evidence_catalog(
    case: AgentStage2Case,
    *,
    call: ToolCall | None = None,
    response: ToolResponse | None = None,
) -> list[str]:
    """Build the input-visible evidence-reference catalog for one Stage2 turn.

    Args:
        case: Stage2 observation and request.
        call: Optional executed tool call.
        response: Optional environment tool response.

    Returns:
        Stable sorted references usable by V2 grounding validation. Supervision
        fields from ``ground_truth`` are intentionally excluded.
    """

    refs = {"instruction.active"}
    images = case.observation.get("images")
    has_fpv = bool(case.images)
    has_bev = "bev" in case.image_roles
    if isinstance(images, dict):
        has_fpv = bool(images.get("fpv") or images.get("front") or has_fpv)
        has_bev = bool(images.get("bev") or has_bev)
    if has_fpv:
        refs.update({"fpv.current", "corridor.left", "corridor.center", "corridor.right"})
    if has_bev:
        refs.add("bev.current")

    observation = case.observation
    if observation.get("destination") is not None or observation.get("goal") is not None:
        refs.add("goal.active")
    route_keys = ("route_session_id", "session_id", "route_id", "route_polyline", "waypoints")
    if has_bev or any(observation.get(key) is not None for key in route_keys):
        refs.add("route.active")

    for payload in (observation, case.metadata.get("nav_observation") if isinstance(case.metadata, dict) else None):
        if isinstance(payload, dict):
            _collect_evidence_refs(payload, refs=refs)

    if call is not None:
        refs.add("tool.latest")
        refs.add(_tool_evidence_ref(call))
        if call.tool == MAP_TOOL_NAME:
            refs.add("route.active")
    if response is not None and isinstance(response.result, dict):
        _collect_evidence_refs(response.result, refs=refs)
    return sorted(refs)


def _stage2_reasoning_target(
    base: ReasoningTargetV2,
    *,
    case: AgentStage2Case,
    call: ToolCall | None,
    response: ToolResponse | None,
) -> ReasoningTargetV2:
    """Augment one adapted Phase1 target with Stage2 tool and control state."""

    trace = base.reasoning
    evidence_payload = trace.evidence_grounding.model_dump(mode="json")
    tool_refs = list(evidence_payload.get("tool_evidence_refs") or [])
    facts = list(evidence_payload.get("relevant_facts") or [])
    if call is not None:
        tool_ref = _tool_evidence_ref(call)
        for ref in ("tool.latest", tool_ref):
            if ref not in tool_refs:
                tool_refs.append(ref)
        facts.append(
            GroundedFactV2(
                fact_code="tool_result_available" if response is not None and response.success else "tool_result_failed",
                evidence_refs=[tool_ref],
            ).model_dump(mode="json")
        )
    evidence_payload["tool_evidence_refs"] = tool_refs
    evidence_payload["relevant_facts"] = facts

    decision_type = _stage2_decision_type(case=case, call=call, response=response, default=base.finish.decision_type)
    decision = trace.decision
    if decision_type != "navigate":
        decision = DecisionV2(
            motion="stop",
            speed="stop",
            social_action="wait" if decision_type == "ask_user" else "none",
            selected_corridor_ref=None,
            fallback="stop",
            confidence=decision.confidence,
            reason_codes=[decision_type],
        )
    updated_trace = ReasoningTraceV2(
        task_grounding=trace.task_grounding,
        evidence_grounding=EvidenceGroundingV2.model_validate(evidence_payload),
        cross_modal_alignment=trace.cross_modal_alignment,
        constraint_applicability=trace.constraint_applicability,
        decision=decision,
    )
    finish = _build_finish_v2(
        case=case,
        decision=decision,
        decision_type=decision_type,
        call=call,
        response=response,
        previous=base.finish,
    )
    return ReasoningTargetV2(reasoning=updated_trace, finish=finish)


def _fallback_reasoning_target(
    *,
    case: AgentStage2Case,
    call: ToolCall | None,
    response: ToolResponse | None,
) -> ReasoningTargetV2:
    """Build a conservative cold-start V2 target without fabricated geometry."""

    catalog = set(stage2_evidence_catalog(case, call=call, response=response))
    route_available = "route.active" in catalog
    fpv_available = "fpv.current" in catalog
    desired_direction = _desired_direction_from_case(case)
    motion = {
        "left": "turn_left",
        "right": "turn_right",
        "stop": "stop",
    }.get(desired_direction, "go_straight")
    decision_type = _stage2_decision_type(case=case, call=call, response=response, default="navigate")
    confidence = _confidence_from_case(case)
    decision = DecisionV2(
        motion="stop" if decision_type != "navigate" else motion,
        speed="stop" if decision_type != "navigate" or motion == "stop" else "normal",
        social_action="wait" if decision_type == "ask_user" else "unknown",
        selected_corridor_ref=None,
        fallback="stop" if decision_type != "navigate" else "slow",
        confidence=confidence,
        reason_codes=["cold_start_without_dense_scene_label"],
    )
    task_refs = ["instruction.active"]
    route_refs = ["route.active"] if route_available else []
    fpv_refs = ["fpv.current"] if fpv_available else []
    tool_refs = []
    facts = [GroundedFactV2(fact_code="active_instruction_available", evidence_refs=task_refs)]
    if route_available:
        facts.append(GroundedFactV2(fact_code="route_context_available", evidence_refs=route_refs))
    if fpv_available:
        facts.append(GroundedFactV2(fact_code="fpv_observation_available", evidence_refs=fpv_refs))
    if call is not None:
        tool_ref = _tool_evidence_ref(call)
        tool_refs = ["tool.latest", tool_ref]
        facts.append(
            GroundedFactV2(
                fact_code="tool_result_available" if response is not None and response.success else "tool_result_failed",
                evidence_refs=[tool_ref],
            )
        )
    trace = ReasoningTraceV2(
        task_grounding=TaskGroundingV2(
            goal_ref="goal.active" if "goal.active" in catalog else None,
            route_ref="route.active" if route_available else None,
            local_objective="follow_instruction" if not route_available else "continue_route",
            instruction_constraints=[],
        ),
        evidence_grounding=EvidenceGroundingV2(
            task_evidence_refs=task_refs,
            route_evidence_refs=route_refs,
            fpv_evidence_refs=fpv_refs,
            tool_evidence_refs=tool_refs,
            relevant_facts=facts,
        ),
        cross_modal_alignment=CrossModalAlignmentV2(
            available=fpv_available and route_available,
            desired_direction=desired_direction,
            route_to_fpv="unknown",
            selected_corridor_ref=None,
            status="insufficient_evidence",
            evidence_refs=[*route_refs, *fpv_refs] if fpv_available and route_available else [],
        ),
        constraint_applicability={"applicable": [], "not_applicable": []},
        decision=decision,
    )
    finish = _build_finish_v2(
        case=case,
        decision=decision,
        decision_type=decision_type,
        call=call,
        response=response,
        previous=None,
    )
    return ReasoningTargetV2(reasoning=trace, finish=finish)


def _build_finish_v2(
    *,
    case: AgentStage2Case,
    decision: DecisionV2,
    decision_type: str,
    call: ToolCall | None,
    response: ToolResponse | None,
    previous: FinishV2 | None,
) -> FinishV2:
    """Build decision-specific V2 finish fields without continuous labels."""

    payload: dict[str, Any] = {
        "decision_type": decision_type,
        "route_action": _route_action(call),
        "tool_used": call is not None,
    }
    if decision_type == "navigate":
        previous_trajectory_type = previous.trajectory_type if previous and previous.decision_type == "navigate" else None
        payload.update(
            {
                "motion": decision.motion,
                "speed": decision.speed,
                "social_action": decision.social_action,
                "confidence_level": decision.confidence,
                "trajectory_type": "stop_hold"
                if decision.motion == "stop"
                else previous_trajectory_type or str(case.ground_truth.get("trajectory_type") or "local_navigation"),
            }
        )
    elif decision_type == "ask_user":
        payload["user_request"] = _user_request_from_tool(case=case, call=call, response=response)
    else:
        default_code = "arrival_confirmed" if decision_type == "complete_task" else "navigation_aborted"
        payload.update(
            {
                "reason_code": case.ground_truth.get("reason_code") or (previous.reason_code if previous else None) or default_code,
                "reason": case.ground_truth.get("reason") or (previous.reason if previous else None),
                "evidence_refs": case.ground_truth.get("terminal_evidence_refs")
                or (list(previous.evidence_refs) if previous else []),
                "retryable": case.ground_truth.get("retryable")
                if "retryable" in case.ground_truth
                else (previous.retryable if previous else None),
            }
        )
    return FinishV2.model_validate(payload)


def _stage2_decision_type(
    *,
    case: AgentStage2Case,
    call: ToolCall | None,
    response: ToolResponse | None,
    default: str,
) -> str:
    """Resolve the runtime control decision from explicit inputs only."""

    raw = str(case.ground_truth.get("decision_type") or default or "navigate").strip().lower()
    raw = {"finish": "complete_task", "stop": "abort_task"}.get(raw, raw)
    if call is not None and call.tool == USER_INTERACTION_TOOL_NAME:
        result = response.result if response is not None and isinstance(response.result, dict) else {}
        requires_user = bool(result.get("requires_user_response", result.get("await_user", True)))
        if requires_user:
            return "ask_user"
    return raw if raw in {"navigate", "ask_user", "complete_task", "abort_task"} else "navigate"


def _user_request_from_tool(
    *,
    case: AgentStage2Case,
    call: ToolCall | None,
    response: ToolResponse | None,
) -> UserRequestV2:
    """Build a bounded user-interaction request from the executed tool turn."""

    args = call.arguments if call is not None else {}
    result = response.result if response is not None and isinstance(response.result, dict) else {}
    raw_choices = result.get("choices") or result.get("options") or args.get("choices") or args.get("options") or []
    choices: list[str] = []
    for item in raw_choices:
        if isinstance(item, dict):
            value = item.get("label") or item.get("text") or item.get("id")
        else:
            value = item
        if value is not None:
            choices.append(str(value))
    raw_policy = str(result.get("motion_policy_while_waiting") or args.get("motion_policy_while_waiting") or "stop")
    policy = raw_policy if raw_policy in {"stop", "slow_down", "handover_to_safety_controller"} else "stop"
    return UserRequestV2(
        question=str(
            result.get("question")
            or result.get("message")
            or args.get("question")
            or args.get("message")
            or case.user_request
        ),
        choices=choices,
        motion_policy_while_waiting=policy,
    )


def _desired_direction_from_case(case: AgentStage2Case) -> str:
    """Normalize explicit local intent into a route direction."""

    raw = str(case.ground_truth.get("intent") or case.observation.get("intent") or "").lower()
    if "left" in raw:
        return "left"
    if "right" in raw:
        return "right"
    if raw in {"stop", "hold", "arrive", "stop_at_goal"}:
        return "stop"
    if raw in {"straight", "go_straight", "continue", "follow_sidewalk_route"}:
        return "straight"
    return "unknown"


def _confidence_from_case(case: AgentStage2Case) -> str:
    """Normalize teacher confidence without inventing high confidence."""

    raw = str(case.ground_truth.get("confidence_level") or "medium").lower()
    return {"normal": "high", "high": "high", "medium": "medium", "low": "low"}.get(raw, "unknown")


def _tool_evidence_ref(call: ToolCall) -> str:
    """Return a stable evidence reference for one tool response."""

    suffix = call.call_id or f"{call.tool}_{call.op or 'call'}"
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(suffix))
    return f"tool.{safe}"


def _collect_evidence_refs(payload: Any, *, refs: set[str]) -> None:
    """Collect explicit reference ids from runtime-visible nested payloads."""

    if isinstance(payload, str):
        if "." in payload:
            refs.add(payload)
        return
    if isinstance(payload, list):
        for item in payload:
            _collect_evidence_refs(item, refs=refs)
        return
    if not isinstance(payload, dict):
        return
    for key in ("ref", "evidence_ref", "asset_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            refs.add(value)
    for key in ("evidence_catalog", "context_assets", "evidence_refs", "trigger_refs"):
        if key in payload:
            _collect_evidence_refs(payload[key], refs=refs)
    for value in payload.values():
        if isinstance(value, (dict, list)):
            _collect_evidence_refs(value, refs=refs)


def _finish_decision_type(text: str) -> str | None:
    """Parse only the finish decision needed by readout gating."""

    match = _FINISH_RE.search(text)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    decision = str(payload.get("decision_type") or "").lower()
    return {"finish": "complete_task", "stop": "abort_task"}.get(decision, decision or None)


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
            "decision_type",
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
