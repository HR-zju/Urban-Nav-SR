"""Export Agentic Stage2 traces to ms-swift agent-template JSONL rows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from mcp_server.agent.schemas import (
    MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION,
    AgentEvalCase,
    AgentStage2Case,
    AgentTrace,
    ToolCall,
)
from mcp_server.agent.contracts import (
    ACTION_CONTRACT_VERSION,
    NAV_OBSERVATION_CONTRACT_VERSION,
    AgentStepResult,
)
from mcp_server.agent.stage2 import (
    NAVIGATION_READOUT_TOKENS,
    READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
    READOUT_PLACEMENT_USER_PROMPT_V1,
    READOUT_PLACEMENTS,
    SYSTEM1_READOUT_TOKENS,
    append_system1_readout,
    final_navigation_user_content,
)
from mcp_server.agent.tool_schema import get_tool_definitions
from mcp_server.runtime.system1_contract import (
    SYSTEM1_TARGET_FIELD_TRAJECTORY_EGO_T,
    SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1,
    canonical_trajectory_policy,
    canonicalize_trajectory_ego_t,
)


DEFAULT_AGENT_SYSTEM_PROMPT = (
    "You are a sidewalk navigation agent. First reason about whether external map, image, "
    "or user-interaction context is needed. When needed, emit a tool call through the "
    "tool_call role. <ToolDecision>.reason_code is optional and should use compact values "
    "such as need_map_context, need_visual_crop, need_user_clarification, or "
    "local_context_sufficient while keeping the free-text reason field. After tool responses, "
    "produce the final navigation CoT and finish block. Raw tool_response content is context; "
    "summarize only its decision-relevant effect in <ToolResultUse>. <Plan> is symbolic local "
    "navigation policy only; do not output CFM action chunks, ego_delta_xyt, or System1Output "
    "in assistant text."
)

_FINISH_RE = re.compile(r"<finish>(.*?)</finish>", re.DOTALL)
_V2_FINISH_SUPERVISION_KEYS = {
    "trajectory",
    "trajectory_norm",
    "trajectory_grid1000",
    "trajectory_coord_frame",
    "trajectory_coordinate",
    "coordinate_frame",
    "pixel_trajectory",
    "trajectory_fpv_t",
    "trajectory_fpv_tv",
    "trajectory_ego_t",
    "action_ego_t",
    "action_mask",
    "semantic_segmentation_path",
    "semantic_mask_path",
}


def _text_content(content) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _json_content(content) -> str:
    if isinstance(content, str):
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return content
        # tool_response 进入 ms-swift row 前必须稳定 JSON 化，避免同义空格影响 diff/review。
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _tool_call_content(content) -> str:
    if isinstance(content, str):
        payload = json.loads(content)
    else:
        payload = content
    # 统一导出为 ms-swift agent-template 期望的 name/arguments 结构。
    call = ToolCall.from_dict(payload if isinstance(payload, dict) else {})
    return json.dumps(call.to_ms_swift_dict(), ensure_ascii=False, sort_keys=True)


def _case_from_trace(trace: AgentTrace) -> AgentStage2Case | None:
    request = trace.request
    if isinstance(request, dict) and request.get("case_id"):
        return AgentStage2Case.from_dict(request)
    return None


def trace_to_ms_swift_stage2_row(
    trace: AgentTrace,
    case: AgentStage2Case | None = None,
    *,
    readout_placement: str = READOUT_PLACEMENT_USER_PROMPT_V1,
) -> dict:
    """Convert one ``AgentTrace`` to a trainable ms-swift agent-template JSONL row.

    Args:
        trace: Source trace.
        case: Optional source case used to recover images and supervision.
        readout_placement: ``user_prompt_v1`` keeps the existing
            ``user [TRAJ_i] -> assistant final`` contract.
            ``assistant_suffix_v2`` exports ``[SYS1_i]`` after final
            ``<finish>`` for System1 CFM training.

    Returns:
        JSON-compatible ms-swift row.

    Raises:
        ValueError: Unknown readout placement or malformed v2 conversion.

    输出字段:
        ``messages``: system/user/assistant/tool_call/tool_response 对话。
        ``tools``: 三个元工具 schema 的 JSON 字符串，避免 pyarrow 类型漂移。
        ``images``: FPV/BEV 图片路径列表。
        ``pixel_trajectory``: 如果 case 或 trace 里存在轨迹监督，则透传。
        ``sidewalk_meta``: trace schema、case_id、metrics 等审计信息。
    """

    if readout_placement not in READOUT_PLACEMENTS:
        raise ValueError(f"Unknown readout placement: {readout_placement}")
    case = case or _case_from_trace(trace)
    messages: list[dict] = []
    has_system = any(turn.role == "system" for turn in trace.turns)
    if not has_system:
        # teacher trace 通常不显式存 system；导出训练 row 时补齐 agent system prompt。
        messages.append({"role": "system", "content": DEFAULT_AGENT_SYSTEM_PROMPT})

    if case is not None and not any(turn.role == "user" for turn in trace.turns):
        # 兼容早期 runner：若 trace 里没有 user turn，则从 case 重建带 <image> 的初始请求。
        image_prefix = "".join("<image>" for _ in case.images)
        messages.append({"role": "user", "content": f"{image_prefix}{case.user_request}"})

    for turn in trace.turns:
        if turn.role == "tool_call":
            # tool_call 是独立 role，content 是 JSON string，不再塞到 assistant content 中。
            messages.append({"role": "tool_call", "content": _tool_call_content(turn.content)})
        elif turn.role == "tool_response":
            # 原始工具返回作为环境上下文，不参与 assistant LM target。
            messages.append({"role": "tool_response", "content": _json_content(turn.content)})
        elif turn.role in {"system", "user", "assistant"}:
            messages.append({"role": turn.role, "content": _text_content(turn.content)})
        else:
            # 未知 role 降级为 assistant 文本，validator 会负责暴露 role sequence 错误。
            messages.append({"role": "assistant", "content": _text_content(turn.content)})

    if readout_placement == READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2:
        messages = _messages_with_assistant_suffix_readout(messages)

    supervision = dict(trace.supervision)
    if case is not None:
        # case ground_truth 优先级更高，保证 Phase1 adapter 透传的 GT 覆盖 trace 临时值。
        supervision.update(case.ground_truth)
    tools = trace.tools or (case.tools if case is not None and case.tools else get_tool_definitions())
    observation_meta = case.to_nav_observation().to_dict() if case is not None else {}
    system2_supervision = _system2_supervision_from_trace(trace, supervision)
    system1_export = _system1_export_supervision(
        supervision,
        readout_placement=readout_placement,
    )
    system1_supervision = _system1_supervision_from_supervision(system1_export)
    step_result = AgentStepResult.from_trace(
        trace,
        case=case,
        validation={"metrics": dict(trace.metrics)},
    )
    row = {
        "schema_version": MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION,
        "stage": "agentic_cold_start",
        "task_type": trace.task_type,
        "messages": messages,
        "tools": json.dumps(tools, ensure_ascii=False, sort_keys=True),
        "images": case.images if case is not None else [],
        "observation_meta": observation_meta,
        "system2_supervision": system2_supervision,
        "system1_supervision": system1_supervision,
        "contract_meta": {
            "observation_contract_version": NAV_OBSERVATION_CONTRACT_VERSION,
            "action_contract_version": ACTION_CONTRACT_VERSION,
            "agent_step_result_schema": step_result.schema_version,
            "system1_readout_placement": readout_placement,
            "system1_readout_tokens": (
                list(SYSTEM1_READOUT_TOKENS)
                if readout_placement == READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2
                else list(NAVIGATION_READOUT_TOKENS)
            ),
            "system1_readout_lm_label": -100,
            "system1_target_field": (
                SYSTEM1_TARGET_FIELD_TRAJECTORY_EGO_T
                if system1_export.get("trajectory_ego_t") is not None
                else None
            ),
            "system1_target_format": (
                SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1
                if system1_export.get("trajectory_ego_t") is not None
                and readout_placement == READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2
                else None
            ),
        },
        "sidewalk_meta": {
            "schema_version": MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION,
            "agent_trace_schema": trace.schema_version,
            "case_id": trace.case_id,
            "task_type": trace.task_type,
            "metrics": dict(trace.metrics),
            "model_run": dict(trace.model_run),
            "artifacts": list(trace.artifacts),
            "agent_step_result": step_result.to_dict(),
        },
    }
    if supervision.get("pixel_trajectory") is not None:
        # 顶层监督字段给 SideWalk trainer 的 trajectory loss 用，不写入 assistant target。
        row["pixel_trajectory"] = supervision["pixel_trajectory"]
    for key in (
        "trajectory_fpv_t",
        "trajectory_fpv_tv",
        "trajectory_ego_t",
        "trajectory_policy",
        "action_ego_t",
        "action_mask",
        "action_stats",
        "semantic_segmentation_path",
        "semantic_mask_path",
    ):
        if system1_export.get(key) is not None:
            # FPV/ego trajectory 与 segmentation 监督继续顶层透传，保持 Phase1 loss 接口不变。
            row[key] = system1_export[key]
    return row


def _system2_supervision_from_trace(trace: AgentTrace, supervision: dict) -> dict:
    """Build text-layer supervision metadata for exported rows.

    Args:
        trace: Source trace.
        supervision: Merged case/trace supervision dictionary.

    Returns:
        JSON-compatible metadata for tool decisions, final navigation decision,
        and expected tool operations.  The canonical LM labels remain in
        ``messages``; this object makes the mapping explicit for review.
    """

    tool_calls = [turn.content for turn in trace.turns if turn.role == "tool_call"]
    return {
        "target_type": "system2_text_actions",
        "expected_tool_ops": supervision.get("expected_tool_ops") or [],
        "tool_call_count": len(tool_calls),
        "has_tool_call": bool(tool_calls),
        "assistant_roles_are_lm_targets": True,
        "tool_response_is_lm_target": False,
    }


def _system1_supervision_from_supervision(supervision: dict) -> dict:
    """Build trajectory/action-head supervision metadata for exported rows.

    Args:
        supervision: Merged supervision dictionary.

    Returns:
        JSON-compatible metadata.  The flat top-level fields are kept for the
        current ms-swift trainer, while this grouped view documents their
        System1 role.
    """

    payload = {
        "target_type": "system1_head_supervision",
        "pixel_trajectory": supervision.get("pixel_trajectory"),
        "trajectory_fpv_t": supervision.get("trajectory_fpv_t"),
        "trajectory_fpv_tv": supervision.get("trajectory_fpv_tv"),
        "trajectory_ego_t": supervision.get("trajectory_ego_t"),
        "trajectory_policy": supervision.get("trajectory_policy"),
        "action_ego_t": supervision.get("action_ego_t"),
        "action_mask": supervision.get("action_mask"),
        "action_stats": supervision.get("action_stats"),
        "semantic_segmentation_path": supervision.get("semantic_segmentation_path"),
        "semantic_mask_path": supervision.get("semantic_mask_path"),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _system1_export_supervision(supervision: dict, *, readout_placement: str) -> dict:
    """Normalize System1 labels for the exported ms-swift contract.

    Args:
        supervision: Merged case/trace supervision dictionary.
        readout_placement: v1/v2 readout placement requested by the exporter.

    Returns:
        A copied supervision dictionary.  For v2 System1 rows,
        ``trajectory_ego_t`` is canonicalized to metric cumulative FLU
        ``[x_forward_m, y_left_m, yaw_rel_ccw_rad]`` so CFM training does not
        silently consume the historical ``x_right/y_front`` convention.
    """

    exported = dict(supervision)
    if readout_placement != READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2:
        return exported
    trajectory_ego_t = exported.get("trajectory_ego_t")
    if trajectory_ego_t is not None:
        policy = dict(exported.get("trajectory_policy") or {})
        source_format = policy.get("system1_target_format")
        canonical = _canonical_system1_trajectory_ego_t(
            trajectory_ego_t,
            source_format=source_format,
        )
        if source_format == SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1:
            yaw_source = str(policy.get("yaw_source") or "")
            if not yaw_source:
                raise ValueError("Canonical trajectory_ego_t requires trajectory_policy.yaw_source.")
        else:
            # 历史 right/front 数据没有可靠逐点 yaw，转换时统一用路径切线。
            yaw_source = "path_tangent"
        exported["trajectory_ego_t"] = canonical
        exported["trajectory_policy"] = canonical_trajectory_policy(
            policy,
            horizon_steps=len(canonical),
            yaw_source=yaw_source,
        )
    return exported


def _canonical_system1_trajectory_ego_t(
    value,
    *,
    source_format: str | None = None,
) -> list[list[float]]:
    """Convert ego trajectory labels to canonical metric cumulative FLU.

    Args:
        value: Sequence shaped ``[T, 3]`` or legacy ``[T, 4]``.
        source_format: Optional declared input format.

    Returns:
        ``[T, 3]`` rows ``[x_forward_m, y_left_m, yaw_rel_ccw_rad]``.

    Raises:
        ValueError: If the input is empty or has an unsupported row width.

    Notes:
        Historical points use ``x_right/y_front``.  The canonical conversion is
        ``x_forward=y_front``, ``y_left=-x_right`` and path-tangent yaw
        ``atan2(delta_y_left, delta_x_forward)``.
    """

    return canonicalize_trajectory_ego_t(value, source_format=source_format)


def _messages_with_assistant_suffix_readout(messages: list[dict]) -> list[dict]:
    """Convert v1 readout placement to v2 assistant-suffix placement.

    Args:
        messages: Exported v1-style message list.

    Returns:
        Copied messages where the final user turn is a text-only final
        navigation prompt and the final assistant has
        ``<readout>[SYS1_i]</readout>`` after ``<finish>``.

    Raises:
        ValueError: The final assistant or v1 readout user turn cannot be
        located.
    """

    converted = [dict(message) for message in messages]
    final_index = _find_final_assistant_message_index(converted)
    if final_index is None:
        raise ValueError("Stage2 row is missing final assistant <finish> turn.")
    readout_index = _find_readout_user_message_index(converted, before=final_index)
    if readout_index is None:
        raise ValueError("Stage2 row is missing v1 final readout user turn.")
    has_tool_roles = any(message.get("role") == "tool_call" for message in converted if isinstance(message, dict))
    converted[readout_index]["content"] = final_navigation_user_content(
        tool_used=has_tool_roles,
        readout_placement=READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
    )
    final_text = _strip_v2_finish_supervision_fields(str(converted[final_index].get("content") or ""))
    converted[final_index]["content"] = append_system1_readout(
        final_text,
        readout_placement=READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
    )
    return converted


def _strip_v2_finish_supervision_fields(text: str) -> str:
    """Remove System1 numeric supervision fields from a v2 ``<finish>`` block.

    Args:
        text: Final assistant text containing a JSON ``<finish>`` payload.

    Returns:
        Assistant text with high-rate trajectory/action/segmentation labels
        removed from ``<finish>``.  Route/tool decision fields stay in text;
        System1 labels remain in top-level row fields and are read out through
        ``assistant_suffix_v2`` tokens.
    """

    match = _FINISH_RE.search(text)
    if match is None:
        return text
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return text
    if not isinstance(payload, dict):
        return text
    stripped = dict(payload)
    for key in _V2_FINISH_SUPERVISION_KEYS:
        stripped.pop(key, None)
    replacement = json.dumps(stripped, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{text[:match.start(1)]}{replacement}{text[match.end(1):]}"


def _find_final_assistant_message_index(messages: list[dict]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "assistant" and "<finish>" in str(message.get("content") or ""):
            return index
    return None


def _find_readout_user_message_index(messages: list[dict], *, before: int) -> int | None:
    for index in range(before - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if all(token in content for token in NAVIGATION_READOUT_TOKENS):
            return index
    return None


def trace_to_ms_swift_row(trace: AgentTrace, case: AgentEvalCase | None = None) -> dict:
    """Compatibility wrapper for the Stage2 exporter.

    新代码应调用 ``trace_to_ms_swift_stage2_row``。保留该函数是为了不破坏早期
    eval helper 和测试。
    """

    stage2_case = AgentStage2Case.from_dict(case.to_dict()) if case is not None else None
    return trace_to_ms_swift_stage2_row(trace, stage2_case)


def write_ms_swift_jsonl(
    traces: Iterable[AgentTrace],
    path: str | Path,
    *,
    cases_by_id: dict[str, AgentEvalCase] | None = None,
    readout_placement: str = READOUT_PLACEMENT_USER_PROMPT_V1,
) -> None:
    """Compatibility wrapper for ``write_ms_swift_stage2_jsonl``."""

    write_ms_swift_stage2_jsonl(
        traces,
        path,
        cases_by_id=cases_by_id,
        readout_placement=readout_placement,
    )


def write_ms_swift_stage2_jsonl(
    traces: Iterable[AgentTrace],
    path: str | Path,
    *,
    cases_by_id: dict[str, AgentStage2Case] | None = None,
    readout_placement: str = READOUT_PLACEMENT_USER_PROMPT_V1,
) -> None:
    """Write ms-swift agent-template JSONL for Stage2 cold-start training.

    Args:
        traces: Source traces.
        path: Output JSONL path.
        cases_by_id: Optional case lookup for supervision/image recovery.
        readout_placement: v1/v2 readout placement exported to each row.

    Returns:
        None.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for trace in traces:
            # cases_by_id 用于把 trace 与原始 supervision/images 重新对齐。
            row = trace_to_ms_swift_stage2_row(
                trace,
                case=(cases_by_id or {}).get(trace.case_id),
                readout_placement=readout_placement,
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_traces_jsonl(traces: Iterable[AgentTrace], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
