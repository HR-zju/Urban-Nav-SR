"""Validators for SideWalk Agentic Stage2 cases, traces, and exported rows.

validator 与 metrics 的职责不同 metrics 给出可聚合分数 validator 给出可审计的
错误列表，适合 build 脚本决定样本是否进入 review/train JSONL。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from mcp_server.agent.finish_schema import finish_trajectory_matches
from mcp_server.agent.schemas import (
    AGENT_STAGE2_CASE_SCHEMA_VERSION,
    AGENT_STAGE2_TRACE_SCHEMA_VERSION,
    MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION,
    AgentStage2Case,
    AgentTrace,
    ToolCall,
)
from mcp_server.agent.stage2 import (
    NAVIGATION_READOUT_TOKENS,
    READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2,
    READOUT_PLACEMENT_USER_PROMPT_V1,
    READOUT_PLACEMENTS,
    SYSTEM1_READOUT_TOKENS,
)
from mcp_server.agent.tool_parser import parse_finish_text
from mcp_server.agent.tool_schema import USER_INTERACTION_TOOL_NAME, validate_tool_call
from mcp_server.eval.metrics import compute_trace_metrics
from mcp_server.runtime.system1_contract import (
    SYSTEM1_AXIS_NAMES_EGO_POSE_XYT_FLU,
    SYSTEM1_COORDINATE_FRAME_CURRENT_EGO_FLU,
    SYSTEM1_REPRESENTATION_CUMULATIVE_RELATIVE_POSE,
    SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1,
    SYSTEM1_YAW_SOURCES,
)


STAGE2_TOOL_TRACE_ROLE_SEQUENCE = ("user", "assistant", "tool_call", "tool_response", "user", "assistant")
STAGE2_NO_TOOL_TRACE_ROLE_SEQUENCE = ("user", "assistant", "user", "assistant")
STAGE2_TRACE_ROLE_SEQUENCE = STAGE2_TOOL_TRACE_ROLE_SEQUENCE
STAGE2_TOOL_ROW_ROLE_SEQUENCE = ("system", "user", "assistant", "tool_call", "tool_response", "user", "assistant")
STAGE2_NO_TOOL_ROW_ROLE_SEQUENCE = ("system", "user", "assistant", "user", "assistant")
STAGE2_ROW_ROLE_SEQUENCE = STAGE2_TOOL_ROW_ROLE_SEQUENCE
FINAL_COT_REQUIRED_BASE_TAGS = ("<Scene>", "<FPVScene>", "<BEVContext>", "<CrossView>", "<Plan>", "<finish>")
FINAL_COT_REQUIRED_TOOL_TAGS = FINAL_COT_REQUIRED_BASE_TAGS + ("<ToolResultUse>",)
FINAL_COT_REQUIRED_TAGS = FINAL_COT_REQUIRED_TOOL_TAGS
USER_WAIT_MOTION_POLICIES = {
    "stop",
    "slow_down",
    "continue_previous_safe_policy",
    "handover_to_safety_controller",
}
FORBIDDEN_CONTEXT_ASSET_KEYS = {
    "action_ego_t",
    "action_mask",
    "action_stats",
    "future_gps",
    "future_pose",
    "future_trajectory",
    "ground_truth",
    "pixel_trajectory",
    "semantic_segmentation_path",
    "teacher_answer",
    "trajectory_ego_t",
    "trajectory_fpv_t",
}


@dataclass(frozen=True)
class Stage2ValidationResult:
    """Validation result returned by Stage2 validators.

    属性:
        valid: 没有 hard errors 时为 True。
        errors: 会导致样本不可训练或不可回放的问题。
        warnings: 不阻塞训练但值得人工 review 的问题。
        metrics: trace validator 会附带协议级 metrics 便于 report 聚合。
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


def validate_stage2_case(case: AgentStage2Case) -> Stage2ValidationResult:
    """Validate one Stage2 case before trace generation.

    输入:
        ``AgentStage2Case``。

    输出:
        ``Stage2ValidationResult``；主要检查 schema、case_id、user_request 和图片字段。
    """

    errors: list[str] = []
    warnings: list[str] = []
    if case.schema_version != AGENT_STAGE2_CASE_SCHEMA_VERSION:
        errors.append(f"schema_version:{case.schema_version}")
    if not case.case_id:
        errors.append("missing_case_id")
    if not case.user_request.strip():
        errors.append("missing_user_request")
    if not case.images:
        warnings.append("missing_images")
    if not isinstance(case.ground_truth, dict):
        # ground_truth 允许为空，但类型必须稳定，便于 exporter 透传监督字段。
        errors.append("ground_truth_not_object")
    return Stage2ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def validate_stage2_trace(trace: AgentTrace, case: AgentStage2Case | None = None) -> Stage2ValidationResult:
    """Validate the normalized Stage2 trace contract.

    输入:
        ``AgentTrace``，可选原始 ``AgentStage2Case``。

    输出:
        ``Stage2ValidationResult``。检查 tool/no-tool role 顺序、tool_call JSON、
        tool_response JSON、readout token 位置、最终 CoT tags 和 ``<finish>`` JSON。
    """

    errors: list[str] = []
    warnings: list[str] = []
    if trace.schema_version != AGENT_STAGE2_TRACE_SCHEMA_VERSION:
        errors.append(f"trace_schema_version:{trace.schema_version}")
    roles = tuple(turn.role for turn in trace.turns)
    has_tool_roles = "tool_call" in roles or "tool_response" in roles
    # tool/no-tool 使用两套固定序列；混入多轮工具或 legacy role 会直接报错。
    expected_roles = STAGE2_TOOL_TRACE_ROLE_SEQUENCE if has_tool_roles else STAGE2_NO_TOOL_TRACE_ROLE_SEQUENCE
    if roles != expected_roles:
        errors.append(f"trace_role_sequence:{roles}")

    call = None
    if has_tool_roles and len(trace.turns) > 2:
        # tool_call 必须是可解析 JSON object，并通过工具 schema 校验。
        call_payload, call_errors = _json_dict(trace.turns[2].content)
        errors.extend(f"tool_call:{item}" for item in call_errors)
        if call_payload is not None:
            call = ToolCall.from_dict(call_payload)
            is_valid, validation_errors = validate_tool_call(call)
            if not is_valid:
                errors.extend(f"tool_call_schema:{item}" for item in validation_errors)
    if has_tool_roles and len(trace.turns) > 3:
        # failed tool 也是合法 response，但必须显式给出 success=false。
        response_payload, response_errors = _json_dict(trace.turns[3].content)
        errors.extend(f"tool_response:{item}" for item in response_errors)
        if response_payload is not None:
            if response_payload.get("tool") != (call.tool if call is not None else trace.turns[3].tool):
                errors.append("tool_response_tool_mismatch")
            if "success" not in response_payload:
                errors.append("tool_response_missing_success")
            _validate_tool_response_contract(response_payload, errors=errors, warnings=warnings)

    readout_index = 4 if has_tool_roles else 2
    final_index = 5 if has_tool_roles else 3
    final_text_for_readout = trace.turns[final_index].content if len(trace.turns) > final_index else ""
    if len(trace.turns) > readout_index:
        # 这里直接检查所有 [TRAJ_i]，保证 trajectory head 的 readout token 完整出现。
        readout = trace.turns[readout_index].content
        if not isinstance(readout, str) or not _contains_all_v1_readout_tokens(readout):
            if not _assistant_suffix_readout_valid(str(final_text_for_readout or "")):
                errors.append("missing_readout_tokens")
    elif not _assistant_suffix_readout_valid(str(final_text_for_readout or "")):
        errors.append("missing_readout_tokens")
    if len(trace.turns) > final_index:
        final_text = trace.turns[final_index].content
        if not isinstance(final_text, str):
            errors.append("final_assistant_not_text")
        else:
            required_tags = FINAL_COT_REQUIRED_TOOL_TAGS if has_tool_roles else FINAL_COT_REQUIRED_BASE_TAGS
            for tag in required_tags:
                if tag not in final_text:
                    errors.append(f"final_missing_tag:{tag}")
            finish, finish_errors = parse_finish_text(final_text)
            errors.extend(f"finish:{item}" for item in finish_errors)
            if finish is not None and case is not None:
                expected_traj = case.ground_truth.get("pixel_trajectory")
                if expected_traj and not finish_trajectory_matches(finish, expected_traj):
                    # 文本 finish 和顶层 trajectory supervision 必须等价，否则 LM/traj loss 目标冲突。
                    errors.append("finish_trajectory_mismatch")
    else:
        errors.append("missing_final_assistant")

    for index, turn in enumerate(trace.turns):
        if turn.role != "assistant" or not isinstance(turn.content, str):
            continue
        if "<tool_response>" in turn.content:
            # 禁止 v1 XML tool_response 回流，Stage2 只能用独立 tool_response role。
            errors.append(f"legacy_tool_response_format:{index}")
        if index != final_index and "<ToolResultUse>" in turn.content:
            # ToolResultUse 只表达最终如何使用工具结果，不应出现在 pre-tool think。
            errors.append(f"tool_result_use_outside_final:{index}")

    metric_result = compute_trace_metrics(trace, case)
    for key in (
        "parse_success",
        "stage2_role_order_valid",
        "trajectory_readout_order_valid",
        "final_cot_validity",
        "tool_result_use_validity",
        "tool_decision_consistency",
    ):
        if metric_result.metrics.get(key) != 1.0:
            # metrics 里的协议失败提升为 validator hard error，build 脚本据此拦截坏样本。
            errors.append(f"metric_failed:{key}")
    errors.extend(metric_result.errors)
    return Stage2ValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        metrics=metric_result.metrics,
    )


def validate_ms_swift_stage2_row(row: dict[str, Any]) -> Stage2ValidationResult:
    """Validate one exported ms-swift Stage2 JSONL row.

    输入:
        exporter 产出的 JSON object。

    输出:
        ``Stage2ValidationResult``。重点检查 ms-swift agent-template 所需字段类型：
        ``tools`` 必须是 JSON 字符串，``tool_call`` / ``tool_response`` content 也必须
        是 JSON 字符串。
    """

    errors: list[str] = []
    warnings: list[str] = []
    if row.get("schema_version") != MS_SWIFT_AGENTIC_STAGE2_SCHEMA_VERSION:
        errors.append(f"row_schema_version:{row.get('schema_version')}")
    messages = row.get("messages")
    if not isinstance(messages, list):
        return Stage2ValidationResult(valid=False, errors=["messages_not_list"])
    roles = tuple(str(message.get("role")) for message in messages if isinstance(message, dict))
    has_tool_roles = "tool_call" in roles or "tool_response" in roles
    readout_placement = _row_readout_placement(row)
    # exported row 比 trace 多 system，因此单独校验 row role sequence。
    expected_roles = STAGE2_TOOL_ROW_ROLE_SEQUENCE if has_tool_roles else STAGE2_NO_TOOL_ROW_ROLE_SEQUENCE
    if roles != expected_roles:
        errors.append(f"row_role_sequence:{roles}")

    tools = row.get("tools")
    if not isinstance(tools, str):
        errors.append("tools_not_json_string")
    else:
        # ms-swift/pyarrow 对 list[dict] 容易类型漂移，因此 tools 必须是 JSON string。
        payload, payload_errors = _json_value(tools)
        errors.extend(f"tools:{item}" for item in payload_errors)
        if payload is not None and not isinstance(payload, list):
            errors.append("tools_json_not_list")

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"message_not_object:{index}")
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"tool_call", "tool_response"}:
            # agent-template 期望 tool role content 是 JSON 字符串，而不是 Python dict。
            payload, payload_errors = _json_dict(content)
            errors.extend(f"{role}:{item}" for item in payload_errors)
            if role == "tool_call" and payload is not None:
                call = ToolCall.from_dict(payload)
                is_valid, validation_errors = validate_tool_call(call)
                if not is_valid:
                    errors.extend(f"tool_call_schema:{item}" for item in validation_errors)
            if role == "tool_response" and payload is not None:
                _validate_tool_response_contract(payload, errors=errors, warnings=warnings)
        if role == "tool" or (isinstance(content, str) and "<tool_response>" in content):
            errors.append("legacy_tool_response_format")
        if role == "assistant" and index != (6 if has_tool_roles else 4) and isinstance(content, str) and "<ToolResultUse>" in content:
            errors.append(f"tool_result_use_outside_final:{index}")
    readout_index = 5 if has_tool_roles else 3
    final_index = 6 if has_tool_roles else 4
    if len(messages) > final_index:
        final_text = messages[final_index].get("content")
        required_tags = FINAL_COT_REQUIRED_TOOL_TAGS if has_tool_roles else FINAL_COT_REQUIRED_BASE_TAGS
        if not isinstance(final_text, str) or not all(tag in final_text for tag in required_tags):
            errors.append("row_final_cot_invalid")
        else:
            finish, finish_errors = parse_finish_text(final_text)
            errors.extend(f"row_finish:{item}" for item in finish_errors)
            if (
                finish is not None
                and row.get("pixel_trajectory") is not None
                and readout_placement != READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2
                and not finish_trajectory_matches(finish, row.get("pixel_trajectory"))
            ):
                errors.append("row_finish_trajectory_mismatch")
        if readout_placement == READOUT_PLACEMENT_ASSISTANT_SUFFIX_V2:
            if len(messages) > readout_index and _contains_all_v1_readout_tokens(str(messages[readout_index].get("content"))):
                errors.append("row_v2_legacy_user_readout_tokens")
            if not isinstance(final_text, str) or not _assistant_suffix_readout_valid(final_text):
                errors.append("row_missing_assistant_suffix_readout")
            _validate_system1_v2_targets(row, errors=errors)
        elif len(messages) > readout_index and not _contains_all_v1_readout_tokens(str(messages[readout_index].get("content"))):
            errors.append("row_missing_readout_tokens")
        elif len(messages) <= readout_index:
            errors.append("row_missing_readout_tokens")
    else:
        errors.append("row_missing_final_assistant")
    if not row.get("images"):
        warnings.append("row_missing_images")
    return Stage2ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def _json_value(text: Any) -> tuple[Any | None, list[str]]:
    if not isinstance(text, str):
        return None, [f"json_text_type:{type(text).__name__}"]
    try:
        return json.loads(text), []
    except json.JSONDecodeError as exc:
        return None, [f"json_decode_error:{exc.msg}"]


def _json_dict(content: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if isinstance(content, dict):
        return dict(content), []
    payload, errors = _json_value(content)
    if errors:
        return None, errors
    if not isinstance(payload, dict):
        return None, ["json_not_object"]
    return payload, []


def _contains_all_v1_readout_tokens(text: str) -> bool:
    return all(token in text for token in NAVIGATION_READOUT_TOKENS)


def _contains_all_v2_readout_tokens(text: str) -> bool:
    return all(token in text for token in SYSTEM1_READOUT_TOKENS)


def _assistant_suffix_readout_valid(text: str) -> bool:
    finish_index = text.rfind("</finish>")
    readout_index = text.find("<readout>", finish_index + len("</finish>")) if finish_index >= 0 else -1
    return finish_index >= 0 and readout_index > finish_index and _contains_all_v2_readout_tokens(text[readout_index:])


def _row_readout_placement(row: dict[str, Any]) -> str:
    contract_meta = row.get("contract_meta") if isinstance(row.get("contract_meta"), dict) else {}
    placement = str(contract_meta.get("system1_readout_placement") or READOUT_PLACEMENT_USER_PROMPT_V1)
    if placement not in READOUT_PLACEMENTS:
        return READOUT_PLACEMENT_USER_PROMPT_V1
    return placement


def _validate_system1_v2_targets(row: dict[str, Any], *, errors: list[str]) -> None:
    """Validate assistant-suffix System1 supervision shape.

    Args:
        row: Exported ms-swift row.
        errors: Mutable hard-error list.

    Returns:
        None.

    Notes:
        v2 CFM rows use metric cumulative FLU ``trajectory_ego_t`` as
        ``[x_forward_m, y_left_m, yaw_rel_ccw_rad]``.  Historical
        ``x_right/y_front`` rows must be converted before a row declares
        ``assistant_suffix_v2`` System1 readout tokens.
    """

    trajectory = row.get("trajectory_ego_t")
    if trajectory is None:
        return
    width = _last_dim_width(trajectory)
    if width != 3:
        errors.append(f"row_system1_trajectory_ego_t_width:{width}")
    contract_meta = row.get("contract_meta") if isinstance(row.get("contract_meta"), dict) else {}
    target_format = contract_meta.get("system1_target_format")
    if target_format != SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1:
        errors.append(f"row_system1_target_format:{target_format}")
    policy = row.get("trajectory_policy") if isinstance(row.get("trajectory_policy"), dict) else {}
    if policy.get("system1_target_format") != SYSTEM1_TARGET_FORMAT_EGO_POSE_XYT_FLU_V1:
        errors.append("row_system1_policy_target_format_not_canonical")
    if policy.get("representation") != SYSTEM1_REPRESENTATION_CUMULATIVE_RELATIVE_POSE:
        errors.append("row_system1_policy_representation_not_cumulative")
    if policy.get("coordinate_frame") != SYSTEM1_COORDINATE_FRAME_CURRENT_EGO_FLU:
        errors.append("row_system1_policy_coordinate_frame_not_flu")
    if policy.get("axis_names") != list(SYSTEM1_AXIS_NAMES_EGO_POSE_XYT_FLU):
        errors.append("row_system1_policy_axis_names_not_canonical")
    if policy.get("normalized") is not False:
        errors.append("row_system1_policy_must_be_metric")
    if policy.get("yaw_source") not in SYSTEM1_YAW_SOURCES:
        errors.append("row_system1_policy_yaw_source_missing")
    system1 = row.get("system1_supervision") if isinstance(row.get("system1_supervision"), dict) else {}
    if system1.get("trajectory_ego_t") is not None and _last_dim_width(system1.get("trajectory_ego_t")) != 3:
        errors.append("row_system1_supervision_trajectory_ego_t_not_canonical")


def _last_dim_width(value: Any) -> int | None:
    if not isinstance(value, list) or not value:
        return None
    first = value[0]
    if isinstance(first, list):
        return len(first)
    return None


def _validate_tool_response_contract(
    payload: dict[str, Any],
    *,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate optional contract fields in one tool response.

    Args:
        payload: Parsed ``tool_response`` content.
        errors: Mutable hard-error list.
        warnings: Mutable warning list.

    Returns:
        None.  This function keeps Stage2 role compatibility and only checks
        optional user-interaction/context-asset contracts when present.
    """

    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    if payload.get("tool") == USER_INTERACTION_TOOL_NAME:
        # user_interaction_tool 是“向用户发起动作”的工具，不是普通文本回答；
        # 需要显式记录是否等待用户以及等待期间的运动策略。
        if "await_user" not in result:
            warnings.append("user_interaction_missing_await_user")
        requires_user = bool(result.get("requires_user_response", result.get("await_user") is True))
        if requires_user and result.get("await_user") is not True:
            errors.append("user_interaction_requires_user_but_not_awaiting")
        if requires_user and result.get("interaction_mode") != "blocking":
            errors.append("user_interaction_requires_user_must_be_blocking")
        if requires_user and result.get("motion_policy_while_waiting") not in USER_WAIT_MOTION_POLICIES:
            errors.append("user_interaction_bad_motion_policy")
        if result.get("await_user") is True:
            if not result.get("dialogue_state"):
                warnings.append("user_interaction_missing_dialogue_state")
            pending = result.get("dialogue_state", {}).get("pending_user_interaction")
            if not isinstance(pending, dict):
                warnings.append("user_interaction_missing_pending_state")
        op = str(result.get("op") or "")
        if op == "present_options":
            _validate_user_options(result, errors=errors, warnings=warnings)
        if op == "answer_query" and not result.get("evidence_refs"):
            warnings.append("user_interaction_answer_query_missing_evidence_refs")
    context_assets = result.get("context_assets")
    if context_assets is not None:
        _validate_context_assets(context_assets, errors=errors, warnings=warnings)


def _validate_user_options(
    result: dict[str, Any],
    *,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate ``present_options`` response payload.

    Args:
        result: Parsed user-interaction result.
        errors: Mutable hard-error list.
        warnings: Mutable warning list.

    Returns:
        None.
    """

    options = result.get("options")
    if not isinstance(options, list) or not options:
        errors.append("user_interaction_present_options_missing_options")
        return
    option_ids: set[str] = set()
    evidence_refs = {str(item) for item in result.get("evidence_refs") or []}
    for index, option in enumerate(options):
        if not isinstance(option, dict):
            errors.append(f"user_interaction_option_not_object:{index}")
            continue
        option_id = option.get("option_id") or option.get("id")
        if option_id is None:
            errors.append(f"user_interaction_option_missing_id:{index}")
        else:
            option_ids.add(str(option_id))
        candidate_id = option.get("candidate_id")
        if candidate_id is not None and not (str(candidate_id) in evidence_refs or option.get("evidence_ref")):
            warnings.append(f"user_interaction_option_candidate_without_evidence_ref:{candidate_id}")
    recommended = result.get("recommended_option_id")
    if recommended is None:
        warnings.append("user_interaction_present_options_missing_recommended_option")
    elif str(recommended) not in option_ids:
        errors.append("user_interaction_recommended_option_not_in_options")


def _validate_context_assets(
    context_assets: Any,
    *,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate route/image context assets returned by tools.

    Args:
        context_assets: Candidate context asset list.
        errors: Mutable hard-error list.
        warnings: Mutable warning list.

    Returns:
        None.
    """

    if not isinstance(context_assets, list):
        errors.append("context_assets_not_list")
        return
    seen: set[str] = set()
    for index, asset in enumerate(context_assets):
        if not isinstance(asset, dict):
            errors.append(f"context_asset_not_object:{index}")
            continue
        asset_id = str(asset.get("asset_id") or "")
        if not asset_id:
            errors.append(f"context_asset_missing_id:{index}")
        elif asset_id in seen:
            errors.append(f"context_asset_duplicate_id:{asset_id}")
        seen.add(asset_id)
        modality = asset.get("modality")
        if modality not in {"text", "structured", "image"}:
            errors.append(f"context_asset_bad_modality:{asset_id or index}")
        if _contains_forbidden_context_asset_key(asset):
            # context_assets 是环境观测，不允许混入 GT / future supervision。
            errors.append(f"context_asset_contains_supervision:{asset_id or index}")
        if asset.get("asset_type") == "next_waypoint" and not asset.get("frame"):
            warnings.append(f"context_asset_next_waypoint_missing_frame:{asset_id or index}")
        if asset.get("modality") == "image":
            payload = asset.get("payload") if isinstance(asset.get("payload"), dict) else {}
            if not payload.get("image_ref") and payload.get("render_status") != "not_implemented":
                errors.append(f"context_asset_image_without_ref:{asset_id or index}")


def _contains_forbidden_context_asset_key(payload: Any) -> bool:
    """Return True when a context asset exposes training-only supervision.

    Args:
        payload: Arbitrary nested context asset payload.

    Returns:
        ``True`` if a key looks like ground-truth / future supervision.
    """

    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in FORBIDDEN_CONTEXT_ASSET_KEYS:
                return True
            if _contains_forbidden_context_asset_key(value):
                return True
    elif isinstance(payload, list):
        return any(_contains_forbidden_context_asset_key(item) for item in payload)
    return False
