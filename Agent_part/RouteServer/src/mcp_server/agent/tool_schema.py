"""Student-visible tool schema for SideWalk Agent data.

本文件只定义学生模型能看到的元工具。旧 RouteServer 内部的 LLM/LVLM teacher
能力不能作为 tool 暴露给学生，否则训练时会把教师推理能力错误建模为环境工具。
"""

from __future__ import annotations

from typing import Any

from mcp_server.agent.schemas import ToolCall


MAP_TOOL_NAME = "map_tool"
IMAGE_TOOL_NAME = "image_tool"
USER_INTERACTION_TOOL_NAME = "user_interaction_tool"

MAP_TOOL_OPS = {
    "geocode",
    "reverse_geocode",
    "search_destination",
    "search_poi",
    "search_category",
    "plan_route",
    "get_route",
    "get_route_context",
    "create_route_session",
    "get_route_session",
    "add_via",
    "add_via_and_replan",
    "remove_via_and_replan",
    "replace_destination",
    "replan_from_current_pose",
}
IMAGE_TOOL_OPS = {"crop", "zoom_in"}
USER_INTERACTION_TOOL_OPS = {
    "ask_user",
    "request_clarification",
    "present_options",
    "confirm_route_change",
    "notify_issue",
    "answer_query",
}

TOOL_OPS_BY_NAME = {
    MAP_TOOL_NAME: MAP_TOOL_OPS,
    IMAGE_TOOL_NAME: IMAGE_TOOL_OPS,
    USER_INTERACTION_TOOL_NAME: USER_INTERACTION_TOOL_OPS,
}

_DESTINATION_ID_ALIASES = (
    "destination_id",
    "candidate_id",
    "chosen_destination_id",
    "selected_destination_id",
    "target_destination_id",
    "target_id",
    "new_destination_id",
    "replacement_destination_id",
    "via_destination_id",
    "via_candidate_id",
    "via_id",
)

_ROUTE_ID_ALIASES = ("route_id", "chosen_route_id", "selected_route_id", "last_route_id")
_SESSION_ID_ALIASES = ("session_id", "route_session_id")
_STOP_INDEX_ALIASES = ("stop_index", "goal_index", "goal_order", "leg_index", "stop_number")
_MODE_ALIASES = ("mode", "travel_mode", "route_mode")
_QUERY_ALIASES = ("query", "keywords", "keyword", "intent", "category", "target_type", "target_types")


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return the minimal student-visible tool registry.

    输出:
        OpenAI/tool-call 风格 JSON schema。后续 ms-swift exporter 会把它写入
        JSONL 的 ``tools`` 字段，训练模型学习选择元工具和 op，而不是记忆大量
        细粒度函数名。
    """

    return [
        {
            "type": "function",
            "function": {
                "name": MAP_TOOL_NAME,
                "description": "Query map data, plan or update a route session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": sorted(MAP_TOOL_OPS),
                            "description": "Map operation name.",
                        },
                        "query": {"type": "string"},
                        "location": {"type": ["string", "object"]},
                        "origin": {"type": ["string", "object"]},
                        "destination": {"type": ["string", "object"]},
                        "destination_id": {"type": "string"},
                        "candidate_id": {"type": "string"},
                        "new_destination_id": {"type": "string"},
                        "via_destination_id": {"type": "string"},
                        "via_id": {"type": "string"},
                        "route_id": {"type": "string"},
                        "session_id": {"type": "string"},
                        "via_point": {"type": "object"},
                        "current_pose": {"type": "object"},
                        "asset_types": {"type": "array", "items": {"type": "string"}},
                        "horizon_m": {"type": "number"},
                        "frame": {"type": "string"},
                        "current_lat": {"type": "number"},
                        "current_lng": {"type": "number"},
                        "origin_lat": {"type": "number"},
                        "origin_lng": {"type": "number"},
                        "stop_index": {"type": "integer"},
                        "goal_index": {"type": "integer"},
                        "mode": {"type": "string", "enum": ["foot", "bike", "car"]},
                    },
                    "required": ["op"],
                    "additionalProperties": True,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": IMAGE_TOOL_NAME,
                "description": "Operate on FPV/BEV images, such as crop or zoom-in.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": sorted(IMAGE_TOOL_OPS)},
                        "image": {"type": "string", "description": "Image key such as fpv or bev."},
                        "image_path": {"type": "string"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                    "required": ["op"],
                    "additionalProperties": True,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": USER_INTERACTION_TOOL_NAME,
                "description": "Ask or confirm information with the human user.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": sorted(USER_INTERACTION_TOOL_OPS)},
                        "question": {"type": "string"},
                        "message": {"type": "string"},
                        "choices": {"type": "array", "items": {"type": "string"}},
                        "options": {"type": "array", "items": {"type": "object"}},
                        "recommended_option_id": {"type": "string"},
                        "requires_user_response": {"type": "boolean"},
                        "interaction_mode": {"type": "string", "enum": ["blocking", "non_blocking"]},
                        "motion_policy_while_waiting": {
                            "type": "string",
                            "enum": [
                                "stop",
                                "slow_down",
                                "continue_previous_safe_policy",
                                "handover_to_safety_controller",
                            ],
                        },
                        "evidence_refs": {"type": "array", "items": {"type": "string"}},
                        "proposal": {"type": "object"},
                    },
                    "required": ["op"],
                    "additionalProperties": True,
                },
            },
        },
    ]


def normalize_map_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Normalize model-emitted map-tool aliases into canonical fields.

    The normalization is deliberately limited to schema-level aliases.  It does
    not infer a destination, route, or operation that the model did not provide.
    It only makes equivalent field spellings executable across parser,
    validator, and replay executor.
    """

    args = dict(arguments or {})
    op = _string_or_none(args.get("op") or args.get("operation") or args.get("action"))
    if op:
        args["op"] = op
    _set_first_string(args, "query", _QUERY_ALIASES)
    _set_first_string(args, "destination_id", _DESTINATION_ID_ALIASES)
    _set_first_string(args, "candidate_id", ("candidate_id", "poi_id", "place_id"))
    _set_first_string(args, "route_id", _ROUTE_ID_ALIASES)
    _set_first_string(args, "session_id", _SESSION_ID_ALIASES)
    _set_first_string(args, "mode", _MODE_ALIASES)
    _set_first_int(args, "stop_index", _STOP_INDEX_ALIASES)
    _set_point_from_scalar_aliases(
        args,
        "origin",
        (
            ("origin_lat", "origin_lng"),
            ("origin_latitude", "origin_longitude"),
            ("from_lat", "from_lng"),
            ("from_latitude", "from_longitude"),
            ("start_lat", "start_lng"),
            ("start_latitude", "start_longitude"),
        ),
    )
    _set_point_from_scalar_aliases(
        args,
        "destination",
        (
            ("destination_lat", "destination_lng"),
            ("destination_latitude", "destination_longitude"),
            ("to_lat", "to_lng"),
            ("to_latitude", "to_longitude"),
            ("target_lat", "target_lng"),
            ("target_latitude", "target_longitude"),
        ),
    )
    _set_point_from_scalar_aliases(
        args,
        "current_pose",
        (
            ("current_lat", "current_lng"),
            ("current_latitude", "current_longitude"),
            ("current_origin_lat", "current_origin_lng"),
            ("current_origin_latitude", "current_origin_longitude"),
        ),
    )
    _set_point_from_scalar_aliases(
        args,
        "via_point",
        (
            ("via_lat", "via_lng"),
            ("via_latitude", "via_longitude"),
            ("waypoint_lat", "waypoint_lng"),
            ("waypoint_latitude", "waypoint_longitude"),
        ),
    )
    return args


def expected_ops_for_tool(tool: str) -> set[str]:
    # metrics/validator 用这个函数判断模型是否调用了 schema 内允许的 op。
    return set(TOOL_OPS_BY_NAME.get(tool, set()))


def validate_tool_call(call: ToolCall | dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a parsed tool call without executing it.

    输入:
        ``ToolCall`` 或 dict。

    输出:
        ``(is_valid, errors)``。这里做轻量 schema 校验，保证 evaluator 能区分
        “解析成功但参数不可执行”和“完全无法解析”。
    """

    call_obj = call if isinstance(call, ToolCall) else ToolCall.from_dict(call)
    errors: list[str] = []
    if call_obj.tool not in TOOL_OPS_BY_NAME:
        # 未注册工具直接拒绝，避免模型调用 teacher-only 或历史内部工具。
        errors.append(f"unknown_tool:{call_obj.tool}")
        return False, errors

    args = call_obj.arguments
    if call_obj.tool == MAP_TOOL_NAME:
        args = normalize_map_tool_arguments(args)
    op = str(args.get("op") or call_obj.op or "").strip() or None
    if not op:
        errors.append("missing_op")
    elif op not in TOOL_OPS_BY_NAME[call_obj.tool]:
        errors.append(f"unsupported_op:{call_obj.tool}.{op}")

    if call_obj.tool == MAP_TOOL_NAME:
        # 这里只做轻量必填项校验；坐标范围/provider 细节交给 executor 和 map_ops。
        if op in {"geocode", "search_destination", "search_poi"} and not (
            args.get("query")
            or args.get("address")
            or args.get("keywords")
            or args.get("keyword")
            or args.get("intent")
            or args.get("category")
        ):
            errors.append("missing_query")
        if op in {"reverse_geocode"} and not (args.get("latitude") is not None and args.get("longitude") is not None):
            errors.append("missing_latitude_longitude")
        if op == "search_category":
            bbox = args.get("bbox") or {}
            has_bbox = all(
                args.get(key) is not None or bbox.get(key) is not None
                for key in ["min_latitude", "min_longitude", "max_latitude", "max_longitude"]
            )
            if not has_bbox:
                errors.append("missing_bbox")
        has_route_pair = bool(args.get("origin") and args.get("destination"))
        has_destination_ref = bool(args.get("destination_id") or args.get("candidate_id"))
        has_route_ref = bool(args.get("route_id"))
        if op == "plan_route" and not (has_route_pair or has_destination_ref or has_route_ref):
            errors.append("missing_origin_destination")
        if op == "get_route_context" and not (args.get("route_id") or args.get("session_id")):
            errors.append("missing_route_or_session_id")
        if op in {"add_via", "add_via_and_replan"} and not (args.get("via_point") or has_destination_ref):
            errors.append("missing_via_point")
        if op == "replace_destination" and not (args.get("destination") or has_destination_ref):
            errors.append("missing_destination")
        if op == "replan_from_current_pose" and not (args.get("origin") or args.get("current_pose")):
            errors.append("missing_replan_origin")
        if op in {
            "add_via",
            "add_via_and_replan",
            "remove_via_and_replan",
            "replace_destination",
            "replan_from_current_pose",
        } and not args.get("session_id") and not (
            has_destination_ref or has_route_ref or args.get("destination") or args.get("via_point")
        ):
            errors.append("missing_session_id")
    elif call_obj.tool == IMAGE_TOOL_NAME:
        # image_tool 当前只允许确定性图像操作，不暴露 inspect/视觉理解工具。
        if op in {"crop", "zoom_in"} and not (args.get("bbox") and (args.get("image") or args.get("image_path"))):
            errors.append("missing_image_or_bbox")
    elif call_obj.tool == USER_INTERACTION_TOOL_NAME:
        # 离线场景不会真的等待用户，但面向用户的 message/question 必须可审计。
        message = args.get("message") or args.get("question")
        if op in USER_INTERACTION_TOOL_OPS and not message:
            errors.append("missing_message")
        if op == "present_options":
            options = args.get("options") or args.get("choices") or []
            if not isinstance(options, list) or not options:
                errors.append("missing_options")
            recommended = args.get("recommended_option_id")
            if recommended is not None and isinstance(options, list):
                option_ids = {
                    str(item.get("option_id"))
                    for item in options
                    if isinstance(item, dict) and item.get("option_id") is not None
                }
                if option_ids and str(recommended) not in option_ids:
                    errors.append("recommended_option_not_in_options")
        requires_user = bool(args.get("requires_user_response", op not in {"answer_query", "notify_issue"}))
        if requires_user:
            mode = str(args.get("interaction_mode") or "")
            if mode and mode != "blocking":
                errors.append("requires_user_response_must_be_blocking")
            # 旧 Stage2 样本没有显式 motion policy；executor 会补 stop 作为
            # blocking interaction 的默认安全策略，schema 层保持向后兼容。

    return not errors, errors


def _set_first_string(args: dict[str, Any], canonical: str, aliases: tuple[str, ...]) -> None:
    if _string_or_none(args.get(canonical)):
        return
    for alias in aliases:
        value = _string_or_none(args.get(alias))
        if value:
            args[canonical] = value
            return


def _set_first_int(args: dict[str, Any], canonical: str, aliases: tuple[str, ...]) -> None:
    if _int_or_none(args.get(canonical)) is not None:
        args[canonical] = _int_or_none(args.get(canonical))
        return
    for alias in aliases:
        value = _int_or_none(args.get(alias))
        if value is not None:
            args[canonical] = value
            return


def _set_point_from_scalar_aliases(
    args: dict[str, Any],
    canonical: str,
    aliases: tuple[tuple[str, str], ...],
) -> None:
    if isinstance(args.get(canonical), dict):
        return
    for lat_key, lng_key in aliases:
        lat = _float_or_none(args.get(lat_key))
        lng = _float_or_none(args.get(lng_key))
        if lat is not None and lng is not None:
            args[canonical] = {"lat": lat, "lng": lng}
            return


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        items = [_string_or_none(item) for item in value]
        items = [item for item in items if item]
        return ", ".join(items) if items else None
    text = str(value).strip()
    return text or None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None
