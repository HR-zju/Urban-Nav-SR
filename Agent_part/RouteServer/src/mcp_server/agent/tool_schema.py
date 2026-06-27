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
    "search_poi",
    "search_category",
    "plan_route",
    "get_route",
    "create_route_session",
    "get_route_session",
    "add_via_and_replan",
    "remove_via_and_replan",
    "replace_destination",
    "replan_from_current_pose",
}
IMAGE_TOOL_OPS = {"crop", "zoom_in"}
USER_INTERACTION_TOOL_OPS = {"ask_user", "request_clarification", "confirm_route_change"}

TOOL_OPS_BY_NAME = {
    MAP_TOOL_NAME: MAP_TOOL_OPS,
    IMAGE_TOOL_NAME: IMAGE_TOOL_OPS,
    USER_INTERACTION_TOOL_NAME: USER_INTERACTION_TOOL_OPS,
}


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
                        "session_id": {"type": "string"},
                        "via_point": {"type": "object"},
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
                        "choices": {"type": "array", "items": {"type": "string"}},
                        "proposal": {"type": "object"},
                    },
                    "required": ["op"],
                    "additionalProperties": True,
                },
            },
        },
    ]


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
    op = call_obj.op
    if not op:
        errors.append("missing_op")
    elif op not in TOOL_OPS_BY_NAME[call_obj.tool]:
        errors.append(f"unsupported_op:{call_obj.tool}.{op}")

    args = call_obj.arguments
    if call_obj.tool == MAP_TOOL_NAME:
        # 这里只做轻量必填项校验；坐标范围/provider 细节交给 executor 和 map_ops。
        if op in {"geocode", "search_poi"} and not (args.get("query") or args.get("address") or args.get("keywords")):
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
        has_route_pair = bool(args.get("origin") and args.get("destination")) or all(
            args.get(key) is not None
            for key in ["from_latitude", "from_longitude", "to_latitude", "to_longitude"]
        )
        if op in {"plan_route", "create_route_session"} and not has_route_pair:
            errors.append("missing_origin_destination")
        if op == "add_via_and_replan" and not args.get("via_point"):
            errors.append("missing_via_point")
        if op == "replace_destination" and not args.get("destination"):
            errors.append("missing_destination")
        if op == "replan_from_current_pose" and not (args.get("origin") or args.get("current_pose")):
            errors.append("missing_replan_origin")
        if op in {
            "get_route_session",
            "add_via_and_replan",
            "remove_via_and_replan",
            "replace_destination",
            "replan_from_current_pose",
        } and not args.get("session_id"):
            errors.append("missing_session_id")
    elif call_obj.tool == IMAGE_TOOL_NAME:
        # image_tool 当前只允许确定性图像操作，不暴露 inspect/视觉理解工具。
        if op in {"crop", "zoom_in"} and not (args.get("bbox") and (args.get("image") or args.get("image_path"))):
            errors.append("missing_image_or_bbox")
    elif call_obj.tool == USER_INTERACTION_TOOL_NAME:
        # 离线场景不会真的等待用户，但 question 必须可审计。
        if op in {"ask_user", "request_clarification", "confirm_route_change"} and not args.get("question"):
            errors.append("missing_question")

    return not errors, errors
