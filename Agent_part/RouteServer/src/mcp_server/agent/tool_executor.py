"""Deterministic executor for student-visible Agent tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp_server.agent.contracts import ContextAsset
from mcp_server.agent.schemas import AgentStage2Case, ToolCall, ToolResponse
from mcp_server.agent.tool_schema import (
    IMAGE_TOOL_NAME,
    MAP_TOOL_NAME,
    USER_INTERACTION_TOOL_NAME,
)
from mcp_server.core import map_ops
from mcp_server.tools.registry import DEFAULT_TOOL_REGISTRY
from mcp_server.tools.specs import normalize_map_tool_arguments


def _as_location(value: Any) -> str:
    """Convert a string or ``{latitude, longitude}`` object to provider location text."""

    if isinstance(value, dict):
        lat = value.get("latitude") if value.get("latitude") is not None else value.get("lat")
        lon = value.get("longitude") if value.get("longitude") is not None else value.get("lon")
        if lat is not None and lon is not None:
            return f"{float(lon)},{float(lat)}"
    return str(value or "").strip()


def _lat_lon(value: Any) -> tuple[float, float]:
    """Return ``(latitude, longitude)`` from dict or ``"lon,lat"`` text."""

    if isinstance(value, dict):
        lat = value.get("latitude") if value.get("latitude") is not None else value.get("lat")
        lon = value.get("longitude") if value.get("longitude") is not None else value.get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    text = str(value or "").strip()
    if "," in text:
        lon_text, lat_text = text.split(",", 1)
        return float(lat_text), float(lon_text)
    raise ValueError(f"Cannot parse coordinate: {value!r}")


class ToolExecutor:
    """Execute validated tool calls against RouteServer core operations.

    输入:
        ``ToolCall``，可选 ``AgentStage2Case``，以及初始化时传入的 ``app_context``。

    输出:
        ``ToolResponse``。执行失败也返回结构化错误，不向上抛给 runner，这样离线
        trace 可以完整记录失败工具调用。
    """

    def __init__(
        self,
        *,
        app_context: Any | None = None,
        artifact_dir: str | Path | None = None,
    ) -> None:
        self.app_context = app_context
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None

    @classmethod
    def supported_capabilities(cls) -> set[tuple[str, str]]:
        """Return capabilities implemented by the live executor.

        Returns:
            Explicit ``(tool, op)`` pairs used by conformance tests.
        """

        return {
            *{(MAP_TOOL_NAME, op) for op in {
                "geocode", "reverse_geocode", "search_destination", "search_poi", "search_category",
                "plan_route", "get_route", "get_route_context", "create_route_session", "get_route_session",
                "add_via", "add_via_and_replan", "remove_via_and_replan", "replace_destination",
                "replan_from_current_pose",
            }},
            (IMAGE_TOOL_NAME, "crop"),
            (IMAGE_TOOL_NAME, "zoom_in"),
            *{(USER_INTERACTION_TOOL_NAME, op) for op in {
                "ask_user", "request_clarification", "present_options", "confirm_route_change",
                "notify_issue", "answer_query",
            }},
        }

    async def execute(self, call: ToolCall, *, case: AgentStage2Case | None = None) -> ToolResponse:
        # executor 只执行 schema 允许的工具；非法调用也转成 tool_response，保证 trace 完整。
        is_valid, errors, canonical_arguments = DEFAULT_TOOL_REGISTRY.validate_call(call)
        if not is_valid:
            return ToolResponse(
                tool=call.tool,
                success=False,
                error={"code": "invalid_tool_call", "messages": errors},
                call_id=call.call_id,
            )
        # executor 只消费 registry 验证后的 canonical 参数，避免 live/replay
        # 对同一 alias 产生不同语义。
        call = ToolCall(
            tool=call.tool,
            arguments=dict(canonical_arguments or {}),
            call_id=call.call_id,
        )
        try:
            # 三类元工具统一从这里分发，方便 validator/test 对 schema 与 executor 做一致性检查。
            if call.tool == MAP_TOOL_NAME:
                return await self._execute_map_tool(call)
            if call.tool == IMAGE_TOOL_NAME:
                return self._execute_image_tool(call, case=case)
            if call.tool == USER_INTERACTION_TOOL_NAME:
                return self._execute_user_interaction_tool(call)
        except Exception as exc:  # pragma: no cover - defensive trace preservation
            return ToolResponse(
                tool=call.tool,
                success=False,
                error={"code": exc.__class__.__name__, "message": str(exc)},
                call_id=call.call_id,
            )
        return ToolResponse(
            tool=call.tool,
            success=False,
            error={"code": "unknown_tool", "message": call.tool},
            call_id=call.call_id,
        )

    def _require_app_context(self, call: ToolCall) -> ToolResponse | None:
        if self.app_context is None:
            return ToolResponse(
                tool=call.tool,
                success=False,
                error={"code": "missing_app_context", "message": "map_tool requires RouteServer app_context"},
                call_id=call.call_id,
            )
        return None

    async def _execute_map_tool(self, call: ToolCall) -> ToolResponse:
        args = normalize_map_tool_arguments(call.arguments)
        op = str(args.get("op"))
        if op == "get_route_context":
            # get_route_context v1 只返回 text/structured ContextAsset，不依赖
            # live provider；真实 BEV/waypoint image 渲染后续通过 artifact 接口补。
            return self._execute_route_context_tool(call)

        missing_context = self._require_app_context(call)
        if missing_context:
            return missing_context

        # map_tool 是单一 student-visible 工具，op 再映射到 RouteServer core map/session API。
        if op == "geocode":
            result = await map_ops.geocode(
                app_context=self.app_context,
                address=str(args.get("address") or args.get("query") or ""),
            )
        elif op == "reverse_geocode":
            result = await map_ops.reverse_geocode(
                app_context=self.app_context,
                latitude=float(args["latitude"]),
                longitude=float(args["longitude"]),
            )
        elif op in {"search_destination", "search_poi"}:
            result = await map_ops.search_poi(
                app_context=self.app_context,
                location=_as_location(args.get("location") or args.get("origin")),
                keywords=str(args.get("keywords") or args.get("query") or ""),
                radius=int(args.get("radius") or 1000),
                sortrule=str(args.get("sortrule") or "distance"),
                limit=int(args.get("limit") or 10),
            )
        elif op == "search_category":
            bbox = args.get("bbox") or {}
            result = await map_ops.search_category(
                app_context=self.app_context,
                category=str(args.get("category") or args.get("query") or ""),
                min_latitude=float(args.get("min_latitude") or bbox.get("min_latitude")),
                min_longitude=float(args.get("min_longitude") or bbox.get("min_longitude")),
                max_latitude=float(args.get("max_latitude") or bbox.get("max_latitude")),
                max_longitude=float(args.get("max_longitude") or bbox.get("max_longitude")),
                subcategories=args.get("subcategories"),
            )
        elif op in {"plan_route", "get_route"} and not args.get("session_id"):
            # 没有 session_id 的 get_route 走一次性 route planning；有 session_id 时读取 route session。
            origin_lat, origin_lon = _lat_lon(
                args.get("origin")
                or {
                    "latitude": args.get("from_latitude"),
                    "longitude": args.get("from_longitude"),
                }
            )
            destination_lat, destination_lon = _lat_lon(
                args.get("destination")
                or {
                    "latitude": args.get("to_latitude"),
                    "longitude": args.get("to_longitude"),
                }
            )
            result = await map_ops.get_route(
                app_context=self.app_context,
                from_latitude=origin_lat,
                from_longitude=origin_lon,
                to_latitude=destination_lat,
                to_longitude=destination_lon,
                mode=str(args.get("mode") or "foot"),
                steps=bool(args.get("steps", False)),
            )
        elif op == "create_route_session":
            from mcp_server.core import route_session_ops

            # route session 类 op 会更新 app_context 内的 session store，测试时必须用 fake/memory store 隔离。
            result = await route_session_ops.create_route_session(
                app_context=self.app_context,
                origin=_as_location(args.get("origin")),
                destination=_as_location(args.get("destination")),
                via_points=args.get("via_points"),
                mode=str(args.get("mode") or "foot"),
                session_name=args.get("session_name"),
            )
        elif op in {"get_route_session", "get_route"}:
            from mcp_server.core import route_session_ops

            result = await route_session_ops.get_route_session_snapshot(
                app_context=self.app_context,
                session_id=str(args.get("session_id")),
                force_replan=bool(args.get("force_replan", False)),
            )
        elif op in {"add_via", "add_via_and_replan"}:
            from mcp_server.core import route_session_ops

            result = await route_session_ops.add_via_point(
                app_context=self.app_context,
                session_id=str(args.get("session_id")),
                new_via_point=dict(args.get("via_point") or {}),
                insert_before_index=args.get("insert_before_index"),
            )
        elif op == "remove_via_and_replan":
            from mcp_server.core import route_session_ops

            result = await route_session_ops.remove_via_point(
                app_context=self.app_context,
                session_id=str(args.get("session_id")),
                via_point_index=args.get("via_point_index"),
                via_point_name=args.get("via_point_name"),
            )
        elif op == "replace_destination":
            from mcp_server.core import route_session_ops

            result = await route_session_ops.replace_destination(
                app_context=self.app_context,
                session_id=str(args.get("session_id")),
                destination=_as_location(args.get("destination")),
            )
        elif op == "replan_from_current_pose":
            from mcp_server.core import route_session_ops

            result = await route_session_ops.replan_route_session_from_origin(
                app_context=self.app_context,
                session_id=str(args.get("session_id")),
                origin=_as_location(args.get("origin") or args.get("current_pose")),
            )
        else:
            return ToolResponse(
                tool=call.tool,
                success=False,
                error={"code": "unsupported_map_op", "message": op},
                call_id=call.call_id,
            )
        return ToolResponse(tool=call.tool, success=True, result=result, call_id=call.call_id)

    def _execute_route_context_tool(self, call: ToolCall) -> ToolResponse:
        """Return deterministic route-context assets for Stage2/Stage3 context injection.

        Args:
            call: ``map_tool.get_route_context`` call.

        Returns:
            ``ToolResponse`` containing text/structured ``context_assets``.
            Image assets are represented only when an executor provides a real
            artifact path; this fallback never fabricates images.
        """

        args = dict(call.arguments)
        asset_types = args.get("asset_types") or [
            "route_summary",
            "route_instruction_text",
            "next_maneuver",
            "next_waypoint",
        ]
        if not isinstance(asset_types, list):
            asset_types = [str(asset_types)]
        route_id = args.get("route_id") or args.get("session_id")
        response_id = call.call_id or f"route_context:{route_id or 'unknown'}"
        assets: list[dict[str, Any]] = []
        for index, asset_type in enumerate(asset_types):
            asset = _route_context_asset(
                str(asset_type),
                index=index,
                call=call,
                response_id=response_id,
            )
            if asset is not None:
                assets.append(asset.to_dict())
        result = {
            "op": "get_route_context",
            "tool_response_id": response_id,
            "route_id": route_id,
            "session_id": args.get("session_id"),
            "horizon_m": args.get("horizon_m"),
            "frame": args.get("frame") or "ego",
            "context_assets": assets,
        }
        return ToolResponse(tool=call.tool, success=True, result=result, call_id=call.call_id)

    def _execute_image_tool(self, call: ToolCall, *, case: AgentStage2Case | None) -> ToolResponse:
        args = dict(call.arguments)
        image_path = args.get("image_path")
        if not image_path and case is not None:
            images = case.observation.get("images") or {}
            if isinstance(images, dict):
                image_path = images.get(str(args.get("image") or "fpv"))
        if not image_path:
            return ToolResponse(
                tool=call.tool,
                success=False,
                error={"code": "missing_image_path", "message": "image_tool requires image_path or case image key"},
                call_id=call.call_id,
            )
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - depends on optional pillow
            return ToolResponse(
                tool=call.tool,
                success=False,
                error={"code": "missing_pillow", "message": str(exc)},
                call_id=call.call_id,
            )

        source = Path(str(image_path))
        bbox = [float(value) for value in args.get("bbox") or []]
        with Image.open(source) as image:
            width, height = image.size
            if all(0.0 <= value <= 1.0 for value in bbox):
                # bbox 支持 0-1 归一化坐标，按原图尺寸转成像素 crop。
                x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
            else:
                x1, y1, x2, y2 = bbox
            crop = image.crop((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
            if self.artifact_dir is None:
                # 无 artifact_dir 时只返回 metadata，不在磁盘产生文件，适合轻量 smoke。
                artifact = {
                    "type": "image_crop",
                    "source_path": str(source),
                    "bbox": [x1, y1, x2, y2],
                    "size": list(crop.size),
                }
            else:
                # 有 artifact_dir 时保存裁剪图，便于人工 review tool artifact。
                self.artifact_dir.mkdir(parents=True, exist_ok=True)
                name = f"{source.stem}_{call.call_id or 'crop'}.jpg"
                target = self.artifact_dir / name
                crop.save(target)
                artifact = {
                    "type": "image_crop",
                    "source_path": str(source),
                    "path": str(target),
                    "bbox": [x1, y1, x2, y2],
                    "size": list(crop.size),
                }
        return ToolResponse(
            tool=call.tool,
            success=True,
            result={"artifact": artifact},
            artifacts=[artifact],
            call_id=call.call_id,
        )

    def _execute_user_interaction_tool(self, call: ToolCall) -> ToolResponse:
        args = dict(call.arguments)
        op = str(args.get("op") or "")
        requires_user = bool(args.get("requires_user_response", op not in {"answer_query", "notify_issue"}))
        interaction_mode = str(args.get("interaction_mode") or ("blocking" if requires_user else "non_blocking"))
        motion_policy = str(
            args.get("motion_policy_while_waiting")
            or ("stop" if requires_user else "continue_previous_safe_policy")
        )
        message = args.get("message") or args.get("question") or ""
        options = _normalize_user_options(args.get("options") or args.get("choices") or [])
        choices = args.get("choices") or [option.get("label") for option in options if option.get("label")]
        interaction_id = args.get("interaction_id") or call.call_id or f"ui:{op or 'interaction'}"
        pending = None
        if requires_user:
            pending = {
                "interaction_id": interaction_id,
                "op": op,
                "message": message,
                "options": options,
                "recommended_option_id": args.get("recommended_option_id"),
                "motion_policy_while_waiting": motion_policy,
            }
        # 离线 evaluator 不真正等待用户，只记录“需要用户确认/澄清”的结构化动作。
        result = {
            "await_user": requires_user,
            "interaction_id": interaction_id,
            "op": op,
            "message": message,
            "question": args.get("question") or message,
            "choices": choices,
            "options": options,
            "recommended_option_id": args.get("recommended_option_id"),
            "requires_user_response": requires_user,
            "interaction_mode": interaction_mode,
            "motion_policy_while_waiting": motion_policy,
            "runtime_state": "WAITING_FOR_USER" if requires_user else "RUNNING",
            "dialogue_state": {"pending_user_interaction": pending} if pending is not None else {},
            "evidence_refs": args.get("evidence_refs") or [],
            "proposal": args.get("proposal") or {},
        }
        return ToolResponse(tool=call.tool, success=True, result=json.loads(json.dumps(result)), call_id=call.call_id)


def _normalize_user_options(options: Any) -> list[dict[str, Any]]:
    """Normalize user-facing choices into auditable option objects.

    Args:
        options: Either ``[{"option_id": ..., "label": ...}]`` or a legacy
            string list from ``choices``.

    Returns:
        List of option dictionaries with stable ``option_id`` values.
    """

    if not isinstance(options, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(options):
        if isinstance(item, dict):
            option = dict(item)
            option.setdefault("option_id", option.get("id") or option.get("candidate_id") or f"option_{index}")
            if option.get("label") is None and option.get("text") is not None:
                option["label"] = option.get("text")
            normalized.append(option)
        else:
            # 兼容早期 choices=["A","B"]，executor 输出统一 option object。
            normalized.append({"option_id": f"option_{index}", "label": str(item)})
    return normalized


def _route_context_asset(
    asset_type: str,
    *,
    index: int,
    call: ToolCall,
    response_id: str,
) -> ContextAsset | None:
    """Build one deterministic route-context asset.

    Args:
        asset_type: Requested asset type.
        index: Position in the requested asset list.
        call: Source tool call.
        response_id: Tool response id used for traceability.

    Returns:
        ``ContextAsset`` for supported text/structured assets.  Unsupported
        image assets return an explicit not-implemented structured asset
        instead of a fabricated image path.
    """

    args = dict(call.arguments)
    asset_id = f"{response_id}:asset:{index}:{asset_type}"
    route_id = args.get("route_id") or args.get("session_id")
    frame = str(args.get("frame") or "ego")
    source_tool = call.tool
    if asset_type == "route_summary":
        return ContextAsset(
            asset_id=asset_id,
            asset_type=asset_type,
            source_tool=source_tool,
            source_tool_response_id=response_id,
            modality="structured",
            payload={
                "route_id": route_id,
                "distance_m": args.get("distance_m"),
                "eta_sec": args.get("eta_sec"),
                "status": "available",
            },
        )
    if asset_type == "route_instruction_text":
        maneuver = args.get("maneuver") or args.get("next_maneuver") or "continue"
        distance = args.get("distance_m") or args.get("next_maneuver_distance_m") or args.get("horizon_m")
        text = args.get("instruction_text") or _route_instruction_text(maneuver=str(maneuver), distance=distance)
        return ContextAsset(
            asset_id=asset_id,
            asset_type=asset_type,
            source_tool=source_tool,
            source_tool_response_id=response_id,
            modality="text",
            payload={"text": text},
        )
    if asset_type == "next_maneuver":
        return ContextAsset(
            asset_id=asset_id,
            asset_type=asset_type,
            source_tool=source_tool,
            source_tool_response_id=response_id,
            modality="structured",
            frame=frame,
            payload={
                "type": args.get("maneuver") or args.get("next_maneuver") or "continue",
                "distance_m": args.get("next_maneuver_distance_m") or args.get("distance_m"),
            },
        )
    if asset_type == "next_waypoint":
        waypoint = args.get("waypoint_ego") or args.get("next_waypoint") or [0.0, float(args.get("horizon_m") or 10.0), 0.0]
        return ContextAsset(
            asset_id=asset_id,
            asset_type=asset_type,
            source_tool=source_tool,
            source_tool_response_id=response_id,
            modality="structured",
            frame=frame,
            payload={
                "coordinate_convention": "x_right_y_front_theta_yaw",
                "waypoint_ego": waypoint,
                "maneuver": args.get("maneuver") or args.get("next_maneuver") or "continue",
            },
        )
    if asset_type in {"route_bev_view", "waypoint_image", "entrance_image"}:
        return ContextAsset(
            asset_id=asset_id,
            asset_type=asset_type,
            source_tool=source_tool,
            source_tool_response_id=response_id,
            modality="image",
            frame=frame,
            payload={
                "image_ref": args.get("image_ref"),
                "render_status": "not_implemented" if not args.get("image_ref") else "provided_by_executor_args",
            },
        )
    return None


def _route_instruction_text(*, maneuver: str, distance: Any) -> str:
    """Return a compact route instruction string for context assets."""

    if distance is not None:
        return f"Continue for about {distance} meters, then {maneuver}."
    return f"Continue and prepare to {maneuver}."
