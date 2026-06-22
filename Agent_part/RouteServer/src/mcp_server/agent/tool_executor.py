"""Deterministic executor for student-visible Agent tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp_server.agent.schemas import AgentStage2Case, ToolCall, ToolResponse
from mcp_server.agent.tool_schema import (
    IMAGE_TOOL_NAME,
    MAP_TOOL_NAME,
    USER_INTERACTION_TOOL_NAME,
    validate_tool_call,
)
from mcp_server.core import map_ops


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

    async def execute(self, call: ToolCall, *, case: AgentStage2Case | None = None) -> ToolResponse:
        is_valid, errors = validate_tool_call(call)
        if not is_valid:
            return ToolResponse(
                tool=call.tool,
                success=False,
                error={"code": "invalid_tool_call", "messages": errors},
                call_id=call.call_id,
            )
        try:
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
        missing_context = self._require_app_context(call)
        if missing_context:
            return missing_context

        args = dict(call.arguments)
        op = str(args.get("op"))
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
        elif op == "search_poi":
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
        elif op == "add_via_and_replan":
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
                x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
            else:
                x1, y1, x2, y2 = bbox
            crop = image.crop((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
            if self.artifact_dir is None:
                artifact = {
                    "type": "image_crop",
                    "source_path": str(source),
                    "bbox": [x1, y1, x2, y2],
                    "size": list(crop.size),
                }
            else:
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
        # 离线 evaluator 不真正等待用户，只记录“需要用户确认/澄清”的结构化动作。
        result = {
            "await_user": True,
            "op": args.get("op"),
            "question": args.get("question"),
            "choices": args.get("choices") or [],
            "proposal": args.get("proposal") or {},
        }
        return ToolResponse(tool=call.tool, success=True, result=json.loads(json.dumps(result)), call_id=call.call_id)
