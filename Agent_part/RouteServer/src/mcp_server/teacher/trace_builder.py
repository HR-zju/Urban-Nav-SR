"""RouteServer teacher wrapper that emits final Agentic Stage2 AgentTrace objects.

这个模块把旧 RouteServer 中已经存在的能力包成 teacher runner：

- ``IntentParser``: 自然语言意图解析。
- ``GoalDecomposer``: 多目标 itinerary 拆解。
- ``DestinationRecommendationEngine``: POI 推荐和候选排序。
- ``run_zero_shot_visual_prompt_pipeline``: FPV visual prompt teacher。

注意：这些 teacher 能力只写入 trace metadata / teacher artifact，不会作为学生
模型可见 tool 暴露。学生仍然只学习 ``map_tool``、``image_tool``、
``user_interaction_tool`` 三个元工具的调用协议。输出 trace 直接采用最终 Stage2
因果顺序：需要工具时工具交互先发生，随后插入 ``[TRAJ_*]`` readout prompt；
显式 no-tool replay 则直接从 ``<ToolDecision>`` 进入 readout prompt。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_server.agent.schemas import AgentStage2Case, AgentTrace, ToolCall, ToolResponse
from mcp_server.llm import GoalDecomposer, IntentParser
from mcp_server.teacher.agentic_stage2_trace_builder import (
    AgenticStage2TraceBuilder,
    AgenticStage2TraceBuilderConfig,
)


def _location_text(value: Any) -> str:
    """Return provider-facing ``"lon,lat"`` text from dict or string."""

    if isinstance(value, dict):
        lat = value.get("latitude") if value.get("latitude") is not None else value.get("lat")
        lon = value.get("longitude") if value.get("longitude") is not None else value.get("lon")
        if lat is not None and lon is not None:
            return f"{float(lon)},{float(lat)}"
    return str(value or "").strip()


def _lat_lon(value: Any) -> tuple[float, float] | None:
    """Return ``(lat, lon)`` for teacher-side map/recommendation calls."""

    if isinstance(value, dict):
        lat = value.get("latitude") if value.get("latitude") is not None else value.get("lat")
        lon = value.get("longitude") if value.get("longitude") is not None else value.get("lon")
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    text = str(value or "").strip()
    if "," in text:
        try:
            lon_text, lat_text = text.split(",", 1)
            return float(lat_text), float(lon_text)
        except Exception:
            return None
    return None


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _resolve_mode(request: str, observation: dict[str, Any], intent: dict[str, Any]) -> str:
    """Prefer explicit user/observation mode over heuristic intent defaults."""

    observed_mode = str(observation.get("mode") or "").strip().lower()
    if observed_mode in {"foot", "bike", "car"}:
        return observed_mode
    if _contains_any(request, ["步行", "走路", "walk", "on foot"]):
        return "foot"
    if _contains_any(request, ["骑行", "bike", "bicycle"]):
        return "bike"
    mode = str(intent.get("mode") or "").strip().lower()
    return mode if mode in {"foot", "bike", "car"} else "foot"


def _safe_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Keep teacher metadata compact enough for review JSONL."""

    if not isinstance(payload, dict):
        return {}
    if payload.get("status"):
        return {
            "status": payload.get("status"),
            "error": payload.get("error"),
        }
    return {
        "provider": payload.get("provider"),
        "parsed_intent": payload.get("parsed_intent"),
        "chosen_destination": payload.get("chosen_destination"),
        "route_summary": payload.get("route_summary"),
        "meta": payload.get("meta"),
        "message": payload.get("message"),
    }


def _is_no_tool_policy(value: Any) -> bool:
    """Return True when a case explicitly requests no-tool replay."""

    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"no_tool", "none", "skip_tool", "local_only", "false"}


@dataclass
class RouteServerTeacherConfig:
    """Configuration for ``RouteServerAgentTeacher``.

    use_llm:
        True 时调用旧 ``IntentParser.parse`` / ``GoalDecomposer.parse``，这些函数
        会在无 API 或失败时自动 fallback。False 时只走 heuristic，适合生成本地
        review 样例和单测。
    execute_tools:
        True 时会把选出的学生可见 tool call 交给 ``ToolExecutor`` 执行。
    enable_visual_prompt:
        True 且 case 中有 FPV、GPS、route polyline 时，调用 visual prompt teacher。
    tool_artifact_dir:
        真实 ``ToolExecutor`` 写入 image crop 等副产物的目录。None 表示只返回
        artifact metadata，不落盘裁剪图。
    """

    use_llm: bool = True
    execute_tools: bool = False
    enable_visual_prompt: bool = False
    enable_recommendation: bool = True
    tool_artifact_dir: str | Path | None = None


class RouteServerAgentTeacher:
    """Build one replayable AgentTrace from one AgentStage2Case.

    输入:
        ``AgentStage2Case``，可选 RouteServer ``app_context``。

    输出:
        ``AgentTrace``，其中 ``tool_call`` 和 ``tool_response`` 是独立 role；
        assistant 文本只承担 LM 监督：先输出 ``<think><AgentState>...``，最终
        输出 Phase1 CoT tags + 可选 ``<ToolResultUse>`` + ``<finish>``。
    """

    def __init__(
        self,
        *,
        app_context: Any | None = None,
        config: RouteServerTeacherConfig | None = None,
    ) -> None:
        self.app_context = app_context
        self.config = config or RouteServerTeacherConfig()
        self.intent_parser = IntentParser()
        self.goal_decomposer = GoalDecomposer()

    async def run_case(self, case: AgentStage2Case) -> AgentTrace:
        intent = await self._parse_intent(case.user_request)
        goals = await self._decompose_goals(case.user_request)
        recommendation = await self._maybe_recommend(case, intent)
        visual_prompt = await self._maybe_run_visual_prompt_teacher(case)

        no_tool_replay = self._uses_no_tool_replay(case)
        call = None
        if not no_tool_replay:
            call = self._select_student_tool_call(
                case,
                intent=intent,
                goals=goals,
                recommendation=recommendation,
                visual_prompt=visual_prompt,
            )
        teacher_internal = {
            "intent_parse": intent,
            "goal_decomposition": goals,
            "recommendation": _safe_summary(recommendation),
            "visual_prompt": _safe_summary(visual_prompt),
            "selected_student_tool_call": call.to_dict() if call is not None else None,
            "tool_policy": "no_tool" if no_tool_replay else "tool",
        }

        response = None
        extra_metadata: dict[str, Any] = {}
        if self.config.execute_tools and call is not None:
            response = await self._execute_selected_tool(call, case)
            if not response.success:
                extra_metadata["tool_execution_warning"] = response.to_dict()

        builder = AgenticStage2TraceBuilder(
            AgenticStage2TraceBuilderConfig(source="routeserver_teacher", compute_metrics=True)
        )
        return builder.build_trace(
            case,
            call=call,
            response=response,
            model_run={
                "runner": "routeserver_teacher",
                "use_llm": bool(self.config.use_llm),
                "execute_tools": bool(self.config.execute_tools),
                "enable_visual_prompt": bool(self.config.enable_visual_prompt),
            },
            intent=intent,
            tool_decision_reason=self._tool_decision_reason(call),
            recommendation=recommendation,
            visual_prompt=visual_prompt,
            teacher_internal=teacher_internal,
            extra_metadata=extra_metadata,
        )

    async def _parse_intent(self, user_request: str) -> dict[str, Any]:
        if self.config.use_llm:
            return await self.intent_parser.parse(user_request)
        parsed = self.intent_parser.heuristic_parse(user_request)
        parsed["intent_source"] = "heuristic_teacher_offline"
        return parsed

    async def _decompose_goals(self, user_request: str) -> dict[str, Any]:
        if self.config.use_llm:
            return await self.goal_decomposer.parse(user_request)
        parsed = self.goal_decomposer.heuristic_decompose(user_request)
        parsed["decomposition_source"] = "heuristic_teacher_offline"
        return parsed

    async def _maybe_recommend(self, case: AgentStage2Case, intent: dict[str, Any]) -> dict[str, Any]:
        if not self.config.enable_recommendation or self.app_context is None:
            return {"status": "skipped_no_app_context"}
        location = _lat_lon(case.observation.get("current_location") or case.observation.get("origin"))
        if location is None:
            return {"status": "skipped_missing_current_location"}
        current_latitude, current_longitude = location
        try:
            # Lazy import：recommendation 依赖 map provider / aiohttp，只有真实执行 teacher 时才加载。
            from mcp_server.recommendation import DestinationRecommendationEngine

            engine = DestinationRecommendationEngine(
                map_client=self.app_context.osm_client,
                provider=self.app_context.provider,
                intent_parser=self.intent_parser,
            )
            return await engine.recommend(
                user_instruction=case.user_request,
                current_latitude=current_latitude,
                current_longitude=current_longitude,
                max_candidates=int(intent.get("max_candidates") or 5),
                search_radius=float(intent.get("radius_m") or 3000),
                mode=str(intent.get("mode") or "foot"),
                compute_chosen_route=False,
            )
        except Exception as exc:
            return {"status": "recommendation_error", "error": str(exc)}

    async def _maybe_run_visual_prompt_teacher(self, case: AgentStage2Case) -> dict[str, Any]:
        if not self.config.enable_visual_prompt:
            return {"status": "skipped_disabled"}
        observation = case.observation
        gps = observation.get("gps") or observation.get("current_pose")
        route_polyline = observation.get("route_polyline") or observation.get("route_polyline_gps")
        coarse_waypoints = observation.get("coarse_waypoints") or observation.get("key_waypoints") or []
        image_path = None
        images = observation.get("images") or {}
        if isinstance(images, dict):
            image_path = images.get("fpv") or images.get("front")
        if not gps or not route_polyline or not image_path:
            return {"status": "skipped_missing_visual_prompt_inputs"}
        try:
            # Lazy import：visual prompt teacher 可能依赖图像和 LVLM 环境。
            from mcp_server.services.visual_prompt_service import run_zero_shot_visual_prompt_pipeline

            return await run_zero_shot_visual_prompt_pipeline(
                gps=gps,
                route_polyline=route_polyline,
                coarse_waypoints=coarse_waypoints,
                image_path=image_path,
                heading_deg=observation.get("heading_deg"),
                navigation_mode=str(observation.get("mode") or "foot"),
                enable_verifier=False,
            )
        except Exception as exc:
            return {"status": "visual_prompt_error", "error": str(exc)}

    async def _execute_selected_tool(self, call: ToolCall, case: AgentStage2Case) -> ToolResponse:
        from mcp_server.agent.tool_executor import ToolExecutor

        executor = ToolExecutor(app_context=self.app_context, artifact_dir=self.config.tool_artifact_dir)
        return await executor.execute(call, case=case)

    def _uses_no_tool_replay(self, case: AgentStage2Case) -> bool:
        """Check explicit no-tool annotations without adding random policy logic.

        Args:
            case: Stage2 case whose ``observation`` / ``ground_truth`` / ``metadata``
                may request no-tool replay.

        Returns:
            True when the teacher should emit ``should_call_tool=false`` and skip
            ``tool_call`` / ``tool_response`` roles.
        """

        for payload in (case.observation, case.ground_truth, case.metadata):
            if not isinstance(payload, dict):
                continue
            if _is_no_tool_policy(payload.get("tool_policy")):
                return True
            if _is_no_tool_policy(payload.get("stage2_tool_policy")):
                return True
            if payload.get("force_no_tool") is True:
                return True
        return False

    def _select_student_tool_call(
        self,
        case: AgentStage2Case,
        *,
        intent: dict[str, Any],
        goals: dict[str, Any],
        recommendation: dict[str, Any],
        visual_prompt: dict[str, Any],
    ) -> ToolCall:
        del goals, visual_prompt
        request = case.user_request.lower()
        observation = case.observation
        session_id = str(observation.get("route_session_id") or observation.get("session_id") or "").strip()
        location = _location_text(observation.get("current_location") or observation.get("origin"))
        destination = observation.get("new_destination") or observation.get("destination")
        via_point = observation.get("via_point")

        if self._needs_image_tool(request, observation):
            return ToolCall(
                tool="image_tool",
                arguments={
                    "op": "crop",
                    "image": str(observation.get("focus_image") or "fpv"),
                    "bbox": observation.get("focus_bbox") or observation.get("bbox") or [0.35, 0.35, 0.75, 0.85],
                    "reason": "inspect_navigation_region",
                },
            )

        if self._needs_clarification(request, observation, intent):
            return ToolCall(
                tool="user_interaction_tool",
                arguments={
                    "op": "request_clarification",
                    "question": "请确认具体目的地、途经点或导航偏好。",
                    "choices": observation.get("clarification_choices") or [],
                },
            )

        if session_id and self._needs_destination_replace(request):
            return ToolCall(
                tool="map_tool",
                arguments={
                    "op": "replace_destination",
                    "session_id": session_id,
                    "destination": _location_text(destination or intent.get("search_keywords")),
                },
            )

        if session_id and self._needs_add_via(request):
            selected_via = dict(via_point or {})
            if not selected_via:
                chosen = recommendation.get("chosen_destination") if isinstance(recommendation, dict) else {}
                selected_via = {
                    "name": chosen.get("name") or str(intent.get("search_keywords") or "via point"),
                    "location": chosen.get("location") or _location_text(destination or intent.get("search_keywords")),
                }
            return ToolCall(
                tool="map_tool",
                arguments={
                    "op": "add_via_and_replan",
                    "session_id": session_id,
                    "via_point": selected_via,
                },
            )

        origin = observation.get("origin") or observation.get("current_location")
        if origin and destination and self._needs_route_session(request, observation):
            return ToolCall(
                tool="map_tool",
                arguments={
                    "op": "create_route_session",
                    "origin": _location_text(origin),
                    "destination": _location_text(destination),
                    "mode": _resolve_mode(request, observation, intent),
                },
            )

        query = str(intent.get("search_keywords") or intent.get("keywords") or case.user_request).strip()
        if isinstance(intent.get("keywords"), list) and not str(intent.get("search_keywords") or "").strip():
            query = "|".join(str(item) for item in intent.get("keywords") if str(item).strip())
        return ToolCall(
            tool="map_tool",
            arguments={
                "op": "search_poi",
                "query": query,
                "location": location,
                "radius": int(intent.get("radius_m") or 1000),
                "limit": int(intent.get("max_candidates") or 5),
            },
        )

    @staticmethod
    def _tool_decision_reason(call: ToolCall | None) -> str:
        if call is None:
            return "available FPV, BEV, and route state are sufficient for the final local navigation decision"
        if call.tool == "map_tool":
            return "map or route state is required before producing the final navigation decision"
        if call.tool == "image_tool":
            return "a focused visual operation is required before final FPV-BEV reasoning"
        if call.tool == "user_interaction_tool":
            return "the request is underspecified and requires user-facing clarification"
        return "external context is required before final navigation"

    @staticmethod
    def _needs_image_tool(request: str, observation: dict[str, Any]) -> bool:
        if observation.get("focus_bbox") or observation.get("bbox"):
            return _contains_any(request, ["放大", "看清", "裁剪", "zoom", "crop", "inspect"])
        return _contains_any(request, ["放大图像", "zoom in", "crop image"])

    @staticmethod
    def _needs_clarification(request: str, observation: dict[str, Any], intent: dict[str, Any]) -> bool:
        if observation.get("requires_clarification"):
            return True
        if _contains_any(request, ["那个地方", "随便", "你决定", "不确定", "where should", "not sure"]):
            return True
        has_location = bool(observation.get("current_location") or observation.get("origin"))
        has_query = bool(intent.get("search_keywords") or observation.get("destination"))
        return not has_location and not has_query

    @staticmethod
    def _needs_destination_replace(request: str) -> bool:
        return _contains_any(request, ["改成", "换成", "不要去", "replace", "change destination", "instead"])

    @staticmethod
    def _needs_add_via(request: str) -> bool:
        return _contains_any(request, ["顺路", "途经", "经过", "先去", "加一个", "add via", "stop by", "on the way"])

    @staticmethod
    def _needs_route_session(request: str, observation: dict[str, Any]) -> bool:
        if observation.get("force_route_session"):
            return True
        return bool(re.search(r"(去|到|前往|导航|route|take me|go to)", request))
