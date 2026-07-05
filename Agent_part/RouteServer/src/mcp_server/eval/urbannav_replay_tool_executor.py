"""Replay tool executor for UrbanNav Module 1 tool-agent runs.

This module exposes frozen UrbanNav POI and route caches through the common
``map_tool`` protocol.  It does not call a live map provider and it does not
select destinations on behalf of the model.  The model must explicitly request
search, route planning, route updates, and then reference returned ids in its
``<finish>`` block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcp_server.agent.schemas import ToolCall, ToolResponse
from mcp_server.agent.tool_schema import MAP_TOOL_NAME, normalize_map_tool_arguments
from mcp_server.core.replay_map_backend import ReplayPoiResult, ReplayRouteResult
from mcp_server.eval.urbannav_runners import (
    ReplayUrbanNavBackendAdapter,
    _append_route,
    _destination_from_mapping,
    _destination_with_route_distance,
    _effective_origin,
    _route_update_metadata,
    _route_update_requires_final,
)
from mcp_server.eval.urbannav_schema import (
    UrbanNavCase,
    UrbanNavDestination,
    UrbanNavPoint,
    UrbanNavPrediction,
    destination_from_any,
    point_from_any,
)


@dataclass
class StoredUrbanNavRoute:
    """Route record stored after a model-requested route tool call.

    Args:
        route_id: Stable id returned to the model.
        destination_id: Candidate id used for this route.
        destination: Destination routed to.
        route: Frozen replay route record.
        leg_index: One-based route leg index in the model-built route state.
        route_phase: Optional route phase, such as ``"via_to_final"``.
    """

    route_id: str
    destination_id: str | None
    destination: UrbanNavDestination
    route: ReplayRouteResult
    leg_index: int
    route_phase: str | None = None


@dataclass
class UrbanNavReplayToolState:
    """Case-local replay tool state.

    Args:
        origin: Starting point for the active route session.
        current_origin: Origin used by the next route operation.
        route_session_id: Case-local session id returned by session tools.
        candidate_by_id: Candidate ids visible to the model.
        latest_candidate_aliases: Short ``poi_1`` aliases for the last search.
        search_results: Stored search responses by search id.
        route_by_id: Stored route records by route id.
        selected_sequence: Ordered committed destinations.
        route_legs: Ordered committed route legs.
        full_route: Concatenated route polyline.
        total_distance_m: Total committed route distance.
        total_duration_s: Total committed route duration when available.
        last_search_id: Last search id returned to the model.
        last_route_id: Last route id returned to the model.
    """

    origin: UrbanNavPoint
    current_origin: UrbanNavPoint
    route_session_id: str
    candidate_by_id: dict[str, UrbanNavDestination] = field(default_factory=dict)
    latest_candidate_aliases: dict[str, str] = field(default_factory=dict)
    search_results: dict[str, ReplayPoiResult] = field(default_factory=dict)
    route_by_id: dict[str, StoredUrbanNavRoute] = field(default_factory=dict)
    selected_sequence: list[UrbanNavDestination] = field(default_factory=list)
    route_legs: list[dict[str, Any]] = field(default_factory=list)
    full_route: list[UrbanNavPoint] = field(default_factory=list)
    total_distance_m: float = 0.0
    total_duration_s: float | None = None
    last_search_id: str | None = None
    last_route_id: str | None = None


class ReplayUrbanNavToolExecutor:
    """Execute UrbanNav map tool calls against frozen replay caches.

    Args:
        backend: Replay backend adapter loaded with ``poi_cache.jsonl`` and
            ``route_cache.jsonl``.
        candidate_limit: Maximum candidates returned by one search call.
        mode: Route mode recorded in route summaries.

    The executor is intentionally case-local.  A new state is created for every
    benchmark row, so model calls cannot leak candidate ids across cases.
    """

    def __init__(
        self,
        *,
        backend: ReplayUrbanNavBackendAdapter,
        candidate_limit: int = 10,
        mode: str = "foot",
    ) -> None:
        """Initialize the replay executor.

        Args:
            backend: Replay backend adapter.
            candidate_limit: Maximum returned candidates per search.
            mode: Route mode string.

        Returns:
            None.
        """

        self.backend = backend
        self.candidate_limit = max(1, int(candidate_limit))
        self.mode = str(mode or "foot")

    def new_state(self, case: UrbanNavCase) -> UrbanNavReplayToolState | None:
        """Create a case-local state.

        Args:
            case: Frozen benchmark case.

        Returns:
            New tool state, or ``None`` when the case has no usable origin.
        """

        origin = _effective_origin(case)
        if origin is None:
            return None
        return UrbanNavReplayToolState(
            origin=origin,
            current_origin=origin,
            route_session_id=f"replay-session:{case.case_id}",
            full_route=[origin],
        )

    def execute(self, *, case: UrbanNavCase, state: UrbanNavReplayToolState, call: ToolCall) -> ToolResponse:
        """Execute one model-emitted tool call.

        Args:
            case: Current benchmark case.
            state: Mutable case-local tool state.
            call: Parsed and schema-validated tool call.

        Returns:
            ``ToolResponse`` with ``success=false`` on execution failure.  The
            caller should append this response to the dialogue; this method
            never fabricates a prediction.
        """

        if call.tool != MAP_TOOL_NAME:
            return self._error(call, "unsupported_tool", {"tool": call.tool})
        op = str(normalize_map_tool_arguments(call.arguments).get("op") or call.op or "").strip()
        try:
            if op in {"search_destination", "search_poi"}:
                return self._search_destination(case=case, state=state, call=call)
            if op in {"plan_route", "create_route_session"}:
                return self._plan_route(case=case, state=state, call=call, op=op)
            if op == "get_route_session":
                return self._get_route_session(case=case, state=state, call=call)
            if op in {"add_via", "add_via_and_replan"}:
                return self._add_via(case=case, state=state, call=call)
            if op == "replace_destination":
                return self._replace_destination(case=case, state=state, call=call)
            if op == "replan_from_current_pose":
                return self._replan_from_current_pose(case=case, state=state, call=call)
        except Exception as exc:
            return self._error(call, "tool_execution_exception", {"detail": str(exc), "op": op})
        return self._error(call, "unsupported_op", {"op": op})

    def resolve_destination(self, state: UrbanNavReplayToolState, destination_id: str | None) -> UrbanNavDestination | None:
        """Resolve a model-visible destination id.

        Args:
            state: Case-local tool state.
            destination_id: Id emitted by the model.

        Returns:
            Matching destination or ``None``.
        """

        key = str(destination_id or "").strip()
        if not key:
            return None
        resolved = state.latest_candidate_aliases.get(key, key)
        return state.candidate_by_id.get(resolved)

    def resolve_route(self, state: UrbanNavReplayToolState, route_id: str | None) -> StoredUrbanNavRoute | None:
        """Resolve a model-visible route id.

        Args:
            state: Case-local tool state.
            route_id: Id emitted by the model.

        Returns:
            Stored route or ``None``.
        """

        key = str(route_id or state.last_route_id or "").strip()
        if not key:
            return None
        return state.route_by_id.get(key)

    def tool_metrics(self, state: UrbanNavReplayToolState) -> dict[str, Any]:
        """Summarize tool state for trace diagnostics.

        Args:
            state: Case-local tool state.

        Returns:
            Tool-use diagnostic metrics.
        """

        return {
            "search_count": len(state.search_results),
            "route_count": len(state.route_by_id),
            "committed_stop_count": len(state.selected_sequence),
            "has_route_session": bool(state.route_session_id),
            "last_search_id": state.last_search_id,
            "last_route_id": state.last_route_id,
        }

    def _search_destination(
        self,
        *,
        case: UrbanNavCase,
        state: UrbanNavReplayToolState,
        call: ToolCall,
    ) -> ToolResponse:
        args = _normalized_args(call)
        stop_index = _stop_index_for_case(case=case, state=state, args=args)
        poi_record = self.backend.search_destinations(case=case, stop_index=stop_index)
        search_id = f"search_{len(state.search_results) + 1}"
        state.search_results[search_id] = poi_record
        state.last_search_id = search_id
        state.latest_candidate_aliases = {}

        candidates_payload: list[dict[str, Any]] = []
        for index, candidate in enumerate(poi_record.candidates[: self.candidate_limit], start=1):
            candidate_id = f"{search_id}_poi_{index}"
            alias = f"poi_{index}"
            payload = _candidate_with_rank(candidate, index).to_dict()
            state.candidate_by_id[candidate_id] = _candidate_with_rank(candidate, index)
            state.candidate_by_id[alias] = _candidate_with_rank(candidate, index)
            state.latest_candidate_aliases[alias] = candidate_id
            candidates_payload.append({"candidate_id": candidate_id, "alias": alias, **payload})

        return ToolResponse(
            tool=call.tool,
            success=True,
            result={
                "op": "search_destination",
                "search_id": search_id,
                "cache_key": poi_record.cache_key,
                "provider": poi_record.provider,
                "query": _search_query_from_args(args),
                "stop_index": stop_index,
                "candidate_count": len(candidates_payload),
                "candidates": candidates_payload,
            },
            call_id=call.call_id,
        )

    def _plan_route(
        self,
        *,
        case: UrbanNavCase,
        state: UrbanNavReplayToolState,
        call: ToolCall,
        op: str,
    ) -> ToolResponse:
        args = _normalized_args(call)
        if op == "create_route_session" and args.get("route_id"):
            route_id = str(args.get("route_id") or "").strip()
            if route_id not in state.route_by_id:
                return self._error(call, "unknown_route", {"route_id": route_id})
            result = self._route_response_payload(state=state, route_id=route_id)
            result["op"] = op
            result["session_id"] = state.route_session_id
            return ToolResponse(tool=call.tool, success=True, result=result, call_id=call.call_id)
        if op == "create_route_session" and not args.get("destination_id") and not args.get("candidate_id"):
            if state.last_route_id:
                result = self._route_response_payload(state=state, route_id=state.last_route_id)
                result["op"] = op
                result["session_id"] = state.route_session_id
                return ToolResponse(tool=call.tool, success=True, result=result, call_id=call.call_id)
        destination_id, destination = self._destination_from_args(state=state, args=args)
        if destination is None:
            return self._error(call, "unknown_destination", {"destination_id": destination_id})
        origin = _point_from_args(args, "origin") or state.current_origin
        stop_index = _stop_index_for_case(case=case, state=state, args=args)
        route_record = self.backend.plan_route(
            case=case,
            destination=destination,
            origin=origin,
            stop_index=stop_index,
        )
        if not _is_routeable(route_record):
            return self._error(
                call,
                "route_not_found",
                {
                    "destination_id": destination_id,
                    "cache_key": route_record.cache_key,
                    "provider_error": route_record.response.get("error") if isinstance(route_record.response, dict) else None,
                },
            )
        route_id = self._store_committed_route(
            state=state,
            destination_id=destination_id,
            destination=destination,
            route_record=route_record,
            origin=origin,
            route_phase=None,
        )
        result = self._route_response_payload(state=state, route_id=route_id)
        result["op"] = op
        if op == "create_route_session":
            result["session_id"] = state.route_session_id
        return ToolResponse(tool=call.tool, success=True, result=result, call_id=call.call_id)

    def _get_route_session(
        self,
        *,
        case: UrbanNavCase,
        state: UrbanNavReplayToolState,
        call: ToolCall,
    ) -> ToolResponse:
        update = _route_update_metadata(case)
        return ToolResponse(
            tool=call.tool,
            success=True,
            result={
                "op": "get_route_session",
                "session_id": state.route_session_id,
                "current_origin": state.current_origin.to_dict(),
                "route_update": update,
                "committed_stop_count": len(state.selected_sequence),
                "last_route_id": state.last_route_id,
            },
            call_id=call.call_id,
        )

    def _add_via(self, *, case: UrbanNavCase, state: UrbanNavReplayToolState, call: ToolCall) -> ToolResponse:
        args = _normalized_args(call)
        destination_id, via_destination = self._destination_from_args(state=state, args=args, destination_key="via_point")
        if via_destination is None:
            return self._error(call, "unknown_via_destination", {"destination_id": destination_id})
        first_leg = self.backend.plan_route(case=case, destination=via_destination, origin=state.current_origin)
        if not _is_routeable(first_leg):
            return self._error(call, "route_not_found", {"destination_id": destination_id, "cache_key": first_leg.cache_key})
        route_id = self._store_committed_route(
            state=state,
            destination_id=destination_id,
            destination=via_destination,
            route_record=first_leg,
            origin=state.current_origin,
            route_phase="update",
        )

        update = _route_update_metadata(case)
        extension: dict[str, Any] | None = None
        if _route_update_requires_final(update):
            final_destination = _destination_from_mapping(update.get("final_destination"))
            if final_destination is not None and via_destination.point is not None:
                final_route = self.backend.plan_route(
                    case=case,
                    destination=final_destination,
                    origin=via_destination.point,
                    route_phase="via_to_final",
                )
                if _is_routeable(final_route):
                    final_route_id = self._store_committed_route(
                        state=state,
                        destination_id="final_destination",
                        destination=final_destination,
                        route_record=final_route,
                        origin=via_destination.point,
                        route_phase="via_to_final",
                    )
                    extension = {"final_route_id": final_route_id, "final_destination": final_destination.to_dict()}
        result = self._route_response_payload(state=state, route_id=route_id)
        result["op"] = "add_via"
        result["session_id"] = state.route_session_id
        if extension:
            result["extension"] = extension
        return ToolResponse(tool=call.tool, success=True, result=result, call_id=call.call_id)

    def _replace_destination(self, *, case: UrbanNavCase, state: UrbanNavReplayToolState, call: ToolCall) -> ToolResponse:
        args = _normalized_args(call)
        destination_id, destination = self._destination_from_args(state=state, args=args)
        if destination is None:
            return self._error(call, "unknown_replacement_destination", {"destination_id": destination_id})
        route_record = self.backend.plan_route(case=case, destination=destination, origin=state.current_origin)
        if not _is_routeable(route_record):
            return self._error(call, "route_not_found", {"destination_id": destination_id, "cache_key": route_record.cache_key})
        route_id = self._store_committed_route(
            state=state,
            destination_id=destination_id,
            destination=destination,
            route_record=route_record,
            origin=state.current_origin,
            route_phase="replace_destination",
        )
        result = self._route_response_payload(state=state, route_id=route_id)
        result["op"] = "replace_destination"
        result["session_id"] = state.route_session_id
        return ToolResponse(tool=call.tool, success=True, result=result, call_id=call.call_id)

    def _replan_from_current_pose(self, *, case: UrbanNavCase, state: UrbanNavReplayToolState, call: ToolCall) -> ToolResponse:
        args = _normalized_args(call)
        current_pose = _point_from_args(args, "current_pose") or _point_from_args(args, "origin")
        if current_pose is not None:
            state.current_origin = current_pose
            if not state.full_route:
                state.full_route = [current_pose]
        destination_id, destination = self._destination_from_args(state=state, args=args)
        if destination is None and state.selected_sequence:
            destination = state.selected_sequence[-1]
            destination_id = "last_selected_destination"
        if destination is None:
            return self._error(call, "missing_replan_destination", {})
        route_record = self.backend.plan_route(case=case, destination=destination, origin=state.current_origin)
        if not _is_routeable(route_record):
            return self._error(call, "route_not_found", {"destination_id": destination_id, "cache_key": route_record.cache_key})
        route_id = self._store_committed_route(
            state=state,
            destination_id=destination_id,
            destination=destination,
            route_record=route_record,
            origin=state.current_origin,
            route_phase="replan_from_current_pose",
        )
        result = self._route_response_payload(state=state, route_id=route_id)
        result["op"] = "replan_from_current_pose"
        result["session_id"] = state.route_session_id
        return ToolResponse(tool=call.tool, success=True, result=result, call_id=call.call_id)

    def _destination_from_args(
        self,
        *,
        state: UrbanNavReplayToolState,
        args: dict[str, Any],
        destination_key: str = "destination",
    ) -> tuple[str | None, UrbanNavDestination | None]:
        destination_id = str(args.get("destination_id") or args.get("candidate_id") or "").strip() or None
        if destination_id:
            return destination_id, self.resolve_destination(state, destination_id)
        raw_destination = (
            args.get(destination_key)
            or args.get("destination")
            or args.get("new_destination")
            or args.get("replacement_destination")
            or args.get("via_point")
            or args.get("waypoint")
        )
        parsed = destination_from_any(raw_destination)
        return None, parsed

    def _store_committed_route(
        self,
        *,
        state: UrbanNavReplayToolState,
        destination_id: str | None,
        destination: UrbanNavDestination,
        route_record: ReplayRouteResult,
        origin: UrbanNavPoint,
        route_phase: str | None,
    ) -> str:
        route_id = f"route_{len(state.route_by_id) + 1}"
        leg_index = len(state.route_legs) + 1
        selected_destination = _destination_with_route_distance(destination, route_record)
        stored = StoredUrbanNavRoute(
            route_id=route_id,
            destination_id=destination_id,
            destination=selected_destination,
            route=route_record,
            leg_index=leg_index,
            route_phase=route_phase,
        )
        state.route_by_id[route_id] = stored
        state.last_route_id = route_id
        state.selected_sequence.append(selected_destination)
        state.route_legs.append(
            {
                "leg_index": leg_index,
                "route_id": route_id,
                "destination_id": destination_id,
                "origin": origin.to_dict(),
                "chosen_destination": selected_destination.to_dict(),
                "route": [point.to_dict() for point in route_record.polyline],
                "route_summary": {
                    "distance_m": route_record.distance_m,
                    "duration_s": route_record.duration_s,
                    "mode": self.mode,
                    "route_phase": route_phase,
                },
            }
        )
        state.full_route = _append_route(state.full_route, list(route_record.polyline))
        state.total_distance_m += float(route_record.distance_m or 0.0)
        if route_record.duration_s is not None:
            state.total_duration_s = float(state.total_duration_s or 0.0) + float(route_record.duration_s)
        if selected_destination.point is not None:
            state.current_origin = selected_destination.point
        return route_id

    def _route_response_payload(self, *, state: UrbanNavReplayToolState, route_id: str) -> dict[str, Any]:
        stored = state.route_by_id[route_id]
        return {
            "route_id": route_id,
            "destination_id": stored.destination_id,
            "destination": stored.destination.to_dict(),
            "cache_key": stored.route.cache_key,
            "distance_m": stored.route.distance_m,
            "duration_s": stored.route.duration_s,
            "mode": self.mode,
            "has_polyline": bool(stored.route.polyline),
            "leg_index": stored.leg_index,
        }

    def _error(self, call: ToolCall, code: str, detail: dict[str, Any]) -> ToolResponse:
        return ToolResponse(
            tool=call.tool or MAP_TOOL_NAME,
            success=False,
            error={"code": code, **dict(detail)},
            call_id=call.call_id,
        )


def build_prediction_from_tool_state(
    *,
    case: UrbanNavCase,
    state: UrbanNavReplayToolState,
    finish: dict[str, Any],
    trace: dict[str, Any],
    executor: ReplayUrbanNavToolExecutor,
) -> UrbanNavPrediction:
    """Convert a model ``<finish>`` payload and tool state into a prediction.

    Args:
        case: Current benchmark case.
        state: Case-local tool state after all tool calls.
        finish: Parsed finish JSON emitted by the model.
        trace: Diagnostic trace to attach to the prediction.
        executor: Executor used to resolve route and destination ids.

    Returns:
        UrbanNav prediction.  If the finish payload does not reference a valid
        destination/route, the returned row is a ``prediction_exception``.
    """

    route_id = str(finish.get("route_id") or finish.get("chosen_route_id") or state.last_route_id or "").strip()
    stored_route = executor.resolve_route(state, route_id)
    destination_id = str(
        finish.get("chosen_destination_id")
        or finish.get("destination_id")
        or finish.get("candidate_id")
        or (stored_route.destination_id if stored_route else "")
        or ""
    ).strip()
    destination = executor.resolve_destination(state, destination_id) if destination_id else None
    if destination is None and stored_route is not None:
        destination = stored_route.destination
    if destination is None:
        return _prediction_exception(case=case, trace=trace, detail="finish_missing_valid_destination")

    if stored_route is None and not state.full_route:
        return _prediction_exception(case=case, trace=trace, detail="finish_missing_valid_route")

    sequence = _finish_sequence(finish=finish, state=state, executor=executor)
    if not sequence and str(case.category or "") == "multi_stop_route":
        sequence = list(state.selected_sequence)
    chosen_destination = sequence[-1] if sequence and str(case.category or "") == "multi_stop_route" else destination
    route_polyline = list(state.full_route)
    route_distance_m = state.total_distance_m if state.total_distance_m > 0 else None
    route_summary = {
        "distance_m": route_distance_m,
        "duration_s": state.total_duration_s,
        "mode": executor.mode,
        "finish": dict(finish),
    }
    return UrbanNavPrediction(
        case_id=case.case_id,
        chosen_destination=chosen_destination,
        candidate_destinations=list(_unique_destinations(state.candidate_by_id.values())),
        destination_sequence=sequence,
        route_legs=list(state.route_legs),
        route_polyline=route_polyline,
        route_distance_m=route_distance_m,
        route_summary=route_summary,
        trace=trace,
    )


def _finish_sequence(
    *,
    finish: dict[str, Any],
    state: UrbanNavReplayToolState,
    executor: ReplayUrbanNavToolExecutor,
) -> list[UrbanNavDestination]:
    raw_ids = finish.get("destination_sequence_ids") or finish.get("chosen_destination_sequence_ids") or []
    if not isinstance(raw_ids, list):
        return []
    output: list[UrbanNavDestination] = []
    for item in raw_ids:
        destination = executor.resolve_destination(state, str(item))
        if destination is not None:
            output.append(destination)
    return output


def _prediction_exception(*, case: UrbanNavCase, trace: dict[str, Any], detail: str) -> UrbanNavPrediction:
    return UrbanNavPrediction(
        case_id=case.case_id,
        trace={
            **dict(trace),
            "runner": "urbannav_tool_agent",
            "case_id": case.case_id,
            "instruction": case.instruction,
            "category": case.category,
            "subcategory": case.subcategory,
            "city": case.city,
            "error": "prediction_exception",
            "detail": detail,
        },
    )


def _candidate_with_rank(candidate: UrbanNavDestination, rank: int) -> UrbanNavDestination:
    payload = candidate.to_dict()
    payload["rank"] = rank
    return UrbanNavDestination.from_dict(payload)


def _is_routeable(route: ReplayRouteResult) -> bool:
    return bool(route.polyline and route.distance_m is not None and route.distance_m > 0)


def _unique_destinations(destinations: list[UrbanNavDestination] | Any) -> list[UrbanNavDestination]:
    output: list[UrbanNavDestination] = []
    seen: set[str] = set()
    for destination in destinations:
        if not isinstance(destination, UrbanNavDestination):
            continue
        key = "|".join(sorted(destination.stable_ids)) or f"{destination.lat}:{destination.lng}:{destination.name}"
        if key in seen:
            continue
        seen.add(key)
        output.append(destination)
    return output


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _search_query_from_args(args: dict[str, Any]) -> Any:
    return (
        args.get("query")
        or args.get("keywords")
        or args.get("keyword")
        or args.get("intent")
        or args.get("category")
        or args.get("target_type")
    )


def _normalized_args(call: ToolCall) -> dict[str, Any]:
    return normalize_map_tool_arguments(call.arguments)


def _point_from_args(args: dict[str, Any], key: str) -> UrbanNavPoint | None:
    return point_from_any(args.get(key))


def _stop_index_for_case(
    *,
    case: UrbanNavCase,
    state: UrbanNavReplayToolState,
    args: dict[str, Any],
) -> int | None:
    if str(case.category or "") != "multi_stop_route":
        return None
    stop_index = _int_or_none(args.get("stop_index"))
    return stop_index if stop_index is not None else len(state.selected_sequence) + 1
