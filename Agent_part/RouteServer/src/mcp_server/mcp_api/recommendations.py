"""MCP endpoints for recommendations."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import Context
from mcp_server.llm import GoalDecomposer, ItineraryCandidateJudge
from mcp_server.map_providers import format_location
from mcp_server.recommendation import DestinationRecommendationEngine
from mcp_server.services.route_session_service import _compute_combined_route_plan
from mcp_server.mcp_api.presentation import _destination_id, _select_itinerary_destination
from mcp_server.mcp_api.app import mcp

@mcp.tool()
async def recommend_destination_and_waypoints(
    user_instruction: str,
    current_latitude: float,
    current_longitude: float,
    ctx: Context,
    max_candidates: int = 5,
    search_radius: float = 3000,
    mode: Optional[str] = None,
    resolution_scope: str = "global_destination",
    anchor_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Parse free-form human instructions, recommend candidate destinations, and return route key waypoints.

    This first version combines a lightweight prompt-engineering parser (LLM when configured,
    heuristic fallback otherwise) with map search and routing to produce practical destination
    recommendations and waypoint sequences for navigation.
    """
    app_context = ctx.request_context.lifespan_context
    engine = DestinationRecommendationEngine(
        map_client=app_context.osm_client,
        provider=app_context.provider,
    )
    return await engine.recommend(
        user_instruction=user_instruction,
        current_latitude=current_latitude,
        current_longitude=current_longitude,
        max_candidates=max_candidates,
        search_radius=search_radius,
        mode=mode,
        resolution_scope=resolution_scope,
        anchor_location=anchor_location,
        ctx=ctx,
    )


@mcp.tool()
async def decompose_itinerary_goals(
    user_instruction: str,
    ctx: Context,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decompose a free-form instruction into an ordered list of itinerary goals.

    This tool does not recommend concrete POIs yet. It extracts the ordered goal
    skeleton that can later be realized as via points plus a final destination.
    """
    decomposer = GoalDecomposer()
    decomposition = await decomposer.parse(user_instruction)

    override_mode = str(mode or "").strip().lower()
    if override_mode in {"car", "foot", "bike"}:
        decomposition["global_mode"] = override_mode
        for goal in decomposition.get("goals", []) or []:
            if isinstance(goal, dict):
                goal["mode"] = override_mode

    return {
        "input": {
            "user_instruction": user_instruction,
            "mode_override": override_mode or None,
        },
        "decomposition": decomposition,
        "timestamp": datetime.now().isoformat(),
    }


@mcp.tool()
async def recommend_itinerary_and_waypoints(
    user_instruction: str,
    current_latitude: float,
    current_longitude: float,
    ctx: Context,
    max_candidates: int = 5,
    search_radius: float = 3000,
    mode: Optional[str] = None,
    max_goals: int = 4,
) -> Dict[str, Any]:
    """
    Decompose a multi-goal instruction, recommend one stop for each ordered goal,
    and build a combined multi-leg route with via points.

    This is the first itinerary-oriented extension on top of the single-destination
    recommendation stack. It keeps each sub-goal provider-retrievable and realizes
    earlier goals as via points plus one final destination.
    """
    app_context = ctx.request_context.lifespan_context
    engine = DestinationRecommendationEngine(
        map_client=app_context.osm_client,
        provider=app_context.provider,
    )
    decomposer = GoalDecomposer()
    candidate_judge = ItineraryCandidateJudge()
    decomposition = await decomposer.parse(user_instruction)

    goals = [
        goal
        for goal in (decomposition.get("goals", []) or [])
        if isinstance(goal, dict) and str(goal.get("request_text") or "").strip()
    ][: max(1, int(max_goals))]

    if not goals:
        single = await engine.recommend(
            user_instruction=user_instruction,
            current_latitude=current_latitude,
            current_longitude=current_longitude,
            max_candidates=max_candidates,
            search_radius=search_radius,
            mode=mode,
            ctx=ctx,
        )
        return {
            "provider": app_context.provider,
            "input": {
                "user_instruction": user_instruction,
                "current_latitude": current_latitude,
                "current_longitude": current_longitude,
            },
            "decomposition": decomposition,
            "goal_recommendations": [],
            "via_points": [],
            "final_destination": single.get("chosen_destination"),
            "route_plan": {
                "route_summary": single.get("route_summary", {}),
                "route_polyline": single.get("route_polyline", []),
                "coarse_waypoints": single.get("key_waypoints", []),
                "legs": [],
            },
            "status": "single_goal_fallback",
        }

    override_mode = str(mode or "").strip().lower()
    global_mode = override_mode if override_mode in {"car", "foot", "bike"} else str(
        decomposition.get("global_mode") or "car"
    ).strip().lower()
    if global_mode not in {"car", "foot", "bike"}:
        global_mode = "car"

    goal_results: List[Dict[str, Any]] = []
    selected_stops: List[Dict[str, Any]] = []
    sequential_lat = float(current_latitude)
    sequential_lon = float(current_longitude)
    used_destination_ids: set[str] = set()

    for goal in goals:
        goal_mode = override_mode if override_mode in {"car", "foot", "bike"} else str(
            goal.get("mode") or global_mode
        ).strip().lower()
        if goal_mode not in {"car", "foot", "bike"}:
            goal_mode = global_mode
        goal_radius = goal.get("radius_m")
        if not isinstance(goal_radius, (int, float)) or float(goal_radius) <= 0:
            goal_radius = search_radius

        goal_instruction = str(goal.get("request_text") or "").strip()
        search_keywords = str(goal.get("search_keywords") or "").strip()
        goal_type = str(goal.get("goal_type") or "").strip()
        if search_keywords:
            goal_instruction = (
                f"{goal_instruction}. Goal type: {goal_type}. "
                f"Use these provider-facing search keywords: {search_keywords}."
            )
        recommendation = await engine.recommend(
            user_instruction=goal_instruction,
            current_latitude=sequential_lat,
            current_longitude=sequential_lon,
            max_candidates=max_candidates,
            search_radius=float(goal_radius),
            mode=goal_mode,
            ctx=ctx,
        )

        top_candidates = [
            candidate
            for candidate in (recommendation.get("recommendations") or [])
            if isinstance(candidate, dict)
            and (
                not _destination_id(candidate)
                or _destination_id(candidate) not in used_destination_ids
            )
        ][: max(1, int(max_candidates))]
        judge_decision = await candidate_judge.select_candidate(
            goal=goal,
            candidates=top_candidates,
            previous_stops=selected_stops,
        )
        chosen = None
        selection_reason = str(judge_decision.get("judge_source") or "llm_judge")
        selected_id = str(judge_decision.get("selected_candidate_id") or "").strip()
        if selected_id:
            for candidate in top_candidates:
                if _destination_id(candidate) == selected_id:
                    chosen = candidate
                    break
        if not chosen:
            fallback_chosen, fallback_reason = _select_itinerary_destination(
                recommendation=recommendation,
                used_destination_ids=used_destination_ids,
            )
            chosen = fallback_chosen
            selection_reason = f"{selection_reason}_fallback_{fallback_reason}"
        if chosen:
            recommendation["chosen_destination"] = chosen
        goal_entry = {
            "goal": goal,
            "recommendation": recommendation,
            "status": "selected" if chosen else "failed",
            "selection_reason": selection_reason,
            "judge_decision": judge_decision,
            "judge_fallback_used": not bool(selected_id and chosen),
        }
        goal_results.append(goal_entry)

        if not chosen:
            break

        destination_id = _destination_id(chosen)
        if destination_id:
            used_destination_ids.add(destination_id)
        selected_stops.append(
            {
                "name": str(chosen.get("name") or ""),
                "location": str(chosen.get("location") or ""),
                "goal_type": str(goal.get("goal_type") or ""),
                "goal_index": goal.get("goal_index"),
            }
        )

        coords = (chosen.get("coordinates") or {})
        next_lat = coords.get("latitude")
        next_lon = coords.get("longitude")
        if isinstance(next_lat, (int, float)) and isinstance(next_lon, (int, float)):
            sequential_lat = float(next_lat)
            sequential_lon = float(next_lon)

    successful_goals = [item for item in goal_results if item.get("status") == "selected"]
    chosen_stops = [item.get("recommendation", {}).get("chosen_destination") for item in successful_goals]
    chosen_stops = [item for item in chosen_stops if isinstance(item, dict)]

    via_points = []
    final_destination = chosen_stops[-1] if chosen_stops else None
    if len(chosen_stops) >= 2:
        for index, stop in enumerate(chosen_stops[:-1]):
            goal = successful_goals[index].get("goal") or {}
            via_points.append(
                {
                    "location": str(stop.get("location") or "").strip(),
                    "name": str(stop.get("name") or goal.get("goal_type") or f"Stop {index + 1}").strip(),
                    "goal_index": goal.get("goal_index"),
                    "goal_type": goal.get("goal_type"),
                }
            )

    route_plan = None
    if final_destination and str(final_destination.get("location") or "").strip():
        route_plan = await _compute_combined_route_plan(
            app_context=app_context,
            origin=format_location(current_longitude, current_latitude),
            destination=str(final_destination["location"]),
            via_points=via_points,
            mode=global_mode,
        )

    status = "failed"
    if goal_results:
        if len(successful_goals) == len(goals):
            status = "completed"
        elif successful_goals:
            status = "partial"

    return {
        "provider": app_context.provider,
        "input": {
            "user_instruction": user_instruction,
            "current_latitude": current_latitude,
            "current_longitude": current_longitude,
            "mode_override": override_mode or None,
        },
        "decomposition": decomposition,
        "goal_recommendations": goal_results,
        "fulfilled_goal_count": len(successful_goals),
        "requested_goal_count": len(goals),
        "via_points": via_points,
        "final_destination": final_destination,
        "route_plan": route_plan,
        "status": status,
        "timestamp": datetime.now().isoformat(),
    }
