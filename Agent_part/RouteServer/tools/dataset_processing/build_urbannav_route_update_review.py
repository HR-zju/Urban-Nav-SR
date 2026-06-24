#!/usr/bin/env python3
"""Build live UrbanNav route-update review cases.

The builder creates frozen update episodes from a live map provider. Each case
first builds an old route, samples a current position on that route, then
constructs an updated route for either destination replacement or via-point
insertion.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Iterable

from mcp_server.core.live_map_backend import LiveMapBackend, LiveRouteRequest
from mcp_server.core.replay_map_backend import ReplayPoiResult, ReplayRouteResult
from mcp_server.eval.urbannav_schema import UrbanNavDestination, UrbanNavPoint


BUILDER_VERSION = "urbannav_route_update_live_v1"
DEFAULT_TASKS = Path(__file__).resolve().parents[2] / "config" / "urbannav_route_update_review_tasks.jsonl"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed route-update builder arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path, help="Directory for the review package.")
    parser.add_argument("--origin-json", required=True, type=Path, help="JSON/JSONL origin file.")
    parser.add_argument("--tasks-json", type=Path, default=DEFAULT_TASKS, help="JSON/JSONL route-update task templates.")
    parser.add_argument("--map-provider", default=None, help="RouteServer map provider override, for example amap.")
    parser.add_argument("--num-cases", type=int, default=8, help="Maximum valid cases to emit.")
    parser.add_argument("--search-radius-m", type=int, default=1500, help="POI search radius.")
    parser.add_argument("--candidate-limit", type=int, default=8, help="Maximum POI candidates routed per goal.")
    parser.add_argument("--acceptable-route-ratio", type=float, default=1.15, help="Near-tie acceptable candidate ratio.")
    return parser.parse_args()


def main() -> None:
    """Build a live route-update review package.

    Returns:
        None.
    """

    args = parse_args()
    asyncio.run(
        build_route_update_review(
            output_root=args.output_root,
            origin_path=args.origin_json,
            tasks_path=args.tasks_json,
            map_provider=args.map_provider,
            num_cases=args.num_cases,
            search_radius_m=args.search_radius_m,
            candidate_limit=args.candidate_limit,
            acceptable_route_ratio=args.acceptable_route_ratio,
        )
    )


async def build_route_update_review(
    *,
    output_root: str | Path,
    origin_path: str | Path,
    tasks_path: str | Path = DEFAULT_TASKS,
    map_provider: str | None = None,
    num_cases: int = 8,
    search_radius_m: int = 1500,
    candidate_limit: int = 8,
    acceptable_route_ratio: float = 1.15,
) -> dict[str, Any]:
    r"""Build route-update GT cases from live POI search and routing.

    For replacement updates, the GT solves:

    \[
    d_{new}^{*}=\arg\min_{d\in D_{new}^{routeable}} c(o_t,d)
    \]

    For add-via updates, the GT solves:

    \[
    v^{*}=\arg\min_{v\in V^{routeable}} c(o_t,v)
    \]

    and stores the updated route:

    \[
    P_{updated}=R(o_t,v^{*})\oplus R(v^{*},d_{old})
    \]

    Args:
        output_root: Output directory containing GT and replay caches.
        origin_path: Origin JSON/JSONL path.
        tasks_path: Route-update task JSON/JSONL path.
        map_provider: Optional live map provider override.
        num_cases: Maximum valid cases to emit.
        search_radius_m: POI search radius.
        candidate_limit: Maximum candidates routed per goal.
        acceptable_route_ratio: Near-tie ratio saved as acceptable updates.

    Returns:
        Output path and validation summary.
    """

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    origins = [dict(item) for item in _iter_json_records(origin_path)]
    tasks = [dict(item) for item in _iter_json_records(tasks_path)]
    backend = LiveMapBackend.from_env(map_provider=map_provider)

    cases: list[dict[str, Any]] = []
    poi_records: list[ReplayPoiResult] = []
    route_records: list[ReplayRouteResult] = []
    rejects: list[dict[str, Any]] = []

    args = {
        "search_radius_m": search_radius_m,
        "candidate_limit": candidate_limit,
        "acceptable_route_ratio": acceptable_route_ratio,
    }
    await backend.connect()
    try:
        pair_index = 0
        for task in tasks:
            for origin_record in origins:
                if len(cases) >= max(0, num_cases):
                    break
                pair_index += 1
                result = await _build_route_update_case(
                    backend=backend,
                    origin_record=origin_record,
                    task=task,
                    pair_index=pair_index,
                    search_radius_m=search_radius_m,
                    candidate_limit=candidate_limit,
                    acceptable_route_ratio=acceptable_route_ratio,
                )
                poi_records.extend(result["poi_records"])
                route_records.extend(result["route_records"])
                if result["case"] is None:
                    rejects.append(result["reject"])
                else:
                    cases.append(result["case"])
            if len(cases) >= max(0, num_cases):
                break
    finally:
        await backend.disconnect()

    paths = _package_paths(root)
    _write_jsonl(cases, paths["task_gt"])
    _write_jsonl((record.to_dict() for record in poi_records), paths["poi_cache"])
    _write_jsonl((record.to_dict() for record in route_records), paths["route_cache"])
    _write_jsonl(rejects, paths["task_rejects"])
    report = _validate_route_update_cases(cases=cases, rejects=rejects)
    report["args"] = args
    paths["validation_report"].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "builder_version": BUILDER_VERSION,
        "task_gt": str(paths["task_gt"]),
        "poi_cache": str(paths["poi_cache"]),
        "route_cache": str(paths["route_cache"]),
        "task_rejects": str(paths["task_rejects"]),
        "validation_report": str(paths["validation_report"]),
        "validation": report,
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


async def _build_route_update_case(
    *,
    backend: LiveMapBackend,
    origin_record: dict[str, Any],
    task: dict[str, Any],
    pair_index: int,
    search_radius_m: int,
    candidate_limit: int,
    acceptable_route_ratio: float,
) -> dict[str, Any]:
    """Build one route-update case.

    Args:
        backend: Connected live map backend.
        origin_record: Origin metadata and coordinate.
        task: Route-update task template.
        pair_index: One-based pair index.
        search_radius_m: POI search radius.
        candidate_limit: Maximum candidates per goal.
        acceptable_route_ratio: Near-tie acceptable candidate ratio.

    Returns:
        Case, replay records, and optional reject.
    """

    provider = backend.provider_name
    case_id = f"{provider}-{origin_record['scene_id']}-{task['subcategory']}-{pair_index:04d}"
    original_origin = UrbanNavPoint(lat=float(origin_record["lat"]), lng=float(origin_record["lng"]))
    previous_goal = _goal(task, "previous_goal")
    update_goal = _goal(task, "update_goal")
    if not previous_goal or not update_goal:
        return _reject_result(case_id, origin_record, task, "missing_previous_or_update_goal")

    poi_records: list[ReplayPoiResult] = []
    route_records: list[ReplayRouteResult] = []

    previous_ranked, previous_poi, previous_routes = await _rank_goal_candidates(
        backend=backend,
        provider=provider,
        case_id=case_id,
        goal_key="previous",
        origin=original_origin,
        keywords=str(previous_goal["keywords"]),
        search_radius_m=search_radius_m,
        candidate_limit=candidate_limit,
    )
    poi_records.append(previous_poi)
    route_records.extend(previous_routes)
    if not previous_ranked:
        return _reject_result(case_id, origin_record, task, "missing_routeable_previous_destination", poi_records=poi_records, route_records=route_records)

    previous_destination = previous_ranked[0]
    old_route = list(previous_destination.reference_route)
    if len(old_route) < 2:
        return _reject_result(case_id, origin_record, task, "old_route_too_short", poi_records=poi_records, route_records=route_records)
    current_origin = _route_point_at_ratio(old_route, float(task.get("current_position_ratio") or 0.45))

    update_ranked, update_poi, update_routes = await _rank_goal_candidates(
        backend=backend,
        provider=provider,
        case_id=case_id,
        goal_key="update",
        origin=current_origin,
        keywords=str(update_goal["keywords"]),
        search_radius_m=search_radius_m,
        candidate_limit=candidate_limit,
    )
    poi_records.append(update_poi)
    route_records.extend(update_routes)
    if not update_ranked:
        return _reject_result(case_id, origin_record, task, "missing_routeable_update_destination", poi_records=poi_records, route_records=route_records)

    updated_destination = update_ranked[0]
    acceptable = _acceptable(update_ranked, acceptable_route_ratio=acceptable_route_ratio)
    update_type = str(task.get("update_type") or "replace_destination")
    updated_route = list(updated_destination.reference_route)
    updated_distance = float(updated_destination.reference_route_distance_m or updated_destination.route_distance_m or 0.0)
    via_destination = updated_destination if update_type == "add_via" else None
    final_destination = previous_destination if update_type == "add_via" else updated_destination
    via_to_final_record = None

    if update_type == "add_via":
        if updated_destination.point is None:
            return _reject_result(case_id, origin_record, task, "via_destination_missing_point", poi_records=poi_records, route_records=route_records)
        via_to_final_key = f"route:{case_id}:via_to_final"
        try:
            via_to_final_record = await backend.plan_route(
                cache_key=via_to_final_key,
                request=LiveRouteRequest(origin=updated_destination.point, destination=previous_destination, mode="foot"),
            )
        except Exception as exc:
            via_to_final_record = ReplayRouteResult(
                cache_key=via_to_final_key,
                provider=provider,
                request={"origin": updated_destination.point.to_dict(), "destination": previous_destination.to_dict(), "mode": "foot"},
                response={"error": str(exc)},
            )
        route_records.append(via_to_final_record)
        if via_to_final_record.distance_m is None or via_to_final_record.distance_m <= 0 or not via_to_final_record.polyline:
            return _reject_result(case_id, origin_record, task, "missing_route_from_via_to_final", poi_records=poi_records, route_records=route_records)
        updated_route = _append_route(updated_route, list(via_to_final_record.polyline))
        updated_distance += float(via_to_final_record.distance_m)

    case = {
        "schema_version": "urbannav_case_v1",
        "id": case_id,
        "city": origin_record["city"],
        "scene_id": origin_record["scene_id"],
        "category": "route_update_add_via" if update_type == "add_via" else "route_update",
        "subcategory": task["subcategory"],
        "difficulty": task.get("difficulty", "hard"),
        "instruction": task.get("updated_instruction") or task["subcategory"],
        "origin": current_origin.to_dict(),
        "gt_destination": updated_destination.to_dict(),
        "acceptable_destinations": [destination.to_dict() for destination in acceptable],
        "ranked_destinations": [destination.to_dict() for destination in update_ranked],
        "gt_route": [point.to_dict() for point in updated_route],
        "gt_distance_m": updated_distance,
        "optimality_scope": {
            "poi_provider": provider,
            "search_radius_m": search_radius_m,
            "candidate_limit": candidate_limit,
            "route_provider": provider,
            "route_mode": "foot",
            "route_strategy": "route_update_current_origin_shortest_route",
            "cost": "route_distance_m" if update_type != "add_via" else "via_then_final_route_distance_m",
            "builder_version": BUILDER_VERSION,
        },
        "cache_keys": {
            "poi_search": [previous_poi.cache_key, update_poi.cache_key],
            "routes": [record.cache_key for record in route_records],
        },
        "metadata": {
            "origin_label": origin_record.get("label"),
            "route_update": {
                "update_type": update_type,
                "previous_instruction": task.get("previous_instruction"),
                "updated_instruction": task.get("updated_instruction"),
                "original_origin": original_origin.to_dict(),
                "current_origin": current_origin.to_dict(),
                "current_position_ratio": float(task.get("current_position_ratio") or 0.45),
                "previous_goal": previous_goal,
                "update_goal": update_goal,
                "previous_destination": previous_destination.to_dict(),
                "via_destination": via_destination.to_dict() if via_destination else None,
                "final_destination": final_destination.to_dict(),
                "old_route": [point.to_dict() for point in old_route],
                "updated_route": [point.to_dict() for point in updated_route],
                "old_route_distance_m": previous_destination.reference_route_distance_m,
                "updated_route_distance_m": updated_distance,
                "via_to_final_route": [point.to_dict() for point in via_to_final_record.polyline] if via_to_final_record else [],
                "via_to_final_distance_m": via_to_final_record.distance_m if via_to_final_record else None,
            },
        },
    }
    return {"case": case, "poi_records": poi_records, "route_records": route_records, "reject": None}


async def _rank_goal_candidates(
    *,
    backend: LiveMapBackend,
    provider: str,
    case_id: str,
    goal_key: str,
    origin: UrbanNavPoint,
    keywords: str,
    search_radius_m: int,
    candidate_limit: int,
) -> tuple[list[UrbanNavDestination], ReplayPoiResult, list[ReplayRouteResult]]:
    """Search and rank routeable candidates for one goal.

    Args:
        backend: Connected live map backend.
        provider: Provider name.
        case_id: Case id.
        goal_key: Goal key used in cache ids.
        origin: Route origin for this goal.
        keywords: POI query keywords.
        search_radius_m: POI search radius.
        candidate_limit: Maximum candidates to route.

    Returns:
        Ranked destinations, POI cache record, and route cache records.
    """

    poi_key = f"poi:{case_id}:{goal_key}"
    poi_record = await backend.search_destinations(
        cache_key=poi_key,
        origin=origin,
        keywords=keywords,
        radius_m=search_radius_m,
        limit=candidate_limit,
    )
    route_records: list[ReplayRouteResult] = []
    ranked: list[UrbanNavDestination] = []
    for candidate_index, candidate in enumerate(poi_record.candidates[:candidate_limit], start=1):
        route_key = f"route:{case_id}:{goal_key}:{candidate_index:03d}"
        try:
            route_record = await backend.plan_route(
                cache_key=route_key,
                request=LiveRouteRequest(origin=origin, destination=candidate, mode="foot"),
            )
        except Exception as exc:
            route_record = ReplayRouteResult(
                cache_key=route_key,
                provider=provider,
                request={"origin": origin.to_dict(), "destination": candidate.to_dict(), "mode": "foot"},
                response={"error": str(exc)},
            )
        route_records.append(route_record)
        if route_record.distance_m is None or route_record.distance_m <= 0 or not route_record.polyline:
            continue
        ranked.append(
            UrbanNavDestination(
                lat=candidate.lat,
                lng=candidate.lng,
                name=candidate.name,
                type=candidate.type,
                provider_poi_id=candidate.provider_poi_id,
                place_id=candidate.place_id,
                reference_route_distance_m=route_record.distance_m,
                reference_route=route_record.polyline,
                route_distance_m=route_record.distance_m,
                score_breakdown={
                    "provider_rank": candidate_index,
                    "route_distance_m": route_record.distance_m,
                    "selection_scope": goal_key,
                },
                raw=candidate.raw,
            )
        )
    ranked.sort(key=lambda item: (item.route_distance_m if item.route_distance_m is not None else float("inf"), item.name or ""))
    return [_with_rank(destination, rank=rank) for rank, destination in enumerate(ranked, start=1)], poi_record, route_records


def _acceptable(ranked: list[UrbanNavDestination], *, acceptable_route_ratio: float) -> list[UrbanNavDestination]:
    """Return near-tie acceptable destinations.

    Args:
        ranked: Ranked routeable destinations.
        acceptable_route_ratio: Near-tie ratio.

    Returns:
        Acceptable destinations.
    """

    if not ranked:
        return []
    best = ranked[0].route_distance_m or ranked[0].reference_route_distance_m or 0.0
    return [
        destination
        for destination in ranked
        if (destination.route_distance_m or float("inf")) <= best * max(1.0, acceptable_route_ratio)
    ] or [ranked[0]]


def _route_point_at_ratio(route: list[UrbanNavPoint], ratio: float) -> UrbanNavPoint:
    """Choose a route point near a fractional progress value.

    Args:
        route: Old route polyline.
        ratio: Fractional progress in ``[0, 1]``.

    Returns:
        Route point used as update-time current origin.
    """

    if not route:
        raise ValueError("route must not be empty")
    clean_ratio = min(0.95, max(0.05, ratio))
    index = min(len(route) - 1, max(0, round((len(route) - 1) * clean_ratio)))
    return route[index]


def _append_route(base: list[UrbanNavPoint], leg: list[UrbanNavPoint]) -> list[UrbanNavPoint]:
    """Append a route leg while removing one adjacent duplicate endpoint.

    Args:
        base: Existing route.
        leg: Route leg to append.

    Returns:
        Concatenated route.
    """

    if not leg:
        return list(base)
    if not base:
        return list(leg)
    if abs(base[-1].lat - leg[0].lat) < 1e-9 and abs(base[-1].lng - leg[0].lng) < 1e-9:
        return [*base, *leg[1:]]
    return [*base, *leg]


def _with_rank(destination: UrbanNavDestination, *, rank: int) -> UrbanNavDestination:
    """Return a copy of a destination with rank.

    Args:
        destination: Destination to copy.
        rank: One-based rank.

    Returns:
        Ranked destination.
    """

    payload = destination.to_dict()
    payload["rank"] = rank
    return UrbanNavDestination.from_dict(payload)


def _goal(task: dict[str, Any], key: str) -> dict[str, Any]:
    """Read a goal dictionary.

    Args:
        task: Task template.
        key: Goal key.

    Returns:
        Goal dictionary, or empty dictionary.
    """

    value = task.get(key)
    return dict(value) if isinstance(value, dict) else {}


def _reject_result(
    case_id: str,
    origin_record: dict[str, Any],
    task: dict[str, Any],
    reason: str,
    *,
    poi_records: list[ReplayPoiResult] | None = None,
    route_records: list[ReplayRouteResult] | None = None,
) -> dict[str, Any]:
    """Create a rejected route-update result.

    Args:
        case_id: Case id.
        origin_record: Origin metadata.
        task: Task template.
        reason: Reject reason.
        poi_records: Partial POI cache records.
        route_records: Partial route cache records.

    Returns:
        Result dictionary.
    """

    return {
        "case": None,
        "poi_records": poi_records or [],
        "route_records": route_records or [],
        "reject": {
            "id": case_id,
            "reason": reason,
            "city": origin_record.get("city"),
            "scene_id": origin_record.get("scene_id"),
            "origin": {"lat": origin_record.get("lat"), "lng": origin_record.get("lng")},
            "subcategory": task.get("subcategory"),
            "update_type": task.get("update_type"),
            "instruction": task.get("updated_instruction"),
        },
    }


def _validate_route_update_cases(*, cases: list[dict[str, Any]], rejects: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate basic route-update invariants for review packages.

    Args:
        cases: Built cases.
        rejects: Reject rows.

    Returns:
        Validation report.
    """

    errors: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id"))
        update = case.get("metadata", {}).get("route_update", {}) if isinstance(case.get("metadata"), dict) else {}
        if not update:
            errors.append({"id": case_id, "error": "missing_route_update_metadata"})
            continue
        if not update.get("old_route") or not update.get("updated_route"):
            errors.append({"id": case_id, "error": "missing_old_or_updated_route"})
        if not update.get("current_origin"):
            errors.append({"id": case_id, "error": "missing_current_origin"})
        if update.get("update_type") == "add_via" and not update.get("final_destination"):
            errors.append({"id": case_id, "error": "missing_final_destination_for_add_via"})
        if not case.get("gt_route"):
            errors.append({"id": case_id, "error": "missing_gt_route"})
    return {
        "valid": not errors,
        "case_count": len(cases),
        "reject_count": len(rejects),
        "errors": errors,
        "builder_version": BUILDER_VERSION,
    }


def _package_paths(root: Path) -> dict[str, Path]:
    """Return output package paths.

    Args:
        root: Output root.

    Returns:
        Named output paths.
    """

    return {
        "task_gt": root / "task_gt.jsonl",
        "poi_cache": root / "poi_cache.jsonl",
        "route_cache": root / "route_cache.jsonl",
        "task_rejects": root / "task_rejects.jsonl",
        "validation_report": root / "validation_report.json",
        "manifest": root / "manifest.json",
    }


def _iter_json_records(path: str | Path) -> Iterable[dict[str, Any]]:
    """Iterate JSON or JSONL records.

    Args:
        path: Input file.

    Returns:
        JSON dictionaries.
    """

    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        return [dict(item) for item in data]
    return [dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def _write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write JSONL records.

    Args:
        records: JSON-ready records.
        path: Output path.

    Returns:
        None.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
