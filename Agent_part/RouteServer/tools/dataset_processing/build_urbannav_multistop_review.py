#!/usr/bin/env python3
"""Build live UrbanNav multi-stop review cases from ordered goal templates."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Iterable

from mcp_server.core.live_map_backend import LiveMapBackend, LiveRouteRequest
from mcp_server.core.replay_map_backend import ReplayPoiResult, ReplayRouteResult
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavDestination, UrbanNavPoint


BUILDER_VERSION = "urbannav_multistop_live_v1"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed multi-stop builder arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path, help="Directory for the review package")
    parser.add_argument("--origin-json", required=True, type=Path, help="JSON/JSONL origin file")
    parser.add_argument("--tasks-json", required=True, type=Path, help="JSON/JSONL ordered-goal task templates")
    parser.add_argument("--map-provider", default=None, help="RouteServer map provider override, for example amap")
    parser.add_argument("--num-cases", type=int, default=5, help="Maximum valid cases to emit")
    parser.add_argument("--search-radius-m", type=int, default=1500, help="POI search radius per stop")
    parser.add_argument("--candidate-limit", type=int, default=8, help="Maximum POI candidates per stop")
    parser.add_argument(
        "--acceptable-route-ratio",
        type=float,
        default=1.15,
        help="Near-tie ratio saved for per-stop acceptable candidates",
    )
    return parser.parse_args()


def main() -> None:
    """Build a live multi-stop review package.

    Returns:
        None.
    """

    args = parse_args()
    asyncio.run(
        build_multistop_review(
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


async def build_multistop_review(
    *,
    output_root: str | Path,
    origin_path: str | Path,
    tasks_path: str | Path,
    map_provider: str | None = None,
    num_cases: int = 5,
    search_radius_m: int = 1500,
    candidate_limit: int = 8,
    acceptable_route_ratio: float = 1.15,
) -> dict[str, Any]:
    r"""Build multi-stop GT cases from live map search and route planning.

    For a task with ordered goals \(g_1,\ldots,g_K\), the builder solves one
    local destination selection problem per goal:

    \[
    d_i^\* = \arg\min_{d \in D_i^{routeable}} c(p_{i-1}, d)
    \]

    where \(p_0\) is the benchmark origin and \(p_i=d_i^\*\). The full route
    is the concatenation of the route legs:

    \[
    R = R(p_0,d_1^\*) \oplus R(d_1^\*,d_2^\*) \oplus \cdots \oplus R(d_{K-1}^\*,d_K^\*)
    \]

    Args:
        output_root: Output directory containing ``task_gt.jsonl`` and caches.
        origin_path: Origin JSON/JSONL file.
        tasks_path: Multi-stop task JSON/JSONL file.
        map_provider: Optional map provider override, such as ``"amap"``.
        num_cases: Maximum valid cases to emit.
        search_radius_m: POI search radius per stop.
        candidate_limit: Maximum candidates routed per stop.
        acceptable_route_ratio: Near-tie ratio saved for per-stop alternatives.

    Returns:
        Dictionary containing output paths and validation summary.
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

    await backend.connect()
    try:
        pair_index = 0
        for task in tasks:
            for origin_record in origins:
                if len(cases) >= max(0, num_cases):
                    break
                pair_index += 1
                result = await _build_multistop_case(
                    backend=backend,
                    origin_record=origin_record,
                    task=task,
                    pair_index=pair_index,
                    search_radius_m=search_radius_m,
                    candidate_limit=candidate_limit,
                    acceptable_route_ratio=acceptable_route_ratio,
                )
                if result["case"] is None:
                    rejects.append(result["reject"])
                    poi_records.extend(result["poi_records"])
                    route_records.extend(result["route_records"])
                    continue
                cases.append(result["case"])
                poi_records.extend(result["poi_records"])
                route_records.extend(result["route_records"])
            if len(cases) >= max(0, num_cases):
                break
    finally:
        await backend.disconnect()

    gt_path = root / "task_gt.jsonl"
    poi_cache_path = root / "poi_cache.jsonl"
    route_cache_path = root / "route_cache.jsonl"
    rejects_path = root / "task_rejects.jsonl"
    report_path = root / "validation_report.json"

    _write_jsonl(cases, gt_path)
    _write_jsonl((record.to_dict() for record in poi_records), poi_cache_path)
    _write_jsonl((record.to_dict() for record in route_records), route_cache_path)
    _write_jsonl(rejects, rejects_path)
    report = _validate_multistop_cases(cases=cases, rejects=rejects)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "task_gt": str(gt_path),
        "poi_cache": str(poi_cache_path),
        "route_cache": str(route_cache_path),
        "task_rejects": str(rejects_path),
        "validation_report": str(report_path),
        "validation": report,
    }


async def _build_multistop_case(
    *,
    backend: LiveMapBackend,
    origin_record: dict[str, Any],
    task: dict[str, Any],
    pair_index: int,
    search_radius_m: int,
    candidate_limit: int,
    acceptable_route_ratio: float,
) -> dict[str, Any]:
    """Build one ordered multi-stop case.

    Args:
        backend: Connected live map backend.
        origin_record: Origin metadata and coordinate.
        task: Task template containing ordered ``goals``.
        pair_index: One-based origin/task pair index.
        search_radius_m: POI search radius per stop.
        candidate_limit: Maximum candidates routed per stop.
        acceptable_route_ratio: Near-tie ratio saved for alternatives.

    Returns:
        Dictionary with case/caches, or a reject and partial caches.
    """

    provider = backend.provider_name
    origin = UrbanNavPoint(lat=float(origin_record["lat"]), lng=float(origin_record["lng"]))
    goals = task.get("goals")
    if not isinstance(goals, list) or not goals:
        case_id = f"{provider}-{origin_record['scene_id']}-{task.get('subcategory', 'invalid')}-{pair_index:04d}"
        return _multistop_reject_result(case_id, origin_record, task, "missing_ordered_goals")

    case_id = f"{provider}-{origin_record['scene_id']}-{task['subcategory']}-{pair_index:04d}"
    current_point = origin
    total_distance_m = 0.0
    gt_sequence: list[UrbanNavDestination] = []
    gt_route_legs: list[dict[str, Any]] = []
    full_route: list[UrbanNavPoint] = [origin]
    poi_records: list[ReplayPoiResult] = []
    route_records: list[ReplayRouteResult] = []
    all_route_keys: list[str] = []
    all_poi_keys: list[str] = []
    selected_destinations: list[UrbanNavDestination] = []

    for goal_index, goal in enumerate(goals, start=1):
        keywords = str(goal.get("keywords") or "").strip()
        if not keywords:
            return _multistop_reject_result(
                case_id,
                origin_record,
                task,
                f"missing_keywords_at_stop_{goal_index}",
                poi_records=poi_records,
                route_records=route_records,
            )

        poi_key = f"poi:{case_id}:stop{goal_index:02d}"
        all_poi_keys.append(poi_key)
        poi_record = await backend.search_destinations(
            cache_key=poi_key,
            origin=current_point,
            keywords=keywords,
            radius_m=search_radius_m,
            limit=candidate_limit,
        )
        poi_records.append(poi_record)
        if not poi_record.candidates:
            return _multistop_reject_result(
                case_id,
                origin_record,
                task,
                f"no_poi_candidates_at_stop_{goal_index}",
                poi_records=poi_records,
                route_records=route_records,
            )

        ranked, stop_route_records = await _rank_routeable_candidates(
            backend=backend,
            provider=provider,
            case_id=case_id,
            stop_index=goal_index,
            origin=current_point,
            candidates=poi_record.candidates[:candidate_limit],
            excluded_destinations=selected_destinations,
        )
        route_records.extend(stop_route_records)
        all_route_keys.extend(record.cache_key for record in stop_route_records)
        if not ranked:
            return _multistop_reject_result(
                case_id,
                origin_record,
                task,
                f"no_routeable_candidates_at_stop_{goal_index}",
                poi_records=poi_records,
                route_records=route_records,
            )

        gt_stop = ranked[0]
        best_distance = gt_stop.route_distance_m or gt_stop.reference_route_distance_m or 0.0
        acceptable = [
            destination
            for destination in ranked
            if (destination.route_distance_m or float("inf")) <= best_distance * max(1.0, acceptable_route_ratio)
        ]
        leg_route = list(gt_stop.reference_route)
        full_route = _append_route(full_route, leg_route)
        total_distance_m += float(gt_stop.reference_route_distance_m or gt_stop.route_distance_m or 0.0)
        gt_sequence.append(gt_stop)
        selected_destinations.append(gt_stop)
        gt_route_legs.append(
            {
                "leg_index": goal_index,
                "goal_label": goal.get("label") or f"stop_{goal_index}",
                "keywords": keywords,
                "origin": current_point.to_dict(),
                "gt_destination": gt_stop.to_dict(),
                "acceptable_destinations": [destination.to_dict() for destination in acceptable],
                "ranked_destinations": [destination.to_dict() for destination in ranked],
                "route": [point.to_dict() for point in leg_route],
                "distance_m": gt_stop.reference_route_distance_m,
                "cache_keys": {
                    "poi_search": [poi_key],
                    "routes": [record.cache_key for record in stop_route_records],
                },
            }
        )
        if gt_stop.point is None:
            return _multistop_reject_result(
                case_id,
                origin_record,
                task,
                f"selected_destination_missing_point_at_stop_{goal_index}",
                poi_records=poi_records,
                route_records=route_records,
            )
        current_point = gt_stop.point

    final_destination = gt_sequence[-1]
    case = {
        "schema_version": "urbannav_case_v1",
        "id": case_id,
        "city": origin_record["city"],
        "scene_id": origin_record["scene_id"],
        "category": "multi_stop_route",
        "subcategory": task["subcategory"],
        "difficulty": task.get("difficulty", "hard"),
        "instruction": task["instruction"],
        "origin": origin.to_dict(),
        "gt_destination": final_destination.to_dict(),
        "gt_destination_sequence": [destination.to_dict() for destination in gt_sequence],
        "gt_route_legs": gt_route_legs,
        "acceptable_destinations": gt_route_legs[-1].get("acceptable_destinations", []),
        "ranked_destinations": gt_route_legs[-1].get("ranked_destinations", []),
        "gt_route": [point.to_dict() for point in full_route],
        "gt_distance_m": total_distance_m,
        "optimality_scope": {
            "poi_provider": provider,
            "search_radius_m": search_radius_m,
            "candidate_limit": candidate_limit,
            "route_provider": provider,
            "route_mode": "foot",
            "route_strategy": "sequential_current_stop_shortest_route",
            "cost": "sum_route_distance_m",
            "builder_version": BUILDER_VERSION,
        },
        "cache_keys": {"poi_search": all_poi_keys, "routes": all_route_keys},
        "metadata": {
            "origin_label": origin_record.get("label"),
            "goal_count": len(gt_sequence),
            "goals": goals,
            "acceptable_route_ratio": acceptable_route_ratio,
            "multi_stop_semantics": "ordered_waypoints",
        },
    }
    return {"case": case, "poi_records": poi_records, "route_records": route_records, "reject": None}


async def _rank_routeable_candidates(
    *,
    backend: LiveMapBackend,
    provider: str,
    case_id: str,
    stop_index: int,
    origin: UrbanNavPoint,
    candidates: list[UrbanNavDestination],
    excluded_destinations: list[UrbanNavDestination],
) -> tuple[list[UrbanNavDestination], list[ReplayRouteResult]]:
    """Rank candidates for one stop by route distance from current position.

    Args:
        backend: Connected live map backend.
        provider: Provider name.
        case_id: Case id.
        stop_index: One-based stop index.
        origin: Current route position.
        candidates: POI candidates to route.
        excluded_destinations: Previously selected stops that cannot be reused.

    Returns:
        Tuple of ranked destinations and route cache records.
    """

    route_records: list[ReplayRouteResult] = []
    ranked: list[UrbanNavDestination] = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        if _is_repeated_destination(candidate, excluded_destinations):
            continue
        route_key = f"route:{case_id}:stop{stop_index:02d}:{candidate_index:03d}"
        try:
            route_record = await backend.plan_route(
                cache_key=route_key,
                request=LiveRouteRequest(origin=origin, destination=candidate, mode="foot"),
            )
        except Exception as exc:
            route_records.append(
                ReplayRouteResult(
                    cache_key=route_key,
                    provider=provider,
                    request={"origin": origin.to_dict(), "destination": candidate.to_dict(), "mode": "foot"},
                    response={"error": str(exc)},
                )
            )
            continue
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
                    "selection_scope": f"stop_{stop_index}",
                },
                raw=candidate.raw,
            )
        )

    ranked.sort(key=lambda item: (item.route_distance_m if item.route_distance_m is not None else float("inf"), item.name or ""))
    return [_with_rank(destination, rank=rank) for rank, destination in enumerate(ranked, start=1)], route_records


def _is_repeated_destination(candidate: UrbanNavDestination, selected: list[UrbanNavDestination]) -> bool:
    """Check whether a candidate repeats an earlier selected stop.

    Args:
        candidate: Candidate destination for the current stop.
        selected: Destinations already selected by earlier stops.

    Returns:
        ``True`` when provider ids match or coordinates are nearly identical.
    """

    candidate_ids = candidate.stable_ids
    candidate_point = candidate.point
    for existing in selected:
        if candidate_ids and candidate_ids.intersection(existing.stable_ids):
            return True
        existing_point = existing.point
        if candidate_point is None or existing_point is None:
            continue
        if abs(candidate_point.lat - existing_point.lat) < 2e-5 and abs(candidate_point.lng - existing_point.lng) < 2e-5:
            return True
    return False


def _with_rank(destination: UrbanNavDestination, *, rank: int) -> UrbanNavDestination:
    """Return a copy of a destination with benchmark rank.

    Args:
        destination: Destination to copy.
        rank: One-based rank.

    Returns:
        Ranked destination.
    """

    payload = destination.to_dict()
    payload["rank"] = rank
    return UrbanNavDestination.from_dict(payload)


def _append_route(base: list[UrbanNavPoint], leg: list[UrbanNavPoint]) -> list[UrbanNavPoint]:
    """Append a route leg while removing one adjacent duplicate endpoint.

    Args:
        base: Existing full route.
        leg: Next route leg.

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


def _multistop_reject_result(
    case_id: str,
    origin_record: dict[str, Any],
    task: dict[str, Any],
    reason: str,
    *,
    poi_records: list[ReplayPoiResult] | None = None,
    route_records: list[ReplayRouteResult] | None = None,
) -> dict[str, Any]:
    """Create a rejected case result.

    Args:
        case_id: Candidate case id.
        origin_record: Origin metadata.
        task: Task template.
        reason: Reject reason.
        poi_records: Partial POI cache records.
        route_records: Partial route cache records.

    Returns:
        Result dictionary compatible with successful builder results.
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
            "instruction": task.get("instruction"),
        },
    }


def _validate_multistop_cases(*, cases: list[dict[str, Any]], rejects: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate basic multi-stop route invariants for review.

    Args:
        cases: Built case payloads.
        rejects: Rejected case records.

    Returns:
        JSON-ready validation report.
    """

    errors: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("id"))
        sequence = case.get("gt_destination_sequence")
        legs = case.get("gt_route_legs")
        route = case.get("gt_route")
        if not isinstance(sequence, list) or len(sequence) < 2:
            errors.append({"id": case_id, "error": "multi_stop_sequence_needs_at_least_two_stops"})
        if not isinstance(legs, list) or len(legs) != len(sequence or []):
            errors.append({"id": case_id, "error": "leg_count_mismatch"})
        if not isinstance(route, list) or len(route) < 2:
            errors.append({"id": case_id, "error": "missing_full_route"})
        for leg in legs or []:
            if not isinstance(leg, dict) or not leg.get("gt_destination") or not leg.get("route"):
                errors.append({"id": case_id, "error": "malformed_leg", "leg": leg.get("leg_index") if isinstance(leg, dict) else None})
    return {
        "valid": not errors,
        "case_count": len(cases),
        "reject_count": len(rejects),
        "errors": errors,
        "builder_version": BUILDER_VERSION,
    }


def _iter_json_records(path: str | Path) -> Iterable[dict[str, Any]]:
    """Iterate JSON or JSONL records.

    Args:
        path: Input file path.

    Returns:
        Iterable of JSON dictionaries.
    """

    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        return [dict(item) for item in data]
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            records.append(dict(json.loads(line)))
    return records


def _write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write JSONL records.

    Args:
        records: JSON-ready records.
        path: Output file path.

    Returns:
        None.
    """

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
