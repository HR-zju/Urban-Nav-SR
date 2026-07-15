#!/usr/bin/env python3
r"""Collect UrbanNav v2 multi-stop and route-update GT cases.

This builder consumes the compiled ``instruction_set.jsonl`` and constructs
real map-grounded complex cases through the same ``UrbanNavLiveProvider``
interface used by the single-stop collector.  It is intentionally separate from
the older review builders so that benchmark-scale collection remains resumable
and provider-agnostic.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

from mcp_server.core.replay_map_backend import ReplayPoiResult, ReplayRouteResult
from mcp_server.core.urbannav_provider import UrbanNavLiveProvider, UrbanNavPoiSearchRequest, UrbanNavRouteRequest
from mcp_server.eval.urbannav_io import iter_json_records, load_urbannav_cases
from mcp_server.eval.urbannav_collection_policy import previous_target_types_for_update
from mcp_server.eval.urbannav_schema import UrbanNavDestination, UrbanNavPoint
from mcp_server.eval.urbannav_validator import validate_urbannav_cases

from collect_urbannav_global_gt import (
    BUILDER_VERSION as GLOBAL_BUILDER_VERSION,
    DEFAULT_CONFIG,
    _append_jsonl,
    _create_provider,
    _dedupe_jsonl,
    _destination_with_route,
    _difficulty,
    _haversine_m,
    _norm_city,
    _sample_region_origins,
    _setting,
    _unlink_outputs,
    _with_rank,
    build_global_collection_plan,
)


BUILDER_VERSION = "urbannav_complex_provider_v2"
DEFAULT_TARGET_MAP = Path(__file__).resolve().parents[2] / "config" / "urbannav_target_type_provider_map_mined_v2.json"
DEFAULT_QUALITY_POLICY = Path(__file__).resolve().parents[2] / "config" / "urbannav_25k_v2_quality_policy.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path, help="Output benchmark package directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Global city/region config JSON.")
    parser.add_argument("--instruction-set", required=True, type=Path, help="Compiled UrbanNav instruction_set.jsonl.")
    parser.add_argument("--target-type-map", type=Path, default=DEFAULT_TARGET_MAP, help="Target type to provider tag map.")
    parser.add_argument("--quality-policy", type=Path, default=None, help="Optional v2 multi-stop quality policy JSON.")
    parser.add_argument(
        "--multi-stop-cache-policy",
        choices=["gt_path", "full_route_matrix"],
        default="gt_path",
        help="Cache only the GT-origin routes or every predecessor/candidate transition.",
    )
    parser.add_argument("--min-inter-stop-distance-m", type=float, default=None, help="Minimum distance between distinct stops.")
    parser.add_argument("--min-leg-route-distance-m", type=float, default=None, help="Minimum route length for every GT leg.")
    parser.add_argument("--max-leg-route-distance-m", type=float, default=None, help="Maximum route length for every GT leg.")
    parser.add_argument("--target-multistop", type=int, default=1500, help="Target valid multi-stop cases.")
    parser.add_argument("--target-route-update", type=int, default=1000, help="Target valid route-update cases.")
    parser.add_argument("--seed", type=int, default=20260629, help="Sampling seed.")
    parser.add_argument("--origins-per-region", type=int, default=64, help="Origins sampled per region.")
    parser.add_argument("--candidate-limit", type=int, default=None, help="Maximum candidate POIs per stop.")
    parser.add_argument("--search-radius-m", type=int, default=None, help="POI search radius in meters.")
    parser.add_argument("--acceptable-route-ratio", type=float, default=None, help="Near-tie acceptable route ratio.")
    parser.add_argument("--throttle-s", type=float, default=0.0, help="Sleep seconds after each attempted case.")
    parser.add_argument("--retry", type=int, default=1, help="Retry count for provider calls.")
    parser.add_argument("--case-concurrency", type=int, default=8, help="Concurrent complex cases.")
    parser.add_argument("--route-concurrency", type=int, default=4, help="Concurrent route calls inside each goal.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing package.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing outputs before running.")
    parser.add_argument("--dry-run", action="store_true", help="Only write plan and manifest.")
    parser.add_argument("--max-attempts", type=int, default=0, help="Optional attempted row cap.")
    parser.add_argument("--provider", choices=["osm", "osm_local", "tomtom"], default=None, help="Provider backend.")
    parser.add_argument("--overpass-url", default=None, help="Override Overpass endpoint.")
    parser.add_argument("--osrm-url", default=None, help="Override OSRM endpoint prefix.")
    parser.add_argument("--osrm-profile", default="foot", help="OSRM profile.")
    parser.add_argument("--poi-index-root", type=Path, default=None, help="Local OSM POI index root.")
    parser.add_argument("--local-router-config", type=Path, default=None, help="Local OSM router config JSON.")
    parser.add_argument("--local-osrm-url", default=None, help="Fallback local OSRM endpoint.")
    parser.add_argument("--local-valhalla-url", default=None, help="Fallback local Valhalla endpoint.")
    parser.add_argument("--valhalla-costing", default="pedestrian", help="Valhalla costing.")
    parser.add_argument("--tomtom-key", default=None, help="TomTom API key.")
    parser.add_argument("--tomtom-search-url", default=None, help="Override TomTom Search API prefix.")
    parser.add_argument("--tomtom-routing-url", default=None, help="Override TomTom Routing API prefix.")
    return parser.parse_args()


def main() -> None:
    """Run complex GT collection.

    Returns:
        None.
    """

    asyncio.run(collect_complex_gt(parse_args()))


async def collect_complex_gt(args: argparse.Namespace) -> dict[str, Any]:
    r"""Collect complex UrbanNav cases.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Manifest dictionary.

    Multi-stop GT is segment-wise optimal:

    \[
    d_i^*=\arg\min_{d\in C_i(p_{i-1})}\ell(p_{i-1}, d),
    \quad p_i=d_i^*
    \]

    Route-update GT freezes both the original route state and the updated route
    after the instruction change, so evaluation can inspect the dynamic context.
    """

    paths = _package_paths(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        _unlink_outputs(paths.values())
    if not args.resume and any(path.exists() for path in paths.values()):
        raise RuntimeError(f"{args.output_root} already contains outputs; use --resume or --overwrite")

    config = _load_json(args.config)
    quality_policy = _load_json(args.quality_policy) if args.quality_policy else {}
    _apply_quality_policy_args(args, quality_policy=quality_policy)
    target_map = _load_target_map(args.target_type_map)
    plan = build_global_collection_plan(config, seed=args.seed, origins_per_region=args.origins_per_region)
    rows = _planned_complex_rows(args.instruction_set, plan=plan, seed=args.seed)
    _write_json(paths["plan"], _plan_summary(rows=rows, args=args, config=config))
    if args.dry_run:
        manifest = _manifest(args, paths, accepted=0, rejected=0, attempted=0, dry_run=True)
        _write_json(paths["manifest"], manifest)
        return manifest

    provider = _create_provider(args, config)
    if args.resume:
        _dedupe_outputs(paths)
    existing_ids = _existing_case_ids(paths["task_gt"]) if args.resume else set()
    scheduled_ids = set(existing_ids)
    accepted_counts = Counter(_existing_case_categories(paths["task_gt"])) if args.resume else Counter()
    rejected = _count_jsonl(paths["task_rejects"]) if args.resume else 0
    attempted = 0
    targets = {"multi_stop_route": int(args.target_multistop), "route_update_add_via": int(args.target_route_update)}

    row_iter = iter(rows)
    pending: set[asyncio.Task[dict[str, Any]]] = set()
    exhausted = False

    def next_row() -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Return the next eligible origin/row pair.

        Returns:
            ``(origin, row)`` or ``None`` when no more work is available.
        """

        nonlocal attempted, exhausted
        while True:
            if args.max_attempts and attempted >= args.max_attempts:
                exhausted = True
                return None
            try:
                origin, row = next(row_iter)
            except StopIteration:
                exhausted = True
                return None
            category = _canonical_complex_category(row)
            if accepted_counts[category] >= targets.get(category, 0):
                continue
            case_id = str(row.get("instruction_id") or row.get("id") or "")
            if not case_id or case_id in scheduled_ids:
                continue
            scheduled_ids.add(case_id)
            attempted += 1
            return origin, row

    await provider.connect()
    try:
        while not _targets_met(accepted_counts, targets) and (pending or not exhausted):
            while len(pending) < max(1, int(args.case_concurrency)) and not _targets_met(accepted_counts, targets) and not exhausted:
                item = next_row()
                if item is None:
                    break
                origin, row = item
                pending.add(
                    asyncio.create_task(
                        _build_complex_case_with_throttle(
                            provider=provider,
                            origin_record=origin,
                            row=row,
                            target_map=target_map,
                            config=config,
                            args=args,
                        )
                    )
                )
            if not pending:
                break
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for finished in done:
                result = finished.result()
                if result["case"] is None:
                    _append_jsonl(paths["task_rejects"], [result["reject"]])
                    _append_jsonl(paths["poi_cache"], [record.to_dict() for record in result.get("poi_records", [])])
                    _append_jsonl(paths["route_cache"], [record.to_dict() for record in result.get("route_records", [])])
                    rejected += 1
                    continue
                case = result["case"]
                category = _canonical_complex_category(case)
                if accepted_counts[category] >= targets.get(category, 0):
                    continue
                _append_jsonl(paths["task_gt"], [case])
                _append_jsonl(paths["poi_cache"], [record.to_dict() for record in result["poi_records"]])
                _append_jsonl(paths["route_cache"], [record.to_dict() for record in result["route_records"]])
                accepted_counts[category] += 1
                if sum(accepted_counts.values()) % 25 == 0:
                    _write_json(
                        paths["manifest"],
                        _manifest(args, paths, accepted=sum(accepted_counts.values()), rejected=rejected, attempted=attempted),
                    )
    finally:
        await provider.disconnect()

    _dedupe_outputs(paths)
    validation = _validate_package(paths)
    manifest = _manifest(
        args,
        paths,
        accepted=_count_jsonl(paths["task_gt"]),
        rejected=_count_jsonl(paths["task_rejects"]),
        attempted=attempted,
        validation=validation,
    )
    _write_json(paths["manifest"], manifest)
    return manifest


async def _build_complex_case_with_throttle(
    *,
    provider: UrbanNavLiveProvider,
    origin_record: dict[str, Any],
    row: dict[str, Any],
    target_map: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build one complex case and apply throttling.

    Args:
        provider: Connected provider.
        origin_record: Origin metadata.
        row: Instruction-set row.
        target_map: Target ontology map.
        config: Raw collection config.
        args: Parsed arguments.

    Returns:
        Build result.
    """

    try:
        if _canonical_complex_category(row) == "multi_stop_route":
            return await _build_multistop_case(provider=provider, origin_record=origin_record, row=row, target_map=target_map, config=config, args=args)
        return await _build_route_update_case(provider=provider, origin_record=origin_record, row=row, target_map=target_map, config=config, args=args)
    finally:
        await asyncio.sleep(max(0.0, float(args.throttle_s)))


async def _build_multistop_case(
    *,
    provider: UrbanNavLiveProvider,
    origin_record: dict[str, Any],
    row: dict[str, Any],
    target_map: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    r"""Build one ordered multi-stop case.

    Args:
        provider: Connected provider.
        origin_record: Origin metadata.
        row: Instruction-set row with ordered target groups.
        target_map: Target ontology map.
        config: Raw config.
        args: Parsed arguments.

    Returns:
        Build result with case and replay cache records, or a reject result.
    """

    case_id = str(row.get("instruction_id") or "")
    origin = UrbanNavPoint(lat=float(origin_record["lat"]), lng=float(origin_record["lng"]))
    groups = row.get("ordered_target_types")
    if not isinstance(groups, list) or len(groups) < 2:
        return _reject_result(case_id, origin_record, row, "missing_ordered_target_types")
    if _target_groups_overlap(groups):
        return _reject_result(case_id, origin_record, row, "overlapping_ordered_target_types")
    goal_labels = [str(item).strip() for item in row.get("ordered_goal_labels") or [] if str(item).strip()]
    if goal_labels and (len(goal_labels) != len(groups) or len(set(label.lower() for label in goal_labels)) != len(goal_labels)):
        return _reject_result(case_id, origin_record, row, "invalid_ordered_goal_labels")

    current = origin
    full_route: list[UrbanNavPoint] = [origin]
    sequence: list[UrbanNavDestination] = []
    selected: list[UrbanNavDestination] = []
    legs: list[dict[str, Any]] = []
    poi_records: list[ReplayPoiResult] = []
    route_records: list[ReplayRouteResult] = []
    total_distance = 0.0
    previous_ranked: list[UrbanNavDestination] = []

    for stop_index, raw_group in enumerate(groups, start=1):
        goal = _goal_from_target_types(raw_group, target_map=target_map)
        if not goal["osm_tags"]:
            return _reject_result(case_id, origin_record, row, f"missing_provider_tags_at_stop_{stop_index}", poi_records=poi_records, route_records=route_records)
        ranked, poi_record, stop_routes = await _search_rank_goal(
            provider=provider,
            origin=current,
            goal=goal,
            origin_record=origin_record,
            case_id=case_id,
            goal_key=f"stop{stop_index:02d}",
            excluded_destinations=selected,
            min_separation_m=float(args.min_inter_stop_distance_m or 0.0),
            config=config,
            args=args,
        )
        poi_records.append(poi_record)
        if previous_ranked:
            ranked = _filter_candidates_against_predecessors(
                ranked,
                previous_ranked,
                min_separation_m=float(args.min_inter_stop_distance_m or 0.0),
            )
        if not ranked:
            return _reject_result(
                case_id,
                origin_record,
                row,
                _goal_failure_reason(poi_record, stop_routes, goal_label=f"stop_{stop_index}"),
                poi_records=poi_records,
                route_records=route_records,
            )
        leg_route_records = list(stop_routes)
        route_matrix: dict[str, Any] | None = None
        if previous_ranked and args.multi_stop_cache_policy == "full_route_matrix":
            matrix_records = await _plan_route_matrix(
                provider=provider,
                predecessor_candidates=previous_ranked,
                destination_candidates=ranked,
                case_id=case_id,
                stop_index=stop_index,
                retry=int(args.retry),
                route_concurrency=int(args.route_concurrency),
            )
            if any(not _route_record_is_valid(record) for record in matrix_records):
                return _reject_result(
                    case_id,
                    origin_record,
                    row,
                    f"incomplete_route_matrix_at_stop_{stop_index}",
                    poi_records=poi_records,
                    route_records=[*route_records, *matrix_records],
                )
            leg_route_records = matrix_records
            route_matrix = {
                "policy": "full_route_matrix",
                "predecessor_candidate_ids": [_destination_key(item) for item in previous_ranked],
                "destination_candidate_ids": [_destination_key(item) for item in ranked],
                "route_keys": [record.cache_key for record in matrix_records],
            }
        route_records.extend(leg_route_records)
        destination = ranked[0]
        if destination.point is None:
            return _reject_result(case_id, origin_record, row, f"selected_destination_missing_point_at_stop_{stop_index}", poi_records=poi_records, route_records=route_records)
        best_distance = float(destination.reference_route_distance_m or destination.route_distance_m or 0.0)
        if best_distance < float(args.min_leg_route_distance_m or 0.0):
            return _reject_result(case_id, origin_record, row, f"leg_route_too_short_at_stop_{stop_index}", poi_records=poi_records, route_records=route_records)
        acceptable = _acceptable(ranked, ratio=float(_setting(args.acceptable_route_ratio, config, "acceptable_route_ratio", 1.15)))
        full_route = _append_route(full_route, destination.reference_route)
        total_distance += best_distance
        sequence.append(destination)
        selected.append(destination)
        legs.append(
            {
                "leg_index": stop_index,
                "target_types": list(raw_group),
                "query_terms": goal["query_terms"],
                "osm_tags": goal["osm_tags"],
                "origin": current.to_dict(),
                "gt_destination": destination.to_dict(),
                "acceptable_destinations": [item.to_dict() for item in acceptable],
                "ranked_destinations": [item.to_dict() for item in ranked],
                "route": [point.to_dict() for point in destination.reference_route],
                "distance_m": best_distance,
                "cache_keys": {
                    "poi_search": [poi_record.cache_key],
                    "routes": [record.cache_key for record in leg_route_records],
                },
                "route_matrix": route_matrix,
            }
        )
        current = destination.point
        previous_ranked = ranked

    case = {
        "schema_version": "urbannav_case_v1",
        "id": case_id,
        "city": origin_record["city"],
        "scene_id": origin_record["region_id"],
        "category": "multi_stop_route",
        "subcategory": row.get("subcategory"),
        "difficulty": _difficulty(total_distance, full_route),
        "instruction": row.get("instruction"),
        "origin": origin.to_dict(),
        "gt_destination": sequence[-1].to_dict(),
        "gt_destination_sequence": [destination.to_dict() for destination in sequence],
        "gt_route_legs": legs,
        "acceptable_destinations": legs[-1]["acceptable_destinations"],
        "ranked_destinations": legs[-1]["ranked_destinations"],
        "gt_route": [point.to_dict() for point in full_route],
        "gt_distance_m": total_distance,
        "optimality_scope": _optimality_scope(provider=provider, config=config, args=args, cost="sum_segment_route_distance_m", route_strategy="ordered_segmentwise_shortest_route"),
        "cache_keys": {
            "poi_search": [record.cache_key for record in poi_records],
            "routes": [record.cache_key for record in route_records],
        },
        "metadata": {
            "country": origin_record.get("country"),
            "locale": origin_record.get("locale"),
            "instruction_id": row.get("instruction_id"),
            "task_id": row.get("task_id"),
            "goal_count": len(sequence),
            "ordered_target_types": groups,
            "ordered_goal_labels": goal_labels,
            "multi_stop_cache_policy": args.multi_stop_cache_policy,
            "min_inter_stop_distance_m": float(args.min_inter_stop_distance_m or 0.0),
            "min_leg_route_distance_m": float(args.min_leg_route_distance_m or 0.0),
            "max_leg_route_distance_m": float(args.max_leg_route_distance_m or 0.0),
            "acceptance_policy": row.get("acceptance_policy"),
            "route_features": _route_features(full_route, total_distance),
        },
    }
    return {"case": case, "poi_records": poi_records, "route_records": route_records, "reject": None}


async def _build_route_update_case(
    *,
    provider: UrbanNavLiveProvider,
    origin_record: dict[str, Any],
    row: dict[str, Any],
    target_map: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    r"""Build one route-update case.

    Args:
        provider: Connected provider.
        origin_record: Origin metadata.
        row: Instruction-set route-update row.
        target_map: Target ontology map.
        config: Raw config.
        args: Parsed arguments.

    Returns:
        Build result with dynamic-route metadata.
    """

    case_id = str(row.get("instruction_id") or "")
    original_origin = UrbanNavPoint(lat=float(origin_record["lat"]), lng=float(origin_record["lng"]))
    update_action = _route_update_action(row)
    previous_goal = _goal_from_target_types(previous_target_types_for_update(row), target_map=target_map)
    update_goal = _goal_from_target_types(row.get("target_types") or [], target_map=target_map)
    if not previous_goal["osm_tags"] or not update_goal["osm_tags"]:
        return _reject_result(case_id, origin_record, row, "missing_previous_or_update_goal_tags")

    poi_records: list[ReplayPoiResult] = []
    route_records: list[ReplayRouteResult] = []
    previous_ranked, previous_poi, previous_routes = await _search_rank_goal(
        provider=provider,
        origin=original_origin,
        goal=previous_goal,
        origin_record=origin_record,
        case_id=case_id,
        goal_key="previous",
        excluded_destinations=[],
        config=config,
        args=args,
    )
    poi_records.append(previous_poi)
    route_records.extend(previous_routes)
    if not previous_ranked:
        return _reject_result(
            case_id,
            origin_record,
            row,
            _goal_failure_reason(previous_poi, previous_routes, goal_label="previous_destination"),
            poi_records=poi_records,
            route_records=route_records,
        )

    previous_destination = previous_ranked[0]
    old_route = list(previous_destination.reference_route)
    if len(old_route) < 2:
        return _reject_result(case_id, origin_record, row, "old_route_too_short", poi_records=poi_records, route_records=route_records)
    current_origin = _route_point_at_ratio(old_route, 0.45)

    update_ranked, update_poi, update_routes = await _search_rank_goal(
        provider=provider,
        origin=current_origin,
        goal=update_goal,
        origin_record=origin_record,
        case_id=case_id,
        goal_key="update",
        excluded_destinations=[previous_destination] if update_action == "replace_destination" else [],
        config=config,
        args=args,
    )
    poi_records.append(update_poi)
    route_records.extend(update_routes)
    if not update_ranked:
        return _reject_result(
            case_id,
            origin_record,
            row,
            _goal_failure_reason(update_poi, update_routes, goal_label="update_destination"),
            poi_records=poi_records,
            route_records=route_records,
        )

    updated_destination = update_ranked[0]
    acceptable = _acceptable(update_ranked, ratio=float(_setting(args.acceptable_route_ratio, config, "acceptable_route_ratio", 1.15)))
    updated_route = list(updated_destination.reference_route)
    updated_distance = float(updated_destination.reference_route_distance_m or updated_destination.route_distance_m or 0.0)
    via_destination = updated_destination if update_action in {"add_via", "insert_before_final"} else None
    final_destination = previous_destination if via_destination is not None else updated_destination
    via_to_final_record: ReplayRouteResult | None = None

    if via_destination is not None:
        if via_destination.point is None:
            return _reject_result(case_id, origin_record, row, "via_destination_missing_point", poi_records=poi_records, route_records=route_records)
        via_to_final_record = await _plan_route(
            provider=provider,
            origin=via_destination.point,
            destination=previous_destination,
            case_id=case_id,
            route_key="via_to_final",
            retry=int(args.retry),
        )
        route_records.append(via_to_final_record)
        if via_to_final_record.distance_m is None or via_to_final_record.distance_m <= 0 or not via_to_final_record.polyline:
            return _reject_result(case_id, origin_record, row, "missing_route_from_via_to_final", poi_records=poi_records, route_records=route_records)
        if float(args.max_leg_route_distance_m or 0.0) > 0 and via_to_final_record.distance_m > float(args.max_leg_route_distance_m):
            return _reject_result(case_id, origin_record, row, "via_to_final_route_too_long", poi_records=poi_records, route_records=route_records)
        updated_route = _append_route(updated_route, via_to_final_record.polyline)
        updated_distance += float(via_to_final_record.distance_m)

    case = {
        "schema_version": "urbannav_case_v1",
        "id": case_id,
        "city": origin_record["city"],
        "scene_id": origin_record["region_id"],
        "category": "route_update_add_via",
        "subcategory": row.get("subcategory"),
        "difficulty": _difficulty(updated_distance, updated_route),
        "instruction": row.get("instruction"),
        "origin": current_origin.to_dict(),
        "gt_destination": updated_destination.to_dict(),
        "acceptable_destinations": [destination.to_dict() for destination in acceptable],
        "ranked_destinations": [destination.to_dict() for destination in update_ranked],
        "gt_route": [point.to_dict() for point in updated_route],
        "gt_distance_m": updated_distance,
        "optimality_scope": _optimality_scope(
            provider=provider,
            config=config,
            args=args,
            cost="updated_route_distance_m",
            route_strategy="route_update_current_state_shortest_route",
        ),
        "cache_keys": {
            "poi_search": [record.cache_key for record in poi_records],
            "routes": [record.cache_key for record in route_records],
        },
        "metadata": {
            "country": origin_record.get("country"),
            "locale": origin_record.get("locale"),
            "instruction_id": row.get("instruction_id"),
            "task_id": row.get("task_id"),
            "acceptance_policy": row.get("acceptance_policy"),
            "route_update": {
                "update_type": update_action,
                "original_origin": original_origin.to_dict(),
                "current_origin": current_origin.to_dict(),
                "current_position_ratio": 0.45,
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


async def _search_rank_goal(
    *,
    provider: UrbanNavLiveProvider,
    origin: UrbanNavPoint,
    goal: dict[str, Any],
    origin_record: dict[str, Any],
    case_id: str,
    goal_key: str,
    excluded_destinations: list[UrbanNavDestination],
    min_separation_m: float = 0.0,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[UrbanNavDestination], ReplayPoiResult, list[ReplayRouteResult]]:
    """Search and rank routeable candidates for one goal.

    Args:
        provider: Connected provider.
        origin: Current origin for this segment.
        goal: Goal query/tags dictionary.
        origin_record: Origin metadata.
        case_id: Case id.
        goal_key: Stable goal key for cache ids.
        excluded_destinations: Previously selected destinations.
        min_separation_m: Minimum distance to every previously selected stop.
        config: Raw config.
        args: Parsed arguments.

    Returns:
        Ranked destinations, POI replay record, and route replay records.
    """

    radius_m = int(_setting(args.search_radius_m, config, "search_radius_m", 1200))
    limit = int(_setting(args.candidate_limit, config, "candidate_limit", 5))
    poi_key = f"poi:{case_id}:{goal_key}"
    poi_record = await _with_provider_retry(
        lambda: provider.search_pois(
            cache_key=poi_key,
            request=UrbanNavPoiSearchRequest(
                origin=origin,
                query=str((goal.get("query_terms") or [""])[0]),
                tags=list(goal.get("osm_tags") or []),
                radius_m=radius_m,
                limit=limit,
                locale=str(origin_record.get("locale") or ""),
                raw={
                    "country": origin_record.get("country"),
                    "city": origin_record.get("city"),
                    "region_id": origin_record.get("region_id"),
                    "target_types": list(goal.get("target_types") or []),
                },
            ),
        ),
        retry=int(args.retry),
        label=f"poi:{case_id}:{goal_key}",
    )
    candidates = _dedupe_destinations(poi_record.candidates)
    route_records = await _plan_candidate_routes(
        provider=provider,
        origin=origin,
        candidates=candidates[:limit],
        case_id=case_id,
        goal_key=goal_key,
        retry=int(args.retry),
        route_concurrency=int(args.route_concurrency),
    )
    ranked: list[UrbanNavDestination] = []
    max_distance_m = float(args.max_leg_route_distance_m or 0.0)
    for candidate_index, candidate, route_record in route_records:
        if _is_repeated_destination(candidate, excluded_destinations, min_separation_m=min_separation_m):
            continue
        if route_record.distance_m is None or route_record.distance_m <= 0 or not route_record.polyline:
            continue
        if max_distance_m > 0 and route_record.distance_m > max_distance_m:
            continue
        ranked.append(_destination_with_route(candidate, route_record, provider_rank=candidate_index))
    ranked.sort(key=lambda item: (item.reference_route_distance_m or float("inf"), item.name or ""))
    return [_with_rank(destination, rank=rank) for rank, destination in enumerate(ranked, start=1)], poi_record, [record for _, _, record in route_records]


async def _plan_candidate_routes(
    *,
    provider: UrbanNavLiveProvider,
    origin: UrbanNavPoint,
    candidates: list[UrbanNavDestination],
    case_id: str,
    goal_key: str,
    retry: int,
    route_concurrency: int,
) -> list[tuple[int, UrbanNavDestination, ReplayRouteResult]]:
    """Plan routes for one candidate set.

    Args:
        provider: Connected provider.
        origin: Route origin.
        candidates: Candidate destinations.
        case_id: Case id.
        goal_key: Goal key for cache ids.
        retry: Retry count.
        route_concurrency: Bounded route concurrency.

    Returns:
        Candidate index, destination, and route replay record tuples.
    """

    semaphore = asyncio.Semaphore(max(1, route_concurrency))

    async def run_one(candidate_index: int, candidate: UrbanNavDestination) -> tuple[int, UrbanNavDestination, ReplayRouteResult]:
        route_key = f"{goal_key}:{candidate_index:03d}"
        async with semaphore:
            route_record = await _plan_route(provider=provider, origin=origin, destination=candidate, case_id=case_id, route_key=route_key, retry=retry)
        return candidate_index, candidate, route_record

    tasks = [asyncio.create_task(run_one(index, candidate)) for index, candidate in enumerate(candidates, start=1)]
    return sorted(await asyncio.gather(*tasks), key=lambda item: item[0])


async def _plan_route_matrix(
    *,
    provider: UrbanNavLiveProvider,
    predecessor_candidates: list[UrbanNavDestination],
    destination_candidates: list[UrbanNavDestination],
    case_id: str,
    stop_index: int,
    retry: int,
    route_concurrency: int,
) -> list[ReplayRouteResult]:
    """Plan every predecessor-to-candidate transition for one later stop.

    Args:
        provider: Connected route provider.
        predecessor_candidates: Frozen candidate set for the previous stop.
        destination_candidates: Frozen candidate set for the current stop.
        case_id: Case identifier.
        stop_index: One-based current stop index.
        retry: Provider retry count.
        route_concurrency: Maximum concurrent route calls.

    Returns:
        Matrix route records in predecessor-major order.
    """

    semaphore = asyncio.Semaphore(max(1, route_concurrency))

    async def run_one(
        predecessor: UrbanNavDestination,
        destination: UrbanNavDestination,
    ) -> ReplayRouteResult:
        if predecessor.point is None:
            return ReplayRouteResult(
                cache_key=_matrix_route_key(case_id, stop_index, predecessor, destination),
                provider=provider.capabilities.provider_name,
                request={"origin": None, "destination": destination.to_dict(), "mode": "foot"},
                response={"error": "predecessor_missing_point"},
            )
        async with semaphore:
            return await _plan_route(
                provider=provider,
                origin=predecessor.point,
                destination=destination,
                case_id=case_id,
                route_key=_matrix_route_suffix(stop_index, predecessor, destination),
                retry=retry,
            )

    tasks = [
        asyncio.create_task(run_one(predecessor, destination))
        for predecessor in predecessor_candidates
        for destination in destination_candidates
    ]
    return await asyncio.gather(*tasks)


def _matrix_route_suffix(
    stop_index: int,
    predecessor: UrbanNavDestination,
    destination: UrbanNavDestination,
) -> str:
    """Build the canonical route-matrix cache suffix."""

    return (
        f"stop{stop_index:02d}:from={_destination_key(predecessor)}"
        f":to={_destination_key(destination)}"
    )


def _matrix_route_key(
    case_id: str,
    stop_index: int,
    predecessor: UrbanNavDestination,
    destination: UrbanNavDestination,
) -> str:
    """Build a complete route-matrix cache key."""

    return f"route:{case_id}:{_matrix_route_suffix(stop_index, predecessor, destination)}"


def _route_record_is_valid(record: ReplayRouteResult) -> bool:
    """Return whether one replay route contains usable geometry."""

    return record.distance_m is not None and record.distance_m > 0 and bool(record.polyline)


def _goal_failure_reason(
    poi_record: ReplayPoiResult,
    route_records: list[ReplayRouteResult],
    *,
    goal_label: str,
) -> str:
    """Classify why a searched goal produced no ranked destination.

    Args:
        poi_record: Frozen POI search response.
        route_records: Candidate route responses.
        goal_label: Stable goal label included in the reason.

    Returns:
        Precise rejection reason separating retrieval, routing, and filtering.
    """

    if not poi_record.candidates:
        return f"no_poi_candidates_at_{goal_label}"
    if not any(_route_record_is_valid(record) for record in route_records):
        return f"no_valid_routes_at_{goal_label}"
    return f"all_candidates_filtered_at_{goal_label}"


async def _plan_route(
    *,
    provider: UrbanNavLiveProvider,
    origin: UrbanNavPoint,
    destination: UrbanNavDestination,
    case_id: str,
    route_key: str,
    retry: int,
) -> ReplayRouteResult:
    """Plan one route with replay-compatible failure records.

    Args:
        provider: Connected provider.
        origin: Route origin.
        destination: Route destination.
        case_id: Case id.
        route_key: Route cache suffix.
        retry: Retry count.

    Returns:
        Route replay record.
    """

    cache_key = f"route:{case_id}:{route_key}"
    try:
        return await _with_provider_retry(
            lambda: provider.plan_route(
                cache_key=cache_key,
                request=UrbanNavRouteRequest(origin=origin, destination=destination, mode="foot"),
            ),
            retry=retry,
            label=cache_key,
        )
    except Exception as exc:
        return ReplayRouteResult(
            cache_key=cache_key,
            provider=provider.capabilities.provider_name,
            request={"origin": origin.to_dict(), "destination": destination.to_dict(), "mode": "foot"},
            response={"error": str(exc)},
        )


def _planned_complex_rows(
    instruction_set: Path,
    *,
    plan: Any,
    seed: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Assign complex instruction rows to sampled origins.

    Args:
        instruction_set: Compiled instruction JSONL.
        plan: Global collection plan.
        seed: Deterministic seed.

    Returns:
        Origin/row pairs.
    """

    rng = random.Random(seed)
    origins_by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for origin in plan.origins:
        origins_by_city[_norm_city(str(origin["city"]))].append(origin)
    for origins in origins_by_city.values():
        rng.shuffle(origins)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in iter_json_records(instruction_set):
        category = _canonical_complex_category(row)
        if category not in {"multi_stop_route", "route_update_add_via"}:
            continue
        planned_city = row.get("planned_city") if isinstance(row.get("planned_city"), dict) else {}
        city_key = _norm_city(str(planned_city.get("city") or row.get("city") or ""))
        origins = origins_by_city.get(city_key)
        if not origins:
            continue
        index = int(row.get("global_index") or 0)
        pairs.append((origins[index % len(origins)], row))
    rng.shuffle(pairs)
    return pairs


def _goal_from_target_types(target_types: Iterable[Any], *, target_map: dict[str, Any]) -> dict[str, Any]:
    """Convert target types to provider-searchable goal metadata.

    Args:
        target_types: Target type names.
        target_map: Loaded target ontology map.

    Returns:
        Goal dictionary with query terms and OSM tags.
    """

    query_terms: list[str] = []
    osm_tags: list[str] = []
    clean_types: list[str] = []
    specs = target_map.get("target_types") if isinstance(target_map.get("target_types"), dict) else {}
    for raw_type in target_types:
        target_type = str(raw_type).strip()
        if not target_type:
            continue
        clean_types.append(target_type)
        spec = specs.get(target_type) if isinstance(specs, dict) else None
        if not isinstance(spec, dict):
            query_terms.append(target_type.replace("_", " "))
            continue
        query_terms.extend(str(item) for item in spec.get("query_terms") or [])
        osm_tags.extend(str(item) for item in spec.get("osm_tags") or [])
    return {
        "target_types": _dedupe_strings(clean_types),
        "query_terms": _dedupe_strings(query_terms) or [clean_types[0].replace("_", " ") if clean_types else ""],
        "osm_tags": _dedupe_strings(osm_tags),
    }


def _route_update_action(row: dict[str, Any]) -> str:
    """Infer the route-update action.

    Args:
        row: Route-update instruction row.

    Returns:
        One of ``add_via``, ``insert_before_final``, or ``replace_destination``.
    """

    action = str(row.get("route_update_action") or "").strip()
    if action:
        return action
    subcategory = str(row.get("subcategory") or "")
    if subcategory.startswith("replace_destination"):
        return "replace_destination"
    if subcategory.startswith("insert_before_final"):
        return "insert_before_final"
    return "add_via"


def _acceptable(ranked: list[UrbanNavDestination], *, ratio: float) -> list[UrbanNavDestination]:
    """Return near-tie acceptable destinations.

    Args:
        ranked: Ranked routeable destinations.
        ratio: Route distance near-tie ratio.

    Returns:
        Acceptable destination list.
    """

    if not ranked:
        return []
    best = ranked[0].reference_route_distance_m or ranked[0].route_distance_m or 0.0
    return [item for item in ranked if (item.reference_route_distance_m or float("inf")) <= best * max(1.0, ratio)] or [ranked[0]]


def _is_repeated_destination(
    candidate: UrbanNavDestination,
    selected: list[UrbanNavDestination],
    *,
    min_separation_m: float = 0.0,
) -> bool:
    """Check whether a candidate repeats a previous stop.

    Args:
        candidate: Candidate destination.
        selected: Previously selected destinations.
        min_separation_m: Minimum allowed candidate-to-selected distance.

    Returns:
        ``True`` when ids match or coordinates are nearly identical.
    """

    point = candidate.point
    for existing in selected:
        if candidate.stable_ids and candidate.stable_ids.intersection(existing.stable_ids):
            return True
        existing_point = existing.point
        if point is None or existing_point is None:
            continue
        if _haversine_m(point, existing_point) < max(2.0, min_separation_m):
            return True
    return False


def _filter_candidates_against_predecessors(
    candidates: list[UrbanNavDestination],
    predecessors: list[UrbanNavDestination],
    *,
    min_separation_m: float,
) -> list[UrbanNavDestination]:
    """Remove current-stop candidates that overlap any predecessor branch."""

    filtered = [
        candidate
        for candidate in candidates
        if not _is_repeated_destination(
            candidate,
            predecessors,
            min_separation_m=min_separation_m,
        )
    ]
    return [_with_rank(item, rank=index) for index, item in enumerate(filtered, start=1)]


def _destination_key(destination: UrbanNavDestination) -> str:
    """Return a cache-key-safe stable destination identifier."""

    identifiers = sorted(destination.stable_ids)
    if identifiers:
        raw = identifiers[0]
    elif destination.point is not None:
        raw = f"coord_{destination.point.lat:.6f}_{destination.point.lng:.6f}"
    else:
        raw = destination.name or "unknown"
    return "".join(character if character.isalnum() or character in {"_", "-", "."} else "_" for character in raw)


def _target_groups_overlap(groups: list[Any]) -> bool:
    """Return whether ordered target-type groups overlap semantically."""

    normalized = [
        {str(item).strip().lower() for item in group if str(item).strip()}
        for group in groups
        if isinstance(group, list)
    ]
    for left in range(len(normalized)):
        for right in range(left + 1, len(normalized)):
            if normalized[left].intersection(normalized[right]):
                return True
    return False


def _apply_quality_policy_args(args: argparse.Namespace, *, quality_policy: dict[str, Any]) -> None:
    """Resolve optional CLI quality settings from the v2 policy."""

    multi = quality_policy.get("multi_stop") if isinstance(quality_policy.get("multi_stop"), dict) else {}
    if args.min_inter_stop_distance_m is None:
        args.min_inter_stop_distance_m = float(multi.get("min_inter_stop_distance_m") or 0.0)
    if args.min_leg_route_distance_m is None:
        args.min_leg_route_distance_m = float(multi.get("min_leg_route_distance_m") or 0.0)
    if args.max_leg_route_distance_m is None:
        args.max_leg_route_distance_m = float(multi.get("max_leg_route_distance_m") or 0.0)
    if args.multi_stop_cache_policy == "gt_path" and multi.get("cache_policy") == "full_route_matrix":
        args.multi_stop_cache_policy = "full_route_matrix"


def _append_route(base: list[UrbanNavPoint], leg: list[UrbanNavPoint]) -> list[UrbanNavPoint]:
    """Append one route leg.

    Args:
        base: Existing route.
        leg: Leg to append.

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


def _route_point_at_ratio(route: list[UrbanNavPoint], ratio: float) -> UrbanNavPoint:
    """Pick a route point at fractional progress.

    Args:
        route: Route polyline.
        ratio: Fraction in ``[0, 1]``.

    Returns:
        Route point.
    """

    if not route:
        raise ValueError("route must not be empty")
    clean_ratio = min(0.95, max(0.05, ratio))
    index = min(len(route) - 1, max(0, round((len(route) - 1) * clean_ratio)))
    return route[index]


def _route_features(route: list[UrbanNavPoint], distance_m: float) -> dict[str, Any]:
    """Compute route audit features.

    Args:
        route: Route polyline.
        distance_m: Route distance.

    Returns:
        Feature dictionary.
    """

    direct = _haversine_m(route[0], route[-1]) if len(route) >= 2 else 0.0
    return {
        "route_distance_m": round(distance_m, 3),
        "direct_distance_m": round(direct, 3),
        "directness": round(direct / distance_m, 4) if distance_m > 0 else None,
        "polyline_points": len(route),
    }


async def _with_provider_retry(factory: Any, *, retry: int, label: str) -> Any:
    """Run one provider call with retries.

    Args:
        factory: Function returning an awaitable.
        retry: Retry count.
        label: Error label.

    Returns:
        Awaited provider result.
    """

    last_error: Exception | None = None
    for attempt in range(max(1, retry + 1)):
        try:
            return await factory()
        except Exception as exc:
            last_error = exc
            if attempt < retry:
                await asyncio.sleep(min(5.0, 0.75 * (2**attempt)))
    raise RuntimeError(f"{label} failed after {retry + 1} attempts: {last_error}") from last_error


def _optimality_scope(
    *,
    provider: UrbanNavLiveProvider,
    config: dict[str, Any],
    args: argparse.Namespace,
    cost: str,
    route_strategy: str,
) -> dict[str, Any]:
    """Build optimality-scope metadata.

    Args:
        provider: Active provider.
        config: Raw config.
        args: Parsed arguments.
        cost: Cost optimized by this case builder.
        route_strategy: Route construction strategy.

    Returns:
        Optimality-scope dictionary.
    """

    return {
        "poi_provider": provider.capabilities.provider_name,
        "search_radius_m": int(_setting(args.search_radius_m, config, "search_radius_m", 1200)),
        "candidate_limit": int(_setting(args.candidate_limit, config, "candidate_limit", 5)),
        "route_provider": provider.capabilities.provider_name,
        "route_mode": "foot",
        "route_strategy": route_strategy,
        "cost": cost,
        "builder_version": BUILDER_VERSION,
        "base_builder_version": GLOBAL_BUILDER_VERSION,
    }


def _reject_result(
    case_id: str,
    origin_record: dict[str, Any],
    row: dict[str, Any],
    reason: str,
    *,
    poi_records: list[ReplayPoiResult] | None = None,
    route_records: list[ReplayRouteResult] | None = None,
) -> dict[str, Any]:
    """Create a rejected complex case result.

    Args:
        case_id: Case id.
        origin_record: Origin metadata.
        row: Instruction row.
        reason: Reject reason.
        poi_records: Partial POI records.
        route_records: Partial route records.

    Returns:
        Result dictionary.
    """

    return {
        "case": None,
        "poi_records": poi_records or [],
        "route_records": route_records or [],
        "reject": {
            "id": case_id,
            "reject_reason": reason,
            "reason": reason,
            "city": origin_record.get("city"),
            "scene_id": origin_record.get("region_id"),
            "origin": {"lat": origin_record.get("lat"), "lng": origin_record.get("lng")},
            "category": _canonical_complex_category(row),
            "subcategory": row.get("subcategory"),
            "instruction": row.get("instruction"),
        },
    }


def _canonical_complex_category(row: dict[str, Any]) -> str:
    """Normalize complex categories.

    Args:
        row: Instruction row or case dictionary.

    Returns:
        Canonical category.
    """

    category = str(row.get("category") or "")
    if category.startswith("route_update"):
        return "route_update_add_via"
    return category


def _targets_met(counts: Counter[str], targets: dict[str, int]) -> bool:
    """Check whether all target counts have been reached.

    Args:
        counts: Accepted counts.
        targets: Desired counts.

    Returns:
        ``True`` when all targets are satisfied.
    """

    return all(counts[key] >= value for key, value in targets.items())


def _package_paths(root: Path) -> dict[str, Path]:
    """Return output paths.

    Args:
        root: Output root.

    Returns:
        Path mapping.
    """

    return {
        "task_gt": root / "task_gt.jsonl",
        "task_rejects": root / "task_rejects.jsonl",
        "poi_cache": root / "poi_cache.jsonl",
        "route_cache": root / "route_cache.jsonl",
        "validation_report": root / "validation_report.json",
        "manifest": root / "manifest.json",
        "plan": root / "collection_plan.json",
    }


def _validate_package(paths: dict[str, Path]) -> dict[str, Any]:
    """Validate output package.

    Args:
        paths: Package path mapping.

    Returns:
        Validation report dictionary.
    """

    cases = load_urbannav_cases(paths["task_gt"]) if paths["task_gt"].exists() else []
    poi_keys = {record.get("cache_key") for record in iter_json_records(paths["poi_cache"])} if paths["poi_cache"].exists() else set()
    route_keys = {record.get("cache_key") for record in iter_json_records(paths["route_cache"])} if paths["route_cache"].exists() else set()
    rejects = list(iter_json_records(paths["task_rejects"])) if paths["task_rejects"].exists() else []
    report = validate_urbannav_cases(cases, poi_cache_keys=set(poi_keys), route_cache_keys=set(route_keys), rejects=rejects)
    paths["validation_report"].write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return report.to_dict()


def _manifest(
    args: argparse.Namespace,
    paths: dict[str, Path],
    *,
    accepted: int,
    rejected: int,
    attempted: int,
    dry_run: bool = False,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build manifest metadata.

    Args:
        args: Parsed arguments.
        paths: Package paths.
        accepted: Accepted count.
        rejected: Rejected count.
        attempted: Attempted count.
        dry_run: Whether this is a dry run.
        validation: Optional validation report.

    Returns:
        Manifest dictionary.
    """

    return {
        "builder_version": BUILDER_VERSION,
        "created_at_unix": time.time(),
        "dry_run": dry_run,
        "target_multistop": args.target_multistop,
        "target_route_update": args.target_route_update,
        "accepted_cases": accepted,
        "rejected_cases": rejected,
        "attempted_pairs_this_run": attempted,
        "case_concurrency": args.case_concurrency,
        "route_concurrency": args.route_concurrency,
        "throttle_s": args.throttle_s,
        "paths": {name: str(path) for name, path in paths.items()},
        "validation": validation,
    }


def _plan_summary(*, rows: list[tuple[dict[str, Any], dict[str, Any]]], args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Build a collection plan summary.

    Args:
        rows: Planned origin/row pairs.
        args: Parsed arguments.
        config: Raw config.

    Returns:
        Plan summary.
    """

    return {
        "config_version": config.get("version"),
        "instruction_set": str(args.instruction_set),
        "target_multistop": args.target_multistop,
        "target_route_update": args.target_route_update,
        "eligible_rows": len(rows),
        "category_counts": dict(Counter(_canonical_complex_category(row) for _, row in rows)),
        "city_counts": dict(Counter(origin.get("city") for origin, _ in rows)),
    }


def _dedupe_outputs(paths: dict[str, Path]) -> None:
    """Dedupe resumable output files.

    Args:
        paths: Output paths.

    Returns:
        None.
    """

    _dedupe_jsonl(paths["task_gt"], key="id")
    _dedupe_jsonl(paths["poi_cache"], key="cache_key")
    _dedupe_jsonl(paths["route_cache"], key="cache_key")
    _dedupe_jsonl(paths["task_rejects"], key="id")


def _existing_case_ids(path: Path) -> set[str]:
    """Read existing case ids.

    Args:
        path: Task GT path.

    Returns:
        Existing id set.
    """

    if not path.exists():
        return set()
    return {str(record.get("id") or record.get("case_id")) for record in iter_json_records(path)}


def _existing_case_categories(path: Path) -> Iterable[str]:
    """Read existing case categories.

    Args:
        path: Task GT path.

    Returns:
        Category iterable.
    """

    if not path.exists():
        return []
    return [_canonical_complex_category(record) for record in iter_json_records(path)]


def _count_jsonl(path: Path) -> int:
    """Count JSONL records.

    Args:
        path: JSONL path.

    Returns:
        Row count.
    """

    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object.

    Args:
        path: JSON path.

    Returns:
        Parsed object.
    """

    return json.loads(path.read_text(encoding="utf-8"))


def _load_target_map(path: Path) -> dict[str, Any]:
    """Load target ontology map.

    Args:
        path: Target map JSON path.

    Returns:
        Target map dictionary.
    """

    payload = _load_json(path)
    if not isinstance(payload.get("target_types"), dict):
        raise ValueError(f"Invalid target type map: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON.

    Args:
        path: Output path.
        payload: JSON payload.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _dedupe_destinations(candidates: Iterable[UrbanNavDestination]) -> list[UrbanNavDestination]:
    """Dedupe destinations by provider id or coordinate key.

    Args:
        candidates: Candidate sequence.

    Returns:
        Deduplicated candidates.
    """

    output: list[UrbanNavDestination] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.provider_poi_id or f"{candidate.name}:{candidate.lat}:{candidate.lng}"
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    """Dedupe strings while preserving order.

    Args:
        values: String iterable.

    Returns:
        Deduplicated strings.
    """

    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


if __name__ == "__main__":
    main()
