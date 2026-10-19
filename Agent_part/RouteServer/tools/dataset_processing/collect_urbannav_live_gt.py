#!/usr/bin/env python3
"""Collect real UrbanNav GT cases from live map APIs."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

from mcp_server.core.live_map_backend import LiveMapBackend, LiveRouteRequest
from mcp_server.core.replay_map_backend import ReplayPoiResult, ReplayRouteResult
from mcp_server.eval.urbannav.urbannav_io import iter_json_records, load_urbannav_cases
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavCase, UrbanNavDestination, UrbanNavPoint
from mcp_server.eval.urbannav.urbannav_validator import validate_urbannav_cases


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "urbannav_live_gt_collection.json"
BUILDER_VERSION = "urbannav_live_gt_large_v1"


@dataclass(frozen=True)
class CollectionPlan:
    r"""Expanded deterministic collection plan.

    Args:
        config: Raw collection configuration.
        origins: Sampled origin records.
        tasks: Task template records.

    The plan expands region centers into nearby origins so benchmark coverage is
    determined by city/region/task quotas instead of by any unrelated visual
    sampling distribution:

    \[
    \mathcal P = \{(o_i,t_j)\mid o_i \in \operatorname{SampleRegion}(r), t_j \in T\}
    \]
    """

    config: dict[str, Any]
    origins: list[dict[str, Any]]
    tasks: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path, help="Benchmark package output directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Collection config JSON.")
    parser.add_argument("--target-cases", type=int, default=10000, help="Target number of valid GT cases.")
    parser.add_argument("--map-provider", default="amap", help="Live map provider, usually amap.")
    parser.add_argument("--seed", type=int, default=20260622, help="Deterministic sampling seed.")
    parser.add_argument("--origins-per-region", type=int, default=45, help="Maximum sampled origins per region.")
    parser.add_argument("--search-radius-m", type=int, default=None, help="Override POI search radius.")
    parser.add_argument("--candidate-limit", type=int, default=None, help="Override routed candidate count.")
    parser.add_argument("--acceptable-route-ratio", type=float, default=None, help="Override near-tie acceptance ratio.")
    parser.add_argument("--min-route-distance-m", type=float, default=None, help="Override minimum reference distance.")
    parser.add_argument("--max-route-distance-m", type=float, default=None, help="Override maximum reference distance.")
    parser.add_argument("--throttle-s", type=float, default=0.05, help="Sleep seconds between live API calls.")
    parser.add_argument("--retry", type=int, default=2, help="Retries for transient live API errors.")
    parser.add_argument("--case-concurrency", type=int, default=1, help="Concurrent origin-task cases.")
    parser.add_argument("--route-concurrency", type=int, default=3, help="Concurrent route requests inside each case.")
    parser.add_argument("--resume", action="store_true", help="Append to existing package and skip existing case ids.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing output files before collecting.")
    parser.add_argument("--dry-run", action="store_true", help="Only write a collection plan and exit without API calls.")
    parser.add_argument("--max-attempts", type=int, default=0, help="Optional cap on attempted origin-task pairs.")
    parser.add_argument(
        "--categories",
        default="poi_navigation,implicit_need_navigation",
        help="Comma-separated categories to collect.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the live GT collection command.

    Returns:
        None.
    """

    args = parse_args()
    asyncio.run(collect_live_gt(args))


async def collect_live_gt(args: argparse.Namespace) -> dict[str, Any]:
    r"""Collect GT cases with a live map backend.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Manifest dictionary describing the collected package.

    The collector writes each accepted case and replay record immediately. A
    failed run can be resumed because case ids are deterministic:

    \[
    case\_id = h(city, region, origin\_index, task, template\_index)
    \]
    """

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    paths = _package_paths(root)
    if args.overwrite:
        _unlink_outputs(paths.values())
    if not args.resume and any(path.exists() for path in paths.values()):
        raise RuntimeError(f"{root} already contains outputs; use --resume or --overwrite")

    config = _load_config(args.config)
    categories = {item.strip() for item in str(args.categories).split(",") if item.strip()}
    plan = build_collection_plan(
        config,
        seed=args.seed,
        origins_per_region=args.origins_per_region,
        categories=categories,
    )
    _write_json(paths["plan"], _plan_summary(plan, args))
    if args.dry_run:
        manifest = _manifest(args, paths, accepted=0, rejected=0, attempted=0, dry_run=True)
        _write_json(paths["manifest"], manifest)
        return manifest

    existing_ids = _existing_case_ids(paths["task_gt"]) if args.resume else set()
    scheduled_ids = set(existing_ids)
    accepted = len(existing_ids)
    rejected = _count_jsonl(paths["task_rejects"]) if args.resume else 0
    attempted = 0
    provider = LiveMapBackend.from_env(map_provider=args.map_provider)
    await provider.connect()
    try:
        pairs = iter(iter_collection_pairs(plan, seed=args.seed))
        pending: set[asyncio.Task[dict[str, Any]]] = set()
        exhausted = False

        def next_pair() -> tuple[dict[str, Any], dict[str, Any], int] | None:
            nonlocal attempted, exhausted
            while True:
                if args.max_attempts and attempted >= args.max_attempts:
                    exhausted = True
                    return None
                try:
                    origin, task, template_index = next(pairs)
                except StopIteration:
                    exhausted = True
                    return None
                case_id = _case_id(origin, task, template_index)
                if case_id in scheduled_ids:
                    continue
                scheduled_ids.add(case_id)
                attempted += 1
                return origin, task, template_index

        case_concurrency = max(1, int(args.case_concurrency))
        while accepted < args.target_cases and (pending or not exhausted):
            while (
                len(pending) < case_concurrency
                and accepted + len(pending) < args.target_cases
                and not exhausted
            ):
                pair = next_pair()
                if pair is None:
                    break
                origin, task, template_index = pair
                pending.add(
                    asyncio.create_task(
                        _build_live_case_with_throttle(
                            backend=provider,
                            origin_record=origin,
                            task=task,
                            template_index=template_index,
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
                    rejected += 1
                    continue
                case = result["case"]
                case_id = str(case.case_id)
                if case_id in existing_ids or accepted >= args.target_cases:
                    continue
                _append_jsonl(paths["task_gt"], [case.to_dict()])
                _append_jsonl(paths["poi_cache"], [record.to_dict() for record in result["poi_records"]])
                _append_jsonl(paths["route_cache"], [record.to_dict() for record in result["route_records"]])
                existing_ids.add(case_id)
                accepted += 1
                if accepted % 100 == 0:
                    _write_json(paths["manifest"], _manifest(args, paths, accepted, rejected, attempted))
    finally:
        await provider.disconnect()

    validation = _validate_package(paths)
    manifest = _manifest(args, paths, accepted, rejected, attempted, validation=validation)
    _write_json(paths["manifest"], manifest)
    return manifest


async def _build_live_case_with_throttle(
    *,
    backend: LiveMapBackend,
    origin_record: dict[str, Any],
    task: dict[str, Any],
    template_index: int,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build one case and apply case-level throttling.

    Args:
        backend: Connected live map backend.
        origin_record: Origin metadata.
        task: Task template record.
        template_index: Template index.
        config: Collection configuration.
        args: Parsed CLI arguments.

    Returns:
        Case build result.
    """

    try:
        return await build_case_from_live_api(
            backend=backend,
            origin_record=origin_record,
            task=task,
            template_index=template_index,
            config=config,
            args=args,
        )
    finally:
        await asyncio.sleep(max(0.0, args.throttle_s))


async def build_case_from_live_api(
    *,
    backend: LiveMapBackend,
    origin_record: dict[str, Any],
    task: dict[str, Any],
    template_index: int,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    r"""Build one GT case from live POI and route APIs.

    Args:
        backend: Connected live map backend.
        origin_record: Sampled origin with city, region, and coordinate.
        task: Task template record.
        template_index: Template index selected for this case.
        config: Collection configuration.
        args: Parsed CLI arguments.

    Returns:
        Dictionary containing ``case`` plus replay records, or a reject row.

    The selected GT is the route-distance optimum among routeable candidates:

    \[
    d^*=\arg\min_{d\in C_{\mathrm{routeable}}} L(R(o,d))
    \]
    """

    origin = UrbanNavPoint(lat=float(origin_record["lat"]), lng=float(origin_record["lng"]))
    case_id = _case_id(origin_record, task, template_index)
    search_radius_m = int(_setting(args.search_radius_m, config, "search_radius_m", 1500))
    candidate_limit = int(_setting(args.candidate_limit, config, "candidate_limit", 10))
    acceptable_ratio = float(_setting(args.acceptable_route_ratio, config, "acceptable_route_ratio", 1.15))
    min_distance = float(_setting(args.min_route_distance_m, config, "min_route_distance_m", 80))
    max_distance = float(_setting(args.max_route_distance_m, config, "max_route_distance_m", 3000))

    poi_records: list[ReplayPoiResult] = []
    candidates: list[UrbanNavDestination] = []
    for query_index, query in enumerate(_task_queries(task), start=1):
        poi_key = f"poi:{case_id}:{query_index:02d}"
        try:
            record = await _with_retry(
                lambda: backend.search_destinations(
                    cache_key=poi_key,
                    origin=origin,
                    keywords=query,
                    radius_m=search_radius_m,
                    limit=max(1, candidate_limit),
                ),
                retry=args.retry,
                label=f"poi:{case_id}:{query_index}",
            )
        except Exception as exc:
            return {"case": None, "reject": _reject(case_id, origin_record, task, "poi_api_error", str(exc))}
        poi_records.append(record)
        candidates.extend(record.candidates)

    merged_candidates = _dedupe_destinations(candidates)
    merged_candidates = [candidate for candidate in merged_candidates if _candidate_satisfies_task(candidate, task)]
    if not merged_candidates:
        return {"case": None, "reject": _reject(case_id, origin_record, task, "no_satisfying_poi_candidates")}

    merged_candidates.sort(
        key=lambda item: (
            item.route_distance_m if item.route_distance_m is not None else float("inf"),
            item.name or "",
        )
    )
    async def plan_candidate(candidate_index: int, candidate: UrbanNavDestination) -> tuple[int, UrbanNavDestination, ReplayRouteResult]:
        route_key = f"route:{case_id}:{candidate_index:02d}"
        try:
            route_record = await _with_retry(
                lambda: backend.plan_route(
                    cache_key=route_key,
                    request=LiveRouteRequest(origin=origin, destination=candidate, mode="foot"),
                ),
                retry=args.retry,
                label=f"route:{case_id}:{candidate_index}",
            )
        except Exception as exc:
            route_record = ReplayRouteResult(
                cache_key=route_key,
                provider=backend.provider_name,
                request={"origin": origin.to_dict(), "destination": candidate.to_dict(), "mode": "foot"},
                response={"error": str(exc)},
            )
        return candidate_index, candidate, route_record

    route_records: list[ReplayRouteResult] = []
    ranked: list[UrbanNavDestination] = []
    route_concurrency = max(1, int(getattr(args, "route_concurrency", 3)))
    route_semaphore = asyncio.Semaphore(route_concurrency)

    async def guarded_plan(candidate_index: int, candidate: UrbanNavDestination) -> tuple[int, UrbanNavDestination, ReplayRouteResult]:
        async with route_semaphore:
            return await plan_candidate(candidate_index, candidate)

    route_results = await asyncio.gather(
        *[
            guarded_plan(candidate_index, candidate)
            for candidate_index, candidate in enumerate(merged_candidates[:candidate_limit], start=1)
        ]
    )
    for candidate_index, candidate, route_record in sorted(route_results, key=lambda item: item[0]):
        route_records.append(route_record)
        if route_record.distance_m is None or route_record.distance_m <= 0 or not route_record.polyline:
            continue
        if not (min_distance <= route_record.distance_m <= max_distance):
            continue
        ranked.append(_destination_with_route(candidate, route_record, candidate_index))

    if not ranked:
        return {"case": None, "reject": _reject(case_id, origin_record, task, "no_routeable_candidates")}

    ranked.sort(key=lambda item: (item.route_distance_m or float("inf"), item.name or ""))
    ranked = [_with_rank(destination, rank=rank) for rank, destination in enumerate(ranked, start=1)]
    gt = ranked[0]
    best_distance = gt.reference_route_distance_m or gt.route_distance_m or 0.0
    if task.get("allow_alternatives", False):
        acceptable = [
            destination
            for destination in ranked
            if (destination.reference_route_distance_m or float("inf")) <= best_distance * max(1.0, acceptable_ratio)
        ]
    else:
        acceptable = [gt]

    instruction = _choose_template(task, template_index)
    distance = float(gt.reference_route_distance_m or 0.0)
    case_payload = {
        "id": case_id,
        "city": origin_record["city"],
        "scene_id": origin_record["region_id"],
        "category": task["category"],
        "subcategory": task["subcategory"],
        "difficulty": _difficulty(distance, gt.reference_route),
        "instruction": instruction,
        "origin": origin.to_dict(),
        "gt_destination": gt.to_dict(),
        "acceptable_destinations": [destination.to_dict() for destination in acceptable],
        "ranked_destinations": [destination.to_dict() for destination in ranked],
        "gt_route": [point.to_dict() for point in gt.reference_route],
        "gt_distance_m": distance,
        "optimality_scope": {
            "poi_provider": backend.provider_name,
            "search_radius_m": search_radius_m,
            "candidate_limit": candidate_limit,
            "route_provider": backend.provider_name,
            "route_mode": "foot",
            "route_strategy": "default",
            "cost": "route_distance_m",
            "builder_version": BUILDER_VERSION,
        },
        "cache_keys": {
            "poi_search": [record.cache_key for record in poi_records],
            "routes": [record.cache_key for record in route_records],
        },
        "metadata": {
            "region_label": origin_record.get("label"),
            "origin_index": origin_record.get("origin_index"),
            "template_index": template_index,
            "queries": _task_queries(task),
            "brand_aliases": task.get("brand_aliases", []),
            "allow_alternatives": bool(task.get("allow_alternatives", False)),
            "route_features": _route_features(gt.reference_route, distance),
        },
    }
    return {
        "case": UrbanNavCase.from_dict(case_payload),
        "poi_records": poi_records,
        "route_records": route_records,
        "reject": None,
    }


def build_collection_plan(
    config: dict[str, Any],
    *,
    seed: int,
    origins_per_region: int,
    categories: set[str],
) -> CollectionPlan:
    """Expand collection config into deterministic origins and tasks.

    Args:
        config: Raw collection configuration.
        seed: Random seed.
        origins_per_region: Maximum origins sampled from each region.
        categories: Categories retained for collection.

    Returns:
        Expanded collection plan.
    """

    rng = random.Random(seed)
    origins: list[dict[str, Any]] = []
    for region in config.get("regions", []):
        origins.extend(_sample_region_origins(region, max_origins=origins_per_region, rng=rng))
    tasks = [dict(task) for task in config.get("tasks", []) if task.get("category") in categories]
    if not origins:
        raise ValueError("No regions/origins configured")
    if not tasks:
        raise ValueError("No task templates selected")
    return CollectionPlan(config=config, origins=origins, tasks=tasks)


def iter_collection_pairs(
    plan: CollectionPlan,
    *,
    seed: int,
) -> Iterable[tuple[dict[str, Any], dict[str, Any], int]]:
    r"""Yield origin-task-template triples in a balanced order.

    Args:
        plan: Expanded collection plan.
        seed: Random seed.

    Returns:
        Iterable of ``(origin, task, template_index)`` triples.

    The iterator covers the full product between sampled origins and tasks,
    while interleaving cities and task categories to keep the output balanced:

    \[
    \mathcal P_c = O_c \times T
    \]
    """

    rng = random.Random(seed)
    origins_by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for origin in plan.origins:
        origins_by_city[str(origin["city"])].append(origin)
    for origins in origins_by_city.values():
        rng.shuffle(origins)

    tasks_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in plan.tasks:
        tasks_by_category[str(task["category"])].append(task)
    for tasks in tasks_by_category.values():
        rng.shuffle(tasks)

    cities = sorted(origins_by_city)
    categories = sorted(tasks_by_category)
    max_origins = max(len(items) for items in origins_by_city.values())
    for origin_offset in range(max_origins):
        for category in categories:
            tasks = tasks_by_category[category]
            if not tasks:
                continue
            for task_offset, task in enumerate(tasks):
                template_count = max(1, len(task.get("templates") or [task.get("instruction", "")]))
                template_index = (origin_offset + task_offset) % template_count
                for city in cities:
                    origins = origins_by_city[city]
                    if origin_offset >= len(origins):
                        continue
                    yield origins[origin_offset], task, template_index


def _sample_region_origins(
    region: dict[str, Any],
    *,
    max_origins: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample origins around one region center.

    Args:
        region: Region record with center coordinate and radius.
        max_origins: Maximum origin count.
        rng: Random generator.

    Returns:
        Origin records with deterministic indexes.
    """

    center = UrbanNavPoint(lat=float(region["lat"]), lng=float(region["lng"]))
    radius_m = float(region.get("radius_m", 1000))
    origins = [dict(region, lat=center.lat, lng=center.lng, origin_index=0)]
    rings = [0.25, 0.5, 0.75, 1.0]
    angles = list(range(0, 360, 30))
    for ring in rings:
        for angle in angles:
            if len(origins) >= max_origins:
                break
            jitter = rng.uniform(-8.0, 8.0)
            distance = radius_m * ring * rng.uniform(0.85, 1.05)
            point = _offset_point(center, distance_m=distance, bearing_deg=angle + jitter)
            origins.append(dict(region, lat=point.lat, lng=point.lng, origin_index=len(origins)))
        if len(origins) >= max_origins:
            break
    return origins


def _offset_point(origin: UrbanNavPoint, *, distance_m: float, bearing_deg: float) -> UrbanNavPoint:
    """Move a point by distance and bearing on a local tangent plane.

    Args:
        origin: Center point.
        distance_m: Distance in meters.
        bearing_deg: Bearing in degrees.

    Returns:
        Offset point.
    """

    bearing = math.radians(bearing_deg)
    d_lat = math.cos(bearing) * distance_m / 111_320.0
    d_lng = math.sin(bearing) * distance_m / (111_320.0 * max(0.2, math.cos(math.radians(origin.lat))))
    return UrbanNavPoint(lat=round(origin.lat + d_lat, 7), lng=round(origin.lng + d_lng, 7))


def _destination_with_route(
    candidate: UrbanNavDestination,
    route_record: ReplayRouteResult,
    provider_rank: int,
) -> UrbanNavDestination:
    """Attach frozen route evidence to a destination.

    Args:
        candidate: Candidate POI.
        route_record: Frozen route record.
        provider_rank: Original provider rank after deduplication.

    Returns:
        Destination with reference route and ranking audit fields.
    """

    return UrbanNavDestination(
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
            "provider_rank": provider_rank,
            "poi_distance_m": candidate.route_distance_m,
            "route_distance_m": route_record.distance_m,
        },
        raw=candidate.raw,
    )


def _with_rank(destination: UrbanNavDestination, *, rank: int) -> UrbanNavDestination:
    """Return a destination copy with rank set.

    Args:
        destination: Destination record.
        rank: One-based benchmark rank.

    Returns:
        Ranked destination.
    """

    payload = destination.to_dict()
    payload["rank"] = rank
    return UrbanNavDestination.from_dict(payload)


def _dedupe_destinations(candidates: Iterable[UrbanNavDestination]) -> list[UrbanNavDestination]:
    """Dedupe POI candidates by provider id or rounded coordinate/name.

    Args:
        candidates: Raw candidate sequence.

    Returns:
        Deduplicated candidates.
    """

    seen: set[str] = set()
    output: list[UrbanNavDestination] = []
    for candidate in candidates:
        if candidate.point is None:
            continue
        key = candidate.provider_poi_id or f"{candidate.name}:{candidate.lat:.6f}:{candidate.lng:.6f}"
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


def _candidate_satisfies_task(candidate: UrbanNavDestination, task: dict[str, Any]) -> bool:
    """Check hard task constraints that can be audited locally.

    Args:
        candidate: Candidate POI.
        task: Task template.

    Returns:
        ``True`` if the candidate should be routed.
    """

    aliases = [str(item).lower() for item in task.get("brand_aliases", []) if str(item).strip()]
    if not aliases:
        return True
    text = f"{candidate.name or ''} {candidate.type or ''} {candidate.raw}".lower()
    return any(alias.lower() in text for alias in aliases)


async def _with_retry(factory: Any, *, retry: int, label: str) -> Any:
    """Run one async API call with bounded retries.

    Args:
        factory: Zero-argument function returning an awaitable.
        retry: Retry count.
        label: Error label.

    Returns:
        Awaited result.

    Raises:
        Exception: The final exception if all attempts fail.
    """

    last_error: Exception | None = None
    for attempt in range(max(1, retry + 1)):
        try:
            return await factory()
        except Exception as exc:
            last_error = exc
            if attempt < retry:
                await asyncio.sleep(min(5.0, 0.5 * (2**attempt)))
    assert last_error is not None
    raise RuntimeError(f"{label} failed after {retry + 1} attempts: {last_error}") from last_error


def _route_features(route: list[UrbanNavPoint], distance_m: float) -> dict[str, Any]:
    """Compute lightweight route-shape features for sampling audit.

    Args:
        route: Reference route.
        distance_m: Reference route distance.

    Returns:
        Feature dictionary.
    """

    direct = _haversine_m(route[0], route[-1]) if len(route) >= 2 else 0.0
    return {
        "route_distance_m": distance_m,
        "direct_distance_m": round(direct, 3),
        "directness": round(direct / distance_m, 4) if distance_m > 0 else None,
        "polyline_points": len(route),
        "turn_count_proxy": _turn_count_proxy(route),
    }


def _turn_count_proxy(route: list[UrbanNavPoint]) -> int:
    """Estimate turn count from polyline bearings.

    Args:
        route: Route polyline.

    Returns:
        Number of bearing changes larger than 35 degrees.
    """

    if len(route) < 3:
        return 0
    bearings = [_bearing(route[i], route[i + 1]) for i in range(len(route) - 1)]
    turns = 0
    for prev, cur in zip(bearings, bearings[1:]):
        delta = abs((cur - prev + 180) % 360 - 180)
        if delta >= 35:
            turns += 1
    return turns


def _bearing(a: UrbanNavPoint, b: UrbanNavPoint) -> float:
    """Compute approximate bearing from point ``a`` to point ``b``.

    Args:
        a: Start point.
        b: End point.

    Returns:
        Bearing in degrees.
    """

    y = math.sin(math.radians(b.lng - a.lng)) * math.cos(math.radians(b.lat))
    x = math.cos(math.radians(a.lat)) * math.sin(math.radians(b.lat)) - math.sin(math.radians(a.lat)) * math.cos(
        math.radians(b.lat)
    ) * math.cos(math.radians(b.lng - a.lng))
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def _haversine_m(a: UrbanNavPoint, b: UrbanNavPoint) -> float:
    """Compute haversine distance in meters.

    Args:
        a: First point.
        b: Second point.

    Returns:
        Distance in meters.
    """

    radius = 6_371_000.0
    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    d_lat = lat2 - lat1
    d_lng = math.radians(b.lng - a.lng)
    hav = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(hav), math.sqrt(1 - hav))


def _difficulty(distance_m: float, route: list[UrbanNavPoint]) -> str:
    """Assign a deterministic difficulty label.

    Args:
        distance_m: Route distance.
        route: Reference route.

    Returns:
        ``easy``, ``medium``, or ``hard``.
    """

    turns = _turn_count_proxy(route)
    if distance_m < 500 and turns <= 3:
        return "easy"
    if distance_m < 1400 and turns <= 8:
        return "medium"
    return "hard"


def _task_queries(task: dict[str, Any]) -> list[str]:
    """Return provider query strings for a task.

    Args:
        task: Task template.

    Returns:
        Non-empty query list.
    """

    queries = [str(item).strip() for item in task.get("queries", []) if str(item).strip()]
    if queries:
        return queries
    keywords = str(task.get("keywords") or "").strip()
    if keywords:
        return [keywords]
    return [str(task.get("subcategory") or task.get("category"))]


def _choose_template(task: dict[str, Any], template_index: int) -> str:
    """Choose one instruction template.

    Args:
        task: Task template.
        template_index: Template index.

    Returns:
        Instruction text.
    """

    templates = [str(item) for item in task.get("templates", []) if str(item).strip()]
    if not templates:
        return str(task.get("instruction") or task.get("subcategory") or "")
    return templates[template_index % len(templates)]


def _case_id(origin: dict[str, Any], task: dict[str, Any], template_index: int) -> str:
    """Build a deterministic case id.

    Args:
        origin: Origin record.
        task: Task record.
        template_index: Selected template index.

    Returns:
        Stable case id.
    """

    return (
        f"urbannav-{origin['city']}-{origin['region_id']}-"
        f"o{int(origin['origin_index']):03d}-{task['subcategory']}-t{template_index:02d}"
    )


def _reject(
    case_id: str,
    origin: dict[str, Any],
    task: dict[str, Any],
    reason: str,
    detail: str | None = None,
) -> dict[str, Any]:
    """Create a rejected-case audit row.

    Args:
        case_id: Case id.
        origin: Origin record.
        task: Task record.
        reason: Reject reason.
        detail: Optional error detail.

    Returns:
        JSON-ready reject record.
    """

    return {
        "id": case_id,
        "reject_reason": reason,
        "reason": reason,
        "detail": detail,
        "city": origin.get("city"),
        "scene_id": origin.get("region_id"),
        "origin": {"lat": origin.get("lat"), "lng": origin.get("lng")},
        "category": task.get("category"),
        "subcategory": task.get("subcategory"),
        "instruction": _choose_template(task, 0),
    }


def _load_config(path: Path) -> dict[str, Any]:
    """Load collection config.

    Args:
        path: Config path.

    Returns:
        Config dictionary.
    """

    return json.loads(path.read_text(encoding="utf-8"))


def _package_paths(root: Path) -> dict[str, Path]:
    """Return output package paths.

    Args:
        root: Output root.

    Returns:
        Mapping from logical name to path.
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
    """Validate a collected package.

    Args:
        paths: Package path mapping.

    Returns:
        Validation report dictionary.
    """

    cases = load_urbannav_cases(paths["task_gt"]) if paths["task_gt"].exists() else []
    poi_keys = {record.get("cache_key") for record in iter_json_records(paths["poi_cache"])} if paths["poi_cache"].exists() else set()
    route_keys = (
        {record.get("cache_key") for record in iter_json_records(paths["route_cache"])}
        if paths["route_cache"].exists()
        else set()
    )
    rejects = list(iter_json_records(paths["task_rejects"])) if paths["task_rejects"].exists() else []
    report = validate_urbannav_cases(cases, poi_cache_keys=set(poi_keys), route_cache_keys=set(route_keys), rejects=rejects)
    paths["validation_report"].write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return report.to_dict()


def _manifest(
    args: argparse.Namespace,
    paths: dict[str, Path],
    accepted: int,
    rejected: int,
    attempted: int,
    *,
    dry_run: bool = False,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a package manifest.

    Args:
        args: Parsed CLI arguments.
        paths: Package paths.
        accepted: Accepted case count.
        rejected: Rejected case count.
        attempted: Attempted live pairs.
        dry_run: Whether this manifest is for a dry run.
        validation: Optional validation report.

    Returns:
        Manifest dictionary.
    """

    return {
        "builder_version": BUILDER_VERSION,
        "created_at_unix": time.time(),
        "dry_run": dry_run,
        "map_provider": args.map_provider,
        "target_cases": args.target_cases,
        "accepted_cases": accepted,
        "rejected_cases": rejected,
        "attempted_pairs_this_run": attempted,
        "paths": {name: str(path) for name, path in paths.items()},
        "validation": validation,
    }


def _plan_summary(plan: CollectionPlan, args: argparse.Namespace) -> dict[str, Any]:
    """Summarize the expanded collection plan.

    Args:
        plan: Expanded collection plan.
        args: Parsed CLI arguments.

    Returns:
        JSON-ready plan summary.
    """

    return {
        "config_version": plan.config.get("version"),
        "target_cases": args.target_cases,
        "origin_count": len(plan.origins),
        "task_count": len(plan.tasks),
        "city_counts": dict(Counter(origin["city"] for origin in plan.origins)),
        "task_counts": dict(Counter(task["category"] for task in plan.tasks)),
        "regions": plan.config.get("regions", []),
        "tasks": plan.tasks,
    }


def _setting(value: Any, config: dict[str, Any], key: str, default: Any) -> Any:
    """Return CLI override, config value, or default.

    Args:
        value: CLI value.
        config: Config dictionary.
        key: Config key.
        default: Default value.

    Returns:
        Selected setting.
    """

    return value if value is not None else config.get(key, default)


def _existing_case_ids(path: Path) -> set[str]:
    """Read existing case ids from a JSONL GT file.

    Args:
        path: GT path.

    Returns:
        Set of case ids.
    """

    if not path.exists():
        return set()
    return {str(record.get("id") or record.get("case_id")) for record in iter_json_records(path)}


def _count_jsonl(path: Path) -> int:
    """Count records in a JSONL file.

    Args:
        path: JSONL path.

    Returns:
        Number of non-empty lines.
    """

    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Append records to JSONL.

    Args:
        path: Output path.
        records: Records to append.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object.

    Args:
        path: Output path.
        payload: JSON-ready payload.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _unlink_outputs(paths: Iterable[Path]) -> None:
    """Remove existing output files.

    Args:
        paths: Paths to remove when present.

    Returns:
        None.
    """

    for path in paths:
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
