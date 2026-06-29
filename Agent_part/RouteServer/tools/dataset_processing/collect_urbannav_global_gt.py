#!/usr/bin/env python3
"""Collect global UrbanNav GT with OSM/Overpass POIs and OSRM routes."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

from mcp_server.core.commercial_global_provider import TomTomUrbanNavProvider
from mcp_server.core.osm_global_provider import OsmOverpassOsrmProvider
from mcp_server.core.osm_local_provider import LocalOsmOsrmProvider
from mcp_server.core.replay_map_backend import ReplayPoiResult, ReplayRouteResult
from mcp_server.core.urbannav_provider import UrbanNavLiveProvider, UrbanNavPoiSearchRequest, UrbanNavRouteRequest
from mcp_server.eval.urbannav_io import iter_json_records, load_urbannav_cases
from mcp_server.eval.urbannav_schema import UrbanNavCase, UrbanNavDestination, UrbanNavPoint
from mcp_server.eval.urbannav_validator import validate_urbannav_cases


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "urbannav_global_gt_collection.json"
BUILDER_VERSION = "urbannav_global_provider_v2"


@dataclass(frozen=True)
class GlobalCollectionPlan:
    r"""Expanded global collection plan.

    Args:
        config: Raw configuration.
        origins: Sampled origin records.
        tasks: Task records.

    The plan enumerates city/task pairs under fixed quotas:

    \[
    \mathcal P = \{(o,t) \mid o \in O_{\mathrm{city}}, t \in T_{\mathrm{eligible}}\}
    \]
    """

    config: dict[str, Any]
    origins: list[dict[str, Any]]
    tasks: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path, help="Output benchmark package directory.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Global collection config JSON.")
    parser.add_argument(
        "--instruction-set",
        type=Path,
        default=None,
        help="Optional UrbanNav instruction_set.jsonl. When set, single-stop rows are collected in instruction-plan order.",
    )
    parser.add_argument("--target-cases", type=int, default=100, help="Target valid cases.")
    parser.add_argument("--seed", type=int, default=20260623, help="Sampling seed.")
    parser.add_argument("--origins-per-region", type=int, default=12, help="Origins sampled per region.")
    parser.add_argument("--candidate-limit", type=int, default=None, help="Override routed candidate count.")
    parser.add_argument("--search-radius-m", type=int, default=None, help="Override POI radius.")
    parser.add_argument("--acceptable-route-ratio", type=float, default=None, help="Override acceptable near-tie ratio.")
    parser.add_argument("--min-route-distance-m", type=float, default=None, help="Override minimum route distance.")
    parser.add_argument("--max-route-distance-m", type=float, default=None, help="Override maximum route distance.")
    parser.add_argument("--throttle-s", type=float, default=0.2, help="Sleep seconds after each attempted case.")
    parser.add_argument("--retry", type=int, default=1, help="Retry count for transient provider errors.")
    parser.add_argument("--case-concurrency", type=int, default=1, help="Concurrent origin-task cases; keep low for public Overpass.")
    parser.add_argument("--route-concurrency", type=int, default=3, help="Concurrent route requests inside each case.")
    parser.add_argument("--resume", action="store_true", help="Resume an existing package.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing output files before running.")
    parser.add_argument("--dry-run", action="store_true", help="Only write collection plan and manifest.")
    parser.add_argument("--max-attempts", type=int, default=0, help="Optional attempted pair cap.")
    parser.add_argument("--provider", choices=["osm", "osm_local", "tomtom"], default=None, help="Live provider backend.")
    parser.add_argument("--overpass-url", default=None, help="Override Overpass endpoint.")
    parser.add_argument("--osrm-url", default=None, help="Override OSRM endpoint prefix.")
    parser.add_argument("--osrm-profile", default="foot", help="OSRM profile.")
    parser.add_argument("--poi-index-root", type=Path, default=None, help="Local OSM POI index root for --provider osm_local.")
    parser.add_argument("--local-router-config", type=Path, default=None, help="Local OSM router config JSON.")
    parser.add_argument("--local-osrm-url", default=None, help="Fallback local OSRM endpoint for --provider osm_local.")
    parser.add_argument("--local-valhalla-url", default=None, help="Fallback local Valhalla endpoint for --provider osm_local.")
    parser.add_argument("--valhalla-costing", default="pedestrian", help="Valhalla costing for --provider osm_local.")
    parser.add_argument("--tomtom-key", default=None, help="TomTom API key for commercial review collection.")
    parser.add_argument("--tomtom-search-url", default=None, help="Override TomTom Search API prefix.")
    parser.add_argument("--tomtom-routing-url", default=None, help="Override TomTom Routing API prefix.")
    return parser.parse_args()


def main() -> None:
    """Run global GT collection.

    Returns:
        None.
    """

    asyncio.run(collect_global_gt(parse_args()))


async def collect_global_gt(args: argparse.Namespace) -> dict[str, Any]:
    r"""Collect global GT cases from the configured open-map provider.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Manifest dictionary.
    """

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    paths = _package_paths(root)
    if args.overwrite:
        _unlink_outputs(paths.values())
    if not args.resume and any(path.exists() for path in paths.values()):
        raise RuntimeError(f"{root} already contains outputs; use --resume or --overwrite")

    config = _load_config(args.config)
    plan = build_global_collection_plan(config, seed=args.seed, origins_per_region=args.origins_per_region)
    _write_json(paths["plan"], _plan_summary(plan, args))
    if args.dry_run:
        manifest = _manifest(args, paths, accepted=0, rejected=0, attempted=0, dry_run=True)
        _write_json(paths["manifest"], manifest)
        return manifest

    provider = _create_provider(args, config)
    if args.resume:
        _dedupe_package_outputs(paths)
    existing_ids = _existing_case_ids(paths["task_gt"]) if args.resume else set()
    scheduled_ids = set(existing_ids)
    accepted = len(existing_ids)
    rejected = _count_jsonl(paths["task_rejects"]) if args.resume else 0
    attempted = 0
    case_concurrency = max(1, int(args.case_concurrency))
    await provider.connect()
    try:
        if args.instruction_set:
            pairs = iter(iter_instruction_collection_pairs(plan, instruction_set=args.instruction_set, seed=args.seed))
        else:
            pairs = iter(iter_global_collection_pairs(plan, seed=args.seed))
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
                        _build_global_case_with_throttle(
                            provider=provider,
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
                case_id = str(case.to_dict().get("id") or case.case_id)
                if case_id in existing_ids or accepted >= args.target_cases:
                    continue
                _append_jsonl(paths["task_gt"], [case.to_dict()])
                _append_jsonl(paths["poi_cache"], [record.to_dict() for record in result["poi_records"]])
                _append_jsonl(paths["route_cache"], [record.to_dict() for record in result["route_records"]])
                existing_ids.add(case_id)
                accepted += 1
                if accepted % 50 == 0:
                    _write_json(paths["manifest"], _manifest(args, paths, accepted=accepted, rejected=rejected, attempted=attempted))
    finally:
        await provider.disconnect()

    _dedupe_package_outputs(paths)
    validation = _validate_package(paths)
    if validation:
        accepted = int(validation.get("total_cases") or accepted)
        rejected = int(validation.get("reject_count") or rejected)
    manifest = _manifest(args, paths, accepted=accepted, rejected=rejected, attempted=attempted, validation=validation)
    _write_json(paths["manifest"], manifest)
    return manifest


async def _build_global_case_with_throttle(
    *,
    provider: UrbanNavLiveProvider,
    origin_record: dict[str, Any],
    task: dict[str, Any],
    template_index: int,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build one case and apply per-attempt throttling.

    Args:
        provider: Connected global provider.
        origin_record: Sampled origin record.
        task: Task configuration.
        template_index: Selected template index.
        config: Raw collection config.
        args: Parsed CLI arguments.

    Returns:
        Case build result.
    """

    try:
        return await build_global_case(
            provider=provider,
            origin_record=origin_record,
            task=task,
            template_index=template_index,
            config=config,
            args=args,
        )
    finally:
        await asyncio.sleep(max(0.0, float(args.throttle_s)))


async def build_global_case(
    *,
    provider: UrbanNavLiveProvider,
    origin_record: dict[str, Any],
    task: dict[str, Any],
    template_index: int,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    r"""Build one global UrbanNav case.

    Args:
        provider: Connected global provider.
        origin_record: Sampled origin record.
        task: Task configuration.
        template_index: Selected template index.
        config: Raw collection config.
        args: Parsed CLI arguments.

    Returns:
        Case plus replay records, or a reject row.
    """

    origin = UrbanNavPoint(lat=float(origin_record["lat"]), lng=float(origin_record["lng"]))
    case_id = _case_id(origin_record, task, template_index)
    radius_m = int(_setting(args.search_radius_m, config, "search_radius_m", 1200))
    candidate_limit = int(_setting(args.candidate_limit, config, "candidate_limit", 5))
    acceptable_ratio = float(_setting(args.acceptable_route_ratio, config, "acceptable_route_ratio", 1.15))
    min_distance = float(_setting(args.min_route_distance_m, config, "min_route_distance_m", 50))
    max_distance = float(_setting(args.max_route_distance_m, config, "max_route_distance_m", 2500))
    query = _task_query(task, locale=str(origin_record.get("locale") or "default"))

    poi_key = f"poi:{case_id}:01"
    try:
        poi_record = await _with_retry(
            lambda: provider.search_pois(
                cache_key=poi_key,
                request=UrbanNavPoiSearchRequest(
                    origin=origin,
                    query=query,
                    tags=_provider_tags(task=task, provider_name=provider.capabilities.provider_name),
                    radius_m=radius_m,
                    limit=candidate_limit,
                    locale=str(origin_record.get("locale") or ""),
                    raw=_provider_request_raw(
                        task=task,
                        provider_name=provider.capabilities.provider_name,
                        origin_record=origin_record,
                    ),
                ),
            ),
            retry=args.retry,
            label=f"poi:{case_id}",
        )
    except Exception as exc:
        return {"case": None, "reject": _reject(case_id, origin_record, task, "poi_api_error", str(exc))}
    candidates = _dedupe_destinations(poi_record.candidates)
    if not candidates:
        return {"case": None, "reject": _reject(case_id, origin_record, task, "no_satisfying_poi_candidates")}

    route_records: list[ReplayRouteResult] = []
    ranked: list[UrbanNavDestination] = []
    route_results = await _plan_candidate_routes(
        provider=provider,
        origin=origin,
        candidates=candidates[:candidate_limit],
        case_id=case_id,
        retry=args.retry,
        route_concurrency=max(1, int(args.route_concurrency)),
    )
    for candidate_index, candidate, route_record in route_results:
        route_records.append(route_record)
        if route_record.distance_m is None or route_record.distance_m <= 0 or not route_record.polyline:
            continue
        if not (min_distance <= route_record.distance_m <= max_distance):
            continue
        ranked.append(_destination_with_route(candidate, route_record, provider_rank=candidate_index))

    if not ranked:
        return {"case": None, "reject": _reject(case_id, origin_record, task, "no_routeable_candidates")}

    ranked.sort(key=lambda item: (item.reference_route_distance_m or float("inf"), item.name or ""))
    ranked = [_with_rank(destination, rank=rank) for rank, destination in enumerate(ranked, start=1)]
    gt = ranked[0]
    best_distance = gt.reference_route_distance_m or gt.route_distance_m or 0.0
    acceptable = [
        destination
        for destination in ranked
        if task.get("allow_alternatives", False)
        and (destination.reference_route_distance_m or float("inf")) <= best_distance * max(1.0, acceptable_ratio)
    ] or [gt]

    case_payload = {
        "id": case_id,
        "city": origin_record["city"],
        "scene_id": origin_record["region_id"],
        "category": task["category"],
        "subcategory": task["subcategory"],
        "difficulty": _difficulty(float(best_distance), gt.reference_route),
        "instruction": _choose_template(task, template_index),
        "origin": origin.to_dict(),
        "gt_destination": gt.to_dict(),
        "acceptable_destinations": [destination.to_dict() for destination in acceptable],
        "ranked_destinations": [destination.to_dict() for destination in ranked],
        "gt_route": [point.to_dict() for point in gt.reference_route],
        "gt_distance_m": best_distance,
        "optimality_scope": {
            "poi_provider": provider.capabilities.provider_name,
            "search_radius_m": radius_m,
            "candidate_limit": candidate_limit,
            "route_provider": provider.capabilities.provider_name,
            "route_mode": "foot",
            "route_strategy": getattr(provider, "route_strategy", getattr(provider, "osrm_profile", "foot")),
            "cost": "route_distance_m",
            "builder_version": BUILDER_VERSION,
        },
        "cache_keys": {"poi_search": [poi_key], "routes": [record.cache_key for record in route_records]},
        "metadata": {
            "country": origin_record.get("country"),
            "locale": origin_record.get("locale"),
            "region_label": origin_record.get("label"),
            "origin_index": origin_record.get("origin_index"),
            "query": query,
            "osm_tags": list(task.get("osm_tags") or []),
            "provider_tags": _provider_tags(task=task, provider_name=provider.capabilities.provider_name),
            "provider": provider.capabilities.raw,
            "route_features": _route_features(gt.reference_route, float(best_distance)),
        },
    }
    return {
        "case": UrbanNavCase.from_dict(case_payload),
        "poi_records": [poi_record],
        "route_records": route_records,
        "reject": None,
    }


async def _plan_candidate_routes(
    *,
    provider: UrbanNavLiveProvider,
    origin: UrbanNavPoint,
    candidates: list[UrbanNavDestination],
    case_id: str,
    retry: int,
    route_concurrency: int,
) -> list[tuple[int, UrbanNavDestination, ReplayRouteResult]]:
    """Plan routes for candidates with bounded concurrency.

    Args:
        provider: Connected global provider.
        origin: Route origin.
        candidates: Candidate destinations.
        case_id: Case id.
        retry: Retry count for route calls.
        route_concurrency: Maximum concurrent route calls for this case.

    Returns:
        Candidate index, candidate, and route replay records in candidate order.
    """

    semaphore = asyncio.Semaphore(max(1, route_concurrency))

    async def run_one(candidate_index: int, candidate: UrbanNavDestination) -> tuple[int, UrbanNavDestination, ReplayRouteResult]:
        route_key = f"route:{case_id}:{candidate_index:02d}"
        async with semaphore:
            try:
                route_record = await _with_retry(
                    lambda: provider.plan_route(
                        cache_key=route_key,
                        request=UrbanNavRouteRequest(origin=origin, destination=candidate, mode="foot"),
                    ),
                    retry=retry,
                    label=f"route:{case_id}:{candidate_index}",
                )
            except Exception as exc:
                route_record = ReplayRouteResult(
                    cache_key=route_key,
                    provider=provider.capabilities.provider_name,
                    request={"origin": origin.to_dict(), "destination": candidate.to_dict(), "mode": "foot"},
                    response={"error": str(exc)},
                )
        return candidate_index, candidate, route_record

    tasks = [asyncio.create_task(run_one(index, candidate)) for index, candidate in enumerate(candidates, start=1)]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda item: item[0])


def build_global_collection_plan(config: dict[str, Any], *, seed: int, origins_per_region: int) -> GlobalCollectionPlan:
    """Expand global config into origins and tasks.

    Args:
        config: Raw collection config.
        seed: Sampling seed.
        origins_per_region: Number of origins per region.

    Returns:
        Expanded plan.
    """

    rng = random.Random(seed)
    origins: list[dict[str, Any]] = []
    for region in config.get("regions", []):
        origins.extend(_sample_region_origins(region, max_origins=origins_per_region, rng=rng))
    tasks = [dict(task) for task in config.get("tasks", [])]
    if not origins or not tasks:
        raise ValueError("Global config must contain regions and tasks")
    return GlobalCollectionPlan(config=config, origins=origins, tasks=tasks)


def _create_provider(args: argparse.Namespace, config: dict[str, Any]) -> UrbanNavLiveProvider:
    """Create the configured live provider.

    Args:
        args: Parsed CLI arguments.
        config: Collection config.

    Returns:
        Live provider instance.

    Raises:
        ValueError: If the provider name is unsupported.
    """

    provider_name = str(args.provider or config.get("provider") or "osm").strip().lower()
    if provider_name in {"osm", "osm_overpass_osrm"}:
        return OsmOverpassOsrmProvider(overpass_url=args.overpass_url, osrm_url=args.osrm_url, osrm_profile=args.osrm_profile)
    if provider_name in {"osm_local", "osm_local_osrm"}:
        if args.poi_index_root is None:
            raise ValueError("--poi-index-root is required for --provider osm_local")
        return LocalOsmOsrmProvider(
            poi_index_root=args.poi_index_root,
            router_config=args.local_router_config,
            default_osrm_url=args.local_osrm_url or args.osrm_url,
            osrm_profile=args.osrm_profile,
            default_valhalla_url=args.local_valhalla_url,
            valhalla_costing=args.valhalla_costing,
        )
    if provider_name in {"tomtom", "tomtom_search_routing"}:
        return TomTomUrbanNavProvider(
            api_key=args.tomtom_key,
            search_url=args.tomtom_search_url,
            routing_url=args.tomtom_routing_url,
        )
    raise ValueError(f"Unsupported global provider: {provider_name}")


def _provider_tags(*, task: dict[str, Any], provider_name: str) -> list[str]:
    """Return provider-specific category tags.

    Args:
        task: Task record.
        provider_name: Active provider name.

    Returns:
        Provider category tags.
    """

    provider_map = task.get("provider_category_map") or {}
    if "tomtom" in provider_name:
        return [str(item) for item in provider_map.get("tomtom", [])]
    return [str(item) for item in task.get("osm_tags") or provider_map.get("osm", [])]


def _provider_request_raw(*, task: dict[str, Any], provider_name: str, origin_record: dict[str, Any]) -> dict[str, Any]:
    """Build provider-specific request metadata.

    Args:
        task: Task record.
        provider_name: Active provider name.
        origin_record: Origin metadata.

    Returns:
        Raw request metadata.
    """

    raw = {
        "country": origin_record.get("country"),
        "city": origin_record.get("city"),
        "region_id": origin_record.get("region_id"),
        "locale": origin_record.get("locale"),
    }
    if "tomtom" not in provider_name:
        return {key: value for key, value in raw.items() if value is not None}
    categories = _provider_tags(task=task, provider_name=provider_name)
    if categories:
        raw["tomtom_category_set"] = categories
    return {key: value for key, value in raw.items() if value is not None}


def iter_global_collection_pairs(
    plan: GlobalCollectionPlan,
    *,
    seed: int,
) -> Iterable[tuple[dict[str, Any], dict[str, Any], int]]:
    r"""Yield balanced origin/task/template triples.

    Args:
        plan: Expanded plan.
        seed: Sampling seed.

    Returns:
        Iterable of triples.

    The iterator enumerates the full city-local Cartesian product between
    sampled origins and task templates while keeping cities interleaved:

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
    tasks = list(plan.tasks)
    rng.shuffle(tasks)
    cities = sorted(origins_by_city)
    max_origins = max(len(items) for items in origins_by_city.values())
    for origin_offset in range(max_origins):
        for task_offset, task in enumerate(tasks):
            template_count = max(1, len(task.get("templates") or [""]))
            template_index = (origin_offset + task_offset) % template_count
            for city in cities:
                origins = origins_by_city[city]
                if origin_offset >= len(origins):
                    continue
                yield origins[origin_offset], task, template_index


def iter_instruction_collection_pairs(
    plan: GlobalCollectionPlan,
    *,
    instruction_set: Path,
    seed: int,
) -> Iterable[tuple[dict[str, Any], dict[str, Any], int]]:
    r"""Yield collection triples directly from a compiled instruction set.

    Args:
        plan: Expanded collection plan with sampled origins.
        instruction_set: ``instruction_set.jsonl`` path.
        seed: Deterministic origin assignment seed.

    Returns:
        Iterable of origin/task/template-index triples.

    This mode preserves the instruction-plan identity:

    \[
    x_i=(I_i, c_i, s_i, T_i)
    \]

    instead of reducing the benchmark back to a small template pool.
    """

    rng = random.Random(seed)
    origins_by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for origin in plan.origins:
        origins_by_city[_norm_city(str(origin["city"]))].append(origin)
    for origins in origins_by_city.values():
        rng.shuffle(origins)
    for row in iter_json_records(instruction_set):
        if row.get("category") not in {"poi_navigation", "implicit_need_navigation", "public_poi_navigation"}:
            continue
        planned_city = row.get("planned_city") if isinstance(row.get("planned_city"), dict) else {}
        city_key = _norm_city(str(planned_city.get("city") or row.get("city") or ""))
        origins = origins_by_city.get(city_key)
        if not origins:
            continue
        index = int(row.get("global_index") or 0)
        origin = origins[index % len(origins)]
        task = _task_from_instruction_row(row)
        yield origin, task, int(row.get("template_index") or 0)


def _task_from_instruction_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one instruction-set row into a collector task.

    Args:
        row: Instruction-set row.

    Returns:
        Collector-compatible task dictionary.
    """

    query_terms = [str(item) for item in row.get("query_terms") or [] if str(item).strip()]
    if not query_terms:
        query_terms = [str(row.get("subcategory") or "")]
    category = str(row.get("category") or "")
    return {
        "case_id": str(row.get("instruction_id") or row.get("id") or ""),
        "category": "poi_navigation" if category == "public_poi_navigation" else category,
        "source_category": row.get("source_category"),
        "subcategory": str(row.get("subcategory") or row.get("task_id") or "unknown"),
        "templates": [str(row.get("instruction") or row.get("canonical_instruction") or row.get("subcategory") or "")],
        "target_types": list(row.get("target_types") or []),
        "ordered_target_types": list(row.get("ordered_target_types") or []),
        "queries_by_locale": {"default": query_terms, "en": query_terms},
        "osm_tags": list(row.get("osm_tags") or []),
        "provider_category_map": row.get("provider_category_map") or {},
        "semantic_tags": list(row.get("semantic_tags") or []),
        "route_update_action": row.get("route_update_action"),
        "allow_alternatives": category != "poi_navigation",
        "acceptance_policy": row.get("acceptance_policy"),
        "instruction_id": row.get("instruction_id"),
        "global_index": row.get("global_index"),
    }


def _sample_region_origins(region: dict[str, Any], *, max_origins: int, rng: random.Random) -> list[dict[str, Any]]:
    """Sample local origins around a region.

    Args:
        region: Region center record.
        max_origins: Maximum origins.
        rng: Random generator.

    Returns:
        Origin records.
    """

    center = UrbanNavPoint(lat=float(region["lat"]), lng=float(region["lng"]))
    origins = [dict(region, lat=center.lat, lng=center.lng, origin_index=0)]
    radius_m = float(region.get("radius_m", 900))
    for idx, angle in enumerate(range(0, 360, 45), start=1):
        if len(origins) >= max_origins:
            break
        point = _offset_point(center, distance_m=radius_m * rng.uniform(0.25, 0.95), bearing_deg=angle + rng.uniform(-10, 10))
        origins.append(dict(region, lat=point.lat, lng=point.lng, origin_index=idx))
    while len(origins) < max_origins:
        point = _offset_point(center, distance_m=radius_m * rng.uniform(0.15, 1.0), bearing_deg=rng.uniform(0, 360))
        origins.append(dict(region, lat=point.lat, lng=point.lng, origin_index=len(origins)))
    return origins


def _offset_point(origin: UrbanNavPoint, *, distance_m: float, bearing_deg: float) -> UrbanNavPoint:
    """Offset a coordinate on a local tangent plane.

    Args:
        origin: Origin point.
        distance_m: Offset distance.
        bearing_deg: Bearing in degrees.

    Returns:
        Offset point.
    """

    bearing = math.radians(bearing_deg)
    d_lat = math.cos(bearing) * distance_m / 111_320.0
    d_lng = math.sin(bearing) * distance_m / (111_320.0 * max(0.2, math.cos(math.radians(origin.lat))))
    return UrbanNavPoint(lat=round(origin.lat + d_lat, 7), lng=round(origin.lng + d_lng, 7))


def _destination_with_route(candidate: UrbanNavDestination, route: ReplayRouteResult, *, provider_rank: int) -> UrbanNavDestination:
    """Attach route evidence to a destination.

    Args:
        candidate: Candidate destination.
        route: Route replay record.
        provider_rank: Provider candidate rank.

    Returns:
        Destination with reference route.
    """

    return UrbanNavDestination(
        lat=candidate.lat,
        lng=candidate.lng,
        name=candidate.name,
        type=candidate.type,
        provider_poi_id=candidate.provider_poi_id,
        place_id=candidate.place_id,
        reference_route_distance_m=route.distance_m,
        reference_route=route.polyline,
        route_distance_m=route.distance_m,
        score_breakdown={"provider_rank": provider_rank, "poi_distance_m": candidate.route_distance_m, "route_distance_m": route.distance_m},
        raw=candidate.raw,
    )


def _with_rank(destination: UrbanNavDestination, *, rank: int) -> UrbanNavDestination:
    """Return a destination copy with rank.

    Args:
        destination: Destination object.
        rank: One-based rank.

    Returns:
        Ranked destination.
    """

    payload = destination.to_dict()
    payload["rank"] = rank
    return UrbanNavDestination.from_dict(payload)


def _dedupe_destinations(candidates: Iterable[UrbanNavDestination]) -> list[UrbanNavDestination]:
    """Dedupe destinations.

    Args:
        candidates: Candidate sequence.

    Returns:
        Deduplicated list.
    """

    seen: set[str] = set()
    output: list[UrbanNavDestination] = []
    for candidate in candidates:
        key = candidate.provider_poi_id or f"{candidate.name}:{candidate.lat}:{candidate.lng}"
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
    return output


async def _with_retry(factory: Any, *, retry: int, label: str) -> Any:
    """Run an async provider call with retries.

    Args:
        factory: Function returning an awaitable.
        retry: Number of retries.
        label: Error label.

    Returns:
        Awaited result.

    Raises:
        RuntimeError: If all attempts fail.
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


def _task_query(task: dict[str, Any], *, locale: str) -> str:
    """Choose a locale query.

    Args:
        task: Task record.
        locale: Locale code.

    Returns:
        Query text.
    """

    queries_by_locale = task.get("queries_by_locale") or {}
    queries = queries_by_locale.get(locale) or queries_by_locale.get("default") or [task.get("subcategory", "")]
    return str(queries[0])


def _choose_template(task: dict[str, Any], template_index: int) -> str:
    """Choose an instruction template.

    Args:
        task: Task record.
        template_index: Template index.

    Returns:
        Instruction string.
    """

    templates = [str(item) for item in task.get("templates", []) if str(item).strip()]
    return templates[template_index % len(templates)] if templates else str(task.get("subcategory") or "")


def _case_id(origin: dict[str, Any], task: dict[str, Any], template_index: int) -> str:
    """Build a deterministic case id.

    Args:
        origin: Origin record.
        task: Task record.
        template_index: Template index.

    Returns:
        Stable case id.
    """

    if task.get("case_id"):
        return str(task["case_id"])
    return f"urbannav-global-{origin['city']}-{origin['region_id']}-o{int(origin['origin_index']):03d}-{task['subcategory']}-t{template_index:02d}"


def _norm_city(value: str) -> str:
    """Normalize city names across configs.

    Args:
        value: City label.

    Returns:
        Normalized city key.
    """

    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _reject(case_id: str, origin: dict[str, Any], task: dict[str, Any], reason: str, detail: str | None = None) -> dict[str, Any]:
    """Create a reject row.

    Args:
        case_id: Case id.
        origin: Origin record.
        task: Task record.
        reason: Reject reason.
        detail: Optional detail.

    Returns:
        JSON-ready reject record.
    """

    return {
        "id": case_id,
        "reject_reason": reason,
        "reason": reason,
        "detail": detail,
        "country": origin.get("country"),
        "city": origin.get("city"),
        "scene_id": origin.get("region_id"),
        "origin": {"lat": origin.get("lat"), "lng": origin.get("lng")},
        "category": task.get("category"),
        "subcategory": task.get("subcategory"),
    }


def _route_features(route: list[UrbanNavPoint], distance_m: float) -> dict[str, Any]:
    """Compute route audit features.

    Args:
        route: Route polyline.
        distance_m: Route distance.

    Returns:
        Feature dictionary.
    """

    direct = _haversine_m(route[0], route[-1]) if len(route) >= 2 else 0.0
    return {"route_distance_m": distance_m, "direct_distance_m": round(direct, 3), "directness": round(direct / distance_m, 4) if distance_m > 0 else None, "polyline_points": len(route)}


def _haversine_m(a: UrbanNavPoint, b: UrbanNavPoint) -> float:
    """Compute haversine distance.

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
    """Assign a difficulty label.

    Args:
        distance_m: Route distance.
        route: Route polyline.

    Returns:
        Difficulty label.
    """

    if distance_m < 400:
        return "easy"
    if distance_m < 1200:
        return "medium"
    return "hard"


def _load_config(path: Path) -> dict[str, Any]:
    """Load JSON config.

    Args:
        path: Config path.

    Returns:
        Config dictionary.
    """

    return json.loads(path.read_text(encoding="utf-8"))


def _package_paths(root: Path) -> dict[str, Path]:
    """Return package output paths.

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
        paths: Package paths.

    Returns:
        Validation report.
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
    """Build manifest.

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
        "target_cases": args.target_cases,
        "accepted_cases": accepted,
        "rejected_cases": rejected,
        "attempted_pairs_this_run": attempted,
        "case_concurrency": args.case_concurrency,
        "route_concurrency": args.route_concurrency,
        "throttle_s": args.throttle_s,
        "paths": {name: str(path) for name, path in paths.items()},
        "validation": validation,
    }


def _plan_summary(plan: GlobalCollectionPlan, args: argparse.Namespace) -> dict[str, Any]:
    """Summarize a collection plan.

    Args:
        plan: Expanded plan.
        args: Parsed arguments.

    Returns:
        Plan summary.
    """

    return {
        "config_version": plan.config.get("version"),
        "target_cases": args.target_cases,
        "case_concurrency": args.case_concurrency,
        "route_concurrency": args.route_concurrency,
        "throttle_s": args.throttle_s,
        "origin_count": len(plan.origins),
        "task_count": len(plan.tasks),
        "country_counts": dict(Counter(origin.get("country") for origin in plan.origins)),
        "city_counts": dict(Counter(origin.get("city") for origin in plan.origins)),
        "task_counts": dict(Counter(task.get("category") for task in plan.tasks)),
        "regions": plan.config.get("regions", []),
        "tasks": plan.tasks,
    }


def _setting(value: Any, config: dict[str, Any], key: str, default: Any) -> Any:
    """Return CLI override, config value, or default.

    Args:
        value: CLI override.
        config: Config dictionary.
        key: Config key.
        default: Default value.

    Returns:
        Selected setting.
    """

    return value if value is not None else config.get(key, default)


def _existing_case_ids(path: Path) -> set[str]:
    """Load existing case ids.

    Args:
        path: GT path.

    Returns:
        Set of ids.
    """

    if not path.exists():
        return set()
    return {str(record.get("id") or record.get("case_id")) for record in iter_json_records(path)}


def _count_jsonl(path: Path) -> int:
    """Count JSONL rows.

    Args:
        path: JSONL path.

    Returns:
        Row count.
    """

    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _dedupe_package_outputs(paths: dict[str, Path]) -> None:
    """Remove duplicate GT and replay rows from resumable outputs.

    Args:
        paths: Package paths.

    Returns:
        None.
    """

    _dedupe_jsonl(paths["task_gt"], key="id")
    _dedupe_jsonl(paths["poi_cache"], key="cache_key")
    _dedupe_jsonl(paths["route_cache"], key="cache_key")


def _dedupe_jsonl(path: Path, *, key: str) -> None:
    """Dedupe a JSONL file by a stable key.

    Args:
        path: JSONL path.
        key: Record key used for deduplication.

    Returns:
        None.
    """

    if not path.exists():
        return
    records: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for record in iter_json_records(path):
        record_key = record.get(key)
        if record_key in seen:
            continue
        seen.add(record_key)
        records.append(record)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")


def _append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Append records to JSONL.

    Args:
        path: Output path.
        records: JSON-ready records.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file.

    Args:
        path: Output path.
        payload: JSON payload.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _unlink_outputs(paths: Iterable[Path]) -> None:
    """Remove output files.

    Args:
        paths: Paths to remove.

    Returns:
        None.
    """

    for path in paths:
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
