#!/usr/bin/env python3
"""Build UrbanNav benchmark packages from mock or live map providers."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

from mcp_server.core.live_map_backend import LiveMapBackend, LiveRouteRequest
from mcp_server.core.replay_map_backend import ReplayPoiResult, ReplayRouteResult
from mcp_server.eval.urbannav.urbannav_io import write_urbannav_cases
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavCase, UrbanNavDestination, UrbanNavPoint
from mcp_server.eval.urbannav.urbannav_validator import validate_urbannav_cases


BUILDER_VERSION = "urbannav_live_v1"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed benchmark-builder arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path, help="Directory for the benchmark package")
    parser.add_argument(
        "--provider",
        choices=["mock", "live"],
        default="live",
        help="Use the deterministic mock builder or a live map provider",
    )
    parser.add_argument(
        "--map-provider",
        default=None,
        help="RouteServer map provider override, e.g. amap. Defaults to MAP_PROVIDER.",
    )
    parser.add_argument("--num-cases", type=int, default=5, help="Maximum valid cases to emit")
    parser.add_argument("--search-radius-m", type=int, default=1500, help="POI search radius in meters")
    parser.add_argument("--candidate-limit", type=int, default=10, help="Maximum POI candidates per task")
    parser.add_argument(
        "--acceptable-route-ratio",
        type=float,
        default=1.15,
        help="Accept routeable same-intent candidates within this ratio of the rank-1 route",
    )
    parser.add_argument(
        "--origins-json",
        type=Path,
        default=None,
        help="Optional JSON/JSONL origins. Each row needs city, scene_id, lat, lng.",
    )
    parser.add_argument(
        "--tasks-json",
        type=Path,
        default=None,
        help="Optional JSON/JSONL task templates. Each row needs instruction and keywords.",
    )
    parser.add_argument("--include-multi-target", action="store_true", help="Append mock/live multi-target task templates")
    parser.add_argument("--include-route-updates", action="store_true", help="Append mock route-update task")
    return parser.parse_args()


def main() -> None:
    """Build the selected UrbanNav benchmark package.

    Returns:
        None.
    """

    args = parse_args()
    if args.provider == "mock":
        builder = _load_mock_builder()
        builder.build_mock_benchmark(
            args.output_root,
            num_cases=args.num_cases,
            include_multi_target=args.include_multi_target,
            include_route_updates=args.include_route_updates,
        )
        return
    asyncio.run(
        build_live_benchmark(
            args.output_root,
            map_provider=args.map_provider,
            num_cases=args.num_cases,
            search_radius_m=args.search_radius_m,
            candidate_limit=args.candidate_limit,
            acceptable_route_ratio=args.acceptable_route_ratio,
            origins_path=args.origins_json,
            tasks_path=args.tasks_json,
            include_multi_target=args.include_multi_target,
        )
    )


async def build_live_benchmark(
    output_root: str | Path,
    *,
    map_provider: str | None = None,
    num_cases: int = 5,
    search_radius_m: int = 1500,
    candidate_limit: int = 10,
    acceptable_route_ratio: float = 1.15,
    origins_path: str | Path | None = None,
    tasks_path: str | Path | None = None,
    include_multi_target: bool = False,
) -> dict[str, Any]:
    r"""Build a benchmark package from a live RouteServer map provider.

    Each origin-template pair defines one search scope. Candidates are first
    filtered by successful route planning, then ranked by route distance:

    \[
    d^\* = \arg\min_{d \in D_{routeable}} cost(o, d)
    \]

    Acceptable destinations are near-ties under the same intent and provider
    scope:

    \[
    A = \{d \mid cost(o,d) \le \rho \cdot cost(o,d^\*)\}
    \]

    Args:
        output_root: Output package directory.
        map_provider: Optional ``MAP_PROVIDER`` override, for example
            ``"amap"``.
        num_cases: Maximum number of valid cases to emit.
        search_radius_m: POI search radius in meters.
        candidate_limit: Maximum POI candidates routed for each task.
        acceptable_route_ratio: Near-tie ratio used to populate
            ``acceptable_destinations``.
        origins_path: Optional JSON/JSONL origins file.
        tasks_path: Optional JSON/JSONL task template file.
        include_multi_target: Whether to append task templates that intentionally
            admit multiple reasonable destinations.

    Returns:
        Dictionary containing output paths and validation summary.
    """

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    backend = LiveMapBackend.from_env(map_provider=map_provider)
    origins = _load_origins(origins_path)
    tasks = _load_tasks(tasks_path)
    if include_multi_target:
        tasks = [*tasks, *_multi_target_live_tasks()]
    cases: list[UrbanNavCase] = []
    poi_records: list[ReplayPoiResult] = []
    route_records: list[ReplayRouteResult] = []
    rejects: list[dict[str, Any]] = []

    await backend.connect()
    try:
        pair_index = 0
        for origin_record, task in _iter_origin_task_pairs(origins, tasks):
            if len(cases) >= max(0, num_cases):
                break
            pair_index += 1
            case_result = await _build_live_case(
                backend=backend,
                origin_record=origin_record,
                task=task,
                pair_index=pair_index,
                search_radius_m=search_radius_m,
                candidate_limit=candidate_limit,
                acceptable_route_ratio=acceptable_route_ratio,
            )
            if case_result["case"] is None:
                rejects.append(case_result["reject"])
                continue
            cases.append(case_result["case"])
            poi_records.append(case_result["poi_record"])
            route_records.extend(case_result["route_records"])
    finally:
        await backend.disconnect()

    gt_path = root / "task_gt.jsonl"
    poi_cache_path = root / "poi_cache.jsonl"
    route_cache_path = root / "route_cache.jsonl"
    rejects_path = root / "task_rejects.jsonl"
    report_path = root / "validation_report.json"

    write_urbannav_cases(cases, gt_path)
    _write_jsonl((record.to_dict() for record in poi_records), poi_cache_path)
    _write_jsonl((record.to_dict() for record in route_records), route_cache_path)
    _write_jsonl(rejects, rejects_path)

    report = validate_urbannav_cases(
        cases,
        poi_cache_keys={record.cache_key for record in poi_records},
        route_cache_keys={record.cache_key for record in route_records},
        rejects=rejects,
    )
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "task_gt": str(gt_path),
        "poi_cache": str(poi_cache_path),
        "route_cache": str(route_cache_path),
        "task_rejects": str(rejects_path),
        "validation_report": str(report_path),
        "validation": report.to_dict(),
    }


async def _build_live_case(
    *,
    backend: LiveMapBackend,
    origin_record: dict[str, Any],
    task: dict[str, Any],
    pair_index: int,
    search_radius_m: int,
    candidate_limit: int,
    acceptable_route_ratio: float,
) -> dict[str, Any]:
    """Build one live case from one origin and one task template.

    Args:
        backend: Connected live map backend.
        origin_record: Origin metadata and coordinate.
        task: Task template with instruction and keywords.
        pair_index: One-based origin-template pair index.
        search_radius_m: POI search radius in meters.
        candidate_limit: Maximum candidates to route.
        acceptable_route_ratio: Near-tie ratio for acceptable destinations.

    Returns:
        Dictionary containing either a case plus cache records, or a reject.
    """

    origin = UrbanNavPoint(lat=float(origin_record["lat"]), lng=float(origin_record["lng"]))
    provider = backend.provider_name
    case_id = f"{provider}-{origin_record['scene_id']}-{task['subcategory']}-{pair_index:04d}"
    poi_key = f"poi:{case_id}"
    poi_record = await backend.search_destinations(
        cache_key=poi_key,
        origin=origin,
        keywords=str(task["keywords"]),
        radius_m=search_radius_m,
        limit=candidate_limit,
    )
    if not poi_record.candidates:
        return {
            "case": None,
            "reject": _reject(case_id, origin_record, task, "no_poi_candidates", poi_key=poi_key),
        }

    route_records: list[ReplayRouteResult] = []
    ranked: list[UrbanNavDestination] = []
    for rank_input, candidate in enumerate(poi_record.candidates[:candidate_limit], start=1):
        route_key = f"route:{case_id}:{rank_input:03d}"
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
                    "provider_rank": rank_input,
                    "poi_distance_m": candidate.route_distance_m,
                    "route_distance_m": route_record.distance_m,
                },
                raw=candidate.raw,
            )
        )

    if not ranked:
        return {
            "case": None,
            "reject": _reject(case_id, origin_record, task, "no_routeable_candidates", poi_key=poi_key),
        }

    ranked.sort(key=lambda item: (item.route_distance_m if item.route_distance_m is not None else float("inf"), item.name or ""))
    ranked = [
        _with_rank(destination, rank=rank)
        for rank, destination in enumerate(ranked, start=1)
    ]
    gt = ranked[0]
    best_distance = gt.route_distance_m or gt.reference_route_distance_m or 0.0
    acceptable = [
        destination
        for destination in ranked
        if (destination.route_distance_m or float("inf")) <= best_distance * max(1.0, acceptable_route_ratio)
    ]
    case_payload = {
        "id": case_id,
        "city": origin_record["city"],
        "scene_id": origin_record["scene_id"],
        "category": task["category"],
        "subcategory": task["subcategory"],
        "difficulty": task["difficulty"],
        "instruction": task["instruction"],
        "origin": origin.to_dict(),
        "gt_destination": gt.to_dict(),
        "acceptable_destinations": [destination.to_dict() for destination in acceptable],
        "ranked_destinations": [destination.to_dict() for destination in ranked],
        "gt_route": [point.to_dict() for point in gt.reference_route],
        "gt_distance_m": gt.reference_route_distance_m,
        "optimality_scope": {
            "poi_provider": provider,
            "search_radius_m": search_radius_m,
            "candidate_limit": candidate_limit,
            "route_provider": provider,
            "route_mode": "foot",
            "route_strategy": "default",
            "cost": "route_distance_m",
            "builder_version": BUILDER_VERSION,
        },
        "cache_keys": {
            "poi_search": [poi_key],
            "routes": [record.cache_key for record in route_records],
        },
        "metadata": {
            "keywords": task["keywords"],
            "origin_label": origin_record.get("label"),
            "acceptable_route_ratio": acceptable_route_ratio,
        },
    }
    return {
        "case": UrbanNavCase.from_dict(case_payload),
        "poi_record": poi_record,
        "route_records": route_records,
        "reject": None,
    }


def _with_rank(destination: UrbanNavDestination, *, rank: int) -> UrbanNavDestination:
    """Return a copy of a destination with a benchmark rank.

    Args:
        destination: Destination to copy.
        rank: One-based rank.

    Returns:
        Ranked destination.
    """

    payload = destination.to_dict()
    payload["rank"] = rank
    return UrbanNavDestination.from_dict(payload)


def _iter_origin_task_pairs(
    origins: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    """Iterate origin/task pairs with origin diversity early.

    Args:
        origins: Origin records.
        tasks: Task templates.

    Returns:
        Iterable of ``(origin, task)`` pairs. The order cycles origins inside
        each task so small smoke runs do not all come from the first origin.
    """

    for task in tasks:
        for origin in origins:
            yield origin, task


def _reject(
    case_id: str,
    origin_record: dict[str, Any],
    task: dict[str, Any],
    reason: str,
    *,
    poi_key: str,
) -> dict[str, Any]:
    """Create a rejected-case audit row.

    Args:
        case_id: Candidate case id.
        origin_record: Origin metadata.
        task: Task template.
        reason: Rejection reason.
        poi_key: POI cache key attempted for this case.

    Returns:
        JSON-ready reject record.
    """

    return {
        "id": case_id,
        "reason": reason,
        "city": origin_record.get("city"),
        "scene_id": origin_record.get("scene_id"),
        "origin": {"lat": origin_record.get("lat"), "lng": origin_record.get("lng")},
        "subcategory": task.get("subcategory"),
        "instruction": task.get("instruction"),
        "cache_keys": {"poi_search": [poi_key], "routes": []},
    }


def _load_origins(path: str | Path | None) -> list[dict[str, Any]]:
    """Load origins or return defaults.

    Args:
        path: Optional JSON/JSONL origin file.

    Returns:
        List of origin dictionaries.
    """

    if path:
        return [dict(item) for item in _iter_json_records(path)]
    return [
        {"city": "hangzhou", "scene_id": "hangzhou-yuquan", "label": "浙大玉泉", "lat": 30.2639, "lng": 120.1236},
        {"city": "hangzhou", "scene_id": "hangzhou-westlake", "label": "西湖湖滨", "lat": 30.2587, "lng": 120.1634},
        {"city": "hangzhou", "scene_id": "hangzhou-xixi", "label": "西溪", "lat": 30.2704, "lng": 120.0648},
    ]


def _load_tasks(path: str | Path | None) -> list[dict[str, Any]]:
    """Load task templates or return defaults.

    Args:
        path: Optional JSON/JSONL task template file.

    Returns:
        List of task dictionaries.
    """

    if path:
        return [dict(item) for item in _iter_json_records(path)]
    return [
        {
            "category": "poi_navigation",
            "subcategory": "convenience_store",
            "difficulty": "easy",
            "instruction": "请帮我找最近的便利店。",
            "keywords": "便利店|超市",
        },
        {
            "category": "implicit_need_navigation",
            "subcategory": "thirsty",
            "difficulty": "medium",
            "instruction": "我有点渴了，帮我找附近可以买饮料的地方。",
            "keywords": "便利店|咖啡|饮品|超市",
        },
        {
            "category": "poi_navigation",
            "subcategory": "restaurant",
            "difficulty": "easy",
            "instruction": "请帮我找最近的餐厅。",
            "keywords": "餐厅|饭店|小吃",
        },
        {
            "category": "poi_navigation",
            "subcategory": "atm",
            "difficulty": "hard",
            "instruction": "帮我找一个附近的 ATM。",
            "keywords": "ATM|自动取款机|银行",
        },
        {
            "category": "implicit_need_navigation",
            "subcategory": "rest",
            "difficulty": "medium",
            "instruction": "我想找个地方坐一会儿休息。",
            "keywords": "咖啡|公园|广场|茶",
        },
    ]


def _multi_target_live_tasks() -> list[dict[str, Any]]:
    """Return live-builder task templates with multiple suitable POI types.

    Returns:
        List of multi-target task templates.
    """

    return [
        {
            "category": "implicit_need_navigation",
            "subcategory": "buy_drink_multi_target",
            "difficulty": "medium",
            "instruction": "我想买点喝的，便利店、咖啡店或饮品店都可以，帮我找一个合适的。",
            "keywords": "便利店|咖啡|饮品|超市",
        },
        {
            "category": "implicit_need_navigation",
            "subcategory": "rest_multi_target",
            "difficulty": "medium",
            "instruction": "我想找个地方坐一会儿，公园、广场或咖啡店都可以。",
            "keywords": "公园|广场|咖啡|茶",
        },
    ]


def _iter_json_records(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield JSON records from JSONL, JSON array, or wrapped JSON.

    Args:
        path: Input path.

    Returns:
        Iterable of dictionaries.
    """

    source = Path(path)
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if source.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("origins", "tasks", "records", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def _write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    """Write records to JSONL.

    Args:
        records: Iterable of JSON-serializable dictionaries.
        path: Output path.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_mock_builder() -> Any:
    """Load the existing mock builder module.

    Returns:
        Imported mock builder module.
    """

    path = Path(__file__).with_name("build_urbannav_mock_benchmark.py")
    spec = importlib.util.spec_from_file_location("build_urbannav_mock_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    if not spec or not spec.loader:
        raise RuntimeError(f"Cannot load mock builder from {path}")
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
