#!/usr/bin/env python3
"""Build a deterministic mock UrbanNav benchmark package.

This builder is Step 6 of the implementation plan. It does not call live map
providers. Its purpose is to produce a small package that exercises the GT
schema, replay cache, validator, and evaluator end to end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mcp_server.core.replay_map_backend import ReplayPoiResult, ReplayRouteResult
from mcp_server.eval.urbannav.urbannav_io import write_urbannav_cases
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavCase
from mcp_server.eval.urbannav.urbannav_validator import validate_urbannav_cases


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with output directory and case count.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path, help="Directory for the mock benchmark package")
    parser.add_argument("--num-cases", type=int, default=5, help="Number of deterministic cases to emit")
    parser.add_argument("--include-multi-target", action="store_true", help="Append one explicit multi-target case")
    parser.add_argument("--include-route-updates", action="store_true", help="Append one explicit route-update case")
    return parser.parse_args()


def main() -> None:
    """Build the mock package and write validation report.

    Returns:
        None.
    """

    args = parse_args()
    build_mock_benchmark(
        args.output_root,
        num_cases=args.num_cases,
        include_multi_target=args.include_multi_target,
        include_route_updates=args.include_route_updates,
    )


def build_mock_benchmark(
    output_root: str | Path,
    *,
    num_cases: int = 5,
    include_multi_target: bool = False,
    include_route_updates: bool = False,
) -> dict[str, Any]:
    r"""Build a deterministic mock benchmark package.

    Mock destination geometry uses a simple eastward route:

    \[
    d_i = (lng_0 + \Delta_i, lat_0)
    \]

    and assigns frozen route distances directly in meters. These distances are
    intentionally not derived from geographic distance; they are benchmark
    fixture values used to test evaluator semantics.

    Args:
        output_root: Output package directory.
        num_cases: Number of base cases to emit. The first five cover the main
            fake review scenarios; larger values repeat the pattern with
            unique ids.
        include_multi_target: Whether to append one case with multiple
            acceptable destinations.
        include_route_updates: Whether to append one case whose effective
            origin/instruction come from route-update metadata.

    Returns:
        Dictionary containing output paths and validation summary.
    """

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    cases: list[UrbanNavCase] = []
    poi_records: list[ReplayPoiResult] = []
    route_records: list[ReplayRouteResult] = []
    rejects: list[dict[str, Any]] = []

    templates = _mock_templates()
    for index in range(max(0, num_cases)):
        template = templates[index % len(templates)]
        case_payload, poi_record, route_items = _build_case_payload(index=index, template=template)
        cases.append(UrbanNavCase.from_dict(case_payload))
        poi_records.append(poi_record)
        route_records.extend(route_items)
    if include_multi_target:
        case_payload, poi_record, route_items = _build_case_payload(
            index=max(0, num_cases),
            template=_multi_target_template(),
            case_id_override="mock-multi-target-001",
        )
        cases.append(UrbanNavCase.from_dict(case_payload))
        poi_records.append(poi_record)
        route_records.extend(route_items)
    if include_route_updates:
        case_payload, poi_record, route_items = _build_route_update_case_payload(index=max(0, num_cases) + 1)
        cases.append(UrbanNavCase.from_dict(case_payload))
        poi_records.append(poi_record)
        route_records.extend(route_items)

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


def _build_case_payload(
    *,
    index: int,
    template: dict[str, Any],
    case_id_override: str | None = None,
) -> tuple[dict[str, Any], ReplayPoiResult, list[ReplayRouteResult]]:
    """Build one mock case payload and its replay cache records.

    Args:
        index: Zero-based case index.
        template: Mock task template.
        case_id_override: Optional explicit case id.

    Returns:
        Tuple of ``(case_payload, poi_record, route_records)``.
    """

    case_id = case_id_override or f"mock-{index + 1:03d}"
    origin = {"lat": 30.0 + 0.001 * (index // 10), "lng": 120.0}
    candidates = []
    acceptable = []
    ranked = []
    route_records: list[ReplayRouteResult] = []
    for rank, candidate_template in enumerate(template["candidates"], start=1):
        poi_id = f"{case_id}-{candidate_template['suffix']}"
        lng = 120.0 + 0.001 * rank
        destination = {
            "provider_poi_id": poi_id,
            "name": candidate_template["name"],
            "type": candidate_template["type"],
            "lat": origin["lat"],
            "lng": lng,
            "rank": rank,
            "reference_route_distance_m": float(candidate_template["distance_m"]),
            "reference_route": [[origin["lng"], origin["lat"]], [lng, origin["lat"]]],
        }
        candidates.append(destination)
        ranked.append(
            {
                "rank": rank,
                "provider_poi_id": poi_id,
                "name": candidate_template["name"],
                "type": candidate_template["type"],
                "lat": origin["lat"],
                "lng": lng,
                "route_distance_m": float(candidate_template["distance_m"]),
                "score_breakdown": {"relevance": 1.0, "route_distance_m": float(candidate_template["distance_m"])},
            }
        )
        if candidate_template.get("acceptable", True):
            acceptable.append(destination)
        route_key = f"route:{case_id}:{poi_id}"
        route_records.append(
            ReplayRouteResult(
                cache_key=route_key,
                provider="mock",
                request={"origin": origin, "destination": destination, "mode": "foot"},
                response={"status": "1", "info": "mock"},
                polyline=[_point(origin["lat"], origin["lng"]), _point(origin["lat"], lng)],
                distance_m=float(candidate_template["distance_m"]),
                duration_s=float(candidate_template["distance_m"]) / 1.2,
            )
        )

    gt = candidates[0]
    poi_key = f"poi:{case_id}:{template['subcategory']}"
    case_payload = {
        "id": case_id,
        "city": template["city"],
        "scene_id": template["scene_id"],
        "category": template["category"],
        "subcategory": template["subcategory"],
        "difficulty": template["difficulty"],
        "instruction": template["instruction"],
        "origin": origin,
        "gt_destination": gt,
        "acceptable_destinations": acceptable,
        "ranked_destinations": ranked,
        "gt_route": gt["reference_route"],
        "gt_distance_m": gt["reference_route_distance_m"],
        "optimality_scope": {
            "poi_provider": "mock",
            "search_radius_m": 1500,
            "candidate_limit": 30,
            "route_provider": "mock",
            "route_mode": "foot",
            "route_strategy": "default",
            "cost": "route_distance_m",
            "builder_version": "urbannav_mock_v1",
        },
        "cache_keys": {
            "poi_search": [poi_key],
            "routes": [record.cache_key for record in route_records],
        },
    }
    poi_record = ReplayPoiResult(
        cache_key=poi_key,
        provider="mock",
        request={"origin": origin, "instruction": template["instruction"], "subcategory": template["subcategory"]},
        response={"status": "1", "info": "mock"},
        candidates=[UrbanNavCase.from_dict(case_payload).ranked_destinations[i] for i in range(len(ranked))],
    )
    return case_payload, poi_record, route_records


def _build_route_update_case_payload(
    *,
    index: int,
) -> tuple[dict[str, Any], ReplayPoiResult, list[ReplayRouteResult]]:
    """Build one mock route-update case.

    Args:
        index: Base index used to vary coordinates.

    Returns:
        Tuple of ``(case_payload, poi_record, route_records)``.
    """

    template = {
        "city": "hangzhou",
        "scene_id": "route-update-lab",
        "category": "route_update",
        "subcategory": "replace_destination",
        "difficulty": "medium",
        "instruction": "先去咖啡店。",
        "candidates": [
            {"suffix": "updated-a", "name": "便利店 Update A", "type": "convenience_store", "distance_m": 80.0},
            {"suffix": "updated-b", "name": "便利店 Update B", "type": "convenience_store", "distance_m": 140.0},
        ],
    }
    case_payload, poi_record, route_records = _build_case_payload(
        index=index,
        template=template,
        case_id_override="mock-route-update-001",
    )
    current_origin = {"lat": case_payload["origin"]["lat"] + 0.0005, "lng": case_payload["origin"]["lng"] + 0.0005}
    case_payload["instruction"] = "我临时想改去可以买水的地方。"
    case_payload["origin"] = current_origin
    case_payload["gt_route"] = [
        current_origin,
        {
            "lat": case_payload["gt_destination"]["lat"],
            "lng": case_payload["gt_destination"]["lng"],
        },
    ]
    case_payload["gt_destination"]["reference_route"] = case_payload["gt_route"]
    for destination in case_payload.get("acceptable_destinations", []):
        destination["reference_route"] = [
            current_origin,
            {"lat": destination["lat"], "lng": destination["lng"]},
        ]
    for destination in case_payload.get("ranked_destinations", []):
        destination["reference_route"] = [
            current_origin,
            {"lat": destination["lat"], "lng": destination["lng"]},
        ]
    case_payload["metadata"] = {
        "route_update": {
            "update_type": "replace_destination",
            "previous_instruction": "先去咖啡店。",
            "updated_instruction": "我临时想改去可以买水的地方。",
            "current_origin": current_origin,
            "previous_destination": {"name": "咖啡店 Previous", "type": "cafe", "lat": current_origin["lat"], "lng": current_origin["lng"] + 0.001},
        }
    }
    updated_route_records: list[ReplayRouteResult] = []
    for record in route_records:
        destination = dict(record.request.get("destination") or {})
        updated_route_records.append(
            ReplayRouteResult(
                cache_key=record.cache_key,
                provider=record.provider,
                request={"origin": current_origin, "destination": destination, "mode": "foot"},
                response=record.response,
                polyline=[
                    _point(current_origin["lat"], current_origin["lng"]),
                    _point(float(destination.get("lat")), float(destination.get("lng"))),
                ],
                distance_m=record.distance_m,
                duration_s=record.duration_s,
            )
        )
    return case_payload, poi_record, updated_route_records


def _mock_templates() -> list[dict[str, Any]]:
    """Return deterministic mock task templates.

    Returns:
        List of task templates used by ``build_mock_benchmark``.
    """

    return [
        {
            "city": "hangzhou",
            "scene_id": "yuquan",
            "category": "poi_navigation",
            "subcategory": "convenience_store",
            "difficulty": "easy",
            "instruction": "请帮我找最近的便利店。",
            "candidates": [{"suffix": "poi-a", "name": "便利店 A", "type": "convenience_store", "distance_m": 100.0}],
        },
        {
            "city": "hangzhou",
            "scene_id": "yuquan",
            "category": "implicit_need_navigation",
            "subcategory": "thirsty",
            "difficulty": "medium",
            "instruction": "我有点渴了，帮我找附近可以买饮料的地方。",
            "candidates": [
                {"suffix": "poi-b1", "name": "便利店 B1", "type": "convenience_store", "distance_m": 150.0},
                {"suffix": "poi-b2", "name": "咖啡店 B2", "type": "cafe", "distance_m": 200.0},
            ],
        },
        {
            "city": "hangzhou",
            "scene_id": "yuquan",
            "category": "poi_navigation",
            "subcategory": "restaurant",
            "difficulty": "easy",
            "instruction": "请帮我找最近的餐厅。",
            "candidates": [{"suffix": "poi-c", "name": "餐厅 C", "type": "restaurant", "distance_m": 110.0}],
        },
        {
            "city": "hangzhou",
            "scene_id": "xixi",
            "category": "poi_navigation",
            "subcategory": "atm",
            "difficulty": "hard",
            "instruction": "帮我找一个附近的 ATM。",
            "candidates": [{"suffix": "poi-d", "name": "ATM D", "type": "atm", "distance_m": 130.0}],
        },
        {
            "city": "hangzhou",
            "scene_id": "xixi",
            "category": "implicit_need_navigation",
            "subcategory": "rest",
            "difficulty": "medium",
            "instruction": "我想找个地方坐一会儿休息。",
            "candidates": [{"suffix": "poi-e", "name": "咖啡店 E", "type": "cafe", "distance_m": 120.0}],
        },
    ]


def _multi_target_template() -> dict[str, Any]:
    """Return a mock case template with multiple acceptable targets.

    Returns:
        Multi-target task template.
    """

    return {
        "city": "hangzhou",
        "scene_id": "multi-target-lab",
        "category": "implicit_need_navigation",
        "subcategory": "buy_drink_multi_target",
        "difficulty": "medium",
        "instruction": "我想买点喝的，便利店或咖啡店都可以。",
        "candidates": [
            {"suffix": "drink-a", "name": "便利店 Drink A", "type": "convenience_store", "distance_m": 100.0},
            {"suffix": "drink-b", "name": "咖啡店 Drink B", "type": "cafe", "distance_m": 108.0},
            {"suffix": "drink-c", "name": "餐厅 Drink C", "type": "restaurant", "distance_m": 180.0, "acceptable": False},
        ],
    }


def _point(lat: float, lng: float) -> Any:
    """Create a lightweight point object for replay records.

    Args:
        lat: Latitude.
        lng: Longitude.

    Returns:
        ``UrbanNavPoint`` instance.
    """

    from mcp_server.eval.urbannav.urbannav_schema import UrbanNavPoint

    return UrbanNavPoint(lat=lat, lng=lng)


def _write_jsonl(records: Any, path: Path) -> None:
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


if __name__ == "__main__":
    main()
