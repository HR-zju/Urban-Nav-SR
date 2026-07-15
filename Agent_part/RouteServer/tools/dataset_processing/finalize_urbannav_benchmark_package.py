#!/usr/bin/env python3
"""Finalize sharded UrbanNav GT collection outputs into one benchmark package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CATEGORY_WEIGHTS = {
    "poi_navigation": 0.40,
    "implicit_need_navigation": 0.35,
    "multi_stop_route": 0.15,
    "route_update_add_via": 0.10,
}
_PACKAGE_FILENAMES = (
    "task_gt.jsonl",
    "poi_cache.jsonl",
    "route_cache.jsonl",
    "task_rejects.jsonl",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments containing input shard roots, final output root, target
        case count, and deterministic balancing options.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True, type=Path, help="Sharded collection root. Can be repeated.")
    parser.add_argument("--output-root", required=True, type=Path, help="Final benchmark package directory.")
    parser.add_argument("--target-cases", type=int, default=15000, help="Number of final GT cases.")
    parser.add_argument("--distribution-plan", type=Path, default=None, help="Optional exact city/category distribution plan.")
    parser.add_argument("--quality-policy", type=Path, default=None, help="Optional v2 eligibility policy applied before selection.")
    parser.add_argument(
        "--semantic-origin-decimals",
        type=int,
        default=5,
        help="Coordinate precision used by semantic duplicate fingerprints.",
    )
    parser.add_argument("--review-cases", type=int, default=300, help="Balanced review sample size.")
    parser.add_argument("--seed", type=int, default=20260630, help="Deterministic balancing seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing final package directory.")
    return parser.parse_args()


def main() -> None:
    """Finalize a benchmark package.

    Returns:
        None. Writes JSONL files, statistics, and manifest to ``output_root``.
    """

    args = parse_args()
    output_root = args.output_root
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace package files")
    output_root.mkdir(parents=True, exist_ok=True)

    package = _load_sharded_outputs(
        args.input_root,
        semantic_origin_decimals=args.semantic_origin_decimals,
    )
    distribution = _load_distribution_plan(args.distribution_plan) if args.distribution_plan else None
    quality_policy = _load_json_object(args.quality_policy) if args.quality_policy else None
    eligible_cases, quality_exclusion_counts = _filter_quality_eligible_cases(
        package.cases,
        quality_policy=quality_policy,
        route_cache=package.route_cache,
    )
    target_cases = int(distribution.get("target_cases") or args.target_cases) if distribution else args.target_cases
    category_quotas = _planned_counts(distribution, "task_distribution", "category") if distribution else None
    city_quotas = _planned_counts(distribution, "city_distribution", "city") if distribution else None
    action_quotas = _planned_counts(distribution, "route_update_action_distribution", "action") if distribution else None
    selected_cases, quotas = _select_balanced_cases(
        eligible_cases,
        target_cases=target_cases,
        category_weights=DEFAULT_CATEGORY_WEIGHTS,
        seed=args.seed,
        exact_category_quotas=category_quotas,
        exact_city_quotas=city_quotas,
        exact_route_update_action_quotas=action_quotas,
    )
    selected_ids = {str(case["id"]) for case in selected_cases}
    poi_keys, route_keys = _referenced_cache_keys(selected_cases)
    selected_poi_cache = [package.poi_cache[key] for key in sorted(poi_keys) if key in package.poi_cache]
    selected_route_cache = [package.route_cache[key] for key in sorted(route_keys) if key in package.route_cache]

    _write_jsonl(output_root / "task_gt.jsonl", selected_cases)
    _write_jsonl(output_root / "poi_cache.jsonl", selected_poi_cache)
    _write_jsonl(output_root / "route_cache.jsonl", selected_route_cache)
    _write_jsonl(output_root / "task_rejects.jsonl", package.rejects)

    review_sample = _select_review_sample(selected_cases, target_cases=args.review_cases, seed=args.seed + 17)
    review_dir = output_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(review_dir / "task_gt_review_sample.jsonl", review_sample)

    stats = _build_statistics(
        raw_cases=package.cases,
        selected_cases=selected_cases,
        selected_poi_cache=selected_poi_cache,
        selected_route_cache=selected_route_cache,
        rejects=package.rejects,
        quotas=quotas,
        semantic_duplicate_count=package.semantic_duplicate_count,
        quality_exclusion_counts=quality_exclusion_counts,
    )
    (output_root / "dataset_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": "urbannav.final_package.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_roots": [str(path) for path in args.input_root],
        "output_root": str(output_root),
        "target_cases": target_cases,
        "selected_case_count": len(selected_cases),
        "selected_case_ids": {
            "count": len(selected_ids),
            "unique": len(selected_ids) == len(selected_cases),
        },
        "category_weights": DEFAULT_CATEGORY_WEIGHTS,
        "category_quotas": quotas,
        "distribution_plan": str(args.distribution_plan) if args.distribution_plan else None,
        "quality_policy": str(args.quality_policy) if args.quality_policy else None,
        "city_quotas": city_quotas,
        "route_update_action_quotas": action_quotas,
        "semantic_origin_decimals": args.semantic_origin_decimals,
        "semantic_duplicate_count": package.semantic_duplicate_count,
        "quality_exclusion_counts": quality_exclusion_counts,
        "files": {
            "ground_truth": "task_gt.jsonl",
            "poi_cache": "poi_cache.jsonl",
            "route_cache": "route_cache.jsonl",
            "rejects": "task_rejects.jsonl",
            "statistics": "dataset_statistics.json",
            "review_sample": "review/task_gt_review_sample.jsonl",
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), **stats["selected"]}, ensure_ascii=False, indent=2))


class ShardedPackage:
    """In-memory records loaded from sharded collection roots.

    Args:
        cases: De-duplicated GT case dictionaries keyed by ``id``.
        poi_cache: POI cache dictionaries keyed by ``cache_key``.
        route_cache: Route cache dictionaries keyed by ``cache_key``.
        rejects: De-duplicated reject rows.
        semantic_duplicate_count: Cases removed by semantic fingerprint after
            case-id de-duplication.
    """

    def __init__(
        self,
        *,
        cases: list[dict[str, Any]],
        poi_cache: dict[str, dict[str, Any]],
        route_cache: dict[str, dict[str, Any]],
        rejects: list[dict[str, Any]],
        semantic_duplicate_count: int = 0,
    ) -> None:
        self.cases = cases
        self.poi_cache = poi_cache
        self.route_cache = route_cache
        self.rejects = rejects
        self.semantic_duplicate_count = semantic_duplicate_count


def _load_sharded_outputs(
    input_roots: list[Path],
    *,
    semantic_origin_decimals: int = 5,
) -> ShardedPackage:
    """Load and de-duplicate sharded GT/cache/reject outputs.

    Args:
        input_roots: Collection roots containing flat files or ``*_shard_*``
            directories.
        semantic_origin_decimals: Origin precision for semantic de-duplication.

    Returns:
        ``ShardedPackage`` with one record per case/cache key.
    """

    cases_by_id: dict[str, dict[str, Any]] = {}
    poi_cache: dict[str, dict[str, Any]] = {}
    route_cache: dict[str, dict[str, Any]] = {}
    rejects_by_key: dict[str, dict[str, Any]] = {}

    for root in input_roots:
        record_dirs = [root] if any((root / name).exists() for name in _PACKAGE_FILENAMES) else []
        record_dirs.extend(sorted(path for path in root.glob("*_shard_*") if path.is_dir()))
        for record_dir in record_dirs:
            for record in _iter_jsonl(record_dir / "task_gt.jsonl"):
                case_id = str(record.get("id") or "")
                if case_id:
                    _insert_consistent(cases_by_id, case_id, record, record_kind="case")
            for record in _iter_jsonl(record_dir / "poi_cache.jsonl"):
                cache_key = str(record.get("cache_key") or "")
                if cache_key:
                    _insert_consistent(poi_cache, cache_key, record, record_kind="POI cache")
            for record in _iter_jsonl(record_dir / "route_cache.jsonl"):
                cache_key = str(record.get("cache_key") or "")
                if cache_key:
                    _insert_consistent(route_cache, cache_key, record, record_kind="route cache")
            for record in _iter_jsonl(record_dir / "task_rejects.jsonl"):
                key = _reject_key(record)
                rejects_by_key.setdefault(key, record)

    semantically_unique: dict[str, dict[str, Any]] = {}
    for case in sorted(cases_by_id.values(), key=lambda item: str(item.get("id") or "")):
        fingerprint = _semantic_case_fingerprint(case, origin_decimals=semantic_origin_decimals)
        semantically_unique.setdefault(fingerprint, case)
    semantic_duplicate_count = len(cases_by_id) - len(semantically_unique)

    return ShardedPackage(
        cases=list(semantically_unique.values()),
        poi_cache=poi_cache,
        route_cache=route_cache,
        rejects=list(rejects_by_key.values()),
        semantic_duplicate_count=semantic_duplicate_count,
    )


def _select_balanced_cases(
    cases: list[dict[str, Any]],
    *,
    target_cases: int,
    category_weights: dict[str, float],
    seed: int,
    exact_category_quotas: dict[str, int] | None = None,
    exact_city_quotas: dict[str, int] | None = None,
    exact_route_update_action_quotas: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    r"""Select a deterministic category-balanced final set.

    The ideal category quota is:

    ``q_c = round(N * w_c / sum_c w_c)``

    If a category does not have enough valid cases, its deficit is
    re-distributed over categories with remaining capacity.

    Args:
        cases: Candidate GT cases.
        target_cases: Desired final number of cases.
        category_weights: Target category mixture.
        seed: Random seed for deterministic within-group ordering.
        exact_category_quotas: Optional exact category quotas.
        exact_city_quotas: Optional exact city quotas. When both exact quota
            maps are supplied, selection enforces their joint matrix.
        exact_route_update_action_quotas: Optional exact action mixture within
            the route-update category.

    Returns:
        Pair of selected cases and actual category quotas.
    """

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_category[str(case.get("category") or "unknown")].append(case)

    if exact_category_quotas is not None:
        quotas = dict(exact_category_quotas)
        if sum(quotas.values()) != target_cases:
            raise ValueError(f"Exact category quotas sum to {sum(quotas.values())}, expected {target_cases}")
    else:
        quotas = _allocate_quotas(
            available={category: len(items) for category, items in by_category.items()},
            target_cases=min(target_cases, len(cases)),
            weights=category_weights,
        )

    if exact_city_quotas is not None:
        if sum(exact_city_quotas.values()) != target_cases:
            raise ValueError(f"Exact city quotas sum to {sum(exact_city_quotas.values())}, expected {target_cases}")
        strata_quotas = _selection_strata_quotas(
            category_quotas=quotas,
            route_update_action_quotas=exact_route_update_action_quotas,
        )
        matrix = _joint_quota_matrix(city_quotas=exact_city_quotas, category_quotas=strata_quotas)
        selected = _select_joint_quota_cases(cases, matrix=matrix, seed=seed)
        return _interleave_cases(selected, seed=seed + 211), dict(quotas)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for category in sorted(quotas):
        category_cases = _round_robin_by_subcategory_city(by_category.get(category, []), seed=seed + len(category))
        for case in category_cases[: quotas[category]]:
            selected.append(case)
            selected_ids.add(str(case.get("id")))

    if len(selected) < min(target_cases, len(cases)):
        remaining = [case for case in cases if str(case.get("id")) not in selected_ids]
        remaining = _round_robin_by_subcategory_city(remaining, seed=seed + 101)
        selected.extend(remaining[: min(target_cases, len(cases)) - len(selected)])

    return _interleave_cases(selected, seed=seed + 211)[:target_cases], Counter(str(case.get("category") or "unknown") for case in selected)


def _allocate_quotas(*, available: dict[str, int], target_cases: int, weights: dict[str, float]) -> dict[str, int]:
    """Allocate category quotas under availability constraints.

    Args:
        available: Available case count by category.
        target_cases: Desired total selected cases.
        weights: Target mixture weights.

    Returns:
        Actual selected quota by category.
    """

    categories = sorted(set(available).union(weights))
    available = {category: max(0, int(available.get(category, 0))) for category in categories}
    quotas = {category: 0 for category in categories}
    remaining_target = min(target_cases, sum(available.values()))
    active = {category for category in categories if available[category] > 0}
    while active and remaining_target > 0:
        capacities = {category: available[category] - quotas[category] for category in active}
        allocation = _largest_remainder_allocation(
            total=remaining_target,
            weights={category: max(0.0, float(weights.get(category, 0.0))) for category in active},
        )
        saturated = {category for category in active if allocation[category] >= capacities[category]}
        if saturated:
            for category in sorted(saturated):
                take = capacities[category]
                quotas[category] += take
                remaining_target -= take
                active.remove(category)
            continue
        for category in sorted(active):
            quotas[category] += allocation[category]
            remaining_target -= allocation[category]
        break
    return quotas


def _largest_remainder_allocation(*, total: int, weights: dict[str, float]) -> dict[str, int]:
    """Allocate an integer total simultaneously using largest remainders."""

    keys = sorted(weights)
    positive_sum = sum(max(0.0, weights[key]) for key in keys)
    normalized = {key: (max(0.0, weights[key]) if positive_sum > 0 else 1.0) for key in keys}
    weight_sum = sum(normalized.values())
    exact = {key: total * normalized[key] / weight_sum for key in keys}
    result = {key: int(exact[key]) for key in keys}
    remainder = total - sum(result.values())
    order = sorted(keys, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def _joint_quota_matrix(
    *,
    city_quotas: dict[str, int],
    category_quotas: dict[str, int],
) -> dict[tuple[str, str], int]:
    """Build a deterministic city-by-category matrix with exact margins."""

    remaining_categories = dict(category_quotas)
    remaining_total = sum(remaining_categories.values())
    matrix: dict[tuple[str, str], int] = {}
    for city in sorted(city_quotas):
        city_total = city_quotas[city]
        allocation = _largest_remainder_allocation(
            total=city_total,
            weights={category: float(count) for category, count in remaining_categories.items()},
        )
        for category in sorted(remaining_categories):
            take = min(allocation[category], remaining_categories[category])
            matrix[(_normalize_city(city), category)] = take
            remaining_categories[category] -= take
            remaining_total -= take
        shortfall = city_total - sum(matrix[(_normalize_city(city), category)] for category in remaining_categories)
        if shortfall:
            for category in sorted(remaining_categories, key=lambda key: (-remaining_categories[key], key)):
                take = min(shortfall, remaining_categories[category])
                matrix[(_normalize_city(city), category)] += take
                remaining_categories[category] -= take
                remaining_total -= take
                shortfall -= take
                if shortfall == 0:
                    break
        if shortfall:
            raise ValueError(f"Cannot allocate city quota for {city}: missing {shortfall}")
    if remaining_total or any(remaining_categories.values()):
        raise ValueError(f"Unallocated category quotas remain: {remaining_categories}")
    return matrix


def _select_joint_quota_cases(
    cases: list[dict[str, Any]],
    *,
    matrix: dict[tuple[str, str], int],
    seed: int,
) -> list[dict[str, Any]]:
    """Select cases under an exact city-by-category quota matrix."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[(_normalize_city(case.get("city")), _selection_stratum(case))].append(case)
    selected: list[dict[str, Any]] = []
    deficits: dict[str, int] = {}
    for cell, quota in sorted(matrix.items()):
        ordered = _round_robin_by_subcategory_city(grouped.get(cell, []), seed=seed + _stable_seed(cell))
        if len(ordered) < quota:
            deficits[f"{cell[0]}|{cell[1]}"] = quota - len(ordered)
        selected.extend(ordered[:quota])
    if deficits:
        raise ValueError(f"Insufficient valid cases for exact city/category quotas: {deficits}")
    return selected


def _selection_strata_quotas(
    *,
    category_quotas: dict[str, int],
    route_update_action_quotas: dict[str, int] | None,
) -> dict[str, int]:
    """Expand route-update category quotas into exact action strata."""

    strata = {
        category: count
        for category, count in category_quotas.items()
        if category != "route_update_add_via"
    }
    update_total = int(category_quotas.get("route_update_add_via", 0))
    if route_update_action_quotas is None:
        strata["route_update_add_via"] = update_total
        return strata
    if sum(route_update_action_quotas.values()) != update_total:
        raise ValueError(
            "Route-update action quotas sum to "
            f"{sum(route_update_action_quotas.values())}, expected {update_total}"
        )
    for action, count in route_update_action_quotas.items():
        strata[f"route_update_add_via/{action}"] = count
    return strata


def _selection_stratum(case: dict[str, Any]) -> str:
    """Return the category or route-update action stratum for one case."""

    category = str(case.get("category") or "unknown")
    if category != "route_update_add_via":
        return category
    metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
    update = metadata.get("route_update") if isinstance(metadata.get("route_update"), dict) else {}
    action = str(update.get("update_type") or case.get("route_update_action") or "").strip()
    return f"{category}/{action}" if action else category


def _round_robin_by_subcategory_city(cases: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Order cases by round-robin over ``(subcategory, city)`` groups.

    Args:
        cases: Candidate GT cases.
        seed: Deterministic shuffle seed.

    Returns:
        Balanced case order.
    """

    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[(str(case.get("subcategory") or "unknown"), str(case.get("city") or "unknown"))].append(case)
    for items in groups.values():
        rng.shuffle(items)
    keys = sorted(groups, key=lambda key: (-len(groups[key]), key[0], key[1]))
    output: list[dict[str, Any]] = []
    while keys:
        next_keys: list[tuple[str, str]] = []
        for key in keys:
            items = groups[key]
            if items:
                output.append(items.pop())
            if items:
                next_keys.append(key)
        keys = next_keys
    return output


def _interleave_cases(cases: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Interleave selected cases across categories for review-friendly ordering.

    Args:
        cases: Selected GT cases.
        seed: Deterministic shuffle seed.

    Returns:
        Interleaved cases.
    """

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_category[str(case.get("category") or "unknown")].append(case)
    rng = random.Random(seed)
    for items in by_category.values():
        rng.shuffle(items)
    categories = sorted(by_category, key=lambda category: -len(by_category[category]))
    output: list[dict[str, Any]] = []
    while categories:
        next_categories: list[str] = []
        for category in categories:
            items = by_category[category]
            if items:
                output.append(items.pop())
            if items:
                next_categories.append(category)
        categories = next_categories
    return output


def _select_review_sample(cases: list[dict[str, Any]], *, target_cases: int, seed: int) -> list[dict[str, Any]]:
    """Build a compact, diverse review subset.

    Args:
        cases: Final selected cases.
        target_cases: Maximum sample size.
        seed: Deterministic ordering seed.

    Returns:
        Review sample cases.
    """

    return _round_robin_by_subcategory_city(cases, seed=seed)[:target_cases]


def _filter_quality_eligible_cases(
    cases: list[dict[str, Any]],
    *,
    quality_policy: dict[str, Any] | None,
    route_cache: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter source cases that cannot satisfy the v2 package contract."""

    if quality_policy is None:
        return list(cases), {}
    multi = quality_policy.get("multi_stop") if isinstance(quality_policy.get("multi_stop"), dict) else {}
    min_stop_distance = float(multi.get("min_inter_stop_distance_m") or 0.0)
    min_leg_distance = float(multi.get("min_leg_route_distance_m") or 0.0)
    max_leg_distance = float(multi.get("max_leg_route_distance_m") or 0.0)
    require_disjoint = bool(multi.get("require_disjoint_target_types", False))
    require_matrix = multi.get("cache_policy") == "full_route_matrix"
    eligible: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for case in cases:
        if str(case.get("category") or "") != "multi_stop_route":
            eligible.append(case)
            continue
        reason = _multi_stop_quality_exclusion_reason(
            case,
            route_cache=route_cache,
            min_stop_distance_m=min_stop_distance,
            min_leg_distance_m=min_leg_distance,
            max_leg_distance_m=max_leg_distance,
            require_disjoint=require_disjoint,
            require_matrix=require_matrix,
        )
        if reason is None:
            eligible.append(case)
        else:
            excluded[reason] += 1
    return eligible, dict(excluded)


def _multi_stop_quality_exclusion_reason(
    case: dict[str, Any],
    *,
    route_cache: dict[str, dict[str, Any]],
    min_stop_distance_m: float,
    min_leg_distance_m: float,
    max_leg_distance_m: float,
    require_disjoint: bool,
    require_matrix: bool,
) -> str | None:
    """Return the first v2 multi-stop exclusion reason, if any."""

    metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
    groups = metadata.get("ordered_target_types")
    labels = [str(item).strip().lower() for item in metadata.get("ordered_goal_labels") or [] if str(item).strip()]
    if not isinstance(groups, list) or len(groups) < 2:
        return "missing_ordered_target_types"
    normalized_groups = [
        {str(item).strip().lower() for item in group if str(item).strip()}
        for group in groups
        if isinstance(group, list)
    ]
    if len(normalized_groups) != len(groups) or any(not group for group in normalized_groups):
        return "invalid_ordered_target_types"
    if require_disjoint and any(
        normalized_groups[left].intersection(normalized_groups[right])
        for left in range(len(normalized_groups))
        for right in range(left + 1, len(normalized_groups))
    ):
        return "overlapping_ordered_target_types"
    if len(labels) != len(groups) or len(set(labels)) != len(labels):
        return "invalid_ordered_goal_labels"
    legs = [item for item in case.get("gt_route_legs") or [] if isinstance(item, dict)]
    if len(legs) != len(groups):
        return "missing_gt_route_legs"
    destinations = [leg.get("gt_destination") for leg in legs]
    for left in range(len(destinations)):
        for right in range(left + 1, len(destinations)):
            distance = _mapping_distance_m(destinations[left], destinations[right])
            if distance is not None and distance < min_stop_distance_m:
                return "inter_stop_distance_too_short"
    if any(float(leg.get("distance_m") or 0.0) < min_leg_distance_m for leg in legs):
        return "leg_route_distance_too_short"
    if max_leg_distance_m > 0 and any(float(leg.get("distance_m") or 0.0) > max_leg_distance_m for leg in legs):
        return "leg_route_distance_too_long"
    if require_matrix:
        for leg in legs[1:]:
            matrix = leg.get("route_matrix") if isinstance(leg.get("route_matrix"), dict) else {}
            route_keys = [str(item) for item in matrix.get("route_keys") or []]
            predecessor_ids = matrix.get("predecessor_candidate_ids") or []
            destination_ids = matrix.get("destination_candidate_ids") or []
            if matrix.get("policy") != "full_route_matrix":
                return "missing_full_route_matrix"
            if len(route_keys) != len(predecessor_ids) * len(destination_ids):
                return "incomplete_route_matrix"
            if any(key not in route_cache for key in route_keys):
                return "missing_route_matrix_cache_key"
    return None


def _mapping_distance_m(first: Any, second: Any) -> float | None:
    """Compute distance between destination/point dictionaries."""

    if not isinstance(first, dict) or not isinstance(second, dict):
        return None
    try:
        lat1 = math.radians(float(first["lat"]))
        lat2 = math.radians(float(second["lat"]))
        dlat = lat2 - lat1
        dlng = math.radians(float(second["lng"]) - float(first["lng"]))
    except (KeyError, TypeError, ValueError):
        return None
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(math.sqrt(hav))


def _referenced_cache_keys(cases: Iterable[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """Collect cache keys referenced by selected cases.

    Args:
        cases: Final selected cases.

    Returns:
        Pair ``(poi_keys, route_keys)``.
    """

    poi_keys: set[str] = set()
    route_keys: set[str] = set()
    for case in cases:
        cache_keys = case.get("cache_keys") if isinstance(case.get("cache_keys"), dict) else {}
        for key in cache_keys.get("poi_search") or []:
            poi_keys.add(str(key))
        for key in cache_keys.get("routes") or []:
            route_keys.add(str(key))
    return poi_keys, route_keys


def _build_statistics(
    *,
    raw_cases: list[dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    selected_poi_cache: list[dict[str, Any]],
    selected_route_cache: list[dict[str, Any]],
    rejects: list[dict[str, Any]],
    quotas: dict[str, int],
    semantic_duplicate_count: int,
    quality_exclusion_counts: dict[str, int],
) -> dict[str, Any]:
    """Build final dataset statistics.

    Args:
        raw_cases: All de-duplicated source cases.
        selected_cases: Final selected cases.
        selected_poi_cache: POI cache rows included in final package.
        selected_route_cache: Route cache rows included in final package.
        rejects: De-duplicated reject records.
        quotas: Actual category quotas used by selection.
        semantic_duplicate_count: Number of source cases removed by semantic
            de-duplication.
        quality_exclusion_counts: Source cases excluded by v2 quality reason.

    Returns:
        JSON-ready statistics dictionary.
    """

    return {
        "raw": {
            "case_count": len(raw_cases),
            "category_distribution": _counter_dict(raw_cases, "category"),
            "city_distribution": _counter_dict(raw_cases, "city"),
            "subcategory_distribution": _nested_subcategory_counts(raw_cases),
            "reject_count": len(rejects),
            "reject_reason_distribution": _counter_dict(rejects, "reject_reason"),
            "semantic_duplicate_count": semantic_duplicate_count,
            "quality_exclusion_counts": quality_exclusion_counts,
        },
        "selected": {
            "case_count": len(selected_cases),
            "category_distribution": _counter_dict(selected_cases, "category"),
            "city_distribution": _counter_dict(selected_cases, "city"),
            "subcategory_distribution": _nested_subcategory_counts(selected_cases),
            "category_quotas": dict(quotas),
            "poi_cache_count": len(selected_poi_cache),
            "route_cache_count": len(selected_route_cache),
            "acceptable_destination_count_distribution": _acceptable_count_distribution(selected_cases),
            "multi_acceptable_destination_cases": sum(1 for case in selected_cases if len(case.get("acceptable_destinations") or []) > 1),
        },
    }


def _counter_dict(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    """Count records by a dictionary key.

    Args:
        records: JSON-like records.
        key: Record key to count.

    Returns:
        Count dictionary sorted by descending count.
    """

    counts = Counter(str(record.get(key) or "unknown") for record in records)
    return dict(counts.most_common())


def _nested_subcategory_counts(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Count subcategories within each category.

    Args:
        records: JSON-like GT case records.

    Returns:
        Nested ``category -> subcategory -> count`` dictionary.
    """

    nested: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        nested[str(record.get("category") or "unknown")][str(record.get("subcategory") or "unknown")] += 1
    return {category: dict(counter.most_common()) for category, counter in sorted(nested.items())}


def _acceptable_count_distribution(cases: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count cases by number of acceptable destinations.

    Args:
        cases: GT case records.

    Returns:
        Histogram keyed by acceptable destination count.
    """

    counts = Counter(str(len(case.get("acceptable_destinations") or [])) for case in cases)
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield dictionaries from a JSONL file if it exists.

    Args:
        path: JSONL path.

    Returns:
        Iterator of JSON dictionaries. Missing files yield no rows.
    """

    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries as JSONL.

    Args:
        path: Output path.
        records: JSON-ready dictionaries.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_distribution_plan(path: Path) -> dict[str, Any]:
    """Load and minimally validate a final-package distribution plan."""

    payload = _load_json_object(path)
    target = int(payload.get("target_cases") or 0)
    if target <= 0:
        raise ValueError(f"Distribution plan has invalid target_cases: {target}")
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _planned_counts(payload: dict[str, Any], list_key: str, item_key: str) -> dict[str, int]:
    """Extract one exact count map from a distribution plan."""

    counts: dict[str, int] = {}
    for item in payload.get(list_key) or []:
        key = str(item.get(item_key) or "").strip()
        if item_key == "city":
            key = _normalize_city(key)
        count = int(item.get("count") or 0)
        if key and count > 0:
            counts[key] = counts.get(key, 0) + count
    if not counts:
        raise ValueError(f"Distribution plan contains no positive {list_key} counts")
    return counts


def _insert_consistent(
    records: dict[str, dict[str, Any]],
    key: str,
    record: dict[str, Any],
    *,
    record_kind: str,
) -> None:
    """Insert one keyed record and reject conflicting duplicate payloads."""

    existing = records.get(key)
    if existing is None:
        records[key] = record
        return
    if _canonical_record(existing) != _canonical_record(record):
        raise ValueError(f"Conflicting {record_kind} records for key={key}")


def _canonical_record(record: dict[str, Any]) -> str:
    """Return deterministic JSON used for conflict detection."""

    ignored = {"created_at", "updated_at", "collected_at"}
    payload = {key: value for key, value in record.items() if key not in ignored}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _semantic_case_fingerprint(case: dict[str, Any], *, origin_decimals: int) -> str:
    """Build a case-id-independent semantic fingerprint."""

    origin = case.get("origin") if isinstance(case.get("origin"), dict) else {}
    normalized_instruction = re.sub(r"\s+", " ", str(case.get("instruction") or "").strip().lower())
    payload = {
        "city": _normalize_city(case.get("city")),
        "scene": str(case.get("scene") or case.get("region_id") or "").strip().lower(),
        "category": str(case.get("category") or "").strip().lower(),
        "subcategory": str(case.get("subcategory") or "").strip().lower(),
        "instruction": normalized_instruction,
        "origin": [
            round(float(origin.get("lat") or 0.0), origin_decimals),
            round(float(origin.get("lng") or 0.0), origin_decimals),
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_city(value: Any) -> str:
    """Normalize city labels used by records and distribution plans."""

    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _stable_seed(value: Any) -> int:
    """Return a process-independent small integer seed."""

    digest = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _reject_key(record: dict[str, Any]) -> str:
    """Build a stable reject de-duplication key.

    Args:
        record: Reject record.

    Returns:
        Stable string key.
    """

    case_id = str(record.get("id") or record.get("case_id") or "")
    reason = str(record.get("reject_reason") or record.get("reason") or "")
    if case_id or reason:
        return f"{case_id}|{reason}"
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    main()
