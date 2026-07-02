#!/usr/bin/env python3
"""Finalize sharded UrbanNav GT collection outputs into one benchmark package."""

from __future__ import annotations

import argparse
import json
import random
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

    package = _load_sharded_outputs(args.input_root)
    selected_cases, quotas = _select_balanced_cases(
        package.cases,
        target_cases=args.target_cases,
        category_weights=DEFAULT_CATEGORY_WEIGHTS,
        seed=args.seed,
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
    )
    (output_root / "dataset_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": "urbannav.final_package.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_roots": [str(path) for path in args.input_root],
        "output_root": str(output_root),
        "target_cases": args.target_cases,
        "selected_case_count": len(selected_cases),
        "selected_case_ids": {
            "count": len(selected_ids),
            "unique": len(selected_ids) == len(selected_cases),
        },
        "category_weights": DEFAULT_CATEGORY_WEIGHTS,
        "category_quotas": quotas,
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
    """

    def __init__(
        self,
        *,
        cases: list[dict[str, Any]],
        poi_cache: dict[str, dict[str, Any]],
        route_cache: dict[str, dict[str, Any]],
        rejects: list[dict[str, Any]],
    ) -> None:
        self.cases = cases
        self.poi_cache = poi_cache
        self.route_cache = route_cache
        self.rejects = rejects


def _load_sharded_outputs(input_roots: list[Path]) -> ShardedPackage:
    """Load and de-duplicate sharded GT/cache/reject outputs.

    Args:
        input_roots: Collection roots containing ``*_shard_*`` directories.

    Returns:
        ``ShardedPackage`` with one record per case/cache key.
    """

    cases_by_id: dict[str, dict[str, Any]] = {}
    poi_cache: dict[str, dict[str, Any]] = {}
    route_cache: dict[str, dict[str, Any]] = {}
    rejects_by_key: dict[str, dict[str, Any]] = {}

    for root in input_roots:
        shard_dirs = sorted(path for path in root.glob("*_shard_*") if path.is_dir())
        for shard_dir in shard_dirs:
            for record in _iter_jsonl(shard_dir / "task_gt.jsonl"):
                case_id = str(record.get("id") or "")
                if case_id:
                    cases_by_id.setdefault(case_id, record)
            for record in _iter_jsonl(shard_dir / "poi_cache.jsonl"):
                cache_key = str(record.get("cache_key") or "")
                if cache_key:
                    poi_cache.setdefault(cache_key, record)
            for record in _iter_jsonl(shard_dir / "route_cache.jsonl"):
                cache_key = str(record.get("cache_key") or "")
                if cache_key:
                    route_cache.setdefault(cache_key, record)
            for record in _iter_jsonl(shard_dir / "task_rejects.jsonl"):
                key = _reject_key(record)
                rejects_by_key.setdefault(key, record)

    return ShardedPackage(
        cases=list(cases_by_id.values()),
        poi_cache=poi_cache,
        route_cache=route_cache,
        rejects=list(rejects_by_key.values()),
    )


def _select_balanced_cases(
    cases: list[dict[str, Any]],
    *,
    target_cases: int,
    category_weights: dict[str, float],
    seed: int,
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

    Returns:
        Pair of selected cases and actual category quotas.
    """

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_category[str(case.get("category") or "unknown")].append(case)

    quotas = _allocate_quotas(
        available={category: len(items) for category, items in by_category.items()},
        target_cases=min(target_cases, len(cases)),
        weights=category_weights,
    )
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

    categories = sorted(available)
    remaining_categories = set(categories)
    quotas = {category: 0 for category in categories}
    remaining_target = target_cases
    while remaining_categories and remaining_target > 0:
        weight_sum = sum(weights.get(category, 0.01) for category in remaining_categories)
        progressed = False
        for category in sorted(list(remaining_categories)):
            desired = round(remaining_target * weights.get(category, 0.01) / weight_sum)
            desired = max(1, desired) if remaining_target > 0 else 0
            capacity = available[category] - quotas[category]
            take = min(desired, capacity)
            quotas[category] += take
            remaining_target -= take
            progressed = progressed or take > 0
            if quotas[category] >= available[category]:
                remaining_categories.remove(category)
        if not progressed:
            break
    if remaining_target > 0:
        for category in categories:
            if remaining_target <= 0:
                break
            capacity = available[category] - quotas[category]
            take = min(capacity, remaining_target)
            quotas[category] += take
            remaining_target -= take
    return quotas


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
) -> dict[str, Any]:
    """Build final dataset statistics.

    Args:
        raw_cases: All de-duplicated source cases.
        selected_cases: Final selected cases.
        selected_poi_cache: POI cache rows included in final package.
        selected_route_cache: Route cache rows included in final package.
        rejects: De-duplicated reject records.
        quotas: Actual category quotas used by selection.

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
