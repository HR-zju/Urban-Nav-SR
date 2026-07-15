#!/usr/bin/env python3
"""Split an UrbanNav instruction pool into deterministic category-balanced shards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


SINGLE_CATEGORIES = {"poi_navigation", "implicit_need_navigation", "public_poi_navigation"}


def parse_args() -> argparse.Namespace:
    """Parse instruction-pool split arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction-set", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--single-shards", type=int, default=8)
    parser.add_argument("--complex-shards", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    """Split the pool and print the split manifest."""

    args = parse_args()
    rows = [json.loads(line) for line in args.instruction_set.read_text(encoding="utf-8").splitlines() if line.strip()]
    single = [row for row in rows if str(row.get("category") or "") in SINGLE_CATEGORIES]
    complex_rows = [row for row in rows if row not in single]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "instruction_set": str(args.instruction_set),
        "single": _write_balanced_shards(single, args.output_root, "single", args.single_shards),
        "complex": _write_balanced_shards(complex_rows, args.output_root, "complex", args.complex_shards),
    }
    (args.output_root / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _write_balanced_shards(
    rows: list[dict[str, Any]],
    output_root: Path,
    prefix: str,
    shard_count: int,
) -> dict[str, Any]:
    """Write category round-robin shards with stable row ordering."""

    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category") or "unknown")].append(row)
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for category in sorted(grouped):
        category_rows = sorted(grouped[category], key=lambda row: str(row.get("instruction_id") or row.get("id") or ""))
        for index, row in enumerate(category_rows):
            buckets[index % shard_count].append(row)
    shard_stats: list[dict[str, Any]] = []
    for shard_index, bucket in enumerate(buckets):
        bucket = _interleave_categories(bucket)
        path = output_root / f"{prefix}_shard_{shard_index:02d}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bucket), encoding="utf-8")
        shard_stats.append(
            {
                "shard": shard_index,
                "path": str(path),
                "rows": len(bucket),
                "category_distribution": dict(Counter(str(row.get("category") or "unknown") for row in bucket)),
            }
        )
    return {
        "rows": len(rows),
        "shards": shard_count,
        "category_distribution": dict(Counter(str(row.get("category") or "unknown") for row in rows)),
        "shard_stats": shard_stats,
    }


def _interleave_categories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Interleave categories while preserving stable within-category order.

    Args:
        rows: Rows assigned to one shard.

    Returns:
        Category round-robin rows. This prevents an interrupted resumable
        collection from containing only the lexicographically first category.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("category") or "unknown")].append(row)
    for category_rows in grouped.values():
        category_rows.sort(key=lambda row: str(row.get("instruction_id") or row.get("id") or ""))
    categories = sorted(grouped)
    interleaved: list[dict[str, Any]] = []
    max_rows = max((len(grouped[category]) for category in categories), default=0)
    for row_index in range(max_rows):
        for category in categories:
            if row_index < len(grouped[category]):
                interleaved.append(grouped[category][row_index])
    return interleaved


if __name__ == "__main__":
    main()
