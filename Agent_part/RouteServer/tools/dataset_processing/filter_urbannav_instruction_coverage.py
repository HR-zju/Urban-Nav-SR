#!/usr/bin/env python3
"""Filter UrbanNav instructions using frozen city-by-target POI coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mcp_server.core.osm_local_index import iter_local_osm_pois
from mcp_server.eval.urbannav_collection_coverage import build_hotspot_target_coverage, filter_instruction_rows_by_coverage


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction-set", required=True, type=Path)
    parser.add_argument("--regions-config", required=True, type=Path)
    parser.add_argument("--router-config", required=True, type=Path)
    parser.add_argument("--poi-index-root", required=True, type=Path)
    parser.add_argument("--target-type-map", required=True, type=Path)
    parser.add_argument("--output-instruction-set", required=True, type=Path)
    parser.add_argument("--output-rejects", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    parser.add_argument("--search-radius-m", type=float, default=2500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build the coverage matrix and write only supported instructions.

    Returns:
        None.
    """

    args = parse_args()
    outputs = [args.output_instruction_set, args.output_rejects, args.output_report]
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError("Coverage outputs exist; pass --overwrite")
    regions_config = _load_object(args.regions_config)
    router_config = _load_object(args.router_config)
    target_map = _load_object(args.target_type_map)
    indexes = router_config.get("indexes") if isinstance(router_config.get("indexes"), dict) else {}
    records_by_asset_region: dict[str, Any] = {}
    for region in regions_config.get("regions", []):
        asset_region_id = str(region.get("asset_region_id") or region.get("region_id") or "")
        if asset_region_id in records_by_asset_region:
            continue
        index_value = indexes.get(asset_region_id) or indexes.get(str(region.get("region_id") or "")) or indexes.get("default")
        if not index_value:
            raise KeyError(f"No POI index configured for {asset_region_id}")
        index_path = Path(str(index_value))
        if not index_path.is_absolute():
            index_path = args.poi_index_root / index_path
        records_by_asset_region[asset_region_id] = list(iter_local_osm_pois(index_path))
    coverage = build_hotspot_target_coverage(
        regions_config.get("regions", []),
        records_by_asset_region=records_by_asset_region,
        search_radius_m=args.search_radius_m,
    )
    rows = [json.loads(line) for line in args.instruction_set.read_text(encoding="utf-8").splitlines() if line.strip()]
    accepted, rejected, report = filter_instruction_rows_by_coverage(rows, coverage=coverage, target_map=target_map)
    _write_jsonl(args.output_instruction_set, accepted)
    _write_jsonl(args.output_rejects, rejected)
    _write_json(args.output_report, report)
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected), "report": str(args.output_report)}, ensure_ascii=False))


def _load_object(path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one indented JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
