#!/usr/bin/env python3
"""Plan POI-dense UrbanNav subregions over shared city-level OSM assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mcp_server.core.osm_local_index import iter_local_osm_pois
from mcp_server.core.urbannav_poi_hotspots import plan_poi_hotspots, split_quota


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions-config", required=True, type=Path, help="Base city-level region config.")
    parser.add_argument("--router-config", required=True, type=Path, help="Base city-level router config.")
    parser.add_argument("--poi-index-root", required=True, type=Path, help="Directory containing city POI indexes.")
    parser.add_argument("--output-regions-config", required=True, type=Path, help="Output hotspot-region config.")
    parser.add_argument("--output-router-config", required=True, type=Path, help="Output router config with hotspot aliases.")
    parser.add_argument("--output-report", required=True, type=Path, help="Output hotspot audit report.")
    parser.add_argument("--hotspots-per-city", type=int, default=5, help="Maximum hotspots per city.")
    parser.add_argument("--hotspot-radius-m", type=int, default=800, help="Origin sampling radius around each hotspot.")
    parser.add_argument("--search-radius-m", type=int, default=2500, help="POI search radius used for hotspot scoring.")
    parser.add_argument("--grid-size-m", type=int, default=1000, help="Grid size used to generate candidate centers.")
    parser.add_argument("--min-separation-m", type=int, default=1800, help="Minimum separation between hotspots.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    return parser.parse_args()


def main() -> None:
    """Generate hotspot region aliases and an audit report.

    Returns:
        None.
    """

    args = parse_args()
    outputs = [args.output_regions_config, args.output_router_config, args.output_report]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs exist; pass --overwrite: {existing}")

    regions_config = _load_object(args.regions_config)
    router_config = _load_object(args.router_config)
    indexes = router_config.get("indexes") if isinstance(router_config.get("indexes"), dict) else {}
    routers = router_config.get("routers") if isinstance(router_config.get("routers"), dict) else {}

    hotspot_regions: list[dict[str, Any]] = []
    report_cities: list[dict[str, Any]] = []
    output_indexes = dict(indexes)
    output_routers = dict(routers)
    for base_region in regions_config.get("regions", []):
        base_region_id = str(base_region["region_id"])
        index_value = indexes.get(base_region_id) or indexes.get("default")
        if not index_value:
            raise KeyError(f"No POI index configured for region {base_region_id}")
        index_path = Path(str(index_value))
        if not index_path.is_absolute():
            index_path = args.poi_index_root / index_path
        records = list(iter_local_osm_pois(index_path))
        hotspots = plan_poi_hotspots(
            records,
            hotspot_count=args.hotspots_per_city,
            search_radius_m=args.search_radius_m,
            grid_size_m=args.grid_size_m,
            min_separation_m=args.min_separation_m,
        )
        if not hotspots:
            raise ValueError(f"No POI hotspots found for region {base_region_id}")
        quotas = split_quota(int(base_region.get("quota") or 0), len(hotspots))
        hotspot_rows: list[dict[str, Any]] = []
        for index, (hotspot, quota) in enumerate(zip(hotspots, quotas, strict=True), start=1):
            hotspot_id = f"{base_region_id}_hotspot_{index:02d}"
            hotspot_region = {
                **base_region,
                "region_id": hotspot_id,
                "label": f"{base_region.get('city', base_region_id)} POI hotspot {index}",
                "lat": round(hotspot.lat, 7),
                "lng": round(hotspot.lng, 7),
                "radius_m": args.hotspot_radius_m,
                "quota": quota,
                "asset_region_id": base_region_id,
            }
            hotspot_regions.append(hotspot_region)
            output_indexes[hotspot_id] = str(index_value)
            output_routers[hotspot_id] = dict(routers.get(base_region_id) or routers.get("default") or {})
            hotspot_rows.append(
                {
                    "region_id": hotspot_id,
                    "lat": hotspot_region["lat"],
                    "lng": hotspot_region["lng"],
                    "sampling_radius_m": args.hotspot_radius_m,
                    "poi_count_within_search_radius": hotspot.poi_count,
                    "supported_tag_count": len(hotspot.tag_signatures),
                    "supported_tags": list(hotspot.tag_signatures),
                }
            )
        report_cities.append(
            {
                "city": base_region.get("city"),
                "base_region_id": base_region_id,
                "poi_index": str(index_path),
                "indexed_poi_count": len(records),
                "hotspots": hotspot_rows,
            }
        )

    output_regions = {
        **regions_config,
        "version": f"{regions_config.get('version', 'urbannav_regions')}_hotspots",
        "description": "POI-density hotspots sharing city-level frozen OSM indexes and routers.",
        "search_radius_m": args.search_radius_m,
        "regions": hotspot_regions,
    }
    output_router = {
        **router_config,
        "description": "Local OSM router config with POI-hotspot aliases over shared city assets.",
        "indexes": output_indexes,
        "routers": output_routers,
    }
    report = {
        "schema_version": "urbannav_poi_hotspot_report_v1",
        "regions_config": str(args.regions_config),
        "router_config": str(args.router_config),
        "hotspots_per_city": args.hotspots_per_city,
        "search_radius_m": args.search_radius_m,
        "hotspot_radius_m": args.hotspot_radius_m,
        "grid_size_m": args.grid_size_m,
        "min_separation_m": args.min_separation_m,
        "city_count": len(report_cities),
        "hotspot_count": len(hotspot_regions),
        "cities": report_cities,
    }
    _write_json(args.output_regions_config, output_regions)
    _write_json(args.output_router_config, output_router)
    _write_json(args.output_report, report)
    print(json.dumps({"hotspots": len(hotspot_regions), "report": str(args.output_report)}, ensure_ascii=False))


def _load_object(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one indented JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
