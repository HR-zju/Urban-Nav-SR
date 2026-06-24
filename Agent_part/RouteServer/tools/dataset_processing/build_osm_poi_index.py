#!/usr/bin/env python3
"""Build local OSM POI JSONL indexes for UrbanNav global GT collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from mcp_server.core.osm_local_index import DEFAULT_OSM_POI_TAGS, LocalOsmPoiRecord, record_has_supported_poi_tag


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", action="append", type=Path, default=[], help="Pre-extracted POI JSONL file.")
    parser.add_argument("--input-pbf", action="append", type=Path, default=[], help="OSM PBF file; requires pyosmium.")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Output local POI JSONL index.")
    parser.add_argument("--country", default=None, help="Country label stored in every emitted record.")
    parser.add_argument("--city", default=None, help="City label stored in every emitted record.")
    parser.add_argument("--region-id", default=None, help="Region id stored in every emitted record.")
    parser.add_argument("--bbox", default=None, help="Optional bbox min_lng,min_lat,max_lng,max_lat.")
    parser.add_argument("--allowed-tags", action="append", default=[], help="Additional allowed OSM tag in key=value form.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file.")
    return parser.parse_args()


def main() -> None:
    """Build a POI index.

    Returns:
        None.
    """

    args = parse_args()
    if args.output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_jsonl} exists; pass --overwrite")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    allowed_tags = set(DEFAULT_OSM_POI_TAGS).union(str(item) for item in args.allowed_tags)
    bbox = _parse_bbox(args.bbox)

    records: list[LocalOsmPoiRecord] = []
    for path in args.input_jsonl:
        records.extend(_iter_jsonl_records(path, country=args.country, city=args.city, region_id=args.region_id, bbox=bbox, allowed_tags=allowed_tags))
    for path in args.input_pbf:
        records.extend(_iter_pbf_node_records(path, country=args.country, city=args.city, region_id=args.region_id, bbox=bbox, allowed_tags=allowed_tags))

    deduped = _dedupe_records(records)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in deduped:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output_jsonl), "records": len(deduped)}, ensure_ascii=False))


def _iter_jsonl_records(
    path: Path,
    *,
    country: str | None,
    city: str | None,
    region_id: str | None,
    bbox: tuple[float, float, float, float] | None,
    allowed_tags: set[str],
) -> Iterable[LocalOsmPoiRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            payload.setdefault("country", country)
            payload.setdefault("city", city)
            payload.setdefault("region_id", region_id)
            record = LocalOsmPoiRecord.from_dict(payload)
            if record is None or not _record_in_bbox(record, bbox):
                continue
            if not record_has_supported_poi_tag(record.tags, allowed_tags=allowed_tags):
                continue
            yield record


def _iter_pbf_node_records(
    path: Path,
    *,
    country: str | None,
    city: str | None,
    region_id: str | None,
    bbox: tuple[float, float, float, float] | None,
    allowed_tags: set[str],
) -> Iterable[LocalOsmPoiRecord]:
    try:
        import osmium  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Reading --input-pbf requires pyosmium. Install osmium/pyosmium or pass --input-jsonl.") from exc

    records: list[LocalOsmPoiRecord] = []

    class Handler(osmium.SimpleHandler):  # type: ignore[misc]
        """pyosmium node handler for benchmark POI extraction."""

        def node(self, node: Any) -> None:
            tags = {str(key): str(value) for key, value in dict(node.tags).items()}
            if not record_has_supported_poi_tag(tags, allowed_tags=allowed_tags):
                return
            if not node.location.valid():
                return
            payload = {
                "osm_id": f"node/{node.id}",
                "name": tags.get("name") or tags.get("brand") or tags.get("operator") or "Unnamed OSM POI",
                "lat": float(node.location.lat),
                "lng": float(node.location.lon),
                "tags": tags,
                "country": country,
                "city": city,
                "region_id": region_id,
            }
            record = LocalOsmPoiRecord.from_dict(payload)
            if record is not None and _record_in_bbox(record, bbox):
                records.append(record)

    handler = Handler()
    handler.apply_file(str(path), locations=False)
    return records


def _dedupe_records(records: Iterable[LocalOsmPoiRecord]) -> list[LocalOsmPoiRecord]:
    by_id: dict[str, LocalOsmPoiRecord] = {}
    for record in records:
        by_id.setdefault(record.osm_id, record)
    return sorted(by_id.values(), key=lambda item: (item.city or "", item.region_id or "", item.name, item.osm_id))


def _record_in_bbox(record: LocalOsmPoiRecord, bbox: tuple[float, float, float, float] | None) -> bool:
    if bbox is None:
        return True
    min_lng, min_lat, max_lng, max_lat = bbox
    return min_lat <= record.lat <= max_lat and min_lng <= record.lng <= max_lng


def _parse_bbox(raw: str | None) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    parts = [float(item.strip()) for item in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be min_lng,min_lat,max_lng,max_lat")
    min_lng, min_lat, max_lng, max_lat = parts
    if min_lng >= max_lng or min_lat >= max_lat:
        raise ValueError("--bbox min values must be smaller than max values")
    return min_lng, min_lat, max_lng, max_lat


if __name__ == "__main__":
    main()
