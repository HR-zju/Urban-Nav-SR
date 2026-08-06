"""Offline city-by-target coverage analysis for UrbanNav collection."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from mcp_server.core.osm_local_index import LocalOsmPoiRecord
from mcp_server.eval.urbannav.urbannav_collection_policy import instruction_target_groups
from mcp_server.eval.urbannav.urbannav_metrics import haversine_m
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavPoint


@dataclass(frozen=True)
class HotspotTargetCoverage:
    """POI tag coverage around one collection hotspot.

    Args:
        region_id: Hotspot region identifier.
        city: Normalized city label.
        effective_radius_m: Conservative coverage radius after subtracting the
            origin sampling radius from the configured search radius.
        tag_counts: Number of nearby POIs for each OSM ``key=value`` tag.
    """

    region_id: str
    city: str
    effective_radius_m: float
    tag_counts: dict[str, int]


def build_hotspot_target_coverage(
    regions: Iterable[dict[str, Any]],
    *,
    records_by_asset_region: dict[str, list[LocalOsmPoiRecord]],
    search_radius_m: float,
) -> list[HotspotTargetCoverage]:
    """Build conservative OSM tag counts for collection hotspots.

    Args:
        regions: Hotspot region records.
        records_by_asset_region: City-level POI records keyed by the shared
            ``asset_region_id`` or region id.
        search_radius_m: Runtime POI search radius.

    Returns:
        Hotspot coverage rows.
    """

    coverage: list[HotspotTargetCoverage] = []
    for region in regions:
        region_id = str(region.get("region_id") or "")
        asset_region_id = str(region.get("asset_region_id") or region_id)
        records = records_by_asset_region.get(asset_region_id, [])
        sampling_radius_m = max(0.0, float(region.get("radius_m") or 0.0))
        effective_radius_m = max(0.0, float(search_radius_m) - sampling_radius_m)
        center = UrbanNavPoint(lat=float(region["lat"]), lng=float(region["lng"]))
        tag_counts: Counter[str] = Counter()
        for record in records:
            point = UrbanNavPoint(lat=record.lat, lng=record.lng)
            if haversine_m(center, point) > effective_radius_m:
                continue
            tag_counts.update(f"{key}={value}" for key, value in record.tags.items())
        coverage.append(
            HotspotTargetCoverage(
                region_id=region_id,
                city=_normalize_city(region.get("city")),
                effective_radius_m=effective_radius_m,
                tag_counts=dict(tag_counts),
            )
        )
    return coverage


def filter_instruction_rows_by_coverage(
    rows: Iterable[dict[str, Any]],
    *,
    coverage: list[HotspotTargetCoverage],
    target_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Filter instructions unsupported by their planned city's POI assets.

    Args:
        rows: Compiled instruction rows.
        coverage: Hotspot tag coverage.
        target_map: Target-type provider ontology.

    Returns:
        Accepted rows, rejected audit rows, and a JSON-ready report.
    """

    coverage_by_city: dict[str, list[HotspotTargetCoverage]] = defaultdict(list)
    for item in coverage:
        coverage_by_city[item.city].append(item)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()
    accepted_by_city_category: Counter[tuple[str, str]] = Counter()
    required_types: set[str] = set()

    for row in rows:
        planned_city = row.get("planned_city") if isinstance(row.get("planned_city"), dict) else {}
        city = _normalize_city(planned_city.get("city") or row.get("city"))
        groups = instruction_target_groups(row)
        required_types.update(target_type for group in groups for target_type in group)
        reason = _unsupported_reason(groups, coverage_by_city.get(city, []), target_map=target_map)
        if reason is None:
            accepted.append(row)
            accepted_by_city_category[(city, str(row.get("category") or "unknown"))] += 1
            continue
        reject_reasons[reason] += 1
        rejected.append(
            {
                "instruction_id": row.get("instruction_id"),
                "city": city,
                "category": row.get("category"),
                "subcategory": row.get("subcategory"),
                "target_groups": groups,
                "reason": reason,
            }
        )

    target_matrix = _target_matrix(
        coverage_by_city=coverage_by_city,
        target_types=sorted(required_types),
        target_map=target_map,
    )
    report = {
        "schema_version": "urbannav_city_target_coverage_v1",
        "input_rows": len(accepted) + len(rejected),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "accepted_by_city_category": {
            city: {
                category: accepted_by_city_category[(city, category)]
                for category in sorted({key[1] for key in accepted_by_city_category if key[0] == city})
            }
            for city in sorted({key[0] for key in accepted_by_city_category})
        },
        "city_target_type_coverage": target_matrix,
    }
    return accepted, rejected, report


def _unsupported_reason(
    groups: list[list[str]],
    city_coverage: list[HotspotTargetCoverage],
    *,
    target_map: dict[str, Any],
) -> str | None:
    """Return a precise pre-collection rejection reason, if any."""

    if not city_coverage:
        return "planned_city_has_no_hotspots"
    if not groups or any(not group for group in groups):
        return "missing_target_type_group"
    group_tags = [_tags_for_target_group(group, target_map=target_map) for group in groups]
    if any(not tags for tags in group_tags):
        return "target_type_has_no_osm_mapping"
    if any(not any(_supports_tags(item, tags) for item in city_coverage) for tags in group_tags):
        return "target_type_absent_from_city"
    if not any(all(_supports_tags(item, tags) for tags in group_tags) for item in city_coverage):
        return "target_groups_not_colocated_in_hotspot"
    return None


def _target_matrix(
    *,
    coverage_by_city: dict[str, list[HotspotTargetCoverage]],
    target_types: list[str],
    target_map: dict[str, Any],
) -> dict[str, Any]:
    """Build city-by-target-type candidate coverage diagnostics."""

    matrix: dict[str, Any] = {}
    for city, city_coverage in sorted(coverage_by_city.items()):
        city_row: dict[str, Any] = {}
        for target_type in target_types:
            tags = _tags_for_target_group([target_type], target_map=target_map)
            counts = [sum(item.tag_counts.get(tag, 0) for tag in tags) for item in city_coverage]
            city_row[target_type] = {
                "hotspots_with_candidates": sum(count > 0 for count in counts),
                "max_candidates_in_hotspot": max(counts, default=0),
            }
        matrix[city] = city_row
    return matrix


def _tags_for_target_group(group: list[str], *, target_map: dict[str, Any]) -> set[str]:
    """Compile one OR-group of target types into OSM tags."""

    specs = target_map.get("target_types") if isinstance(target_map.get("target_types"), dict) else {}
    tags: set[str] = set()
    for target_type in group:
        spec = specs.get(target_type) if isinstance(specs, dict) else None
        if isinstance(spec, dict):
            tags.update(str(item) for item in spec.get("osm_tags") or [] if str(item).strip())
    return tags


def _supports_tags(coverage: HotspotTargetCoverage, tags: set[str]) -> bool:
    """Return whether a hotspot contains any POI matching an OR-tag set."""

    return any(coverage.tag_counts.get(tag, 0) > 0 for tag in tags)


def _normalize_city(value: Any) -> str:
    """Normalize a city label for deterministic joins."""

    return "_".join(str(value or "").strip().lower().replace("-", " ").split())
