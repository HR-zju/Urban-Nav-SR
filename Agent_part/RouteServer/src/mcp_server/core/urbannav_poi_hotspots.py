"""Deterministic POI-density hotspot planning for UrbanNav collection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from mcp_server.core.osm_local_index import DEFAULT_OSM_POI_TAGS, LocalOsmPoiRecord
from mcp_server.eval.urbannav.urbannav_metrics import haversine_m
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavPoint


@dataclass(frozen=True)
class PoiHotspot:
    """One deterministic high-density POI collection hotspot.

    Args:
        lat: Hotspot center latitude.
        lng: Hotspot center longitude.
        poi_count: Number of indexed POIs within the search radius.
        tag_signatures: Distinct benchmark-supported ``key=value`` tags found
            within the search radius.
    """

    lat: float
    lng: float
    poi_count: int
    tag_signatures: tuple[str, ...]


def plan_poi_hotspots(
    records: Iterable[LocalOsmPoiRecord],
    *,
    hotspot_count: int,
    search_radius_m: float,
    grid_size_m: float,
    min_separation_m: float,
) -> list[PoiHotspot]:
    r"""Select POI-dense and semantically diverse collection centers.

    Args:
        records: Frozen city-level POI records.
        hotspot_count: Maximum number of returned hotspots.
        search_radius_m: Radius used to score POI coverage around a center.
        grid_size_m: Grid width used to generate deterministic center
            candidates.
        min_separation_m: Minimum distance between selected centers.

    Returns:
        Selected hotspots in greedy selection order.

    Candidate cells are scored greedily by newly covered tag signatures,
    total tag diversity, and POI density:

    \[
    s(h)=(|T_h\\setminus T_S|, |T_h|, |P_h|)
    \]

    where \(T_S\) is the set of signatures covered by already selected
    hotspots. Stable coordinate tie-breakers make the result reproducible.
    """

    values = list(records)
    if hotspot_count <= 0 or not values:
        return []
    if search_radius_m <= 0 or grid_size_m <= 0:
        raise ValueError("search_radius_m and grid_size_m must be positive")
    if min_separation_m < 0:
        raise ValueError("min_separation_m must be non-negative")

    reference_lat = sum(record.lat for record in values) / len(values)
    grid: dict[tuple[int, int], list[LocalOsmPoiRecord]] = {}
    for record in values:
        cell = _grid_cell(record.lat, record.lng, reference_lat=reference_lat, grid_size_m=grid_size_m)
        grid.setdefault(cell, []).append(record)

    candidates: list[PoiHotspot] = []
    for cell_records in grid.values():
        center_record = _cell_medoid(cell_records)
        nearby = _records_within(values, lat=center_record.lat, lng=center_record.lng, radius_m=search_radius_m)
        candidates.append(
            PoiHotspot(
                lat=center_record.lat,
                lng=center_record.lng,
                poi_count=len(nearby),
                tag_signatures=tuple(sorted({signature for record in nearby for signature in _tag_signatures(record)})),
            )
        )

    selected: list[PoiHotspot] = []
    covered_tags: set[str] = set()
    remaining = list(candidates)
    while remaining and len(selected) < hotspot_count:
        eligible = [
            candidate
            for candidate in remaining
            if all(_distance_m(candidate, chosen) >= min_separation_m for chosen in selected)
        ]
        if not eligible:
            break
        best = max(
            eligible,
            key=lambda candidate: (
                len(set(candidate.tag_signatures) - covered_tags),
                len(candidate.tag_signatures),
                candidate.poi_count,
                -round(candidate.lat, 7),
                -round(candidate.lng, 7),
            ),
        )
        selected.append(best)
        covered_tags.update(best.tag_signatures)
        remaining.remove(best)
    return selected


def split_quota(total: int, parts: int) -> list[int]:
    """Split an integer quota deterministically across regions.

    Args:
        total: Total quota.
        parts: Number of regions.

    Returns:
        Per-region quotas whose sum equals ``total``.
    """

    if parts <= 0:
        raise ValueError("parts must be positive")
    quotient, remainder = divmod(max(0, int(total)), parts)
    return [quotient + (1 if index < remainder else 0) for index in range(parts)]


def _grid_cell(lat: float, lng: float, *, reference_lat: float, grid_size_m: float) -> tuple[int, int]:
    """Project latitude/longitude into a deterministic local meter grid."""

    y_m = lat * 111_320.0
    x_m = lng * 111_320.0 * max(0.2, abs(math.cos(math.radians(reference_lat))))
    return math.floor(y_m / grid_size_m), math.floor(x_m / grid_size_m)


def _cell_medoid(records: list[LocalOsmPoiRecord]) -> LocalOsmPoiRecord:
    """Return the record nearest the arithmetic center of one grid cell."""

    mean_lat = sum(record.lat for record in records) / len(records)
    mean_lng = sum(record.lng for record in records) / len(records)
    center = UrbanNavPoint(lat=mean_lat, lng=mean_lng)
    return min(
        records,
        key=lambda record: (
            haversine_m(center, UrbanNavPoint(lat=record.lat, lng=record.lng)),
            record.osm_id,
        ),
    )


def _records_within(
    records: list[LocalOsmPoiRecord],
    *,
    lat: float,
    lng: float,
    radius_m: float,
) -> list[LocalOsmPoiRecord]:
    """Return records within a haversine radius of a center."""

    center = UrbanNavPoint(lat=lat, lng=lng)
    return [
        record
        for record in records
        if haversine_m(center, UrbanNavPoint(lat=record.lat, lng=record.lng)) <= radius_m
    ]


def _tag_signatures(record: LocalOsmPoiRecord) -> set[str]:
    """Return benchmark-supported OSM tag signatures for one POI."""

    return {
        f"{key}={value}"
        for key, value in record.tags.items()
        if f"{key}={value}" in DEFAULT_OSM_POI_TAGS
    }


def _distance_m(left: PoiHotspot, right: PoiHotspot) -> float:
    """Return the haversine distance between hotspot centers."""

    return haversine_m(
        UrbanNavPoint(lat=left.lat, lng=left.lng),
        UrbanNavPoint(lat=right.lat, lng=right.lng),
    )
