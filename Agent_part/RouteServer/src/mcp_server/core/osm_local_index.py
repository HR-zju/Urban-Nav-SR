"""Local OSM POI index for reproducible UrbanNav benchmark construction."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable

from mcp_server.eval.urbannav_metrics import haversine_m
from mcp_server.eval.urbannav_schema import UrbanNavDestination, UrbanNavPoint


DEFAULT_OSM_POI_TAGS = {
    "amenity=restaurant",
    "amenity=fast_food",
    "amenity=food_court",
    "amenity=cafe",
    "amenity=bar",
    "amenity=bank",
    "amenity=atm",
    "amenity=pharmacy",
    "amenity=clinic",
    "amenity=hospital",
    "amenity=library",
    "shop=convenience",
    "shop=supermarket",
    "shop=books",
    "tourism=hotel",
    "tourism=hostel",
    "leisure=park",
    "office=coworking",
    "healthcare=clinic",
}


@dataclass(frozen=True)
class LocalOsmPoiRecord:
    """One POI record extracted from a frozen OSM snapshot.

    Args:
        osm_id: Stable OSM element id, such as ``node/123``.
        name: Display name. Brand or operator may be used when ``name`` is
            absent.
        lat: Latitude.
        lng: Longitude.
        tags: OSM tags retained for filtering and audit.
        category: Normalized category derived from tags.
        country: Optional country split label.
        city: Optional city split label.
        region_id: Optional region label.
        raw: Original source record.
    """

    osm_id: str
    name: str
    lat: float
    lng: float
    tags: dict[str, str] = field(default_factory=dict)
    category: str | None = None
    country: str | None = None
    city: str | None = None
    region_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LocalOsmPoiRecord | None":
        """Parse one local POI record.

        Args:
            payload: POI dictionary from JSONL, GeoJSON properties, or an OSM
                extraction script.

        Returns:
            Parsed record, or ``None`` when coordinates are missing.
        """

        lat = _float_or_none(payload.get("lat") or payload.get("latitude"))
        lng = _float_or_none(payload.get("lng") or payload.get("lon") or payload.get("longitude"))
        if lat is None or lng is None:
            return None
        tags = _string_dict(payload.get("tags") or {})
        osm_id = str(payload.get("osm_id") or payload.get("id") or payload.get("@id") or f"local:{lat:.7f}:{lng:.7f}")
        name = str(payload.get("name") or tags.get("name") or tags.get("brand") or tags.get("operator") or "Unnamed OSM POI")
        category = str(payload.get("category") or _category_from_tags(tags) or "")
        return cls(
            osm_id=osm_id,
            name=name,
            lat=float(lat),
            lng=float(lng),
            tags=tags,
            category=category or None,
            country=_str_or_none(payload.get("country")),
            city=_str_or_none(payload.get("city")),
            region_id=_str_or_none(payload.get("region_id")),
            raw=dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the POI record.

        Returns:
            JSON-ready dictionary.
        """

        return {
            "osm_id": self.osm_id,
            "name": self.name,
            "lat": self.lat,
            "lng": self.lng,
            "tags": dict(self.tags),
            "category": self.category,
            "country": self.country,
            "city": self.city,
            "region_id": self.region_id,
        }

    def to_destination(self, origin: UrbanNavPoint) -> UrbanNavDestination:
        """Convert the local POI to an UrbanNav destination.

        Args:
            origin: Search origin used to attach direct-distance audit.

        Returns:
            Normalized destination record.
        """

        point = UrbanNavPoint(lat=self.lat, lng=self.lng)
        return UrbanNavDestination(
            lat=self.lat,
            lng=self.lng,
            name=self.name,
            type=self.category or _category_from_tags(self.tags) or "osm_poi",
            provider_poi_id=f"osm:{self.osm_id}",
            route_distance_m=round(haversine_m(origin, point), 3),
            raw={"osm_local": self.to_dict()},
        )


class LocalOsmPoiIndex:
    r"""Grid-backed local radius index over frozen OSM POIs.

    Args:
        records: POI records.
        cell_deg: Grid cell size in decimal degrees.
        source_id: Human-readable source id for replay audit.

    Querying first enumerates grid cells intersecting the radius envelope, then
    applies exact haversine filtering:

    \[
    C(o,r,T)=\{p_i \mid \operatorname{hav}(o,p_i)\le r \land tag(p_i)\cap T \ne \varnothing\}
    \]
    """

    def __init__(self, records: Iterable[LocalOsmPoiRecord], *, cell_deg: float = 0.01, source_id: str = "osm_local") -> None:
        """Initialize the local POI index.

        Args:
            records: POI records.
            cell_deg: Grid cell size.
            source_id: Source identifier.

        Returns:
            None.
        """

        self.records = list(records)
        self.cell_deg = max(float(cell_deg), 1e-4)
        self.source_id = source_id
        self._grid: dict[tuple[int, int], list[LocalOsmPoiRecord]] = {}
        for record in self.records:
            self._grid.setdefault(self._cell(record.lat, record.lng), []).append(record)

    @classmethod
    def from_path(cls, path: Path, *, cell_deg: float = 0.01, source_id: str | None = None) -> "LocalOsmPoiIndex":
        """Load an index from a JSONL file or a directory of JSONL files.

        Args:
            path: JSONL file or directory.
            cell_deg: Grid cell size.
            source_id: Optional source id.

        Returns:
            Loaded POI index.
        """

        records: list[LocalOsmPoiRecord] = []
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for item in files:
            records.extend(iter_local_osm_pois(item))
        return cls(records, cell_deg=cell_deg, source_id=source_id or str(path))

    def search(
        self,
        *,
        origin: UrbanNavPoint,
        tags: list[str],
        query: str,
        radius_m: int,
        limit: int,
    ) -> list[UrbanNavDestination]:
        """Search local POIs around an origin.

        Args:
            origin: Search center.
            tags: OSM tag constraints in ``key=value`` form.
            query: Human-readable query fallback.
            radius_m: Radius in meters.
            limit: Maximum destinations.

        Returns:
            Matching destinations sorted by direct distance.
        """

        tag_pairs = [_parse_osm_tag(tag) for tag in tags]
        tag_pairs = [item for item in tag_pairs if item is not None]
        candidates: list[tuple[float, LocalOsmPoiRecord]] = []
        for record in self._candidate_records(origin=origin, radius_m=radius_m):
            if tag_pairs and not _record_matches_tags(record, tag_pairs):
                continue
            if not tag_pairs and not _record_matches_query(record, query):
                continue
            distance = haversine_m(origin, UrbanNavPoint(lat=record.lat, lng=record.lng))
            if distance <= radius_m:
                candidates.append((distance, record))
        candidates.sort(key=lambda item: (item[0], item[1].name))
        return [record.to_destination(origin) for _, record in candidates[: max(1, int(limit))]]

    def _candidate_records(self, *, origin: UrbanNavPoint, radius_m: int) -> Iterable[LocalOsmPoiRecord]:
        """Yield records from grid cells intersecting a radius envelope.

        Args:
            origin: Search origin.
            radius_m: Radius in meters.

        Returns:
            Iterable of candidate records.
        """

        lat_delta = radius_m / 111_320.0
        lng_delta = radius_m / (111_320.0 * max(0.2, abs(math.cos(math.radians(origin.lat)))))
        min_cell = self._cell(origin.lat - lat_delta, origin.lng - lng_delta)
        max_cell = self._cell(origin.lat + lat_delta, origin.lng + lng_delta)
        for lat_idx in range(min_cell[0], max_cell[0] + 1):
            for lng_idx in range(min_cell[1], max_cell[1] + 1):
                yield from self._grid.get((lat_idx, lng_idx), [])

    def _cell(self, lat: float, lng: float) -> tuple[int, int]:
        """Return a grid cell id.

        Args:
            lat: Latitude.
            lng: Longitude.

        Returns:
            Integer grid cell.
        """

        return (math.floor(lat / self.cell_deg), math.floor(lng / self.cell_deg))


def iter_local_osm_pois(path: Path) -> Iterable[LocalOsmPoiRecord]:
    """Iterate local OSM POI records from JSONL.

    Args:
        path: JSONL file.

    Returns:
        Iterable of parsed POI records.
    """

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            record = LocalOsmPoiRecord.from_dict(payload)
            if record is not None:
                yield record


def record_has_supported_poi_tag(tags: dict[str, str], allowed_tags: set[str] | None = None) -> bool:
    """Check whether OSM tags describe a benchmark-relevant POI.

    Args:
        tags: OSM tags.
        allowed_tags: Optional allowed ``key=value`` set.

    Returns:
        ``True`` when at least one supported tag is present.
    """

    allowed = allowed_tags or DEFAULT_OSM_POI_TAGS
    return any(f"{key}={value}" in allowed for key, value in tags.items())


def _record_matches_tags(record: LocalOsmPoiRecord, tag_pairs: list[tuple[str, str]]) -> bool:
    return any(record.tags.get(key) == value for key, value in tag_pairs)


def _record_matches_query(record: LocalOsmPoiRecord, query: str) -> bool:
    text = f"{record.name} {record.category or ''} {' '.join(record.tags.values())}".lower()
    return all(part in text for part in str(query or "").lower().split() if part)


def _parse_osm_tag(tag: str) -> tuple[str, str] | None:
    if "=" not in tag:
        return None
    key, value = tag.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        return None
    return key, value


def _category_from_tags(tags: dict[str, str]) -> str | None:
    for key in ("amenity", "shop", "tourism", "leisure", "office", "healthcare"):
        value = tags.get(key)
        if value:
            return f"{key}:{value}"
    return None


def _string_dict(payload: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in payload.items() if value is not None}


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
