"""Tests for deterministic UrbanNav POI hotspot planning."""

from mcp_server.core.osm_local_index import LocalOsmPoiRecord
from mcp_server.core.urbannav_poi_hotspots import plan_poi_hotspots, split_quota


def _record(osm_id: str, lat: float, lng: float, tag: tuple[str, str]) -> LocalOsmPoiRecord:
    """Build one synthetic POI record."""

    return LocalOsmPoiRecord(osm_id=osm_id, name=osm_id, lat=lat, lng=lng, tags={tag[0]: tag[1]})


def test_plan_poi_hotspots_prefers_diverse_separated_clusters() -> None:
    """Hotspot selection should cover separated clusters and new tag types."""

    records = [
        _record("node/1", 40.0000, -74.0000, ("amenity", "cafe")),
        _record("node/2", 40.0002, -74.0001, ("amenity", "restaurant")),
        _record("node/3", 40.0200, -74.0000, ("amenity", "library")),
        _record("node/4", 40.0202, -74.0001, ("shop", "supermarket")),
    ]

    hotspots = plan_poi_hotspots(
        records,
        hotspot_count=2,
        search_radius_m=500,
        grid_size_m=500,
        min_separation_m=1000,
    )

    assert len(hotspots) == 2
    assert {tag for hotspot in hotspots for tag in hotspot.tag_signatures} == {
        "amenity=cafe",
        "amenity=restaurant",
        "amenity=library",
        "shop=supermarket",
    }
    assert hotspots[0].lat != hotspots[1].lat


def test_split_quota_preserves_total() -> None:
    """Quota splitting should preserve the requested total."""

    assert split_quota(11, 3) == [4, 4, 3]
