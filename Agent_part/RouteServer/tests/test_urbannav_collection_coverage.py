"""Tests for UrbanNav city-target coverage filtering."""

from mcp_server.core.osm_local_index import LocalOsmPoiRecord
from mcp_server.eval.urbannav_collection_coverage import build_hotspot_target_coverage, filter_instruction_rows_by_coverage


TARGET_MAP = {
    "target_types": {
        "cafe": {"osm_tags": ["amenity=cafe"]},
        "library": {"osm_tags": ["amenity=library"]},
        "pharmacy": {"osm_tags": ["amenity=pharmacy"]},
    }
}


def test_filter_requires_all_targets_in_one_hotspot() -> None:
    """A multi-stop row should require all goal groups in one hotspot."""

    records = {
        "city_asset": [
            LocalOsmPoiRecord(osm_id="node/1", name="Cafe", lat=40.0, lng=-74.0, tags={"amenity": "cafe"}),
            LocalOsmPoiRecord(osm_id="node/2", name="Library", lat=40.001, lng=-74.0, tags={"amenity": "library"}),
        ]
    }
    regions = [
        {
            "region_id": "city_hotspot_01",
            "asset_region_id": "city_asset",
            "city": "city",
            "lat": 40.0,
            "lng": -74.0,
            "radius_m": 100,
        }
    ]
    coverage = build_hotspot_target_coverage(regions, records_by_asset_region=records, search_radius_m=500)
    rows = [
        {
            "instruction_id": "ok",
            "category": "multi_stop_route",
            "planned_city": {"city": "city"},
            "ordered_target_types": [["cafe"], ["library"]],
        },
        {
            "instruction_id": "bad",
            "category": "multi_stop_route",
            "planned_city": {"city": "city"},
            "ordered_target_types": [["cafe"], ["pharmacy"]],
        },
    ]

    accepted, rejected, report = filter_instruction_rows_by_coverage(rows, coverage=coverage, target_map=TARGET_MAP)

    assert [row["instruction_id"] for row in accepted] == ["ok"]
    assert rejected[0]["reason"] == "target_type_absent_from_city"
    assert report["accepted_rows"] == 1
