"""Tests for extracting node, way, and relation POIs from OSM snapshots."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from mcp_server.core.osm_local_index import DEFAULT_OSM_POI_TAGS


ROUTE_SERVER_ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    """Load the POI index builder script as a test module."""

    path = ROUTE_SERVER_ROOT / "tools" / "dataset_processing" / "build_osm_poi_index.py"
    spec = importlib.util.spec_from_file_location("build_osm_poi_index", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pbf_extraction_includes_nodes_ways_and_relations(tmp_path: Path) -> None:
    """Tagged OSM elements should all become stable POI records."""

    osm_path = tmp_path / "fixture.osm"
    osm_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <node id="1" lat="30.0000" lon="120.0000"><tag k="amenity" v="cafe"/><tag k="name" v="Node Cafe"/></node>
  <node id="2" lat="30.0010" lon="120.0010"/>
  <node id="3" lat="30.0010" lon="120.0020"/>
  <node id="4" lat="30.0020" lon="120.0020"/>
  <node id="5" lat="30.0020" lon="120.0010"/>
  <node id="6" lat="30.0030" lon="120.0030"/>
  <node id="7" lat="30.0030" lon="120.0040"/>
  <node id="8" lat="30.0040" lon="120.0040"/>
  <node id="9" lat="30.0040" lon="120.0030"/>
  <way id="10"><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="5"/><nd ref="2"/><tag k="shop" v="supermarket"/><tag k="name" v="Way Market"/></way>
  <way id="11"><nd ref="6"/><nd ref="7"/><nd ref="8"/><nd ref="9"/><nd ref="6"/></way>
  <relation id="20"><member type="way" ref="11" role="outer"/><tag k="type" v="multipolygon"/><tag k="amenity" v="school"/><tag k="name" v="Relation School"/></relation>
</osm>
""",
        encoding="utf-8",
    )
    builder = _load_builder()

    records = list(
        builder._iter_pbf_records(
            osm_path,
            country="china",
            city="hangzhou",
            region_id="hangzhou_test",
            bbox=None,
            allowed_tags=set(DEFAULT_OSM_POI_TAGS),
        )
    )

    by_id = {record.osm_id: record for record in records}
    assert set(by_id) == {"node/1", "way/10", "relation/20"}
    assert by_id["way/10"].name == "Way Market"
    assert by_id["relation/20"].tags["amenity"] == "school"
    assert 30.001 <= by_id["way/10"].lat <= 30.002
    assert 120.003 <= by_id["relation/20"].lng <= 120.004
