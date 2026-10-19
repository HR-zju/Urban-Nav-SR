"""Tests for local OSM POI index and provider."""

from __future__ import annotations

import json
import asyncio

from mcp_server.core.osm_local_index import LocalOsmPoiIndex, LocalOsmPoiRecord
from mcp_server.core.osm_local_provider import LocalOsmOsrmProvider
from mcp_server.core.urbannav_provider import UrbanNavPoiSearchRequest, UrbanNavRouteRequest
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavPoint


def test_local_osm_poi_index_filters_by_radius_and_tags() -> None:
    """Local index should return nearby POIs matching requested OSM tags."""

    origin = UrbanNavPoint(lat=35.0, lng=139.0)
    index = LocalOsmPoiIndex(
        [
            LocalOsmPoiRecord(
                osm_id="node/1",
                name="Cafe A",
                lat=35.0005,
                lng=139.0005,
                tags={"amenity": "cafe", "name": "Cafe A"},
                city="tokyo",
            ),
            LocalOsmPoiRecord(
                osm_id="node/2",
                name="Bank B",
                lat=35.0006,
                lng=139.0006,
                tags={"amenity": "bank", "name": "Bank B"},
                city="tokyo",
            ),
        ]
    )

    results = index.search(origin=origin, tags=["amenity=cafe"], query="cafe", radius_m=200, limit=5)

    assert [item.name for item in results] == ["Cafe A"]
    assert results[0].provider_poi_id == "osm:node/1"


def test_local_osm_provider_uses_local_index_and_osrm_response(tmp_path, monkeypatch) -> None:
    """Provider should search local JSONL and normalize OSRM routes."""

    async def run_test() -> None:
        index_path = tmp_path / "tokyo_shinjuku.jsonl"
        index_path.write_text(
            json.dumps(
                {
                    "osm_id": "node/1",
                    "name": "Cafe A",
                    "lat": 35.0005,
                    "lng": 139.0005,
                    "tags": {"amenity": "cafe", "name": "Cafe A"},
                    "city": "tokyo",
                    "region_id": "tokyo_shinjuku",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        router_config = tmp_path / "routers.json"
        router_config.write_text(
            json.dumps(
                {
                    "indexes": {"tokyo_shinjuku": "tokyo_shinjuku.jsonl"},
                    "routers": {"tokyo_shinjuku": {"osrm_url": "http://127.0.0.1:5999", "profile": "foot"}},
                }
            ),
            encoding="utf-8",
        )
        provider = LocalOsmOsrmProvider(poi_index_root=tmp_path, router_config=router_config)

        poi_record = await provider.search_pois(
            cache_key="poi:test",
            request=UrbanNavPoiSearchRequest(
                origin=UrbanNavPoint(lat=35.0, lng=139.0),
                query="cafe",
                tags=["amenity=cafe"],
                radius_m=200,
                limit=5,
                raw={"region_id": "tokyo_shinjuku", "city": "tokyo"},
            ),
        )

        assert poi_record.provider == "osm_local_osrm"
        assert len(poi_record.candidates) == 1

        def fake_request_osrm(origin, destination, osrm_url, profile):
            assert osrm_url == "http://127.0.0.1:5999"
            assert profile == "foot"
            return {
                "routes": [
                    {
                        "distance": 100.0,
                        "duration": 80.0,
                        "geometry": {"coordinates": [[139.0, 35.0], [139.0005, 35.0005]]},
                    }
                ]
            }

        monkeypatch.setattr(provider, "_request_osrm", fake_request_osrm)
        route_record = await provider.plan_route(
            cache_key="route:test",
            request=UrbanNavRouteRequest(
                origin=UrbanNavPoint(lat=35.0, lng=139.0),
                destination=poi_record.candidates[0],
                mode="foot",
            ),
        )

        assert route_record.distance_m == 100.0
        assert len(route_record.polyline) == 2

    asyncio.run(run_test())


def test_local_osm_provider_supports_valhalla_router(tmp_path, monkeypatch) -> None:
    """Provider should normalize Valhalla routes when router_type is valhalla."""

    async def run_test() -> None:
        index_path = tmp_path / "london_soho.jsonl"
        index_path.write_text(
            json.dumps(
                {
                    "osm_id": "node/10",
                    "name": "Cafe London",
                    "lat": 51.5005,
                    "lng": -0.1205,
                    "tags": {"amenity": "cafe", "name": "Cafe London"},
                    "city": "london",
                    "region_id": "london_soho",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        router_config = tmp_path / "routers.json"
        router_config.write_text(
            json.dumps(
                {
                    "default_router_type": "valhalla",
                    "indexes": {"london_soho": "london_soho.jsonl"},
                    "routers": {
                        "london_soho": {
                            "router_type": "valhalla",
                            "valhalla_url": "http://127.0.0.1:6999",
                            "costing": "pedestrian",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        provider = LocalOsmOsrmProvider(poi_index_root=tmp_path, router_config=router_config)
        poi_record = await provider.search_pois(
            cache_key="poi:london",
            request=UrbanNavPoiSearchRequest(
                origin=UrbanNavPoint(lat=51.5, lng=-0.12),
                query="cafe",
                tags=["amenity=cafe"],
                radius_m=200,
                limit=5,
                raw={"region_id": "london_soho", "city": "london"},
            ),
        )

        def fake_request_valhalla(origin, destination, valhalla_url, costing):
            assert valhalla_url == "http://127.0.0.1:6999"
            assert costing == "pedestrian"
            return {
                "trip": {
                    "summary": {"length": 0.12, "time": 95.0},
                    "legs": [{"shape": [[-0.12, 51.5], [-0.1205, 51.5005]]}],
                }
            }

        monkeypatch.setattr(provider, "_request_valhalla", fake_request_valhalla)
        route_record = await provider.plan_route(
            cache_key="route:london",
            request=UrbanNavRouteRequest(
                origin=UrbanNavPoint(lat=51.5, lng=-0.12),
                destination=poi_record.candidates[0],
                mode="foot",
            ),
        )

        assert provider.capabilities.provider_name == "osm_local_valhalla"
        assert route_record.distance_m == 120.0
        assert route_record.duration_s == 95.0
        assert len(route_record.polyline) == 2

    asyncio.run(run_test())


def test_local_osm_provider_rejects_graph_router_for_benchmark(tmp_path) -> None:
    """Benchmark provider must reject self-implemented graph routing."""

    async def run_test() -> None:
        index_path = tmp_path / "tokyo_shinjuku.jsonl"
        index_path.write_text(
            json.dumps(
                {
                    "osm_id": "node/1",
                    "name": "Cafe A",
                    "lat": 35.0008,
                    "lng": 139.0,
                    "tags": {"amenity": "cafe", "name": "Cafe A"},
                    "city": "tokyo",
                    "region_id": "tokyo_shinjuku",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        router_config = tmp_path / "routers.json"
        router_config.write_text(
            json.dumps(
                {
                    "default_router_type": "graph",
                    "indexes": {"tokyo_shinjuku": "tokyo_shinjuku.jsonl"},
                    "routers": {
                        "tokyo_shinjuku": {
                            "router_type": "graph",
                            "graph_path": str(tmp_path / "tokyo_shinjuku.walk_graph.pkl"),
                            "profile": "foot",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        provider = LocalOsmOsrmProvider(poi_index_root=tmp_path, router_config=router_config)
        poi_record = await provider.search_pois(
            cache_key="poi:graph",
            request=UrbanNavPoiSearchRequest(
                origin=UrbanNavPoint(lat=35.0, lng=139.0),
                query="cafe",
                tags=["amenity=cafe"],
                radius_m=200,
                limit=5,
                raw={"region_id": "tokyo_shinjuku", "city": "tokyo"},
            ),
        )
        try:
            await provider.plan_route(
                cache_key="route:graph",
                request=UrbanNavRouteRequest(
                    origin=UrbanNavPoint(lat=35.0, lng=139.0),
                    destination=poi_record.candidates[0],
                    mode="foot",
                ),
            )
        except ValueError as exc:
            assert "router_type=graph is not allowed" in str(exc)
        else:
            raise AssertionError("graph router must be rejected for benchmark GT construction")

    asyncio.run(run_test())
