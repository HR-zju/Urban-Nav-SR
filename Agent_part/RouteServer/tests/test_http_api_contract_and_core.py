from __future__ import annotations

import os
import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

_HAS_AIOHTTP = importlib.util.find_spec("aiohttp") is not None
if not _HAS_AIOHTTP:
    sys.modules.setdefault("aiohttp", SimpleNamespace())

from mcp_server.core import map_ops, route_session_ops
from mcp_server.services import route_session_service


class FakeMapOpsClient:
    async def geocode(self, query: str) -> list[dict]:
        return [{"display_name": query, "lat": "30.0", "lon": "120.0"}]

    async def reverse_geocode(self, lat: float, lon: float) -> dict:
        return {"lat": lat, "lon": lon, "display_name": "reverse"}

    async def search_semantic_pois(
        self,
        location: str,
        keywords: str,
        radius: int = 1000,
        sortrule: str = "distance",
        limit: int = 10,
    ) -> list[dict]:
        return [
            {
                "id": "poi-1",
                "name": keywords,
                "location": location,
                "distance_m": radius / 2,
                "sortrule": sortrule,
                "rank_limit": limit,
            }
        ]

    async def get_route(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        mode: str = "car",
        steps: bool = False,
        overview: str = "simplified",
        annotations: bool = False,
    ) -> dict:
        del from_lat, from_lon, to_lat, to_lon, overview, annotations
        route_steps = [
            {
                "maneuver": {"instruction": "go", "location": [120.0, 30.0]},
                "distance": 12.0,
                "duration": 3.0,
                "name": "road",
            }
        ] if steps else []
        return {
            "routes": [
                {
                    "distance": 100.0,
                    "duration": 80.0,
                    "geometry": {"type": "LineString"},
                    "legs": [{"steps": route_steps}],
                }
            ],
            "waypoints": [{"name": "origin"}, {"name": "destination"}],
            "mode": mode,
        }

    async def search_features_by_category(self, bbox, category, subcategories=None) -> list[dict]:
        del bbox, subcategories
        return [
            {
                "id": 1,
                "type": "node",
                "lat": 30.0,
                "lon": 120.0,
                "tags": {"name": "Cafe", category: "cafe"},
            },
            {
                "id": 2,
                "type": "way",
                "center": {"lat": 30.1, "lon": 120.1},
                "tags": {"name": "Library", category: "library"},
            },
        ]


class FakeRouteMapClient:
    async def get_coarse_waypoints(self, origin: str, destination: str, mode: str = "foot") -> dict:
        return {
            "provider": "fake",
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "route_summary": {
                "distance_m": 100.0,
                "duration_s": 80.0,
            },
            "coarse_waypoints": [
                {
                    "location": origin,
                    "coordinate": _location_coordinate(origin),
                    "instruction": "start",
                    "road": "",
                    "action": "start",
                },
                {
                    "location": destination,
                    "coordinate": _location_coordinate(destination),
                    "instruction": "continue",
                    "road": "",
                    "action": "continue",
                },
            ],
            "route_polyline": [
                _location_coordinate(origin),
                _location_coordinate(destination),
            ],
        }


def _location_coordinate(location: str) -> dict[str, float]:
    lon_text, lat_text = str(location).split(",", 1)
    return {
        "longitude": float(lon_text),
        "latitude": float(lat_text),
    }


@unittest.skipIf(not _HAS_AIOHTTP, "aiohttp is not installed in this environment")
class HTTPAPIContractTest(unittest.TestCase):
    def test_app_routes_remain_backward_compatible(self) -> None:
        from mcp_server import api_http

        app = api_http.create_app()
        routes = {(route.method, route.resource.canonical) for route in app.router.routes()}

        expected = {
            ("GET", "/api/health"),
            ("POST", "/api/recommendations"),
            ("POST", "/api/routes/plan"),
            ("POST", "/api/route-sessions"),
            ("GET", "/api/route-sessions/{session_id}"),
            ("POST", "/api/route-sessions/{session_id}/via-points"),
            ("POST", "/api/route-sessions/{session_id}/replan"),
            ("POST", "/api/waypoints/push"),
            ("GET", "/ws/route-sessions/{session_id}"),
        }
        self.assertTrue(expected.issubset(routes), sorted(expected - routes))


class RouteSessionCoreTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.env_patch = patch.dict(os.environ, {"ROUTE_SESSION_STORE": "memory"})
        self.env_patch.start()
        route_session_service.ROUTE_SESSIONS.clear()
        route_session_service.ROUTE_SESSIONS_LOADED = False

    def tearDown(self) -> None:
        route_session_service.ROUTE_SESSIONS.clear()
        route_session_service.ROUTE_SESSIONS_LOADED = False
        self.env_patch.stop()

    async def test_route_session_core_preserves_http_payload_shape(self) -> None:
        app_context = SimpleNamespace(osm_client=FakeRouteMapClient(), provider="fake")

        created = await route_session_ops.create_route_session(
            app_context=app_context,
            origin="120.0000,30.0000",
            destination="120.0100,30.0100",
            via_points=[],
            mode="foot",
            session_name="contract",
        )
        self.assertEqual(created["session"]["session_name"], "contract")
        self.assertFalse(created["route_plan_cache_hit"])
        self.assertIn("route_plan", created)
        self.assertIn("session_store", created)

        session_id = created["session"]["session_id"]
        snapshot = await route_session_ops.get_route_session_snapshot(
            app_context=app_context,
            session_id=session_id,
        )
        self.assertTrue(snapshot["route_plan_cache_hit"])
        self.assertEqual(snapshot["session"]["session_id"], session_id)

        updated = await route_session_ops.add_via_point(
            app_context=app_context,
            session_id=session_id,
            new_via_point={"location": "120.0050,30.0050", "name": "Rest stop"},
        )
        self.assertTrue(updated["session_changed"])
        self.assertEqual(updated["change_reason"], "via_point_added")
        self.assertEqual(updated["session"]["via_points"][0]["name"], "Rest stop")

        replanned = await route_session_ops.replan_route_session_from_origin(
            app_context=app_context,
            session_id=session_id,
            origin="120.0010,30.0010",
        )
        self.assertFalse(replanned["route_plan_cache_hit"])
        self.assertEqual(replanned["session"]["origin"], "120.0010,30.0010")


class MapOpsCoreTest(unittest.IsolatedAsyncioTestCase):
    async def test_map_ops_preserve_mcp_payload_shapes(self) -> None:
        app_context = SimpleNamespace(osm_client=FakeMapOpsClient(), provider="fake")

        geocoded = await map_ops.geocode(app_context=app_context, address="rest stop")
        self.assertEqual(geocoded[0]["coordinates"], {"latitude": 30.0, "longitude": 120.0})

        reverse = await map_ops.reverse_geocode(app_context=app_context, latitude=30.0, longitude=120.0)
        self.assertEqual(reverse["display_name"], "reverse")

        pois = await map_ops.search_poi(
            app_context=app_context,
            location="120.0,30.0",
            keywords="bench",
            radius=500,
            sortrule="distance",
            limit=3,
        )
        self.assertEqual(pois["provider"], "fake")
        self.assertEqual(pois["query"]["keywords"], "bench")
        self.assertEqual(pois["count"], 1)

        route = await map_ops.get_route(
            app_context=app_context,
            from_latitude=30.0,
            from_longitude=120.0,
            to_latitude=30.1,
            to_longitude=120.1,
            mode="foot",
            steps=True,
        )
        self.assertEqual(route["summary"]["mode"], "foot")
        self.assertEqual(route["directions"][0]["instruction"], "go")

        category = await map_ops.search_category(
            app_context=app_context,
            category="amenity",
            min_latitude=29.9,
            min_longitude=119.9,
            max_latitude=30.2,
            max_longitude=120.2,
            subcategories=["cafe"],
        )
        self.assertEqual(category["query"]["subcategories"], ["cafe"])
        self.assertEqual(category["count"], 2)
        self.assertEqual(category["results"][0]["coordinates"], {"latitude": 30.0, "longitude": 120.0})


if __name__ == "__main__":
    unittest.main()
