from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image

from mcp_server.route_context.bev_map_provider import BEVMapProviderSession, build_osm_bev_semantic_layers
from mcp_server.services.visual_prompt_service import _bev_map_heading_rotation_deg, _build_bev_map_overlay_payload


class FakeMapClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[float, float, float, float]]] = []
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def search_features_by_category(self, *, bbox, category, subcategories=None):
        del subcategories
        self.calls.append((str(category), tuple(float(value) for value in bbox)))
        if category == "highway":
            return [
                {
                    "id": 1,
                    "type": "way",
                    "center": {"lat": 30.00009, "lon": 120.0},
                    "tags": {"highway": "footway", "footway": "sidewalk"},
                },
                {
                    "id": 2,
                    "type": "node",
                    "lat": 30.00005,
                    "lon": 120.00005,
                    "tags": {"highway": "crossing"},
                },
                {
                    "id": 3,
                    "type": "way",
                    "center": {"lat": 29.99995, "lon": 120.0},
                    "tags": {"highway": "residential"},
                },
            ]
        if category == "entrance":
            return [
                {
                    "id": 4,
                    "type": "node",
                    "lat": 30.00004,
                    "lon": 120.00002,
                    "tags": {"entrance": "main"},
                }
            ]
        if category == "barrier":
            return [
                {
                    "id": 5,
                    "type": "node",
                    "lat": 30.0,
                    "lon": 120.00004,
                    "tags": {"barrier": "fence"},
                }
            ]
        return []


class BEVMapProviderTest(unittest.IsolatedAsyncioTestCase):
    async def test_projects_osm_features_into_semantic_bev_layers(self) -> None:
        layers = await build_osm_bev_semantic_layers(
            map_client=FakeMapClient(),
            gps={"lat": 30.0, "lng": 120.0},
            heading_deg=0.0,
            local_bev_context={
                "bev_rotation_deg": 0.0,
                "bev_bbox": {"x_min_m": -15.0, "x_max_m": 15.0, "y_min_m": -10.0, "y_max_m": 30.0},
            },
            provider_name="osm",
            radius_m=50.0,
        )

        self.assertEqual(layers["source"], "osm_overpass_semantic_bev_v1")
        self.assertGreaterEqual(layers["feature_counts"]["sidewalk"], 1)
        self.assertGreaterEqual(layers["feature_counts"]["crossing"], 1)
        self.assertGreaterEqual(layers["feature_counts"]["road"], 1)
        self.assertGreaterEqual(layers["feature_counts"]["entrance"], 1)
        self.assertGreaterEqual(layers["feature_counts"]["barrier"], 1)
        self.assertGreater(layers["sidewalk_points_bev"][0]["y_m"], 0.0)

    async def test_session_reuses_raw_feature_cache_across_nearby_frames(self) -> None:
        client = FakeMapClient()
        session = BEVMapProviderSession(map_client=client, provider_name="osm", cache_cell_m=5000.0)
        await session.connect()
        try:
            first = await session.build_semantic_layers(
                gps={"lat": 30.0, "lng": 120.0},
                heading_deg=0.0,
                local_bev_context={
                    "bev_rotation_deg": 0.0,
                    "bev_bbox": {"x_min_m": -15.0, "x_max_m": 15.0, "y_min_m": -10.0, "y_max_m": 30.0},
                },
                radius_m=50.0,
            )
            first_call_count = len(client.calls)
            second = await session.build_semantic_layers(
                gps={"lat": 30.00001, "lng": 120.00001},
                heading_deg=10.0,
                local_bev_context={
                    "bev_rotation_deg": 10.0,
                    "bev_bbox": {"x_min_m": -15.0, "x_max_m": 15.0, "y_min_m": -10.0, "y_max_m": 30.0},
                },
                radius_m=50.0,
            )
        finally:
            await session.disconnect()

        self.assertGreater(first_call_count, 0)
        self.assertEqual(len(client.calls), first_call_count)
        self.assertFalse(first["cache"]["hit"])
        self.assertTrue(second["cache"]["hit"])
        self.assertEqual(session.query_count, 1)
        self.assertEqual(session.cache_hits, 1)

    async def test_bev_visual_input_uses_real_osm_tile_screenshot_path(self) -> None:
        def fake_tile(*, x: int, y: int, zoom: int) -> Image.Image:
            del x, y, zoom
            return Image.new("RGB", (256, 256), (210, 220, 210))

        with patch("mcp_server.services.visual_prompt_service._fetch_osm_tile_image", side_effect=fake_tile):
            payload = _build_bev_map_overlay_payload(
                local_bev_context={
                    "_origin_gps": {"lat": 30.0, "lng": 120.0},
                    "bev_rotation_deg": 15.0,
                    "navigation_path_source": "offline_trajectory",
                    "navigation_path_gps": [
                        {"lat": 30.0, "lng": 120.0},
                        {"lat": 30.0001, "lng": 120.0},
                    ],
                },
                llm_max_bytes=1_000_000,
                llm_jpeg_quality=80,
            )

        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["source"], "openstreetmap_raster_tiles")
        self.assertEqual(payload["legend"]["source"], "real_openstreetmap_raster_screenshot")
        self.assertTrue(str(payload["bev_map_overlay_image_data_url"]).startswith("data:image/jpeg;base64,"))
        self.assertEqual(payload["llm_image_size"], [640, 640])
        self.assertEqual(payload["alignment_order"], "rotate_mosaic_then_center_crop")
        self.assertIn("rotation_center_px", payload)
        self.assertGreater(payload["crop_side_px"], 0.0)
        self.assertEqual(payload["navigation_path_overlay"]["source"], "offline_trajectory")
        self.assertEqual(payload["navigation_path_overlay"]["path_points_drawn"], 2)
        self.assertEqual(payload["legend"]["ego_marker"], "orange circle; short white tick points forward/up")

    async def test_bev_tile_heading_convention_and_offset_are_configurable(self) -> None:
        with patch.dict(
            "os.environ",
            {"BEV_MAP_HEADING_CONVENTION": "enu_yaw", "BEV_MAP_HEADING_OFFSET_DEG": "15"},
        ):
            rotation, debug = _bev_map_heading_rotation_deg(0.0)

        self.assertEqual(rotation, 105.0)
        self.assertEqual(debug["heading_convention"], "enu_yaw")
        self.assertEqual(debug["compass_heading_deg"], 90.0)
        self.assertEqual(debug["heading_offset_deg"], 15.0)


if __name__ == "__main__":
    unittest.main()
