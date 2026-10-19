from __future__ import annotations

import unittest

from mcp_server.core.commercial_global_provider import (
    _parse_tomtom_route,
    _parse_tomtom_search_results,
    _tomtom_category_set,
)
from mcp_server.core.urbannav_provider import UrbanNavPoiSearchRequest
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavPoint


class TomTomUrbanNavProviderTest(unittest.TestCase):
    """Offline tests for TomTom response normalization."""

    def test_search_results_normalize_to_destinations(self) -> None:
        """POI search rows should become UrbanNav destinations.

        Returns:
            None.
        """

        candidates = _parse_tomtom_search_results(
            [
                {
                    "id": "poi-1",
                    "dist": 42.5,
                    "position": {"lat": 51.5007, "lon": -0.1246},
                    "poi": {
                        "name": "Mock Cafe",
                        "categories": ["cafe"],
                        "categorySet": [{"id": 7315}],
                    },
                }
            ],
            origin=UrbanNavPoint(lat=51.5, lng=-0.125),
            limit=5,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].provider_poi_id, "poi-1")
        self.assertEqual(candidates[0].name, "Mock Cafe")
        self.assertEqual(candidates[0].type, "tomtom_category=7315")
        self.assertEqual(candidates[0].route_distance_m, 42.5)

    def test_route_response_normalizes_summary_and_points(self) -> None:
        """Route rows should keep exact benchmark endpoints.

        Returns:
            None.
        """

        origin = UrbanNavPoint(lat=51.5, lng=-0.125)
        destination = UrbanNavPoint(lat=51.501, lng=-0.124)
        parsed = _parse_tomtom_route(
            {
                "routes": [
                    {
                        "summary": {"lengthInMeters": 321, "travelTimeInSeconds": 240},
                        "legs": [
                            {
                                "points": [
                                    {"latitude": 51.5008, "longitude": -0.1248},
                                    {"latitude": 51.5010, "longitude": -0.1240},
                                ]
                            }
                        ],
                    }
                ]
            },
            origin=origin,
            destination=destination,
        )

        self.assertEqual(parsed["distance_m"], 321.0)
        self.assertEqual(parsed["duration_s"], 240.0)
        self.assertEqual(parsed["polyline"][0], origin)
        self.assertEqual(parsed["polyline"][-1], destination)

    def test_category_set_request_metadata(self) -> None:
        """TomTom categorySet should come from request raw metadata.

        Returns:
            None.
        """

        request = UrbanNavPoiSearchRequest(
            origin=UrbanNavPoint(lat=51.5, lng=-0.125),
            query="coffee",
            tags=["7315"],
            radius_m=1000,
            limit=5,
            raw={"tomtom_category_set": ["7315", "9361"]},
        )

        self.assertEqual(_tomtom_category_set(request), "7315,9361")


if __name__ == "__main__":
    unittest.main()
