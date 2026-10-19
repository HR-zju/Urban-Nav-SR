from __future__ import annotations

import tempfile
import unittest
import importlib.util
from pathlib import Path

from mcp_server.core.live_map_backend import LiveMapBackend, LiveRouteRequest
from mcp_server.core.replay_map_backend import ReplayMapBackend
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavDestination, UrbanNavPoint
from mcp_server.eval.urbannav.urbannav_io import load_urbannav_cases, load_urbannav_predictions
from mcp_server.eval.urbannav.urbannav_metrics import evaluate_predictions
from mcp_server.eval.urbannav.urbannav_runners import OracleUrbanNavRunner, ReplayUrbanNavBackendAdapter, RouteServerRunnerConfig, RouteServerUrbanNavRunner, run_urbannav_cases, run_urbannav_cases_async
from mcp_server.eval.urbannav.urbannav_validator import validate_urbannav_package


def _load_mock_builder():
    """Load the mock builder CLI module from its file path.

    Returns:
        Imported module object exposing ``build_mock_benchmark``.
    """

    path = Path(__file__).resolve().parents[1] / "tools" / "dataset_processing" / "build_urbannav_mock_benchmark.py"
    spec = importlib.util.spec_from_file_location("build_urbannav_mock_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


build_mock_benchmark = _load_mock_builder().build_mock_benchmark


class UrbanNavValidatorReplayBuilderTest(unittest.TestCase):
    """Tests for Step 4-6 UrbanNav benchmark infrastructure."""

    def test_mock_builder_outputs_valid_replayable_package(self) -> None:
        """Build a mock package and validate all cache references.

        Returns:
            None.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "mock_benchmark"
            result = build_mock_benchmark(output, num_cases=5)

            report = validate_urbannav_package(
                gt_path=result["task_gt"],
                poi_cache_path=result["poi_cache"],
                route_cache_path=result["route_cache"],
                rejects_path=result["task_rejects"],
            )

            self.assertTrue(report.valid, report.to_dict())
            self.assertEqual(report.total_cases, 5)
            self.assertEqual(report.cache_missing_count, 0)

    def test_replay_backend_loads_mock_cache(self) -> None:
        """Replay backend returns frozen POI and route records by key.

        Returns:
            None.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "mock_benchmark"
            result = build_mock_benchmark(output, num_cases=1)
            backend = ReplayMapBackend(
                poi_cache_path=result["poi_cache"],
                route_cache_path=result["route_cache"],
            )
            case = load_urbannav_cases(result["task_gt"])[0]
            poi_key = case.cache_keys["poi_search"][0]
            route_key = case.cache_keys["routes"][0]

            poi = backend.search_destinations(cache_key=poi_key)
            route = backend.plan_route(cache_key=route_key)

            self.assertEqual(poi.cache_key, poi_key)
            self.assertEqual(len(poi.candidates), 1)
            self.assertEqual(route.cache_key, route_key)
            self.assertEqual(route.distance_m, 100.0)
            self.assertEqual(len(route.polyline), 2)

    def test_mock_package_can_feed_evaluator_with_fixture_predictions(self) -> None:
        """Generated mock GT can be evaluated with compatible predictions.

        Returns:
            None.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "mock_benchmark"
            result = build_mock_benchmark(output, num_cases=5)
            cases = load_urbannav_cases(result["task_gt"])
            predictions = load_urbannav_predictions("tests/fixtures/urbannav_fake_pred_5.jsonl")
            # Builder ids are mock-001... while fake predictions are fake-001...
            # This test uses the evaluator to confirm missing predictions are
            # explicit rather than silently ignored.
            rows, summary = evaluate_predictions(cases, predictions)

            self.assertEqual(summary["total_cases"], 5)
            self.assertEqual(summary["missing_prediction_count"], 5)
            self.assertTrue(all("missing_prediction" in row.errors for row in rows))

    def test_oracle_runner_scores_perfect_on_mock_package(self) -> None:
        """Oracle runner produces all-one main metrics on generated mock GT.

        Returns:
            None.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "mock_benchmark"
            result = build_mock_benchmark(output, num_cases=5)
            cases = load_urbannav_cases(result["task_gt"])
            predictions = run_urbannav_cases(OracleUrbanNavRunner(), cases)
            _, summary = evaluate_predictions(cases, predictions)

            self.assertEqual(summary["destination_success"], 1.0)
            self.assertEqual(summary["hit@5"], 1.0)
            self.assertEqual(summary["route_success"], 1.0)
            self.assertEqual(summary["median_route_ratio"], 1.0)
            self.assertEqual(summary["spl"], 1.0)

    def test_mock_builder_can_emit_multi_target_and_route_update_cases(self) -> None:
        """Mock builder emits Step 10 case types without breaking validation.

        Returns:
            None.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "mock_benchmark"
            result = build_mock_benchmark(
                output,
                num_cases=1,
                include_multi_target=True,
                include_route_updates=True,
            )
            cases = load_urbannav_cases(result["task_gt"])
            report = validate_urbannav_package(
                gt_path=result["task_gt"],
                poi_cache_path=result["poi_cache"],
                route_cache_path=result["route_cache"],
                rejects_path=result["task_rejects"],
            )

            self.assertTrue(report.valid, report.to_dict())
            self.assertTrue(any(len(case.acceptable_destinations) > 1 for case in cases))
            self.assertTrue(any(case.metadata.get("route_update") for case in cases))

    def test_routeserver_replay_runner_scores_on_mock_package(self) -> None:
        """RouteServer replay runner produces evaluable predictions.

        Returns:
            None.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "mock_benchmark"
            result = build_mock_benchmark(
                output,
                num_cases=2,
                include_multi_target=True,
                include_route_updates=True,
            )
            cases = load_urbannav_cases(result["task_gt"])
            backend = ReplayUrbanNavBackendAdapter(
                ReplayMapBackend(
                    poi_cache_path=result["poi_cache"],
                    route_cache_path=result["route_cache"],
                )
            )
            runner = RouteServerUrbanNavRunner(
                backend=backend,
                config=RouteServerRunnerConfig(selection_strategy="route_distance", use_llm_intent=False),
            )
            predictions = run_urbannav_cases_async(runner, cases)
            _, summary = evaluate_predictions(cases, predictions)

            self.assertEqual(summary["missing_prediction_count"], 0)
            self.assertGreaterEqual(summary["destination_success"], 0.75)
            self.assertGreaterEqual(summary["route_success"], 0.75)


class FakeLiveMapClient:
    """Fake async map client for live backend normalization tests."""

    async def connect(self) -> None:
        """Connect the fake client.

        Returns:
            None.
        """

    async def disconnect(self) -> None:
        """Disconnect the fake client.

        Returns:
            None.
        """

    async def geocode(self, query):
        """Return no fake geocoding results.

        Args:
            query: Geocode query.

        Returns:
            Empty result list.
        """

        return []

    async def reverse_geocode(self, lat, lon):
        """Return a fake reverse-geocode result.

        Args:
            lat: Latitude.
            lon: Longitude.

        Returns:
            Fake reverse-geocode dictionary.
        """

        return {"display_name": "fake"}

    async def search_semantic_pois(self, location, keywords, radius=1000, sortrule="distance", limit=10):
        """Return fake normalized POI records.

        Args:
            location: Provider location string.
            keywords: Keyword query.
            radius: Search radius.
            sortrule: Sort rule.
            limit: Candidate limit.

        Returns:
            List with one normalized POI dictionary.
        """

        return [
            {
                "provider_poi_id": "fake-poi-1",
                "name": "Fake Cafe",
                "type": "cafe",
                "location": "120.001,30.001",
                "distance_m": 100,
            }
        ][:limit]

    async def get_coarse_waypoints(self, origin, destination, mode="foot"):
        """Return a fake normalized route response.

        Args:
            origin: Origin ``"lng,lat"`` string.
            destination: Destination ``"lng,lat"`` string.
            mode: Route mode.

        Returns:
            Normalized route response dictionary.
        """

        return {
            "provider": "fake",
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "route_summary": {"distance_m": 123, "duration_s": 98, "step_count": 1},
            "route_polyline": [[120.0, 30.0], [120.001, 30.001]],
        }

    async def get_route(self, from_lat, from_lon, to_lat, to_lon, mode="car", steps=False, overview="overview", annotations=True):
        """Return no raw fake routes.

        Args:
            from_lat: Origin latitude.
            from_lon: Origin longitude.
            to_lat: Destination latitude.
            to_lon: Destination longitude.
            mode: Route mode.
            steps: Whether to include steps.
            overview: Route overview mode.
            annotations: Whether to include annotations.

        Returns:
            Empty route response.
        """

        return {"routes": []}

    async def get_nearby_pois(self, lat, lon, radius=1000, categories=None):
        """Return no fake nearby POIs.

        Args:
            lat: Latitude.
            lon: Longitude.
            radius: Radius.
            categories: Optional categories.

        Returns:
            Empty result list.
        """

        return []

    async def search_features_by_category(self, bbox, category, subcategories=None):
        """Return no fake features.

        Args:
            bbox: Bounding box.
            category: Feature category.
            subcategories: Optional subcategories.

        Returns:
            Empty result list.
        """

        return []


class UrbanNavLiveBackendTest(unittest.IsolatedAsyncioTestCase):
    """Tests for Step 7 live map backend normalization."""

    async def test_live_backend_normalizes_poi_and_route_records(self) -> None:
        """Fake live provider responses become replay-compatible records.

        Returns:
            None.
        """

        backend = LiveMapBackend(client=FakeLiveMapClient(), provider_name="fake")
        origin = UrbanNavPoint(lat=30.0, lng=120.0)
        poi = await backend.search_destinations(
            cache_key="poi:fake",
            origin=origin,
            keywords="咖啡",
            radius_m=500,
            limit=5,
        )
        route = await backend.plan_route(
            cache_key="route:fake",
            request=LiveRouteRequest(
                origin=origin,
                destination=UrbanNavDestination(provider_poi_id="fake-poi-1", lat=30.001, lng=120.001, name="Fake Cafe"),
            ),
        )

        self.assertEqual(poi.provider, "fake")
        self.assertEqual(poi.candidates[0].provider_poi_id, "fake-poi-1")
        self.assertEqual(route.provider, "fake")
        self.assertEqual(route.distance_m, 123.0)
        self.assertEqual(len(route.polyline), 2)


if __name__ == "__main__":
    unittest.main()
