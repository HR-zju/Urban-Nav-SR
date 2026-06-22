from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcp_server.eval.urbannav_io import (
    load_urbannav_cases,
    load_urbannav_predictions,
    write_urbannav_cases,
    write_urbannav_predictions,
)
from mcp_server.eval.urbannav_metrics import (
    UrbanNavEvalConfig,
    evaluate_case,
    evaluate_predictions,
    match_destination,
)
from mcp_server.eval.urbannav_schema import UrbanNavCase, UrbanNavDestination, UrbanNavPoint, UrbanNavPrediction


class UrbanNavSchemaIOTest(unittest.TestCase):
    def test_case_roundtrip_jsonl(self) -> None:
        case = _case()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "cases.jsonl"
            write_urbannav_cases([case], path)
            loaded = load_urbannav_cases(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].case_id, "case-1")
        self.assertEqual(loaded[0].acceptable_destinations[0].provider_poi_id, "poi-a")
        self.assertEqual(loaded[0].optimality_scope.cost, "route_distance_m")

    def test_prediction_nested_payload_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pred.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "case-1",
                        "prediction": {
                            "chosen_destination": {"provider_poi_id": "poi-a", "name": "Store A"},
                            "route_polyline": [[120.0, 30.0], [120.001, 30.0]],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            predictions = load_urbannav_predictions(path)

        self.assertEqual(predictions[0].case_id, "case-1")
        self.assertEqual(predictions[0].chosen_destination.provider_poi_id, "poi-a")
        self.assertEqual(len(predictions[0].route_polyline), 2)

    def test_multistop_prediction_roundtrip_jsonl(self) -> None:
        prediction = UrbanNavPrediction(
            case_id="case-multi",
            chosen_destination=UrbanNavDestination(provider_poi_id="poi-b", name="Bank", lat=30.2, lng=120.2),
            destination_sequence=[
                UrbanNavDestination(provider_poi_id="poi-a", name="Restaurant", lat=30.1, lng=120.1),
                UrbanNavDestination(provider_poi_id="poi-b", name="Bank", lat=30.2, lng=120.2),
            ],
            route_legs=[
                {"leg_index": 1, "chosen_destination": {"provider_poi_id": "poi-a"}},
                {"leg_index": 2, "chosen_destination": {"provider_poi_id": "poi-b"}},
            ],
            route_polyline=[
                UrbanNavPoint(lat=30.0, lng=120.0),
                UrbanNavPoint(lat=30.1, lng=120.1),
                UrbanNavPoint(lat=30.2, lng=120.2),
            ],
            route_distance_m=300.0,
            route_summary={"distance_m": 300.0},
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "pred.jsonl"
            write_urbannav_predictions([prediction], path)
            loaded = load_urbannav_predictions(path)

        self.assertEqual(loaded[0].case_id, "case-multi")
        self.assertEqual(len(loaded[0].destination_sequence), 2)
        self.assertEqual(loaded[0].destination_sequence[-1].provider_poi_id, "poi-b")
        self.assertEqual(len(loaded[0].route_legs), 2)


class UrbanNavDestinationMatchTest(unittest.TestCase):
    def test_stable_id_match_wins(self) -> None:
        match = match_destination(
            UrbanNavDestination(provider_poi_id="poi-a", name="Different"),
            [_acceptable("poi-a", rank=1, distance=100.0)],
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.method, "stable_id")

    def test_coordinate_fallback_requires_name_and_type(self) -> None:
        acceptable = [_acceptable("poi-a", rank=1, distance=100.0)]
        ok = match_destination(
            UrbanNavDestination(name="Store A", type="convenience_store", lat=30.0, lng=120.0011),
            acceptable,
            tolerance_m=30.0,
        )
        bad = match_destination(
            UrbanNavDestination(name="Other", type="convenience_store", lat=30.0, lng=120.0011),
            acceptable,
            tolerance_m=30.0,
        )
        self.assertIsNotNone(ok)
        self.assertEqual(ok.method, "name_type_coordinate")
        self.assertIsNone(bad)


class UrbanNavMetricsTest(unittest.TestCase):
    def test_matched_non_rank1_destination_uses_own_reference_distance(self) -> None:
        case = _case()
        prediction = UrbanNavPrediction.from_dict(
            {
                "id": "case-1",
                "chosen_destination": {"provider_poi_id": "poi-b", "name": "Store B"},
                "candidate_destinations": [
                    {"provider_poi_id": "wrong", "name": "Wrong"},
                    {"provider_poi_id": "poi-b", "name": "Store B"},
                ],
                "route_polyline": [[120.0, 30.0], [120.002, 30.0]],
                "route_summary": {"distance_m": 240.0},
            }
        )

        row = evaluate_case(case, prediction, config=UrbanNavEvalConfig(destination_tolerance_m=30, endpoint_tolerance_m=30))

        self.assertTrue(row.destination_success)
        self.assertTrue(row.hit_at_5)
        self.assertTrue(row.route_success)
        self.assertEqual(row.matched_destination_rank, 2)
        self.assertEqual(row.reference_route_distance_m, 200.0)
        self.assertAlmostEqual(row.route_ratio, 1.2)
        self.assertAlmostEqual(row.spl, 200.0 / 240.0)

    def test_wrong_destination_fails_route_even_with_route(self) -> None:
        case = _case()
        prediction = UrbanNavPrediction.from_dict(
            {
                "id": "case-1",
                "chosen_destination": {"provider_poi_id": "wrong", "name": "Wrong"},
                "candidate_destinations": [{"provider_poi_id": "wrong", "name": "Wrong"}],
                "route_polyline": [[120.0, 30.0], [120.001, 30.0]],
                "route_summary": {"distance_m": 100.0},
            }
        )

        row = evaluate_case(case, prediction)

        self.assertFalse(row.destination_success)
        self.assertFalse(row.hit_at_5)
        self.assertFalse(row.route_success)
        self.assertEqual(row.spl, 0.0)
        self.assertIsNone(row.route_ratio)

    def test_summary_main_metrics(self) -> None:
        case = _case()
        ok_prediction = UrbanNavPrediction.from_dict(
            {
                "id": "case-1",
                "chosen_destination": {"provider_poi_id": "poi-a"},
                "candidate_destinations": [{"provider_poi_id": "poi-a"}],
                "route_polyline": [[120.0, 30.0], [120.001, 30.0]],
                "route_summary": {"distance_m": 100.0},
            }
        )
        rows, summary = evaluate_predictions([case], [ok_prediction])

        self.assertEqual(len(rows), 1)
        self.assertEqual(summary["total_cases"], 1)
        self.assertEqual(summary["destination_success"], 1.0)
        self.assertEqual(summary["hit@5"], 1.0)
        self.assertEqual(summary["route_success"], 1.0)
        self.assertEqual(summary["median_route_ratio"], 1.0)
        self.assertEqual(summary["spl"], 1.0)

    def test_multistop_extended_metrics(self) -> None:
        case = UrbanNavCase.from_dict(
            {
                "id": "multi-1",
                "category": "multi_stop_route",
                "instruction": "先吃饭，再取钱。",
                "origin": {"lat": 30.0, "lng": 120.0},
                "gt_destination": {"provider_poi_id": "bank-a", "lat": 30.0, "lng": 120.002},
                "gt_destination_sequence": [
                    {"provider_poi_id": "food-a", "name": "Food", "type": "restaurant", "lat": 30.0, "lng": 120.001, "rank": 1},
                    {"provider_poi_id": "bank-a", "name": "Bank", "type": "bank", "lat": 30.0, "lng": 120.002, "rank": 1},
                ],
                "gt_route_legs": [
                    {
                        "leg_index": 1,
                        "gt_destination": {"provider_poi_id": "food-a", "name": "Food", "type": "restaurant", "lat": 30.0, "lng": 120.001, "reference_route_distance_m": 100.0},
                        "acceptable_destinations": [{"provider_poi_id": "food-a", "name": "Food", "type": "restaurant", "lat": 30.0, "lng": 120.001, "reference_route_distance_m": 100.0}],
                        "distance_m": 100.0,
                    },
                    {
                        "leg_index": 2,
                        "gt_destination": {"provider_poi_id": "bank-a", "name": "Bank", "type": "bank", "lat": 30.0, "lng": 120.002, "reference_route_distance_m": 120.0},
                        "acceptable_destinations": [{"provider_poi_id": "bank-a", "name": "Bank", "type": "bank", "lat": 30.0, "lng": 120.002, "reference_route_distance_m": 120.0}],
                        "distance_m": 120.0,
                    },
                ],
                "gt_route": [[120.0, 30.0], [120.001, 30.0], [120.002, 30.0]],
                "gt_distance_m": 220.0,
            }
        )
        prediction = UrbanNavPrediction.from_dict(
            {
                "id": "multi-1",
                "chosen_destination": {"provider_poi_id": "bank-a", "lat": 30.0, "lng": 120.002},
                "pred_destination_sequence": [
                    {"provider_poi_id": "food-a", "lat": 30.0, "lng": 120.001},
                    {"provider_poi_id": "bank-a", "lat": 30.0, "lng": 120.002},
                ],
                "pred_route_legs": [
                    {"leg_index": 1, "route": [[120.0, 30.0], [120.001, 30.0]], "route_summary": {"distance_m": 100.0}},
                    {"leg_index": 2, "route": [[120.001, 30.0], [120.002, 30.0]], "route_summary": {"distance_m": 120.0}},
                ],
                "route_polyline": [[120.0, 30.0], [120.001, 30.0], [120.002, 30.0]],
                "route_summary": {"distance_m": 220.0},
            }
        )

        rows, summary = evaluate_predictions([case], [prediction])
        multi = rows[0].extended_metrics["multi_stop"]

        self.assertTrue(multi["sequence_success"])
        self.assertTrue(multi["leg_route_success"])
        self.assertEqual(multi["stop_recall"], 1.0)
        self.assertEqual(summary["multi_stop_case_count"], 1)
        self.assertEqual(summary["multi_stop_sequence_success"], 1.0)

    def test_route_update_extended_metrics(self) -> None:
        case = UrbanNavCase.from_dict(
            {
                "id": "update-1",
                "category": "route_update",
                "instruction": "我临时想改去可以买水的地方。",
                "origin": {"lat": 30.0005, "lng": 120.0005},
                "gt_destination": {"provider_poi_id": "water-a", "name": "Water", "type": "convenience_store", "lat": 30.0, "lng": 120.001},
                "acceptable_destinations": [
                    {
                        "provider_poi_id": "water-a",
                        "name": "Water",
                        "type": "convenience_store",
                        "lat": 30.0,
                        "lng": 120.001,
                        "reference_route_distance_m": 80.0,
                        "reference_route": [[120.0005, 30.0005], [120.001, 30.0]],
                    }
                ],
                "gt_route": [[120.0005, 30.0005], [120.001, 30.0]],
                "gt_distance_m": 80.0,
                "metadata": {
                    "route_update": {
                        "update_type": "replace_destination",
                        "current_origin": {"lat": 30.0005, "lng": 120.0005},
                        "previous_destination": {"provider_poi_id": "coffee-old", "name": "Coffee", "type": "cafe", "lat": 30.002, "lng": 120.0},
                    }
                },
            }
        )
        prediction = UrbanNavPrediction.from_dict(
            {
                "id": "update-1",
                "chosen_destination": {"provider_poi_id": "water-a", "lat": 30.0, "lng": 120.001},
                "route_polyline": [[120.0005, 30.0005], [120.001, 30.0]],
                "route_summary": {"distance_m": 80.0},
            }
        )

        rows, summary = evaluate_predictions([case], [prediction])
        update = rows[0].extended_metrics["route_update"]

        self.assertTrue(update["route_update_success"])
        self.assertTrue(update["current_origin_used"])
        self.assertTrue(update["previous_destination_avoided"])
        self.assertEqual(summary["route_update_case_count"], 1)
        self.assertEqual(summary["route_update_success"], 1.0)

    def test_complex_examples_cover_non_rank1_and_fallback_semantics(self) -> None:
        cases = load_urbannav_cases("tests/fixtures/urbannav_complex_gt_6.jsonl")
        predictions = load_urbannav_predictions("tests/fixtures/urbannav_complex_pred_6.jsonl")
        rows, summary = evaluate_predictions(cases, predictions)
        by_id = {row.case_id: row for row in rows}

        self.assertEqual(summary["total_cases"], 6)
        self.assertAlmostEqual(summary["destination_success"], 4 / 6)
        self.assertEqual(summary["hit@5"], 1.0)
        self.assertAlmostEqual(summary["route_success"], 4 / 6)
        self.assertTrue(by_id["complex-001"].destination_success)
        self.assertEqual(by_id["complex-001"].matched_destination_rank, 2)
        self.assertAlmostEqual(by_id["complex-001"].route_ratio or 0.0, 1.0)
        self.assertTrue(by_id["complex-002"].destination_success)
        self.assertEqual(by_id["complex-002"].matched_destination_rank, 2)
        self.assertAlmostEqual(by_id["complex-002"].route_ratio or 0.0, 1.25)
        self.assertFalse(by_id["complex-003"].destination_success)
        self.assertTrue(by_id["complex-003"].hit_at_5)
        self.assertEqual(by_id["complex-004"].destination_match_method, "name_type_coordinate")
        self.assertTrue(by_id["complex-005"].destination_success)
        self.assertEqual(by_id["complex-005"].matched_destination_rank, 2)
        self.assertFalse(by_id["complex-006"].destination_success)
        self.assertTrue(by_id["complex-006"].hit_at_5)


def _case() -> UrbanNavCase:
    return UrbanNavCase.from_dict(
        {
            "id": "case-1",
            "city": "hangzhou",
            "scene_id": "yuquan",
            "category": "implicit_need_navigation",
            "subcategory": "thirsty",
            "difficulty": "easy",
            "instruction": "我有点渴了，帮我找附近可以买饮料的地方。",
            "origin": {"lat": 30.0, "lng": 120.0},
            "gt_destination": {"provider_poi_id": "poi-a", "name": "Store A", "type": "convenience_store", "lat": 30.0, "lng": 120.001},
            "acceptable_destinations": [
                _acceptable("poi-a", rank=1, distance=100.0).to_dict(),
                _acceptable("poi-b", rank=2, distance=200.0, lng=120.002, name="Store B").to_dict(),
            ],
            "ranked_destinations": [
                {"rank": 1, "provider_poi_id": "poi-a", "route_distance_m": 100.0},
                {"rank": 2, "provider_poi_id": "poi-b", "route_distance_m": 200.0},
            ],
            "gt_route": [[120.0, 30.0], [120.001, 30.0]],
            "gt_distance_m": 100.0,
            "optimality_scope": {
                "poi_provider": "mock",
                "search_radius_m": 1500,
                "candidate_limit": 30,
                "route_provider": "mock",
                "route_mode": "foot",
                "cost": "route_distance_m",
            },
            "cache_keys": {"poi_search": ["poi:case-1"], "routes": ["route:case-1:poi-a"]},
        }
    )


def _acceptable(
    provider_poi_id: str,
    *,
    rank: int,
    distance: float,
    lng: float = 120.001,
    name: str = "Store A",
) -> UrbanNavDestination:
    return UrbanNavDestination(
        provider_poi_id=provider_poi_id,
        name=name,
        type="convenience_store",
        lat=30.0,
        lng=lng,
        rank=rank,
        reference_route_distance_m=distance,
        reference_route=[],
    )


if __name__ == "__main__":
    unittest.main()
