"""Tests for fixed-denominator UrbanNav v2 metrics."""

from __future__ import annotations

import json

from mcp_server.eval.urbannav_metrics_v2 import (
    UrbanNavEvalV2Config,
    evaluate_predictions_v2,
    write_eval_v2_artifacts,
)
from mcp_server.eval.urbannav_schema import UrbanNavCase, UrbanNavPrediction


def _single_case(case_id: str = "single-1") -> UrbanNavCase:
    """Build a single-destination case with one acceptable POI."""

    return UrbanNavCase.from_dict(
        {
            "id": case_id,
            "category": "poi_navigation",
            "instruction": "Find a cafe.",
            "origin": {"lat": 30.0, "lng": 120.0},
            "gt_destination": {
                "provider_poi_id": "poi-ok",
                "name": "Cafe",
                "type": "cafe",
                "lat": 30.0,
                "lng": 120.001,
                "reference_route_distance_m": 100.0,
            },
            "acceptable_destinations": [
                {
                    "provider_poi_id": "poi-ok",
                    "name": "Cafe",
                    "type": "cafe",
                    "lat": 30.0,
                    "lng": 120.001,
                    "reference_route_distance_m": 100.0,
                }
            ],
            "gt_route": [[120.0, 30.0], [120.001, 30.0]],
            "gt_distance_m": 100.0,
        }
    )


def test_fixed_denominator_ranking_and_mrr() -> None:
    """Missing candidates score zero and rank three receives partial MRR."""

    ranked = UrbanNavPrediction.from_dict(
        {
            "id": "single-1",
            "chosen_destination": {"provider_poi_id": "poi-ok", "lat": 30.0, "lng": 120.001},
            "candidate_destinations": [
                {"provider_poi_id": "wrong-1"},
                {"provider_poi_id": "wrong-2"},
                {"provider_poi_id": "poi-ok"},
            ],
            "route_polyline": [[120.0, 30.0], [120.001, 30.0]],
            "route_summary": {"distance_m": 100.0},
        }
    )
    empty = UrbanNavPrediction.from_dict(
        {
            "id": "single-2",
            "chosen_destination": {"provider_poi_id": "wrong"},
            "candidate_destinations": [],
        }
    )
    rows, summary, sensitivity = evaluate_predictions_v2(
        [_single_case("single-1"), _single_case("single-2")],
        [ranked, empty],
    )

    assert rows[0]["task_hit@1"] == 0.0
    assert rows[0]["task_hit@3"] == 1.0
    assert rows[0]["task_mrr@5"] == 1 / 3
    assert rows[1]["candidate_coverage"] == 0.0
    assert summary["summary"]["task_hit@3"] == 0.5
    assert summary["summary"]["task_mrr@5"] == 1 / 6
    assert sensitivity["ranking"]["fixed_denominator_cases"] == 2


def test_multistop_ranking_averages_required_stops_and_checks_continuity() -> None:
    """Multi-stop scores average stop ranks and require connected route legs."""

    case = UrbanNavCase.from_dict(
        {
            "id": "multi-1",
            "category": "multi_stop_route",
            "instruction": "Go to food and then a bank.",
            "origin": {"lat": 30.0, "lng": 120.0},
            "gt_destination": {"provider_poi_id": "bank", "lat": 30.0, "lng": 120.002},
            "gt_destination_sequence": [
                {"provider_poi_id": "food", "lat": 30.0, "lng": 120.001},
                {"provider_poi_id": "bank", "lat": 30.0, "lng": 120.002},
            ],
            "gt_route_legs": [
                {
                    "leg_index": 1,
                    "gt_destination": {"provider_poi_id": "food", "lat": 30.0, "lng": 120.001},
                    "acceptable_destinations": [
                        {
                            "provider_poi_id": "food",
                            "lat": 30.0,
                            "lng": 120.001,
                            "reference_route_distance_m": 100.0,
                        }
                    ],
                    "distance_m": 100.0,
                },
                {
                    "leg_index": 2,
                    "gt_destination": {"provider_poi_id": "bank", "lat": 30.0, "lng": 120.002},
                    "acceptable_destinations": [
                        {
                            "provider_poi_id": "bank",
                            "lat": 30.0,
                            "lng": 120.002,
                            "reference_route_distance_m": 100.0,
                        }
                    ],
                    "distance_m": 100.0,
                },
            ],
            "gt_route": [[120.0, 30.0], [120.001, 30.0], [120.002, 30.0]],
            "gt_distance_m": 200.0,
        }
    )
    prediction = UrbanNavPrediction.from_dict(
        {
            "id": "multi-1",
            "chosen_destination": {"provider_poi_id": "bank", "lat": 30.0, "lng": 120.002},
            "pred_destination_sequence": [
                {"provider_poi_id": "food", "lat": 30.0, "lng": 120.001},
                {"provider_poi_id": "bank", "lat": 30.0, "lng": 120.002},
            ],
            "pred_route_legs": [
                {
                    "candidate_destinations": [{"provider_poi_id": "food"}],
                    "origin": {"lat": 30.0, "lng": 120.0},
                    "route": [[120.0, 30.0], [120.001, 30.0]],
                    "route_summary": {"distance_m": 100.0},
                },
                {
                    "candidate_destinations": [
                        {"provider_poi_id": "wrong"},
                        {"provider_poi_id": "bank"},
                    ],
                    "origin": {"lat": 30.0, "lng": 120.001},
                    "route": [[120.001, 30.0], [120.002, 30.0]],
                    "route_summary": {"distance_m": 100.0},
                },
            ],
            "route_polyline": [[120.0, 30.0], [120.001, 30.0], [120.002, 30.0]],
            "route_summary": {"distance_m": 200.0},
        }
    )
    rows, summary, _ = evaluate_predictions_v2([case], [prediction])

    assert rows[0]["task_hit@1"] == 0.5
    assert rows[0]["task_hit@3"] == 1.0
    assert rows[0]["task_mrr@5"] == 0.75
    assert rows[0]["task_success_v2"] is True
    assert rows[0]["task_spl_v2"] == 1.0
    assert summary["summary"]["multi_stop_leg_continuity_success"] == 1.0


def test_artifact_writer_keeps_v2_outputs_separate(tmp_path) -> None:
    """The writer creates only the requested v2 artifact directory."""

    rows, summary, sensitivity = evaluate_predictions_v2([_single_case()], [])
    output = tmp_path / "eval_v2"
    paths = write_eval_v2_artifacts(
        rows=rows,
        summary=summary,
        sensitivity=sensitivity,
        output_dir=output,
        run_id="test-v2",
        dataset_path="task_gt.jsonl",
        predictions_path="predictions.jsonl",
        config=UrbanNavEvalV2Config(),
    )

    assert set(paths) == {"rows", "summary", "metric_sensitivity", "manifest"}
    assert json.loads((output / "summary.json").read_text())["schema_version"] == "urbannav_eval_v2"
    assert json.loads((output / "manifest.json").read_text())["config"]["fixed_denominator"] is True


def test_prediction_exception_forces_all_v2_outcomes_to_failure() -> None:
    """An exception row cannot retain destination or ranking credit."""

    prediction = UrbanNavPrediction.from_dict(
        {
            "id": "single-1",
            "chosen_destination": {"provider_poi_id": "poi-ok", "lat": 30.0, "lng": 120.001},
            "candidate_destinations": [{"provider_poi_id": "poi-ok"}],
            "route_polyline": [[120.0, 30.0], [120.001, 30.0]],
            "route_summary": {"distance_m": 100.0},
            "trace": {"error": "prediction_exception", "exception": "upstream failure"},
        }
    )
    rows, summary, _ = evaluate_predictions_v2([_single_case()], [prediction])

    assert rows[0]["destination_success"] is False
    assert rows[0]["task_hit@3"] == 0.0
    assert rows[0]["task_mrr@5"] == 0.0
    assert rows[0]["task_success_v2"] is False
    assert rows[0]["task_spl_v2"] == 0.0
    assert summary["summary"]["prediction_exception_count"] == 1
