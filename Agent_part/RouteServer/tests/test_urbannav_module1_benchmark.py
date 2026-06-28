from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcp_server.eval.urbannav_io import write_urbannav_cases, write_urbannav_predictions
from mcp_server.eval.urbannav_module1_benchmark import (
    UrbanNavModule1BenchmarkConfig,
    module1_prediction_contract,
    run_module1_benchmark,
)
from mcp_server.eval.urbannav_schema import UrbanNavCase, UrbanNavDestination, UrbanNavPoint, UrbanNavPrediction


class UrbanNavModule1BenchmarkTest(unittest.TestCase):
    def test_oracle_runner_writes_complete_report_package(self) -> None:
        """Oracle runner should create predictions, rows, summary and manifest."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            dataset = tmp / "task_gt.jsonl"
            output = tmp / "module1_oracle"
            write_urbannav_cases([_case()], dataset)

            result = run_module1_benchmark(
                UrbanNavModule1BenchmarkConfig(
                    dataset_path=dataset,
                    output_dir=output,
                    runner="oracle",
                    run_id="unit-oracle",
                )
            )

            self.assertEqual(result.case_count, 1)
            self.assertEqual(result.prediction_count, 1)
            self.assertTrue((output / "predictions.jsonl").exists())
            self.assertTrue((output / "eval_rows.jsonl").exists())
            self.assertTrue((output / "summary.json").exists())
            self.assertTrue((output / "manifest.json").exists())
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["summary"]["destination_success"], 1.0)
            self.assertEqual(summary["summary"]["route_success"], 1.0)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["module"], "urbannav_module1_instruction_to_destination_route")
            self.assertIn("prediction_contract", manifest)

    def test_external_predictions_are_normalized_and_evaluated(self) -> None:
        """External models can join the benchmark by writing prediction JSONL."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            dataset = tmp / "task_gt.jsonl"
            predictions = tmp / "predictions_external.jsonl"
            output = tmp / "module1_external"
            write_urbannav_cases([_case()], dataset)
            write_urbannav_predictions(
                [
                    UrbanNavPrediction(
                        case_id="case-1",
                        chosen_destination=UrbanNavDestination(provider_poi_id="poi-a", name="Store A", lat=30.0, lng=120.001),
                        candidate_destinations=[UrbanNavDestination(provider_poi_id="poi-a", name="Store A")],
                        route_polyline=[UrbanNavPoint(lat=30.0, lng=120.0), UrbanNavPoint(lat=30.0, lng=120.001)],
                        route_summary={"distance_m": 100.0},
                    )
                ],
                predictions,
            )

            result = run_module1_benchmark(
                UrbanNavModule1BenchmarkConfig(
                    dataset_path=dataset,
                    output_dir=output,
                    runner="external",
                    predictions_path=predictions,
                )
            )

            self.assertEqual(result.summary["summary"]["destination_success"], 1.0)
            self.assertEqual(result.summary["summary"]["hit@5"], 1.0)
        self.assertEqual(result.summary["summary"]["spl"], 1.0)

    def test_oracle_runner_records_concurrency_in_manifest(self) -> None:
        """Concurrency config should be recorded for reproducibility."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            dataset = tmp / "task_gt.jsonl"
            output = tmp / "module1_oracle"
            write_urbannav_cases([_case()], dataset)

            run_module1_benchmark(
                UrbanNavModule1BenchmarkConfig(
                    dataset_path=dataset,
                    output_dir=output,
                    runner="oracle",
                    runner_concurrency=4,
                )
            )

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["config"]["runner_concurrency"], 4)

    def test_prediction_contract_documents_model_interface(self) -> None:
        """Contract should expose the fields needed by external model runners."""

        contract = module1_prediction_contract()
        self.assertEqual(contract["schema_version"], "urbannav_prediction_contract_v1")
        self.assertIn("id", contract["required"])
        self.assertIn("route_polyline", contract["recommended"])
        self.assertIn("destination_success", contract["main_metrics"])


def _case() -> UrbanNavCase:
    return UrbanNavCase(
        case_id="case-1",
        instruction="Please find a nearby convenience store.",
        origin=UrbanNavPoint(lat=30.0, lng=120.0),
        gt_destination=UrbanNavDestination(provider_poi_id="poi-a", name="Store A", type="convenience_store", lat=30.0, lng=120.001),
        acceptable_destinations=[
            UrbanNavDestination(
                provider_poi_id="poi-a",
                name="Store A",
                type="convenience_store",
                lat=30.0,
                lng=120.001,
                rank=1,
                reference_route_distance_m=100.0,
                reference_route=[UrbanNavPoint(lat=30.0, lng=120.0), UrbanNavPoint(lat=30.0, lng=120.001)],
            )
        ],
        ranked_destinations=[
            UrbanNavDestination(
                provider_poi_id="poi-a",
                name="Store A",
                type="convenience_store",
                lat=30.0,
                lng=120.001,
                rank=1,
                route_distance_m=100.0,
            )
        ],
        gt_route=[UrbanNavPoint(lat=30.0, lng=120.0), UrbanNavPoint(lat=30.0, lng=120.001)],
        gt_distance_m=100.0,
        category="poi_navigation",
        subcategory="convenience_store",
        city="unit_city",
        scene_id="unit_scene",
        difficulty="easy",
    )


if __name__ == "__main__":
    unittest.main()
