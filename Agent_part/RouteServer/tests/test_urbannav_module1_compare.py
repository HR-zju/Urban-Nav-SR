from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mcp_server.eval.urbannav.urbannav_module1_compare import Module1ComparisonConfig, Module1RunSpec, compare_module1_runs, comparison_to_markdown, score_summary


class UrbanNavModule1CompareTest(unittest.TestCase):
    def test_relative_index_and_paired_comparison(self) -> None:
        """Comparison report should rank runs and compute paired deltas."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            base = _write_run(
                tmp / "base",
                summary_metrics={
                    "total_cases": 2,
                    "destination_success": 0.5,
                    "hit@5": 1.0,
                    "route_success": 0.5,
                    "median_route_ratio": 1.0,
                    "spl": 0.5,
                },
                rows=[
                    _row("case-1", ds=True, hit=True, rs=True, spl=1.0, ratio=1.0),
                    _row("case-2", ds=False, hit=True, rs=False, spl=0.0, ratio=None),
                ],
            )
            stronger = _write_run(
                tmp / "stronger",
                summary_metrics={
                    "total_cases": 2,
                    "destination_success": 1.0,
                    "hit@5": 1.0,
                    "route_success": 1.0,
                    "median_route_ratio": 1.0,
                    "spl": 1.0,
                },
                rows=[
                    _row("case-1", ds=True, hit=True, rs=True, spl=1.0, ratio=1.0),
                    _row("case-2", ds=True, hit=True, rs=True, spl=1.0, ratio=1.0),
                ],
            )

            report = compare_module1_runs(
                Module1ComparisonConfig(
                    runs=[
                        Module1RunSpec(label="base", path=base),
                        Module1RunSpec(label="stronger", path=stronger),
                    ],
                    baseline_label="base",
                )
            )

        self.assertEqual(report["baseline_label"], "base")
        self.assertEqual(report["ranking"][0]["label"], "stronger")
        self.assertGreater(report["ranking"][0]["relative_index"], 100.0)
        paired = report["paired_vs_baseline"]["stronger"]
        self.assertEqual(paired["common_cases"], 2)
        self.assertEqual(paired["wins"], 1)
        self.assertEqual(paired["draws"], 1)
        self.assertEqual(paired["losses"], 0)
        self.assertIn("UrbanNav Module 1 Relative Comparison", comparison_to_markdown(report))

    def test_score_summary_penalizes_route_ratio(self) -> None:
        """Route efficiency should decline when median route ratio is worse."""

        good = score_summary(
            {
                "destination_success": 1.0,
                "hit@5": 1.0,
                "route_success": 1.0,
                "median_route_ratio": 1.0,
                "spl": 1.0,
            }
        )
        worse = score_summary(
            {
                "destination_success": 1.0,
                "hit@5": 1.0,
                "route_success": 1.0,
                "median_route_ratio": 2.0,
                "spl": 0.5,
            }
        )
        self.assertGreater(good, worse)


def _write_run(path: Path, *, summary_metrics: dict, rows: list[dict]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "urbannav_module1_benchmark_v1",
        "run_id": path.name,
        "summary": summary_metrics,
        "by_category": {
            "poi_navigation": summary_metrics,
        },
    }
    (path / "summary.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    (path / "eval_rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _row(case_id: str, *, ds: bool, hit: bool, rs: bool, spl: float, ratio: float | None) -> dict:
    return {
        "case_id": case_id,
        "destination_success": ds,
        "hit@5": hit,
        "route_success": rs,
        "route_ratio": ratio,
        "spl": spl,
    }


if __name__ == "__main__":
    unittest.main()
