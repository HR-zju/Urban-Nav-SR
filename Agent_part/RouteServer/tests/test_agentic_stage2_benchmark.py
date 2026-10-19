from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mcp_server.eval.agent_tools.agentic_stage2_benchmark import AgenticStage2BenchmarkConfig, collect_agentic_stage2_benchmark, run_ms_swift_stage2_pipeline


ROUTESERVER_ROOT = Path(__file__).resolve().parents[1]
AGENT_TOOLS_ROOT = ROUTESERVER_ROOT.parents[2] / "UrbanNav-AgentTools"


class AgenticStage2BenchmarkCollectorTest(unittest.TestCase):
    def test_collects_per_case_errors_and_artifact_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output_root = tmp / "ms_swift"
            report_dir = tmp / "urban_review"
            _write_fake_ms_swift_benchmark(output_root)

            result = collect_agentic_stage2_benchmark(
                AgenticStage2BenchmarkConfig(
                    run_id="unit-step8",
                    heldout_manifest=tmp / "heldout.jsonl",
                    ms_swift_output_root=output_root,
                    report_output_dir=report_dir,
                    base_model="/models/base",
                    stage2a_checkpoint="/models/stage2a",
                    stage2b_checkpoint="/models/stage2b",
                )
            )

            self.assertEqual(result["num_cases"], 2)
            self.assertGreaterEqual(result["num_errors"], 1)
            per_case = _read_jsonl(report_dir / "per_case_report.jsonl")
            errors = _read_jsonl(report_dir / "error_report.jsonl")
            artifact_index = json.loads((report_dir / "artifact_index.json").read_text(encoding="utf-8"))
            summary = json.loads((report_dir / "summary_report.json").read_text(encoding="utf-8"))

            self.assertEqual([row["case_id"] for row in per_case], ["case-tool", "case-no-tool"])
            self.assertEqual(summary["tool_policy_counts"], {"no_tool": 1, "tool": 1})
            self.assertTrue(any(row["error"] == "base:final_protocol_parse_failed" for row in errors))
            self.assertEqual(len(artifact_index["cases"]), 2)
            self.assertIn("final_refs", artifact_index["cases"][0]["reference"])
            self.assertTrue((report_dir / "summary_report.md").exists())

    def test_collect_rejects_reference_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output_root = tmp / "ms_swift"
            _write_fake_ms_swift_benchmark(output_root)
            _write_jsonl(output_root / "benchmark" / "sidewalk_agentic_stage2_decision_refs.jsonl", [_stage2_row("only")])

            with self.assertRaises(ValueError):
                collect_agentic_stage2_benchmark(
                    AgenticStage2BenchmarkConfig(
                        run_id="unit-mismatch",
                        heldout_manifest=tmp / "heldout.jsonl",
                        ms_swift_output_root=output_root,
                        report_output_dir=tmp / "review",
                        base_model="/models/base",
                    )
                )

    def test_pipeline_dry_run_writes_command_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            result = run_ms_swift_stage2_pipeline(
                AgenticStage2BenchmarkConfig(
                    run_id="unit-dry-run",
                    heldout_manifest=tmp / "heldout.jsonl",
                    ms_swift_output_root=tmp / "ms_swift",
                    report_output_dir=tmp / "review",
                    base_model="/models/base",
                    ms_swift_root="/workspace/ms-swift",
                    env={"MAX_TOKENS": "16"},
                ),
                actions=("prepare",),
                dry_run=True,
            )
            self.assertEqual(result[0].returncode, 0)
            self.assertIn("prepare", Path(result[0].stdout_path).read_text(encoding="utf-8"))


class AgenticStage2BenchmarkCliTest(unittest.TestCase):
    def test_cli_collects_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            output_root = tmp / "ms_swift"
            report_dir = tmp / "review"
            _write_fake_ms_swift_benchmark(output_root)
            cmd = [
                sys.executable,
                str(ROUTESERVER_ROOT / "tools" / "evaluation" / "run_sidewalk_agentic_stage2_benchmark.py"),
                "--action",
                "collect",
                "--run-id",
                "cli-unit",
                "--heldout-manifest",
                str(tmp / "heldout.jsonl"),
                "--ms-swift-output-root",
                str(output_root),
                "--report-output-dir",
                str(report_dir),
                "--base-model",
                "/models/base",
                "--stage2a-checkpoint",
                "/models/stage2a",
                "--stage2b-checkpoint",
                "/models/stage2b",
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                [str(ROUTESERVER_ROOT / "src"), str(AGENT_TOOLS_ROOT / "src")]
            )
            completed = subprocess.run(cmd, cwd=ROUTESERVER_ROOT, env=env, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["collection"]["num_cases"], 2)
            self.assertTrue((report_dir / "artifact_index.json").exists())


def _write_fake_ms_swift_benchmark(output_root: Path) -> None:
    decision_refs = [_stage2_row("case-tool", "tool"), _stage2_row("case-no-tool", "no_tool")]
    final_refs = [_stage2_row("case-tool", "tool"), _stage2_row("case-no-tool", "no_tool")]
    _write_jsonl(output_root / "benchmark" / "sidewalk_agentic_stage2_decision_refs.jsonl", decision_refs)
    _write_jsonl(output_root / "benchmark" / "sidewalk_agentic_stage2_final_refs.jsonl", final_refs)
    for run_name in ("base_zero_shot", "stage2a", "stage2b"):
        run_dir = output_root / run_name
        _write_jsonl(run_dir / "decision_predictions.jsonl", [_prediction("bad"), _prediction("ok")])
        _write_jsonl(run_dir / "final_text_predictions.jsonl", [_prediction("bad"), _prediction("ok")])
        _write_jsonl(run_dir / "decision_agentic_metrics.jsonl", [_agentic_metric(0.0), _agentic_metric(1.0)])
        _write_jsonl(run_dir / "final_agentic_metrics.jsonl", [_final_agentic_metric(0.0), _final_agentic_metric(1.0)])
        _write_jsonl(run_dir / "final_text_traj_metrics.jsonl", [_traj_metric(0.0), _traj_metric(1.0)])
        _write_json(run_dir / "decision_agentic_summary.json", {"num_rows": 2, "metrics": {"protocol_parse_success": 0.5}})
        _write_json(run_dir / "final_agentic_summary.json", {"num_rows": 2, "metrics": {"protocol_parse_success": 0.5}})
        _write_json(run_dir / "final_text_traj_summary.json", {"num_samples": 2, "parse_success": 0.5, "ade": 0.1})
    stage2b = output_root / "stage2b"
    _write_jsonl(stage2b / "final_head_predictions.jsonl", [{"trajectory_norm": [[0.5, 0.9]]}, {"trajectory_norm": [[0.6, 0.8]]}])
    _write_jsonl(stage2b / "final_head_traj_metrics.jsonl", [_traj_metric(1.0), _traj_metric(1.0)])
    _write_json(stage2b / "final_head_traj_summary.json", {"num_samples": 2, "parse_success": 1.0, "strict_success": 1.0, "ade": 0.08, "fde": 0.12})


def _stage2_row(case_id: str, policy: str = "tool") -> dict:
    return {
        "schema_version": "sidewalk_agentic_stage2_sft_v1",
        "messages": [{"role": "user", "content": "request"}, {"role": "assistant", "content": "answer"}],
        "images": [f"/images/{case_id}.jpg"],
        "sidewalk_meta": {
            "case_id": case_id,
            "source_id": f"source-{case_id}",
            "clip_id": "clip-a",
            "frame_idx": 1,
            "stage2_tool_policy": policy,
            "stage2_expected_should_call_tool": policy != "no_tool",
            "stage2_expected_tool": "map_tool" if policy != "no_tool" else None,
            "stage2_expected_op": "search_poi" if policy != "no_tool" else None,
        },
    }


def _prediction(label: str) -> dict:
    return {"response": label, "source_id": "source", "clip_id": "clip-a", "frame_idx": 1}


def _agentic_metric(parse: float) -> dict:
    return {
        "protocol_parse_success": parse,
        "tool_decision_present": parse,
        "tool_op_match": parse,
    }


def _final_agentic_metric(parse: float) -> dict:
    return {
        "protocol_parse_success": parse,
        "final_cot_tag_validity": parse,
        "tool_result_use_valid": parse,
    }


def _traj_metric(parse: float) -> dict:
    return {"parse_success": parse, "strict_success": parse, "ade": 0.1, "fde": 0.2}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
