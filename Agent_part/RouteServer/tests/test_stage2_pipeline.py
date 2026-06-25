from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mcp_server.agent.schemas import AgentStage2Case, ToolCall, ToolResponse
from mcp_server.eval.exporters import trace_to_ms_swift_stage2_row
from mcp_server.eval.stage2_pipeline import (
    Stage2SplitConfig,
    apply_stage2_splits,
    sample_stage2_review_rows,
    summarize_stage2_slices,
    write_stage2_split_outputs,
)
from mcp_server.eval.stage2_validator import (
    validate_ms_swift_stage2_row,
    validate_stage2_case,
    validate_stage2_trace,
)
from mcp_server.teacher.agentic_stage2_trace_builder import AgenticStage2TraceBuilder


ROUTESERVER_ROOT = Path(__file__).resolve().parents[1]


class Stage2PipelineHelperTest(unittest.TestCase):
    def test_split_slice_review_and_split_outputs(self) -> None:
        cases = [_sample_case("tool"), _sample_case("no-tool"), _sample_case("failed")]
        cases[1].ground_truth["tool_policy"] = "no_tool"
        split_counts = apply_stage2_splits(
            cases,
            Stage2SplitConfig(train_ratio=0.0, val_ratio=0.0, heldout_ratio=1.0, seed="unit-test"),
        )

        builder = AgenticStage2TraceBuilder()
        call = ToolCall(tool="map_tool", arguments={"op": "search_poi", "query": "bench", "location": "120,30"})
        traces = [
            builder.build_trace(cases[0], call=call),
            builder.build_trace(cases[1], call=None),
            builder.build_trace(
                cases[2],
                call=call,
                response=ToolResponse(
                    tool="map_tool",
                    success=False,
                    error={"message": "fake provider unavailable"},
                    call_id=call.call_id,
                ),
            ),
        ]
        rows = [trace_to_ms_swift_stage2_row(trace, case=case) for case, trace in zip(cases, traces)]
        case_validations = [{"case_id": case.case_id, **validate_stage2_case(case).to_dict()} for case in cases]
        trace_validations = [
            {"case_id": trace.case_id, **validate_stage2_trace(trace, case=case).to_dict()}
            for case, trace in zip(cases, traces)
        ]
        row_validations = [
            {"case_id": case.case_id, **validate_ms_swift_stage2_row(row).to_dict()}
            for case, row in zip(cases, rows)
        ]

        summary = summarize_stage2_slices(
            cases,
            traces,
            rows,
            case_validations=case_validations,
            trace_validations=trace_validations,
            row_validations=row_validations,
        )
        review_rows = sample_stage2_review_rows(
            cases,
            traces,
            rows,
            case_validations=case_validations,
            trace_validations=trace_validations,
            row_validations=row_validations,
            per_slice=1,
        )

        self.assertEqual(split_counts, {"train": 0, "val": 0, "heldout": 3})
        self.assertEqual(summary["by_tool_policy"]["tool"]["count"], 1)
        self.assertEqual(summary["by_tool_policy"]["no_tool"]["count"], 1)
        self.assertEqual(summary["by_tool_policy"]["failed_tool"]["count"], 1)
        self.assertEqual(summary["by_op"]["search_poi"]["count"], 2)
        self.assertGreaterEqual(len(review_rows), 3)
        self.assertIn("final_response_preview", review_rows[0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs = write_stage2_split_outputs(cases=cases, traces=traces, rows=rows, output_dir=tmp_dir)
            self.assertEqual(outputs["heldout"]["count"], 3)
            heldout_rows = Path(outputs["heldout"]["ms_swift"]).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(heldout_rows), 3)


class Stage2PipelineCliTest(unittest.TestCase):
    def test_cli_builds_phase1_outputs_with_fake_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            archive_path = tmp / "phase1_archive.jsonl"
            archive_path.write_text(
                json.dumps(_sample_phase1_archive_row(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            output_dir = tmp / "outputs"
            split_dir = output_dir / "splits"
            cmd = [
                sys.executable,
                str(ROUTESERVER_ROOT / "tools" / "evaluation" / "build_sidewalk_agent_stage2_review_samples.py"),
                "--phase1-archive-jsonl",
                str(archive_path),
                "--limit",
                "1",
                "--execute-tools",
                "--tool-context",
                "fake",
                "--stage2-cases-out",
                str(output_dir / "sidewalk_agent_stage2_cases_smoke.jsonl"),
                "--traces-out",
                str(output_dir / "sidewalk_agent_stage2_traces_smoke.jsonl"),
                "--ms-swift-out",
                str(output_dir / "sidewalk_agentic_stage2_ms_swift_smoke.jsonl"),
                "--report-out",
                str(output_dir / "sidewalk_agent_stage2_report_smoke.json"),
                "--split-output-dir",
                str(split_dir),
                "--review-samples-out",
                str(output_dir / "sidewalk_agent_stage2_review_samples_smoke.jsonl"),
                "--review-index-out",
                str(output_dir / "sidewalk_agent_stage2_review_index_smoke.csv"),
                "--review-per-slice",
                "2",
                "--train-ratio",
                "0",
                "--val-ratio",
                "0",
                "--heldout-ratio",
                "1",
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROUTESERVER_ROOT / "src")
            subprocess.run(cmd, cwd=ROUTESERVER_ROOT, env=env, check=True)

            report = json.loads((output_dir / "sidewalk_agent_stage2_report_smoke.json").read_text(encoding="utf-8"))
            self.assertEqual(report["input_mode"], "phase1_archive")
            self.assertEqual(report["split"]["counts"], {"train": 0, "val": 0, "heldout": 1})
            self.assertEqual(report["validation"]["cases"]["invalid"], 0)
            self.assertEqual(report["validation"]["traces"]["invalid"], 0)
            self.assertEqual(report["validation"]["ms_swift_rows"]["invalid"], 0)
            self.assertEqual(report["slices"]["by_tool_policy"]["tool"]["count"], 1)
            self.assertEqual(report["review"]["sample_count"], 1)
            self.assertTrue((split_dir / "sidewalk_agent_stage2_cases_heldout.jsonl").exists())
            self.assertTrue((split_dir / "sidewalk_agent_stage2_traces_heldout.jsonl").exists())
            self.assertTrue((split_dir / "sidewalk_agentic_stage2_ms_swift_heldout.jsonl").exists())


def _sample_case(suffix: str) -> AgentStage2Case:
    return AgentStage2Case(
        case_id=f"case-{suffix}",
        user_request="找一个可以休息的地方，然后继续沿人行道前进",
        observation={
            "images": {"fpv": f"/data/{suffix}_fpv.jpg", "bev": f"/data/{suffix}_bev.jpg"},
            "current_location": "120.0,30.0",
        },
        ground_truth={
            "expected_tool_ops": ["search_poi"],
            "pixel_trajectory": [[round(0.5 + index * 0.002, 4), round(0.9 - index * 0.025, 4)] for index in range(16)],
            "intent": "go_straight",
        },
    )


def _sample_phase1_archive_row() -> dict:
    trajectory = [[round(0.5 + index * 0.002, 4), round(0.9 - index * 0.025, 4)] for index in range(16)]
    response = (
        "<think>\n"
        '<Scene>{"summary":"urban sidewalk scene","visibility":"clear","occlusion_sources":[],"uncertainty":"low"}</Scene>\n'
        '<FPVScene>{"navigable_support":["sidewalk"],"conditional_support":[],"risk_objects":[],"visibility":"clear","local_risk":"low"}</FPVScene>\n'
        '<BEVContext>{"scene_summary":"local BEV route context","route_summary":"route continues forward","route_direction_bin":"front","intent":"go_straight","target_direction":"front","bev_anchor":"front route segment"}</BEVContext>\n'
        '<CrossView>{"bev_anchor":"front route","fpv_anchor":"sidewalk center","near_reference":"curb","blind_area_context":{"left":"clear","right":"clear","front":"clear"},"fpv_route_region":"center","route_to_fpv_direction":"forward_center","safe_corridor":"center_corridor","alignment":"aligned","confidence":"high"}</CrossView>\n'
        '<Plan>{"motion_primitive":"go_straight","lateral_preference":"center","avoidance_side":"none","speed_policy":"normal","primary_route":"follow sidewalk","safety_policy":"stay on visible support","contingency":"slow if blocked"}</Plan>\n'
        "</think>\n"
        f'<finish>{json.dumps({"trajectory": trajectory, "intent": "go_straight", "confidence_level": "normal", "trajectory_type": "forward", "trajectory_validity": "valid"}, ensure_ascii=False, separators=(",", ":"))}</finish>'
    )
    return {
        "messages": [
            {"role": "system", "content": "Phase1 system prompt"},
            {
                "role": "user",
                "content": (
                    "<image><image>\n"
                    "Task: Sidewalk navigation structured reasoning and trajectory prediction\n"
                    "Instruction: Navigate safely on the sidewalk.\n"
                ),
            },
            {"role": "assistant", "content": response},
        ],
        "images": ["media/fpv/000001.jpg", "media/bev/000001.jpg"],
        "pixel_trajectory": trajectory,
        "trajectory_fpv_t": [[point[0], point[1], round(index * 0.25, 2)] for index, point in enumerate(trajectory)],
        "semantic_segmentation_path": "sidecars/seg/000001.png",
        "sidewalk_meta": {
            "archive_schema_version": "sidewalk_phase1_archive_v1",
            "source_id": "clip-a_000001",
            "clip_id": "clip-a",
            "frame_name": "000001",
            "lat": 30.0,
            "lng": 120.0,
            "intent_gt": "go_straight",
            "trajectory_source": "calibrated_pose_projection",
        },
    }


if __name__ == "__main__":
    unittest.main()
