from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "model_processing" / "sidewalk_phase1_cot_teacher.py"
_SPEC = importlib.util.spec_from_file_location("sidewalk_phase1_cot_teacher", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

TeacherConfig = _MODULE.TeacherConfig
ValidationResult = _MODULE.ValidationResult
build_cot_manifest_row = _MODULE.build_cot_manifest_row
generate_cot_dataset = _MODULE.generate_cot_dataset
load_evidence = _MODULE.load_evidence
render_template_answer = _MODULE.render_template_answer
validate_teacher_answer = _MODULE.validate_teacher_answer


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_source_row(root: Path) -> dict:
    trajectory = [[0.5, 0.9], [0.5, 0.7], [0.5, 0.55], [0.5, 0.45]]
    label_path = root / "trajectory_labels" / "000001.json"
    meta_path = root / "bev_patch_meta" / "000001.json"
    _write_json(
        label_path,
        {
            "schema_version": "sidewalk_trajectory_label_v1",
            "trajectory_source": "calibrated_pose_projection",
            "trajectory_norm": trajectory,
            "route_theta_deg": 0.0,
            "intent_gt": "go_straight",
            "anchor": "front",
        },
    )
    _write_json(
        meta_path,
        {
            "schema_version": "sidewalk_bev_patch_meta_v1",
            "route_polyline_local_m": [[0.0, 0.0], [0.0, 2.0], [0.0, 4.0], [0.0, 6.0]],
            "route_target_local_m": [0.0, 6.0],
            "route_theta_deg": 0.0,
            "intent_gt": "go_straight",
            "target_direction": "front",
            "confidence_level_gt": "normal",
        },
    )
    return {
        "id": "clip_a_000001",
        "clip_id": "clip_a",
        "frame_idx": 1,
        "fpv_image_path": str(root / "images" / "000001.jpg"),
        "bev_image_path": str(root / "bev_images" / "000001_bev.jpg"),
        "pixel_trajectory": trajectory,
        "image_width": 640,
        "image_height": 360,
        "instruction": "Navigate safely on the sidewalk.",
        "trajectory_label_path": str(label_path),
        "bev_patch_meta_path": str(meta_path),
        "route_context": {"route_theta_deg": 0.0, "intent_gt": "go_straight"},
        "sidewalk_meta": {"trajectory_source": "calibrated_pose_projection"},
    }


class SidewalkPhase1CotTeacherTest(unittest.TestCase):
    def test_template_answer_is_valid_and_preserves_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            row = _make_source_row(Path(tmp))
            evidence = load_evidence(row)
            answer = render_template_answer(row, evidence)
            validation = validate_teacher_answer(answer, row, evidence, TeacherConfig(expected_points=4))

            self.assertTrue(validation.valid, validation.errors)
            self.assertTrue(answer.startswith("<think>\n<Scene>"))
            self.assertIn("</Plan>\n</think>\n<finish>", answer)
            self.assertIn("navigable_support", validation.sections["FPVScene"])
            self.assertEqual(validation.sections["BEVContext"]["intent"], "go_straight")
            self.assertEqual(validation.sections["BEVContext"]["route_direction_bin"], "front")
            self.assertNotIn("route_theta_deg", validation.sections["BEVContext"])
            self.assertNotIn("route_theta_bin", validation.sections["BEVContext"])
            self.assertNotIn("ego_cell", validation.sections["BEVContext"])
            self.assertNotIn("target_cell", validation.sections["BEVContext"])
            self.assertNotIn("grid_size", validation.sections["BEVContext"])
            self.assertEqual(validation.sections["finish"]["trajectory"], row["pixel_trajectory"])
            self.assertEqual(validation.sections["finish"]["trajectory_type"], "straight_center")
            self.assertEqual(validation.sections["finish"]["trajectory_validity"], "valid")
            self.assertEqual(validation.sections["CrossView"]["fpv_route_region"], "center")
            self.assertEqual(validation.sections["Plan"]["speed_policy"], "normal")
            self.assertNotIn("route_theta_deg", validation.sections["finish"])

    def test_validator_rejects_teacher_that_changes_gt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            row = _make_source_row(Path(tmp))
            evidence = load_evidence(row)
            answer = render_template_answer(row, evidence)
            answer = answer.replace('"intent":"go_straight"', '"intent":"turn_left"', 1)
            validation = validate_teacher_answer(answer, row, evidence, TeacherConfig(expected_points=4))

            self.assertFalse(validation.valid)
            self.assertTrue(any(error.startswith("bev_intent_mismatch") for error in validation.errors))

    def test_manifest_row_uses_training_prompt_without_geometry_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            row = _make_source_row(Path(tmp))
            evidence = load_evidence(row)
            answer = render_template_answer(row, evidence)
            validation = validate_teacher_answer(answer, row, evidence, TeacherConfig(expected_points=4))
            manifest_row = build_cot_manifest_row(
                row,
                answer=answer,
                validation=validation,
                teacher_source="template_after_llm_failure",
                teacher_model="unit-test",
            )

            self.assertIn('<Scene>{"summary":"urban sidewalk scene"', manifest_row["prompt"])
            self.assertIn("occlusion_sources", manifest_row["prompt"])
            self.assertIn('<FPVScene>{"navigable_support"', manifest_row["prompt"])
            self.assertIn('<BEVContext>{"scene_summary"', manifest_row["prompt"])
            self.assertNotIn("route_theta_deg", manifest_row["prompt"])
            self.assertNotIn("route_theta_bin", manifest_row["prompt"])
            self.assertIn("fpv_route_region", manifest_row["prompt"])
            self.assertIn("motion_primitive", manifest_row["prompt"])
            self.assertIn("trajectory_validity", manifest_row["prompt"])
            self.assertIn("FPV image size: 640x360 px", manifest_row["prompt"])
            self.assertIn('<finish>{"trajectory":[[0.5,0.9]]', manifest_row["prompt"])
            self.assertIn("pixel conversion is x_px=x*width, y_px=y*height", manifest_row["prompt"])
            self.assertIn("First identify safe visible navigation support in <FPVScene>.navigable_support", manifest_row["prompt"])
            self.assertIn("final point should still be on near-field visible ground support", manifest_row["prompt"])
            self.assertIn("Every trajectory point must lie on visible walkable ground support", manifest_row["prompt"])
            self.assertIn("Do not place any point in sky", manifest_row["prompt"])
            self.assertNotIn("ego_cell", manifest_row["prompt"])
            self.assertNotIn("target_cell", manifest_row["prompt"])
            self.assertNotIn("grid_size", manifest_row["prompt"])
            self.assertNotIn("BEV cells", manifest_row["prompt"])
            self.assertNotIn("trajectory_norm", manifest_row["prompt"])
            self.assertNotIn("[[0.5, 0.9]", manifest_row["prompt"])
            self.assertEqual(manifest_row["response"], answer)
            self.assertEqual(manifest_row["cot_teacher"]["schema_version"], "sidewalk_phase1_cot_teacher_v5")

    def test_validator_rejects_free_text_inside_think(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            row = _make_source_row(Path(tmp))
            evidence = load_evidence(row)
            answer = render_template_answer(row, evidence).replace("<think>\n", "<think>\nThis repeats the structured fields.\n", 1)
            validation = validate_teacher_answer(answer, row, evidence, TeacherConfig(expected_points=4))

            self.assertFalse(validation.valid)
            self.assertIn("text_before_nested_tag:Scene", validation.errors)

    def test_validator_does_not_require_bev_grid_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            row = _make_source_row(Path(tmp))
            evidence = load_evidence(row)
            answer = render_template_answer(row, evidence)
            validation = validate_teacher_answer(answer, row, evidence, TeacherConfig(expected_points=4))

            self.assertTrue(validation.valid, validation.errors)
            self.assertNotIn("ego_cell", validation.sections["BEVContext"])
            self.assertNotIn("target_cell", validation.sections["BEVContext"])

    def test_generate_cot_dataset_retries_and_writes_valid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _make_source_row(root)
            source = root / "phase1_source_train.jsonl"
            output = root / "phase1_cot_train.jsonl"
            report = root / "phase1_cot_report_train.json"
            source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            async def fake_call(row_arg, evidence_arg, config_arg, *, previous_answer=None, errors=None, api_key=None):
                if previous_answer is None:
                    return "invalid teacher answer", {"model": "fake-vlm"}
                return render_template_answer(row_arg, evidence_arg), {"model": "fake-vlm"}

            original = _MODULE.call_llm_teacher
            _MODULE.call_llm_teacher = fake_call
            try:
                summary = asyncio.run(
                    generate_cot_dataset(
                        str(source),
                        str(output),
                        report_json=str(report),
                        config=TeacherConfig(expected_points=4, max_retries=1, fallback_policy="error"),
                    )
                )
            finally:
                _MODULE.call_llm_teacher = original

            self.assertEqual(summary["written_rows"], 1)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["cot_teacher"]["teacher_source"], "llm_teacher")
            self.assertEqual(rows[0]["cot_teacher"]["teacher_model"], "fake-vlm")
            saved_report = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["written_rows"], 1)

    def test_generate_cot_dataset_parallel_workers_write_valid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row_a = _make_source_row(root)
            row_b = json.loads(json.dumps(row_a))
            row_b["id"] = "clip_a_000002"
            row_b["frame_idx"] = 2
            source = root / "phase1_source_train.jsonl"
            output = root / "phase1_cot_train.jsonl"
            report = root / "phase1_cot_report_train.json"
            source.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in (row_a, row_b)) + "\n",
                encoding="utf-8",
            )
            seen_keys: list[str | None] = []

            async def fake_call(row_arg, evidence_arg, config_arg, *, previous_answer=None, errors=None, api_key=None):
                seen_keys.append(api_key)
                await asyncio.sleep(0.01)
                return render_template_answer(row_arg, evidence_arg), {"model": "fake-vlm"}

            original = _MODULE.call_llm_teacher
            _MODULE.call_llm_teacher = fake_call
            try:
                summary = asyncio.run(
                    generate_cot_dataset(
                        str(source),
                        str(output),
                        report_json=str(report),
                        config=TeacherConfig(expected_points=4, max_retries=0, fallback_policy="error"),
                        concurrency=2,
                        api_keys=("key-a", "key-b"),
                    )
                )
            finally:
                _MODULE.call_llm_teacher = original

            self.assertEqual(summary["written_rows"], 2)
            self.assertEqual(summary["concurrency"], 2)
            self.assertEqual(summary["api_key_count"], 2)
            self.assertTrue((Path(str(output) + ".state.jsonl")).is_file())
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["id"] for row in rows}, {"clip_a_000001", "clip_a_000002"})
            self.assertIn("key-a", seen_keys)
            self.assertIn("key-b", seen_keys)


if __name__ == "__main__":
    unittest.main()
