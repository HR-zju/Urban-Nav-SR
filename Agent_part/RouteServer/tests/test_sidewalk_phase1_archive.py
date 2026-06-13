from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


_ARCHIVE_PATH = Path(__file__).resolve().parents[1] / "tools" / "model_processing" / "sidewalk_phase1_archive.py"
_ARCHIVE_SPEC = importlib.util.spec_from_file_location("sidewalk_phase1_archive", _ARCHIVE_PATH)
if _ARCHIVE_SPEC is None or _ARCHIVE_SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_ARCHIVE_PATH}")
_ARCHIVE = importlib.util.module_from_spec(_ARCHIVE_SPEC)
sys.modules[_ARCHIVE_SPEC.name] = _ARCHIVE
_ARCHIVE_SPEC.loader.exec_module(_ARCHIVE)

_TEACHER_PATH = Path(__file__).resolve().parents[1] / "tools" / "model_processing" / "sidewalk_phase1_cot_teacher.py"
_TEACHER_SPEC = importlib.util.spec_from_file_location("sidewalk_phase1_cot_teacher_for_archive_test", _TEACHER_PATH)
if _TEACHER_SPEC is None or _TEACHER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_TEACHER_PATH}")
_TEACHER = importlib.util.module_from_spec(_TEACHER_SPEC)
sys.modules[_TEACHER_SPEC.name] = _TEACHER
_TEACHER_SPEC.loader.exec_module(_TEACHER)

ArchiveConfig = _ARCHIVE.ArchiveConfig
archive_cot_dataset = _ARCHIVE.archive_cot_dataset
load_evidence = _TEACHER.load_evidence
render_template_answer = _TEACHER.render_template_answer
validate_teacher_answer = _TEACHER.validate_teacher_answer
TeacherConfig = _TEACHER.TeacherConfig


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_cot_row(root: Path) -> dict:
    trajectory = [[0.5, 0.9], [0.5, 0.7], [0.5, 0.55], [0.5, 0.45]]
    fpv = root / "images" / "000001.jpg"
    bev = root / "bev_images" / "000001_bev.jpg"
    fpv.parent.mkdir(parents=True)
    bev.parent.mkdir(parents=True)
    Image.new("RGB", (640, 360), (20, 20, 20)).save(fpv)
    Image.new("RGB", (512, 512), (200, 200, 200)).save(bev)

    label_path = root / "phase1" / "clips" / "clip_a" / "trajectory_labels" / "000001.json"
    meta_path = root / "phase1" / "clips" / "clip_a" / "bev_patch_meta" / "000001.json"
    qa_path = root / "phase1" / "clips" / "clip_a" / "qa" / "000001_qa.json"
    _write_json(
        label_path,
        {
            "schema_version": "sidewalk_phase1_trajectory_v2",
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
    _write_json(qa_path, {"schema_version": "sidewalk_bev_fpv_qa_v1", "samples": []})
    row = {
        "id": "clip_a_000001",
        "clip_id": "clip_a",
        "frame_idx": 1,
        "frame_name": "000001",
        "fpv_image_path": str(fpv),
        "bev_image_path": str(bev),
        "pixel_trajectory": trajectory,
        "trajectory_fpv_t": [[0.5, 0.9, 0.0], [0.5, 0.7, 0.1], [0.5, 0.55, 0.2], [0.5, 0.45, 0.3]],
        "trajectory_fpv_tv": [
            [0.5, 0.9, 0.0, 0.3],
            [0.5, 0.7, 0.1, 0.4],
            [0.5, 0.55, 0.2, 0.4],
            [0.5, 0.45, 0.3, 0.4],
        ],
        "trajectory_ego_t": [[0.0, 0.0, 0.0, 0.3], [0.0, 1.0, 0.1, 0.4], [0.0, 2.0, 0.2, 0.4], [0.0, 3.0, 0.3, 0.4]],
        "trajectory_policy": {"schema_version": "sidewalk_time_conditioned_policy_v1"},
        "image_width": 640,
        "image_height": 360,
        "instruction": "Navigate safely on the sidewalk.",
        "system_prompt": "System prompt",
        "prompt": "<image><image>\nTask: Sidewalk navigation structured reasoning and trajectory prediction",
        "trajectory_label_path": str(label_path),
        "bev_patch_meta_path": str(meta_path),
        "qa_bundle_path": str(qa_path),
        "route_context": {"route_theta_deg": 0.0, "intent_gt": "go_straight"},
        "sidewalk_meta": {
            "clip_id": "clip_a",
            "frame_idx": 1,
            "intent_gt": "go_straight",
            "route_theta_deg": 0.0,
            "trajectory_source": "calibrated_pose_projection",
        },
    }
    evidence = load_evidence(row)
    answer = render_template_answer(row, evidence)
    validation = validate_teacher_answer(answer, row, evidence, TeacherConfig(expected_points=4))
    if not validation.valid:
        raise AssertionError(validation.errors)
    row["response"] = answer
    row["cot_teacher"] = {"teacher_source": "template_after_llm_failure", "schema_version": "sidewalk_phase1_cot_teacher_v5"}
    return row


class SidewalkPhase1ArchiveTest(unittest.TestCase):
    def test_archive_writes_ms_swift_ready_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _make_cot_row(root)
            source = root / "phase1_cot_train.jsonl"
            source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            archive = root / "archive"
            summary = archive_cot_dataset(
                str(source),
                str(archive),
                config=ArchiveConfig(split="train", media_mode="hardlink", path_mode="absolute", expected_points=4, write_review_html=True),
            )

            self.assertEqual(summary["accepted_rows"], 1)
            self.assertEqual(summary["rejected_rows"], 0)
            train_rows = [json.loads(line) for line in (archive / "manifests" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(train_rows), 1)
            out = train_rows[0]
            self.assertEqual([m["role"] for m in out["messages"]], ["system", "user", "assistant"])
            self.assertEqual(len(out["images"]), 2)
            self.assertTrue(Path(out["images"][0]).is_file())
            self.assertTrue(Path(out["images"][1]).is_file())
            self.assertEqual(out["pixel_trajectory"], row["pixel_trajectory"])
            self.assertIn("cot_sections_path", out["sidewalk_meta"])
            self.assertTrue((archive / "manifests" / "train.review.csv").is_file())
            self.assertTrue((archive / "dataset_card.json").is_file())
            sample_pages = list((archive / "review" / "samples").glob("*.html"))
            self.assertEqual(len(sample_pages), 1)
            sample_html = sample_pages[0].read_text(encoding="utf-8")
            self.assertIn("FPV + GT trajectory", sample_html)
            self.assertIn("<svg viewBox='0 0 100 100'", sample_html)

    def test_archive_rejects_invalid_cot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _make_cot_row(root)
            row["response"] = row["response"].replace('"intent":"go_straight"', '"intent":"turn_left"', 1)
            source = root / "phase1_cot_train.jsonl"
            source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            archive = root / "archive"
            summary = archive_cot_dataset(
                str(source),
                str(archive),
                config=ArchiveConfig(split="train", media_mode="reference", path_mode="absolute", expected_points=4),
            )

            self.assertEqual(summary["accepted_rows"], 0)
            self.assertEqual(summary["rejected_rows"], 1)
            self.assertEqual((archive / "manifests" / "train.jsonl").read_text(encoding="utf-8"), "")
            errors = [json.loads(line) for line in (archive / "manifests" / "train.errors.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any("bev_intent_mismatch" in item for item in errors[0]["errors"]))

    def test_archive_can_write_relative_media_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _make_cot_row(root)
            source = root / "phase1_cot_train.jsonl"
            source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            archive = root / "archive"
            archive_cot_dataset(
                str(source),
                str(archive),
                config=ArchiveConfig(split="train", media_mode="copy", path_mode="relative", expected_points=4),
            )

            train_rows = [json.loads(line) for line in (archive / "manifests" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertFalse(Path(train_rows[0]["images"][0]).is_absolute())
            self.assertTrue(train_rows[0]["images"][0].startswith("../media/"))


if __name__ == "__main__":
    unittest.main()
