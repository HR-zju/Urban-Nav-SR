from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "model_processing" / "sidewalk_phase1_dataset.py"
_SPEC = importlib.util.spec_from_file_location("sidewalk_phase1_dataset", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

Phase1Config = _MODULE.Phase1Config
build_phase1_dataset = _MODULE.build_phase1_dataset
local_xy_m = _MODULE.local_xy_m
project_pose_to_fpv = _MODULE.project_pose_to_fpv
smooth_forward_fpv_trajectory = _MODULE.smooth_forward_fpv_trajectory
CameraParams = _MODULE.CameraParams


class SidewalkPhase1DatasetTest(unittest.TestCase):
    def test_local_xy_heading_convention(self) -> None:
        # Facing north: north is forward, east is right.
        x, y = local_xy_m(30.0, 120.0, 30.0001, 120.0, 0.0)
        self.assertAlmostEqual(x, 0.0, delta=0.1)
        self.assertGreater(y, 10.0)

        x_east, y_east = local_xy_m(30.0, 120.0, 30.0, 120.0001, 90.0)
        self.assertAlmostEqual(x_east, 0.0, delta=0.1)
        self.assertGreater(y_east, 9.0)

    def test_build_phase1_dataset_writes_source_and_qa(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sidewalk" / "chunk_0" / "clip_a"
            images = root / "images"
            bevs = root / "bev_images"
            images.mkdir(parents=True)
            bevs.mkdir(parents=True)
            trajectory = []
            for idx in range(8):
                name = f"{idx:06d}"
                Image.new("RGB", (640, 360), (20, 20, 20)).save(images / f"{name}.jpg")
                Image.new("RGB", (512, 512), (200, 200, 200)).save(bevs / f"{name}_bev.jpg")
                trajectory.append(
                    {
                        "timestamp": idx,
                        "lat": 30.0 + idx * 0.00001,
                        "lng": 120.0,
                        "heading_deg": 0.0,
                        "image_path": str(images / f"{name}.jpg"),
                    }
                )
            (root / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")

            out = Path(tmp) / "phase1"
            report = build_phase1_dataset(
                str(root),
                str(out),
                config=Phase1Config(num_points=4, frame_step=2, future_frames=4),
                limit=2,
                allow_heuristic_trajectory=True,
            )

            self.assertEqual(report["frames_written"], 2)
            source = out / "manifests" / "phase1_source_train.jsonl"
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(rows[0]["pixel_trajectory"]), 4)
            self.assertEqual(len(rows[0]["trajectory_fpv_t"]), 4)
            self.assertEqual(len(rows[0]["trajectory_fpv_tv"]), 4)
            self.assertEqual(len(rows[0]["trajectory_ego_t"]), 4)
            self.assertEqual(rows[0]["trajectory_fpv_t"][0][2], 0.0)
            self.assertEqual(rows[0]["trajectory_policy"]["dynamic_obstacle_audit"]["status"], "not_evaluated_no_dynamic_masks")
            self.assertEqual(rows[0]["sidewalk_meta"]["trajectory_source"], "heuristic_gnss_projection")
            self.assertTrue(Path(rows[0]["qa_bundle_path"]).exists())
            label = json.loads(Path(rows[0]["trajectory_label_path"]).read_text(encoding="utf-8"))
            self.assertEqual(label["schema_version"], "sidewalk_phase1_trajectory_v2")
            self.assertIn("trajectory_fpv_t", label)
            self.assertIn("trajectory_policy", label)
            bev_meta = json.loads(Path(rows[0]["bev_patch_meta_path"]).read_text(encoding="utf-8"))
            self.assertIn("route_polyline_local_m", bev_meta)
            self.assertIn("route_target_local_m", bev_meta)
            self.assertNotIn("grid_size", bev_meta)
            self.assertNotIn("ego_cell_gt", bev_meta)
            self.assertNotIn("target_cell_gt", bev_meta)
            self.assertNotIn("route_polyline_cells", bev_meta)
            qa = json.loads(Path(rows[0]["qa_bundle_path"]).read_text(encoding="utf-8"))
            self.assertEqual(qa["schema_version"], "sidewalk_bev_fpv_qa_v1")
            self.assertIn("gt_trajectory_fpv_t", qa)
            task_types = {item["task_type"] for item in qa["samples"]}
            self.assertIn("bev_route_direction", task_types)
            self.assertIn("trajectory_intent_check", task_types)
            answers_by_type = {item["task_type"]: item["teacher_answer"] for item in qa["samples"]}
            self.assertIn("<BEVContext>", answers_by_type["bev_route_direction"])
            self.assertNotIn("route_theta_deg", answers_by_type["bev_route_direction"])
            self.assertNotIn("route_theta_bin", answers_by_type["bev_route_direction"])
            self.assertIn("<finish>", answers_by_type["trajectory_intent_check"])
            self.assertNotIn("<BEVIntent>", answers_by_type["bev_route_direction"])
            self.assertNotIn("<TrajectorySupport>", answers_by_type["trajectory_intent_check"])

    def test_build_phase1_dataset_uses_calibrated_internnav_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sidewalk" / "chunk_0" / "clip_a"
            images = root / "images"
            bevs = root / "bev_images"
            images.mkdir(parents=True)
            bevs.mkdir(parents=True)
            for idx in range(6):
                name = f"{idx:06d}"
                Image.new("RGB", (640, 360), (20, 20, 20)).save(images / f"{name}.jpg")
                Image.new("RGB", (512, 512), (200, 200, 200)).save(bevs / f"{name}_bev.jpg")
            np.save(root / "frame_positions.npy", np.asarray([[float(i), 0.0] for i in range(6)], dtype=np.float32))
            np.save(root / "frame_orientations.npy", np.asarray([[1.0, 0.0, 0.0, 0.0] for _ in range(6)], dtype=np.float32))
            np.save(root / "frame_times.npy", np.asarray([idx * 50_000_000 for idx in range(6)], dtype=np.int64))
            np.save(root / "frame_speeds.npy", np.asarray([0.2 + idx * 0.1 for idx in range(6)], dtype=np.float32))
            camera = {
                "K": [[100.0, 0.0, 320.0], [0.0, 100.0, 180.0], [0.0, 0.0, 1.0]],
                "image_width": 640,
                "image_height": 360,
                "camera_height_meters": 1.0,
            }
            (root.parent / "camera_params.json").write_text(json.dumps(camera), encoding="utf-8")

            out = Path(tmp) / "phase1"
            report = build_phase1_dataset(
                str(root),
                str(out),
                config=Phase1Config(num_points=4, frame_step=2, future_frames=4),
                limit=1,
            )

            self.assertEqual(report["frames_written"], 1)
            rows = [
                json.loads(line)
                for line in (out / "manifests" / "phase1_source_train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            row = rows[0]
            self.assertEqual(row["sidewalk_meta"]["trajectory_source"], "calibrated_pose_projection")
            trajectory = row["pixel_trajectory"]
            self.assertAlmostEqual(trajectory[0][0], 0.5, places=5)
            self.assertAlmostEqual(trajectory[0][1], 0.9, places=5)
            self.assertAlmostEqual(trajectory[1][0], 0.5, places=5)
            self.assertAlmostEqual(trajectory[1][1], 230.0 / 360.0, places=5)
            self.assertAlmostEqual(row["route_context"]["route_theta_deg"], 0.0, places=3)
            self.assertEqual(row["route_context"]["intent_gt"], "go_straight")
            self.assertEqual(row["trajectory_fpv_t"][0][2], 0.0)
            self.assertAlmostEqual(row["trajectory_fpv_t"][1][2], 0.1, places=5)
            self.assertAlmostEqual(row["trajectory_fpv_tv"][1][3], 0.4, places=5)
            self.assertEqual(row["trajectory_policy"]["primary_temporal_target"], "trajectory_fpv_t")

    def test_future_horizon_uses_timestamp_window_before_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sidewalk" / "chunk_0" / "clip_a"
            images = root / "images"
            bevs = root / "bev_images"
            images.mkdir(parents=True)
            bevs.mkdir(parents=True)
            frame_count = 70
            # 80ms spacing is 12.5fps.  The old fixed future_frames=50 path
            # would produce a ~4s horizon; the timestamp horizon should keep it
            # near 2.5s to match chunk0's 20Hz/50-frame target.
            for idx in range(frame_count):
                name = f"{idx:06d}"
                Image.new("RGB", (640, 360), (20, 20, 20)).save(images / f"{name}.jpg")
                Image.new("RGB", (512, 512), (200, 200, 200)).save(bevs / f"{name}_bev.jpg")
            np.save(root / "frame_positions.npy", np.asarray([[float(i), 0.0] for i in range(frame_count)], dtype=np.float32))
            np.save(root / "frame_orientations.npy", np.asarray([[1.0, 0.0, 0.0, 0.0] for _ in range(frame_count)], dtype=np.float32))
            np.save(root / "frame_times.npy", np.asarray([idx * 80_000_000 for idx in range(frame_count)], dtype=np.int64))
            camera = {
                "K": [[100.0, 0.0, 320.0], [0.0, 100.0, 180.0], [0.0, 0.0, 1.0]],
                "image_width": 640,
                "image_height": 360,
                "camera_height_meters": 1.0,
            }
            (root.parent / "camera_params.json").write_text(json.dumps(camera), encoding="utf-8")

            out = Path(tmp) / "phase1"
            report = build_phase1_dataset(
                str(root),
                str(out),
                config=Phase1Config(num_points=4, frame_step=1, future_frames=50, future_horizon_sec=2.5),
                limit=1,
            )

            self.assertEqual(report["frames_written"], 1)
            row = json.loads((out / "manifests" / "phase1_source_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
            last_dt = row["trajectory_fpv_t"][-1][2]
            self.assertLessEqual(last_dt, 2.5)
            self.assertGreater(last_dt, 2.3)

    def test_projection_matches_internnav_formula(self) -> None:
        camera = CameraParams(
            K=np.asarray([[100.0, 0.0, 320.0], [0.0, 100.0, 180.0], [0.0, 0.0, 1.0]], dtype=np.float64),
            image_width=640,
            image_height=360,
            camera_height_meters=1.0,
        )
        u, v = project_pose_to_fpv((0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (2.0, 0.0), camera)
        self.assertAlmostEqual(u, 320.0, places=5)
        self.assertAlmostEqual(v, 230.0, places=5)

    def test_smooth_forward_trajectory_repairs_near_field_bottom_kink(self) -> None:
        trajectory = np.asarray(
            [[0.5, 0.9], [0.5005, 0.999], [0.501, 0.85], [0.503, 0.72]],
            dtype=np.float32,
        )
        smoothed = smooth_forward_fpv_trajectory(
            trajectory,
            config=Phase1Config(num_points=4, trajectory_min_y_step=0.005),
        )

        self.assertAlmostEqual(smoothed[0, 0], 0.5, places=5)
        self.assertAlmostEqual(smoothed[0, 1], 0.9, places=5)
        self.assertLess(smoothed[1, 1], smoothed[0, 1])
        self.assertTrue(np.all(np.diff(smoothed[:, 1]) < 0.0))

    def test_internnav_one_based_image_names_do_not_shift_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sidewalk" / "chunk_0" / "clip_a"
            images = root / "front_images"
            bevs = root / "bev_images"
            images.mkdir(parents=True)
            bevs.mkdir(parents=True)
            for idx in range(1, 7):
                name = f"{idx:06d}"
                Image.new("RGB", (640, 360), (20, 20, 20)).save(images / f"{name}.jpg")
                Image.new("RGB", (512, 512), (200, 200, 200)).save(bevs / f"{name}_bev.jpg")
            np.save(root / "frame_positions.npy", np.asarray([[float(i), 0.0] for i in range(6)], dtype=np.float32))
            np.save(root / "frame_orientations.npy", np.asarray([[1.0, 0.0, 0.0, 0.0] for _ in range(6)], dtype=np.float32))
            (root.parent / "camera_params.json").write_text(
                json.dumps(
                    {
                        "K": [[100.0, 0.0, 320.0], [0.0, 100.0, 180.0], [0.0, 0.0, 1.0]],
                        "image_width": 640,
                        "image_height": 360,
                        "camera_height_meters": 1.0,
                    }
                ),
                encoding="utf-8",
            )

            out = Path(tmp) / "phase1"
            report = build_phase1_dataset(
                str(root),
                str(out),
                config=Phase1Config(num_points=4, frame_step=1, future_frames=4),
                limit=2,
            )
            self.assertEqual(report["frames_written"], 2)
            rows = [
                json.loads(line)
                for line in (out / "manifests" / "phase1_source_train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(rows[0]["fpv_image_path"].endswith("000001.jpg"))
            self.assertTrue(rows[1]["fpv_image_path"].endswith("000002.jpg"))

    def test_internnav_maps_high_rate_fpv_to_low_rate_bev_by_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sidewalk" / "chunk_0" / "clip_a"
            images = root / "images"
            bevs = root / "bev_images"
            images.mkdir(parents=True)
            bevs.mkdir(parents=True)
            for idx in range(6):
                name = f"{idx:06d}"
                Image.new("RGB", (640, 360), (20, 20, 20)).save(images / f"{name}.jpg")
            for idx in (0, 3):
                name = f"{idx:06d}"
                Image.new("RGB", (512, 512), (200, 200, 200)).save(bevs / f"{name}_bev.jpg")
            bev_index = {
                "schema_version": "sidewalk_bev_index_v1",
                "entries": [
                    {
                        "frame_idx": 0,
                        "frame_name": "000000",
                        "timestamp": 0,
                        "bev_image_path": str(bevs / "000000_bev.jpg"),
                    },
                    {
                        "frame_idx": 3,
                        "frame_name": "000003",
                        "timestamp": 150_000_000,
                        "bev_image_path": str(bevs / "000003_bev.jpg"),
                    },
                ],
            }
            (root / "bev_index.json").write_text(json.dumps(bev_index), encoding="utf-8")
            np.save(root / "frame_times.npy", np.asarray([idx * 50_000_000 for idx in range(6)], dtype=np.int64))
            np.save(root / "frame_positions.npy", np.asarray([[float(i), 0.0] for i in range(6)], dtype=np.float32))
            np.save(root / "frame_orientations.npy", np.asarray([[1.0, 0.0, 0.0, 0.0] for _ in range(6)], dtype=np.float32))
            (root.parent / "camera_params.json").write_text(
                json.dumps(
                    {
                        "K": [[100.0, 0.0, 320.0], [0.0, 100.0, 180.0], [0.0, 0.0, 1.0]],
                        "image_width": 640,
                        "image_height": 360,
                        "camera_height_meters": 1.0,
                    }
                ),
                encoding="utf-8",
            )

            out = Path(tmp) / "phase1"
            report = build_phase1_dataset(
                str(root),
                str(out),
                config=Phase1Config(num_points=4, future_frames=3),
                limit=3,
            )

            self.assertEqual(report["frames_written"], 3)
            rows = [
                json.loads(line)
                for line in (out / "manifests" / "phase1_source_train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(rows[0]["bev_image_path"].endswith("000000_bev.jpg"))
            self.assertEqual(rows[0]["sidewalk_meta"]["bev_frame_idx"], 0)
            self.assertTrue(rows[1]["bev_image_path"].endswith("000000_bev.jpg"))
            self.assertEqual(rows[1]["sidewalk_meta"]["bev_frame_idx"], 0)
            self.assertEqual(rows[1]["sidewalk_meta"]["bev_mapping_source"], "nearest_timestamp")
            self.assertTrue(rows[2]["bev_image_path"].endswith("000003_bev.jpg"))
            self.assertEqual(rows[2]["sidewalk_meta"]["bev_frame_idx"], 3)


if __name__ == "__main__":
    unittest.main()
