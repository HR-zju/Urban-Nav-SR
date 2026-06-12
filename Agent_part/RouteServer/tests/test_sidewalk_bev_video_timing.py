from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "dataset_processing" / "process_sidewalk_dataset.py"
_SPEC = importlib.util.spec_from_file_location("process_sidewalk_dataset", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_bev_video_timing = _MODULE._bev_video_timing
_select_indices_by_fps = _MODULE._select_indices_by_fps
_route_path_for_bev_frame = _MODULE._route_path_for_bev_frame
process_sidewalk_ride = _MODULE.process_sidewalk_ride
from frodobot_bev_enhanced import _rotate_point_about_center
from mcp_server.services.visual_prompt_service import _rotate_point_about_center as _service_rotate_point_about_center


class SidewalkBevVideoTimingTest(unittest.TestCase):
    def test_timestamps_override_stale_fallback_fps(self) -> None:
        timestamps_ns = [0, 50_000_000, 100_000_000, 150_000_000]
        durations, encode_fps = _bev_video_timing(
            num_frames=4,
            fallback_fps=2.0,
            timestamps_ns=timestamps_ns,
        )

        self.assertEqual(len(durations), 4)
        self.assertAlmostEqual(encode_fps, 20.0, places=4)
        self.assertTrue(all(abs(duration - 0.05) < 1e-6 for duration in durations))

    def test_fallback_fps_is_used_without_timestamps(self) -> None:
        durations, encode_fps = _bev_video_timing(
            num_frames=3,
            fallback_fps=2.0,
            timestamps_ns=None,
        )

        self.assertEqual(durations, [0.5, 0.5, 0.5])
        self.assertEqual(encode_fps, 2.0)

    def test_select_indices_by_timestamp_fps(self) -> None:
        timestamps_ns = [idx * 50_000_000 for idx in range(21)]
        self.assertEqual(_select_indices_by_fps(timestamps_ns, 2.0), [0, 10, 20])

    def test_route_overlay_rotation_matches_pil_screen_coordinates(self) -> None:
        x, y = _rotate_point_about_center(70.0, 50.0, 50.0, 50.0, 90.0)
        self.assertAlmostEqual(x, 50.0, places=6)
        self.assertAlmostEqual(y, 30.0, places=6)

        sx, sy = _service_rotate_point_about_center(
            x=70.0,
            y=50.0,
            center_x=50.0,
            center_y=50.0,
            angle_deg=90.0,
        )
        self.assertAlmostEqual(sx, 50.0, places=6)
        self.assertAlmostEqual(sy, 30.0, places=6)

    def test_global_full_route_scope_keeps_complete_route(self) -> None:
        route = [(30.0, 120.0), (30.0001, 120.0), (30.0002, 120.0)]
        overlay, meta = _route_path_for_bev_frame(
            route,
            frame_pos=1,
            route_scope="global_full",
            lookahead_s=1.0,
            bev_sample_fps=2.0,
        )

        self.assertEqual(overlay, route)
        self.assertEqual(meta["scope"], "global_full")
        self.assertEqual(meta["start_sample_index"], 0)
        self.assertEqual(meta["end_sample_index"], 2)

    def test_process_sidewalk_ride_creates_missing_output_root_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "clip"
            output = Path(tmp) / "missing" / "output"
            root.mkdir()
            np.save(root / "frame_gps.npy", np.asarray([[30.0, 120.0], [30.0, 120.00001]], dtype=np.float64))
            np.save(root / "frame_orientations.npy", np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64))
            np.save(root / "frame_times.npy", np.asarray([0, 50_000_000], dtype=np.int64))
            (root / "front.mp4").write_bytes(b"not a real video")

            try:
                process_sidewalk_ride(root, output, bev_sample_fps=2.0)
            except Exception:
                pass

            self.assertTrue(output.exists())

    def test_pose_ekf_does_not_override_bev_position_by_default(self) -> None:
        class FakePoseEKF:
            @classmethod
            def init_from_latlng_heading(cls, lat, lng, heading_deg, pos_var=25.0, theta_var=0.5):
                return cls(float(lat), float(lng), float(heading_deg))

            def __init__(self, lat: float, lng: float, heading_deg: float) -> None:
                self.lat = lat
                self.lng = lng
                self.heading_deg = heading_deg

            def predict(self, omega, dt, q_pos=1e-3, q_theta=1e-2) -> None:
                return None

            def update_gps_and_yaw(self, lat, lng, yaw_deg=None, r_xy=9.0, r_theta=0.5) -> None:
                if yaw_deg is not None:
                    self.heading_deg = float(yaw_deg)

            def get_latlng(self):
                # Simulate the historical failure mode: EKF x/y remains pinned at
                # the first frame because there is no translational odometry input.
                return self.lat, self.lng

            def get_heading_deg(self):
                return self.heading_deg

        original_ekf = _MODULE.PoseEKF
        original_render = _MODULE.render_bev_map
        _MODULE.PoseEKF = FakePoseEKF
        _MODULE.render_bev_map = lambda **kwargs: Image.new("RGB", (64, 64), (240, 240, 240))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "clip"
                root.mkdir()
                frames = 81
                gps = np.asarray([[30.0 + idx * 0.00001, 120.0] for idx in range(frames)], dtype=np.float64)
                np.save(root / "frame_gps.npy", gps)
                np.save(root / "frame_orientations.npy", np.asarray([[0.0, 0.0, 0.0, 1.0] for _ in range(frames)], dtype=np.float64))
                np.save(root / "frame_times.npy", np.asarray([idx * 50_000_000 for idx in range(frames)], dtype=np.int64))
                (root / "front.mp4").write_bytes(b"not a real video")

                process_sidewalk_ride(root, root, bev_sample_fps=2.0, use_pose_ekf=True)

                trajectory = json.loads((root / "trajectory.json").read_text(encoding="utf-8"))
                samples = json.loads((root / "samples.json").read_text(encoding="utf-8"))
                bev_sample_trajectory = json.loads((root / "bev_sample_trajectory.json").read_text(encoding="utf-8"))
                bev_index = json.loads((root / "bev_index.json").read_text(encoding="utf-8"))

                self.assertEqual(len(trajectory), frames)
                self.assertLess(len(samples), frames)
                self.assertEqual(len(bev_sample_trajectory), len(samples))
                self.assertEqual(len(bev_index["entries"]), len(samples))
                self.assertEqual(
                    [row["frame_idx"] for row in bev_sample_trajectory],
                    [row["frame_idx"] for row in bev_index["entries"]],
                )
                self.assertGreater(trajectory[-1]["lat"] - trajectory[0]["lat"], 0.00003)
                self.assertEqual(trajectory[0]["position_source"], "gps_smoothed")
                self.assertTrue(str(trajectory[0]["attitude_source"]).startswith("ekf_"))
                meta = json.loads((root / "bev_images" / "000000_bev.jpg.bevmeta.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["position_source"], "gps_smoothed")
                self.assertFalse(meta["use_ekf_position"])
        finally:
            _MODULE.PoseEKF = original_ekf
            _MODULE.render_bev_map = original_render


if __name__ == "__main__":
    unittest.main()
