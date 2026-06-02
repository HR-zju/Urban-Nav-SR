from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
