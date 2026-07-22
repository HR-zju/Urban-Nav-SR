from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from review_tool.gt_overlay import (
    GroundTruthOverlay,
    _continuous_quaternions,
    _slerp_one,
    _yaw_from_quaternion,
)


def straight_poses(count=101, step=0.05):
    return [
        SimpleNamespace(
            t=index * step,
            x=index * step,
            y=0.0,
            qx=0.0,
            qy=0.0,
            qz=0.0,
            qw=1.0,
        )
        for index in range(count)
    ]


def overlay_config():
    return {
        "enabled": True,
        "horizons_sec": [2.0, 1.0],
        "trajectory_sample_fps": 20,
        "camera_time_offset_ms": 0.0,
        "intrinsics": [300.0, 300.0, 320.0, 240.0],
        "distortion": [0.0, 0.0, 0.0],
        "camera_height": 0.68,
        "camera_x_offset": -0.1,
        "pitch_deg": 0.0,
        "odom_lateral_convention": "left_positive",
        "path_half_width": 0.2,
    }


class GroundTruthOverlayTests(unittest.TestCase):
    def test_slerp_handles_equivalent_opposite_quaternion_signs(self):
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        q1 = -np.array(
            [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
        )
        quaternions = _continuous_quaternions(np.stack([q0, q1]))
        midpoint = _slerp_one(0.5, np.array([0.0, 1.0]), quaternions)
        self.assertAlmostEqual(
            math.degrees(_yaw_from_quaternion(midpoint)), 45.0, places=6
        )

    def test_draws_straight_future_path(self):
        overlay = GroundTruthOverlay(straight_poses(), overlay_config(), 0.0)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertTrue(overlay.draw(image, camera_time=1.0))
        self.assertGreater(int(np.count_nonzero(image)), 0)
        self.assertEqual(overlay.summary()["applied_source_frames"], 1)

    def test_skips_camera_time_outside_pose_coverage(self):
        overlay = GroundTruthOverlay(straight_poses(), overlay_config(), 0.0)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertFalse(overlay.draw(image, camera_time=-1.0))
        self.assertEqual(overlay.summary()["skipped_source_frames"], 1)


if __name__ == "__main__":
    unittest.main()
