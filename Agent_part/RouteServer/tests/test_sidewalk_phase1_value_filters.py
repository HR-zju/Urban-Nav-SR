from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BEHAVIOR = _load_module(
    "sidewalk_phase1_behavior_filter_vlm",
    "tools/model_processing/sidewalk_phase1_behavior_filter_vlm.py",
)
MOTION = _load_module(
    "sidewalk_phase1_motion_saliency_filter",
    "tools/model_processing/sidewalk_phase1_motion_saliency_filter.py",
)


class SidewalkPhase1ValueFiltersTest(unittest.TestCase):
    def test_behavior_parser_accepts_binary_and_words(self) -> None:
        self.assertEqual(BEHAVIOR.parse_behavior_answer("1"), 1)
        self.assertEqual(BEHAVIOR.parse_behavior_answer("2"), 2)
        self.assertEqual(BEHAVIOR.parse_behavior_answer("The segment is interesting."), 1)
        self.assertEqual(BEHAVIOR.parse_behavior_answer("boring"), 2)
        self.assertIsNone(BEHAVIOR.parse_behavior_answer("unclear"))

    def test_behavior_frame_collection_keeps_fpv_temporal_order(self) -> None:
        row = {
            "id": "sample",
            "images": [
                "/data/clip/fpv/000000.jpg",
                "/data/clip/bev/000000_bev.jpg",
                "/data/clip/fpv/000001.jpg",
                "/data/clip/bev/000001_bev.jpg",
                "/data/clip/fpv/000002.jpg",
            ],
        }
        frames = BEHAVIOR.collect_row_fpv_images(row, max_frames=2)
        self.assertEqual(frames, ["/data/clip/fpv/000000.jpg", "/data/clip/fpv/000002.jpg"])

    def test_motion_features_capture_acceleration_and_turning(self) -> None:
        row = {
            "id": "motion",
            "trajectory_ego_t": [
                [0.0, 0.0, 0.0, 0.5],
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 2.0, 2.0, 1.5],
                [2.0, 2.0, 3.0, 0.2],
            ],
        }
        features = MOTION.compute_motion_features(row, default_dt_sec=1.0)
        self.assertTrue(features["valid"])
        self.assertGreater(features["accel_max"], 0.0)
        self.assertGreater(features["turn_max_deg"], 0.0)
        self.assertAlmostEqual(features["path_length"], 1.0 + 2**0.5 + 1.0, places=5)

    def test_motion_selection_uses_behavior_priority_weight(self) -> None:
        rows = [
            {
                "id": "boring_fast",
                "clip_id": "clip",
                "frame_idx": 10,
                "behavior_filter": {"label": 2},
                "trajectory_ego_t": [[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 1.0, 2.0], [0.0, 4.0, 2.0, 0.0]],
            },
            {
                "id": "interesting_turn",
                "clip_id": "clip",
                "frame_idx": 30,
                "behavior_filter": {"label": 1},
                "trajectory_ego_t": [[0.0, 0.0, 0.0, 0.5], [0.0, 1.0, 1.0, 0.5], [1.0, 2.0, 2.0, 0.5]],
            },
        ]
        records = MOTION.score_rows(
            rows,
            behavior_by_id={},
            default_dt_sec=1.0,
            accel_weights=[0.35, 0.45, 0.2],
            turn_weights=[0.35, 0.45, 0.2],
            combine_mode="max",
            scale_percentile=100.0,
            interesting_weight=2.0,
            boring_weight=0.1,
        )
        selected = MOTION.select_records(
            records,
            top_k=1,
            top_fraction=0.0,
            min_score=None,
            min_frame_gap=1,
            keep_all=False,
        )
        self.assertEqual(selected, {"interesting_turn"})


if __name__ == "__main__":
    unittest.main()
