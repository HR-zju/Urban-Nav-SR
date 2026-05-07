from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "mcp_server" / "brain" / "fpv_segmentation.py"
_SPEC = importlib.util.spec_from_file_location("fpv_segmentation_validation", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

validate_rendered_prompt_against_segmentation = _MODULE.validate_rendered_prompt_against_segmentation
build_safe_corridor_prior = _MODULE.build_safe_corridor_prior
corridor_width_ratio_limits = _MODULE.corridor_width_ratio_limits
repair_rendered_prompt_with_safe_corridor = _MODULE.repair_rendered_prompt_with_safe_corridor


def _segmentation_context(mask: np.ndarray) -> dict:
    return {
        "mask": mask,
        "id_to_name": {1: "road", 2: "sidewalk", 36: "curb", 38: "occlusion"},
        "traversable_ids": {2},
        "conditional_traversable_ids": {1, 36},
        "allowed_traversable_ids": {1, 2, 36},
        "boundary_ids": {1, 2, 36},
        "forbidden_ids": {38},
        "disallowed_surface_ids": set(),
        "curb_ids": {36},
        "navigation_mode": "foot",
    }


class FpvSegmentationValidationTest(unittest.TestCase):
    def test_accepts_road_and_curb_as_conditional_connector_for_foot(self) -> None:
        mask = np.full((80, 100), 2, dtype=np.int32)
        mask[:, :20] = 1
        mask[:, 20:28] = 36
        prompt = {
            "type": "path_corridor",
            "path_samples": [
                {"center_uv": [50, 72], "left_edge_uv": [12, 72], "right_edge_uv": [52, 72]},
                {"center_uv": [50, 30], "left_edge_uv": [14, 30], "right_edge_uv": [52, 30]},
            ],
            "coverage_polygon_uv": [[12, 72], [14, 30], [52, 30], [52, 72], [12, 72]],
            "termination_semantics": "continue_visible_corridor",
        }

        validation = validate_rendered_prompt_against_segmentation(prompt, _segmentation_context(mask))

        self.assertIsInstance(validation, dict)
        self.assertTrue(validation["valid"])
        self.assertGreater(validation["polygon_conditional_ratio"], 0.0)
        self.assertEqual(validation["polygon_forbidden_ratio"], 0.0)

    def test_accepts_conditional_connector_prefix_when_it_joins_preferred_sidewalk(self) -> None:
        mask = np.full((100, 120), 38, dtype=np.int32)
        mask[55:100, 20:100] = 1
        mask[30:55, 20:100] = 2
        prompt = {
            "type": "path_corridor",
            "path_samples": [
                {"center_uv": [60, 92], "left_edge_uv": [40, 92], "right_edge_uv": [80, 92], "support_label": "conditional_connector"},
                {"center_uv": [60, 76], "left_edge_uv": [40, 76], "right_edge_uv": [80, 76], "support_label": "conditional_connector"},
                {"center_uv": [60, 62], "left_edge_uv": [40, 62], "right_edge_uv": [80, 62], "support_label": "sidewalk"},
                {"center_uv": [60, 45], "left_edge_uv": [40, 45], "right_edge_uv": [80, 45], "support_label": "sidewalk"},
            ],
            "coverage_polygon_uv": [[40, 92], [40, 76], [40, 62], [40, 45], [80, 45], [80, 62], [80, 76], [80, 92], [40, 92]],
            "termination_semantics": "continue_visible_corridor",
        }

        validation = validate_rendered_prompt_against_segmentation(prompt, _segmentation_context(mask))

        self.assertIsInstance(validation, dict)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["first_preferred_sample_index"], 3)
        self.assertEqual(len(validation["conditional_connector_samples"]), 3)
        self.assertEqual(validation["preferred_run_missing_samples"], [])

    def test_rejects_conditional_connector_that_never_joins_preferred_sidewalk(self) -> None:
        mask = np.full((100, 120), 38, dtype=np.int32)
        mask[30:100, 20:100] = 1
        prompt = {
            "type": "path_corridor",
            "path_samples": [
                {"center_uv": [60, 92], "left_edge_uv": [40, 92], "right_edge_uv": [80, 92], "support_label": "conditional_connector"},
                {"center_uv": [60, 76], "left_edge_uv": [40, 76], "right_edge_uv": [80, 76], "support_label": "conditional_connector"},
                {"center_uv": [60, 62], "left_edge_uv": [40, 62], "right_edge_uv": [80, 62], "support_label": "conditional_connector"},
            ],
            "coverage_polygon_uv": [[40, 92], [40, 76], [40, 62], [80, 62], [80, 76], [80, 92], [40, 92]],
            "termination_semantics": "continue_visible_corridor",
        }

        validation = validate_rendered_prompt_against_segmentation(prompt, _segmentation_context(mask))

        self.assertIsInstance(validation, dict)
        self.assertFalse(validation["valid"])
        self.assertIn("centerline_preferred_run_missing", validation["reason"])

    def test_rejects_corridor_that_hits_forbidden_obstacle(self) -> None:
        mask = np.full((80, 100), 2, dtype=np.int32)
        mask[25:75, 45:65] = 38
        prompt = {
            "type": "path_corridor",
            "path_samples": [
                {"center_uv": [50, 72], "left_edge_uv": [35, 72], "right_edge_uv": [65, 72]},
                {"center_uv": [50, 30], "left_edge_uv": [35, 30], "right_edge_uv": [65, 30]},
            ],
            "coverage_polygon_uv": [[35, 72], [35, 30], [65, 30], [65, 72], [35, 72]],
            "termination_semantics": "continue_visible_corridor",
        }

        validation = validate_rendered_prompt_against_segmentation(prompt, _segmentation_context(mask))

        self.assertIsInstance(validation, dict)
        self.assertFalse(validation["valid"])
        self.assertGreater(validation["polygon_forbidden_ratio"], 0.0)
        self.assertIn("polygon_hits_forbidden_ratio", validation["reason"])

    def test_accepts_corridor_fully_inside_sidewalk(self) -> None:
        mask = np.full((80, 100), 2, dtype=np.int32)
        prompt = {
            "type": "path_corridor",
            "path_samples": [
                {"center_uv": [50, 72], "left_edge_uv": [35, 72], "right_edge_uv": [65, 72]},
                {"center_uv": [50, 30], "left_edge_uv": [35, 30], "right_edge_uv": [65, 30]},
            ],
            "coverage_polygon_uv": [[35, 72], [35, 30], [65, 30], [65, 72], [35, 72]],
            "termination_semantics": "continue_visible_corridor",
        }

        validation = validate_rendered_prompt_against_segmentation(prompt, _segmentation_context(mask))

        self.assertIsInstance(validation, dict)
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["polygon_traversable_ratio"], 1.0)

    def test_rejects_prompt_that_does_not_start_from_lower_middle_band(self) -> None:
        mask = np.full((80, 100), 2, dtype=np.int32)
        prompt = {
            "type": "path_corridor",
            "path_samples": [
                {"center_uv": [50, 40], "left_edge_uv": [35, 40], "right_edge_uv": [65, 40]},
                {"center_uv": [50, 20], "left_edge_uv": [35, 20], "right_edge_uv": [65, 20]},
            ],
            "coverage_polygon_uv": [[35, 40], [35, 20], [65, 20], [65, 40], [35, 40]],
            "termination_semantics": "continue_visible_corridor",
        }

        validation = validate_rendered_prompt_against_segmentation(prompt, _segmentation_context(mask))

        self.assertIsInstance(validation, dict)
        self.assertFalse(validation["valid"])
        self.assertFalse(validation["start_valid"])
        self.assertIn("start_point_not_in_lower_middle_band", validation["reason"])

    def test_rejects_overly_wide_corridor_even_on_sidewalk(self) -> None:
        mask = np.full((80, 100), 2, dtype=np.int32)
        prompt = {
            "type": "path_corridor",
            "path_samples": [
                {"center_uv": [50, 72], "left_edge_uv": [3, 72], "right_edge_uv": [97, 72]},
                {"center_uv": [50, 30], "left_edge_uv": [4, 30], "right_edge_uv": [96, 30]},
            ],
            "coverage_polygon_uv": [[3, 72], [4, 30], [96, 30], [97, 72], [3, 72]],
            "termination_semantics": "continue_visible_corridor",
        }

        validation = validate_rendered_prompt_against_segmentation(prompt, _segmentation_context(mask))

        self.assertIsInstance(validation, dict)
        self.assertFalse(validation["valid"])
        self.assertFalse(validation["corridor_width_valid"])
        self.assertGreater(validation["max_corridor_width_ratio_observed"], 0.40)
        self.assertIn("corridor_width_too_wide", validation["reason"])

    def test_corridor_width_limit_uses_single_env_value(self) -> None:
        with patch.dict("os.environ", {"VISUAL_PROMPT_CORRIDOR_WIDTH_RATIO": "0.50"}):
            max_ratio, preferred_ratio = corridor_width_ratio_limits("foot")

        self.assertEqual(max_ratio, 0.50)
        self.assertEqual(preferred_ratio, 0.375)

    def test_builds_safe_corridor_prior_centered_and_inset(self) -> None:
        mask = np.full((100, 120), 38, dtype=np.int32)
        mask[:, 30:90] = 2

        prior = build_safe_corridor_prior(_segmentation_context(mask))

        self.assertIsInstance(prior, dict)
        samples = prior["path_samples"]
        self.assertGreaterEqual(len(samples), 2)
        for sample in samples:
            self.assertAlmostEqual(sample["center_uv"][0], 59.5, delta=1.0)
            self.assertGreater(sample["left_edge_uv"][0], 30.0)
            self.assertLess(sample["right_edge_uv"][0], 90.0)

    def test_safe_corridor_prior_shifts_to_clear_lane_around_obstacle(self) -> None:
        mask = np.full((100, 120), 38, dtype=np.int32)
        mask[:, 20:110] = 2
        mask[45:75, 50:72] = 38

        prior = build_safe_corridor_prior(_segmentation_context(mask))

        self.assertIsInstance(prior, dict)
        obstacle_rows = [
            sample
            for sample in prior["path_samples"]
            if 45 <= float(sample["center_uv"][1]) <= 80
        ]
        self.assertTrue(obstacle_rows)
        self.assertTrue(any(float(sample["center_uv"][0]) < 50.0 or float(sample["center_uv"][0]) > 72.0 for sample in obstacle_rows))
        self.assertEqual(prior["debug"]["clearance_source"], "distance_transform_edt")

    def test_repairs_off_center_and_edge_hugging_corridor(self) -> None:
        mask = np.full((100, 120), 38, dtype=np.int32)
        mask[:, 30:90] = 2
        prompt = {
            "type": "path_corridor",
            "path_samples": [
                {"center_uv": [35, 92], "left_edge_uv": [30, 92], "right_edge_uv": [89, 92]},
                {"center_uv": [36, 70], "left_edge_uv": [30, 70], "right_edge_uv": [89, 70]},
                {"center_uv": [37, 55], "left_edge_uv": [30, 55], "right_edge_uv": [89, 55]},
            ],
            "coverage_polygon_uv": [[30, 92], [30, 70], [30, 55], [89, 55], [89, 70], [89, 92], [30, 92]],
            "termination_semantics": "continue_visible_corridor",
        }
        before = validate_rendered_prompt_against_segmentation(prompt, _segmentation_context(mask))

        repaired, repair_debug = repair_rendered_prompt_with_safe_corridor(prompt, _segmentation_context(mask))
        after = validate_rendered_prompt_against_segmentation(repaired, _segmentation_context(mask))

        self.assertFalse(before["valid"])
        self.assertIn("edge_margin_too_low", before["reason"])
        self.assertTrue(repair_debug["applied"])
        self.assertTrue(after["valid"])
        self.assertGreater(after["min_edge_margin_ratio_observed"], 0.0)

    def test_repair_replans_full_path_when_prompt_hits_obstacle(self) -> None:
        mask = np.full((100, 120), 38, dtype=np.int32)
        mask[:, 20:110] = 2
        mask[45:75, 50:72] = 38
        prompt = {
            "type": "path_corridor",
            "path_samples": [
                {"center_uv": [60, 92], "left_edge_uv": [45, 92], "right_edge_uv": [75, 92]},
                {"center_uv": [60, 78], "left_edge_uv": [45, 78], "right_edge_uv": [75, 78]},
                {"center_uv": [60, 62], "left_edge_uv": [45, 62], "right_edge_uv": [75, 62]},
            ],
            "coverage_polygon_uv": [[45, 92], [45, 78], [45, 62], [75, 62], [75, 78], [75, 92], [45, 92]],
            "termination_semantics": "continue_visible_corridor",
        }
        before = validate_rendered_prompt_against_segmentation(prompt, _segmentation_context(mask))

        repaired, repair_debug = repair_rendered_prompt_with_safe_corridor(
            prompt,
            _segmentation_context(mask),
            validation=before,
        )
        after = validate_rendered_prompt_against_segmentation(repaired, _segmentation_context(mask))

        self.assertFalse(before["valid"])
        self.assertIn("forbidden", before["reason"])
        self.assertTrue(repair_debug["applied"])
        self.assertEqual(repair_debug["source"], "segmentation_obstacle_aware_replan_v1")
        self.assertTrue(after["valid"])


if __name__ == "__main__":
    unittest.main()
