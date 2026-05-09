from __future__ import annotations

import unittest

from mcp_server.qa.bundle import build_visual_prompt_qa_bundle
from mcp_server.qa.pov_candidates import build_pov_prompt_candidate_bundle


class VisualPromptQABundleTest(unittest.TestCase):
    def test_bundle_contains_baseline_tasks(self) -> None:
        route_context = {
            "pose_match": {"route_confidence": 0.9},
            "progress": {"distance_to_next_turn_m": 20.0, "next_turn_type": "go_straight"},
            "local_semantics": {"turn_type": "go_straight", "instruction_hint": "continue forward"},
        }
        local_bev = {
            "local_segment_bev": [{"x_m": 0.0, "y_m": 0.0}, {"x_m": 0.0, "y_m": 8.0}],
            "route_polyline_local": [{"x_m": 0.0, "y_m": 0.0}, {"x_m": 0.0, "y_m": 8.0}],
            "connector_bev": [],
            "corridor_sector_deg": {"center_deg": 0.0, "span_deg": 40.0},
            "turn_sector_deg": {"center_deg": 0.0, "span_deg": 35.0},
            "semantic_bev_layers": {},
            "lookahead_m": 20.0,
        }
        rendered_prompt = {
            "type": "path_corridor",
            "path_samples": [
                {"center_uv": [500, 760], "left_edge_uv": [450, 760], "right_edge_uv": [550, 760]},
                {"center_uv": [500, 420], "left_edge_uv": [460, 420], "right_edge_uv": [540, 420]},
            ],
        }
        candidate_bundle = build_pov_prompt_candidate_bundle(
            route_context=route_context,
            local_bev_context=local_bev,
            image_width=1000,
            image_height=800,
            segmentation_context=None,
            rendered_prompt=rendered_prompt,
        )
        bundle = build_visual_prompt_qa_bundle(
            route_context=route_context,
            local_bev_context=local_bev,
            image_width=1000,
            image_height=800,
            fpv_segmentation_summary=None,
            segmentation_validation=None,
            rendered_prompt=rendered_prompt,
            arbiter_decision={"reject_all": False, "reason": "ok"},
            candidate_bundle=candidate_bundle,
        )
        task_types = {sample["task_type"] for sample in bundle["samples"]}
        self.assertIn("bev_route_theta", task_types)
        self.assertIn("point_prompt", task_types)
        self.assertIn("prompt_validity", task_types)
        self.assertEqual(bundle["task_counts"]["bev_component_exists"], 3)
        self.assertEqual(bundle["task_counts"]["bev_component_theta_range"], 2)


if __name__ == "__main__":
    unittest.main()
