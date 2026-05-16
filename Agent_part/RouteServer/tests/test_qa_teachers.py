from __future__ import annotations

import unittest

from mcp_server.qa import teachers


class QATeachersTest(unittest.TestCase):
    def test_bev_route_theta_from_local_segment(self) -> None:
        local_bev = {
            "local_segment_bev": [
                {"x_m": 0.0, "y_m": 0.0},
                {"x_m": 2.0, "y_m": 6.0},
            ],
            "route_polyline_local": [],
            "corridor_sector_deg": {"center_deg": 0.0, "span_deg": 40.0},
        }
        teacher = teachers.teacher_bev_route_theta(local_bev)
        self.assertGreater(teacher["theta"]["theta_deg"], 0.0)
        self.assertEqual(teacher["theta"]["source"], "local_segment_bev")

    def test_connector_component_teacher(self) -> None:
        local_bev = {
            "connector_bev": [
                {"x_m": -1.0, "y_m": 4.0},
                {"x_m": -2.0, "y_m": 8.0},
            ],
            "semantic_bev_layers": {},
        }
        exists = teachers.teacher_bev_component_exists(local_bev, "connector")
        geometry = teachers.teacher_bev_component_theta_range(local_bev, "connector")
        self.assertTrue(exists["exists"])
        self.assertTrue(geometry["exists"])
        self.assertLess(geometry["theta"]["theta_deg"], 0.0)

    def test_point_prompt_teacher(self) -> None:
        rendered_prompt = {
            "type": "path_corridor",
            "path_samples": [
                {"center_uv": [400, 700]},
                {"center_uv": [600, 400]},
            ],
        }
        teacher = teachers.teacher_point_prompt(rendered_prompt, image_width=1000)
        self.assertFalse(teacher["reject_all"])
        self.assertEqual(teacher["target_point_uv"]["u"], 600.0)
        self.assertGreater(teacher["theta"]["theta_deg"], 0.0)

    def test_fpv_scene_stage_uses_compact_segmentation_summary_fields(self) -> None:
        summary = {
            "present_classes": [
                {"class_name": "sidewalk", "pixel_ratio": 0.4},
                {"class_name": "vehicle", "pixel_ratio": 0.05},
            ],
            "traversable_class_names": ["sidewalk", "crosswalk"],
            "forbidden_class_names": ["vehicle", "person"],
        }
        teacher = teachers.teacher_fpv_scene_stage(summary)
        self.assertEqual(teacher["scene_stage"], "mixed_corridor_obstacles")

    def test_crossview_topology_grounding_aligns_bev_and_fpv_evidence(self) -> None:
        rendered_prompt = {
            "type": "path_corridor",
            "path_samples": [
                {"center_uv": [500, 760]},
                {"center_uv": [520, 420]},
            ],
        }
        teacher = teachers.teacher_crossview_topology_grounding(
            route_context={"local_semantics": {"turn_type": "go_straight"}},
            local_bev_context={
                "local_segment_bev": [
                    {"x_m": 0.0, "y_m": 0.0},
                    {"x_m": 0.2, "y_m": 8.0},
                ],
                "route_polyline_local": [],
            },
            rendered_prompt=rendered_prompt,
            fpv_segmentation_summary=None,
            segmentation_validation={"valid": True, "reason": "segmentation_support_ok"},
            som_summary={
                "regions": [
                    {
                        "region_id": "som_1",
                        "label": 1,
                        "region_role": "traversable",
                        "dominant_class_name": "sidewalk",
                        "center_uv": [520, 430],
                        "bbox_uv": [420, 300, 620, 780],
                        "route_intent_score": 0.9,
                    }
                ],
                "conditional_regions": [],
            },
            safe_corridor_prior=None,
            candidate_bundle=None,
            image_width=1000,
            image_height=800,
        )

        self.assertEqual(teacher["alignment_status"], "aligned")
        self.assertLessEqual(teacher["theta_delta_deg"], 25.0)
        self.assertEqual(teacher["anchored_fpv_masks"][0]["region_id"], "som_1")
        self.assertIn("som_summary", teacher["evidence_sources"])

    def test_crossview_topology_grounding_rejects_invalid_segmentation_support(self) -> None:
        teacher = teachers.teacher_crossview_topology_grounding(
            route_context={"local_semantics": {"turn_type": "go_straight"}},
            local_bev_context={
                "local_segment_bev": [
                    {"x_m": 0.0, "y_m": 0.0},
                    {"x_m": 0.0, "y_m": 8.0},
                ],
                "route_polyline_local": [],
            },
            rendered_prompt={
                "type": "path_corridor",
                "path_samples": [{"center_uv": [500, 760]}, {"center_uv": [500, 420]}],
            },
            fpv_segmentation_summary=None,
            segmentation_validation={"valid": False, "reason": "center_path_hits_forbidden_surface"},
            som_summary=None,
            safe_corridor_prior=None,
            candidate_bundle=None,
            image_width=1000,
            image_height=800,
        )

        self.assertEqual(teacher["alignment_status"], "reject")
        self.assertEqual(teacher["failure_mode"], "center_path_hits_forbidden_surface")


if __name__ == "__main__":
    unittest.main()
