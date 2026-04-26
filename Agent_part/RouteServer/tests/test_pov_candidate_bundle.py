from __future__ import annotations

import unittest

from mcp_server.qa.pov_candidates import build_pov_prompt_candidate_bundle, pov_candidate_bundle_to_dict


class PoVCandidateBundleTest(unittest.TestCase):
    def test_builds_candidate_from_rendered_prompt(self) -> None:
        bundle = build_pov_prompt_candidate_bundle(
            route_context={},
            local_bev_context={
                "local_segment_bev": [
                    {"x_m": 0.0, "y_m": 0.0},
                    {"x_m": 0.0, "y_m": 8.0},
                ],
                "connector_bev": [],
                "corridor_sector_deg": {"center_deg": 0.0, "span_deg": 40.0},
                "turn_sector_deg": {"center_deg": 0.0, "span_deg": 35.0},
            },
            image_width=1000,
            image_height=800,
            segmentation_context=None,
            rendered_prompt={
                "type": "path_corridor",
                "path_samples": [
                    {"center_uv": [500, 760], "left_edge_uv": [450, 760], "right_edge_uv": [550, 760]},
                    {"center_uv": [510, 420], "left_edge_uv": [470, 420], "right_edge_uv": [550, 420]},
                ],
                "span_px": 80,
            },
        )
        payload = pov_candidate_bundle_to_dict(bundle)
        self.assertEqual(payload["selected_prompt_id"], "final_rendered_prompt")
        self.assertEqual(len(payload["fpv_candidates"]), 1)
        self.assertGreaterEqual(payload["fpv_candidates"][0]["route_alignment_score"], 0.8)


if __name__ == "__main__":
    unittest.main()
