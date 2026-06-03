from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "model_processing" / "run_sidewalk_phase1_cot_baseline.py"
_SPEC = importlib.util.spec_from_file_location("run_sidewalk_phase1_cot_baseline", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class SidewalkPhase1CotBaselineTest(unittest.TestCase):
    def test_parse_cot_answer_extracts_finish_trajectory(self) -> None:
        answer = "\n".join(
            [
                "<think>",
                '<Scene>{"summary":"clear sidewalk","visibility":"clear","occlusion_sources":[],"uncertainty":"low"}</Scene>',
                '<FPVScene>{"navigable_support":["sidewalk"],"conditional_support":[],"risk_objects":[],"visibility":"clear","local_risk":"low"}</FPVScene>',
                '<BEVContext>{"scene_summary":"local sidewalk","route_summary":"front route","navigation_state":"following_sidewalk","topology_state":"same_sidewalk_segment","grid_size":[16,16],"ego_cell":[8,12],"target_cell":[8,8],"route_theta_deg":0.0,"route_theta_bin":"[-5,5)","route_direction_bin":"front","intent":"go_straight","target_direction":"front","bev_anchor":"front","topology_evidence":"route continues forward"}</BEVContext>',
                '<CrossView>{"bev_anchor":"front","fpv_anchor":"center corridor","near_reference":"sidewalk edge","blind_area_context":{"left":"visible","right":"visible","front":"support continues"},"fpv_route_region":"center","route_to_fpv_direction":"forward_center","safe_corridor":"center_corridor","alignment":"aligned","confidence":"high"}</CrossView>',
                '<Plan>{"motion_primitive":"go_straight","lateral_preference":"center","avoidance_side":"none","speed_policy":"normal","primary_route":"go forward","safety_policy":"stay centered","contingency":"stop"}</Plan>',
                "</think>",
                '<finish>{"trajectory":[[0.5,0.9],[0.5,0.7],[0.51,0.5]],"intent":"go_straight","confidence_level":"normal","trajectory_type":"straight_center","trajectory_validity":"valid"}</finish>',
            ]
        )
        sections, errors = _MODULE.parse_cot_answer(answer)

        self.assertEqual(errors, [])
        self.assertEqual(sections["finish"]["intent"], "go_straight")
        self.assertEqual(sections["trajectory_norm"], [[0.5, 0.9], [0.5, 0.7], [0.51, 0.5]])

    def test_trajectory_to_rendered_prompt_uses_source_pixels(self) -> None:
        rendered = _MODULE.trajectory_to_rendered_prompt(
            [[0.5, 0.9], [0.25, 0.5]],
            image_width=1920,
            image_height=1080,
        )

        self.assertEqual(rendered["type"], "arrow_path")
        self.assertEqual(rendered["path_uv"], [[960.0, 972.0], [480.0, 540.0]])
        self.assertEqual(rendered["center_uv"], [480.0, 540.0])

    def test_curve_metrics_handle_different_point_counts(self) -> None:
        metrics = _MODULE.compute_trajectory_metrics(
            pred_norm=[[0.5, 0.9], [0.5, 0.7], [0.5, 0.5]],
            gt_norm=[[0.5, 0.9], [0.5, 0.5]],
            curve_samples=16,
        )

        self.assertIsNotNone(metrics)
        self.assertFalse(metrics["point_count_match"])
        self.assertIn("curve_ade_norm", metrics)
        self.assertLess(metrics["curve_ade_norm"], 1e-5)

    def test_sanity_warnings_flag_skyward_local_path(self) -> None:
        warnings = _MODULE.trajectory_sanity_warnings(
            pred_norm=[[0.48, 0.97], [0.48, 0.55], [0.48, 0.08]],
            gt_norm=[[0.5, 0.9], [0.5, 0.86], [0.5, 0.82]],
        )

        self.assertTrue(any(item.startswith("trajectory_extends_above_local_ground_band") for item in warnings))
        self.assertTrue(any(item.startswith("endpoint_much_higher_than_geometry_gt") for item in warnings))

    def test_baseline_system_prompt_forbids_sky_trajectory(self) -> None:
        prompt = _MODULE.build_baseline_system_prompt()

        self.assertIn("The <FPVScene> section is mandatory", prompt)
        self.assertIn("navigable_support, conditional_support, and risk_objects", prompt)
        self.assertIn("Every point must stay on visible walkable ground support", prompt)
        self.assertIn("near-field lower half", prompt)
        self.assertIn("Never extend trajectory points into sky", prompt)

    def test_phase1_source_samples_use_exported_bev_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fpv = root / "images" / "000001.jpg"
            bev = root / "bev_images" / "000001_bev.jpg"
            fpv.parent.mkdir(parents=True)
            bev.parent.mkdir(parents=True)
            fpv.write_bytes(b"fake")
            bev.write_bytes(b"fake")
            row = {
                "id": "clip_000001",
                "frame_idx": 1,
                "frame_name": "000001",
                "fpv_image_path": str(fpv),
                "bev_image_path": str(bev),
                "pixel_trajectory": [[0.5, 0.9], [0.5, 0.5]],
                "image_width": 640,
                "image_height": 360,
                "instruction": "Navigate safely.",
            }
            source = root / "phase1_source_test.jsonl"
            source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

            samples = _MODULE.build_phase1_source_samples(
                source_jsonl=source,
                limit=None,
                frame_start=None,
                frame_end=None,
                frame_step=1,
            )

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].fpv_image_path, fpv)
            self.assertEqual(samples[0].bev_image_path, bev)


if __name__ == "__main__":
    unittest.main()
