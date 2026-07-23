from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from PIL import Image

from mcp_server.route_context.schema import VisualPromptDecision
from mcp_server.services import visual_prompt_service


class VisualPromptCoreContractTest(unittest.IsolatedAsyncioTestCase):
    """Protect the non-QA output contract of the visual-prompt pipeline."""

    async def test_pipeline_returns_core_artifacts_without_legacy_qa_fields(self) -> None:
        """Run the local pipeline shell while replacing route and LVLM dependencies.

        Returns:
            None.
        """
        route_context = {
            "pose_match": {"off_route": False},
            "progress": {"remaining_distance_m": 40.0},
            "local_semantics": {"short_action_summary": "Continue straight."},
        }
        local_bev_context = {
            "bev_rotation_deg": 0.0,
            "bev_bbox": {
                "x_min_m": -15.0,
                "x_max_m": 15.0,
                "y_min_m": -10.0,
                "y_max_m": 30.0,
            },
        }
        route_state = SimpleNamespace(route_context=object(), local_bev_context=object())
        generated_decision = VisualPromptDecision(
            selected_candidate_id="direct",
            selected_prompt_type="path_corridor",
            instruction="Follow the marked path.",
            confidence=0.9,
            reason="The route continues straight ahead.",
            reject_all=False,
            fallback_used=False,
            arbiter_source="test_generator",
            rendered_prompt={
                "type": "path_corridor",
                "center_uv": [84.0, 60.0],
                "path_uv": [[80.0, 110.0], [82.0, 85.0], [84.0, 60.0]],
                "span_px": 24.0,
                "termination_semantics": "continue",
                "support_confidence": 0.9,
            },
            debug={},
        )

        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "frame.jpg"
            Image.new("RGB", (160, 120), (32, 64, 96)).save(image_path)

            with (
                patch.object(visual_prompt_service, "build_route_state", return_value=route_state),
                patch.object(
                    visual_prompt_service,
                    "route_context_to_dict",
                    return_value=route_context,
                ),
                patch.object(
                    visual_prompt_service,
                    "local_bev_context_to_dict",
                    return_value=local_bev_context,
                ),
                patch.object(
                    visual_prompt_service,
                    "_build_inputs_for_A",
                    return_value={
                        "question": "{}",
                        "system_prompt": "test system prompt",
                        "alignment_report": None,
                        "overlay_legend": None,
                        "summary_source": "test",
                    },
                ),
                patch.object(
                    visual_prompt_service,
                    "_run_A_generator",
                    new_callable=AsyncMock,
                ) as generator,
            ):
                generator.return_value = ({"provider": "test"}, generated_decision)
                result = await visual_prompt_service.run_zero_shot_visual_prompt_pipeline(
                    gps={"lat": 30.0, "lng": 120.0},
                    route_polyline=[
                        {"latitude": 30.0, "longitude": 120.0},
                        {"latitude": 30.001, "longitude": 120.0},
                    ],
                    coarse_waypoints=[],
                    image_path=str(image_path),
                    request_id="core-contract-test",
                )

        self.assertEqual(result["request_id"], "core-contract-test")
        self.assertIs(result["route_context"], route_context)
        self.assertEqual(result["local_bev_context"]["_origin_gps"], {"lat": 30.0, "lng": 120.0})
        self.assertFalse(result["arbiter_decision"]["reject_all"])
        self.assertEqual(result["arbiter_decision"]["arbiter_source"], "test_generator")
        self.assertEqual(
            result["arbiter_decision"]["rendered_prompt"]["path_uv"],
            [[80.0, 110.0], [82.0, 85.0], [84.0, 60.0]],
        )
        self.assertTrue(result["overlay_svg"].startswith("<svg"))
        self.assertTrue(result["overlay_svg_data_url"].startswith("data:image/svg+xml"))
        self.assertEqual(result["som_candidate_hints"], [])
        self.assertIsNone(result["fpv_segmentation_summary"])
        self.assertIsNone(result["som_summary"])
        self.assertIn("telemetry", result)
        self.assertNotIn("qa_bundle", result)
        self.assertNotIn("pov_prompt_candidate_bundle", result)
        generator.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
