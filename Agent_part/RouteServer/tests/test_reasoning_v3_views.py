from __future__ import annotations

from mcp_server.agent.reasoning_v3_prompt_policy import STUDENT_FULL_COT_PROFILE_V3
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_source import build_reasoning_source_v3
from mcp_server.data.reasoning_v3_views import build_reasoning_training_views_v3
from reasoning_v3_test_utils import accepted_annotation


def test_v3_export_emits_profile_bound_full_cot_without_qa() -> None:
    source = _source()
    annotation = accepted_annotation(source, corridor="corridor.right", direction="right")

    views = build_reasoning_training_views_v3(source, annotation)

    assert len(views) == 1
    full = views[0]
    assert full.view_type == "full_cot"
    assert full.task_type == "reasoning_v3_3_full_cot"
    assert full.trajectory_ego_t is not None
    assert full.loss_mask["system1_action"] is True
    assert full.prompt_profile_id == STUDENT_FULL_COT_PROFILE_V3.profile_id
    assert full.prompt_policy_digest == STUDENT_FULL_COT_PROFILE_V3.policy_digest
    assert full.prompt_policy_digest == "d169323133591122e0282ce0fe4f8bca7c0bab97d41c28ab109e7a46e29f990a"
    assert full.section_loss_mask.decision is True
    assert full.section_loss_mask.navigation_context_grounding is True


def _source():
    return build_reasoning_source_v3(
        {
            "id": "views-1",
            "dataset_source": "sidewalk",
            "images": ["current.jpg", "bev.jpg"],
            "image_roles": ["fpv.current", "bev.current"],
            "instruction": "Turn right toward the entrance.",
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [2.0, -1.0, -0.2]],
            "trajectory_fpv_t": [[0.5, 0.9], [0.8, 0.7]],
            "trajectory_fpv_coord_frame": "normalized_0_1",
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )
