"""Focused tests for section-specific Reasoning V4 image routing."""

from __future__ import annotations

from mcp_server.contracts.direction_profile import (
    FPV_DEGREE10_AFFORDANCE7_V1,
    FPV_DEGREE10_TERMINAL_CENTROID_V2,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionContextV4,
    GroundingOutputsV4,
    InputModalitiesV4,
    NavObservationV4,
)
from mcp_server.contracts.reasoning_v4 import (
    DirectionAffordanceV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    SceneGroundingV4,
    TaskDirectionV4,
    TaskGroundingV4,
)
from mcp_server.teacher.reasoning_v4_image_policy import (
    REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID,
    REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID,
    get_reasoning_v4_section_image_policy,
)
from mcp_server.teacher.reasoning_v4_profiles import (
    get_reasoning_v4_teacher_profile,
)
from mcp_server.teacher.reasoning_v4_teacher import (
    build_reasoning_v4_section_question,
)


_FPV_ROLES = (
    "fpv.history_4",
    "fpv.history_3",
    "fpv.history_2",
    "fpv.history_1",
    "fpv.current",
)


def _observation(*, with_bev: bool = True) -> NavObservationV4:
    """Build one validated five-frame observation with optional current BEV."""

    image_roles = list(_FPV_ROLES)
    if with_bev:
        image_roles.append("bev.current")
    evidence_catalog = [
        {"ref": "instruction.active", "modality": "instruction"},
        *[
            {"ref": role, "modality": "fpv"}
            for role in _FPV_ROLES
        ],
    ]
    if with_bev:
        evidence_catalog.append(
            {
                "ref": "bev.current",
                "modality": "bev",
                "bev_route_evidence": {
                    "frame": "north_up",
                    "ego_forward": "up",
                    "route_overlay_visible": True,
                    "route_direction_authority": "visual_conditional",
                    "local_safety_authority": False,
                },
            }
        )
    return NavObservationV4(
        source_group_id="foxglove:image-policy",
        dataset_source="foxglove",
        source_record_id="image-policy",
        split_group_id="clip-image-policy",
        images=[
            f"/tmp/{role.replace('.', '_')}.jpg"
            for role in image_roles
        ],
        image_roles=image_roles,
        instruction="Follow the route.",
        evidence_catalog=evidence_catalog,
        input_modalities=InputModalitiesV4(
            history_fpv=True,
            instruction=True,
            bev=with_bev,
        ),
        direction_context=DirectionContextV4(
            action_profile_id=FPV_DEGREE10_TERMINAL_CENTROID_V2,
            action_profile_version="2.0",
            affordance_probe_profile_id=FPV_DEGREE10_AFFORDANCE7_V1,
            affordance_probe_profile_version="1.0",
        ),
        temporal_evidence_eligible=True,
    )


def test_routed_policy_matches_section_visual_responsibilities() -> None:
    """Each section receives only the image roles required by its contract."""

    policy = get_reasoning_v4_section_image_policy(
        REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID
    )
    observation = _observation()

    assert policy.select(
        observation,
        "SceneGrounding",
    ).image_roles == _FPV_ROLES
    assert policy.select(
        observation,
        "InteractionConstraintGrounding",
    ).image_roles == _FPV_ROLES
    assert policy.select(
        observation,
        "LocalTraversability",
    ).image_roles == ("fpv.current",)
    assert policy.select(
        observation,
        "NavigationContextGrounding",
    ).image_roles == ("fpv.current", "bev.current")
    assert policy.select(
        observation,
        "Decision",
    ).image_roles == ("fpv.current", "bev.current")


def test_single_frame_projection_closes_temporal_and_bev_flags() -> None:
    """Omitted image roles cannot remain advertised through modality flags."""

    policy = get_reasoning_v4_section_image_policy(
        REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID
    )
    projected = policy.project_observation(
        _observation(),
        "LocalTraversability",
    )

    assert projected.image_roles == ["fpv.current"]
    assert projected.input_modalities.history_fpv is False
    assert projected.input_modalities.bev is False
    assert projected.temporal_evidence_eligible is False
    assert {str(entry.ref) for entry in projected.evidence_catalog} == {
        "instruction.active",
        "fpv.current",
    }


def test_optional_bev_falls_back_to_current_fpv() -> None:
    """Navigation and Decision remain valid when the source has no BEV."""

    policy = get_reasoning_v4_section_image_policy(
        REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID
    )
    observation = _observation(with_bev=False)

    for section_name in (
        "NavigationContextGrounding",
        "Decision",
    ):
        selection = policy.select(observation, section_name)
        assert selection.image_roles == ("fpv.current",)


def test_all_visible_and_routed_profiles_have_distinct_identity() -> None:
    """Cache identity changes whenever image routing behavior changes."""

    baseline = get_reasoning_v4_teacher_profile(
        "Decision",
        section_image_policy_id=(
            REASONING_V4_ALL_VISIBLE_IMAGES_POLICY_ID
        ),
    )
    routed = get_reasoning_v4_teacher_profile(
        "Decision",
        section_image_policy_id=(
            REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID
        ),
    )

    assert baseline.profile_id != routed.profile_id
    assert baseline.policy_digest != routed.policy_digest
    assert (
        baseline.section_image_policy_digest
        != routed.section_image_policy_digest
    )


def test_decision_question_serializes_only_routed_current_visuals() -> None:
    """Provider prompt image_order mirrors the projected Decision payload."""

    policy = get_reasoning_v4_section_image_policy(
        REASONING_V4_SECTION_ROUTED_VISUAL_POLICY_ID
    )
    profile = get_reasoning_v4_teacher_profile(
        "Decision",
        section_image_policy_id=policy.policy_id,
    )
    observation = policy.project_observation(
        _observation(),
        "Decision",
    )
    groundings = GroundingOutputsV4(
        task_grounding=TaskGroundingV4(
            objective="follow_route",
            requirements=["route_adherence"],
        ),
        scene_grounding=SceneGroundingV4(
            scene_type="urban_sidewalk",
            walkable_support="paved_sidewalk",
            traffic_context="no_active_control_visible",
        ),
        local_traversability=LocalTraversabilityV4(
            direction_affordances=[
                DirectionAffordanceV4(
                    direction_label="straight",
                    affordance="direct",
                    clearance="sufficient",
                )
            ]
        ),
        navigation_context_grounding=NavigationContextGroundingV4(
            task_direction=TaskDirectionV4(
                status="aligned",
                source="bev_route",
                direction="straight",
            )
        ),
        interaction_constraint_grounding=(
            InteractionConstraintGroundingV4()
        ),
    )
    question = build_reasoning_v4_section_question(
        observation,
        profile,
        groundings=groundings,
    )

    assert profile.section_image_policy_id == policy.policy_id
    assert '"requested_section":"Decision"' in question
    assert '"validated_grounding_sections"' in question
    assert '"role":"fpv.current"' in question
    assert '"role":"bev.current"' in question
    assert '"role":"fpv.history_1"' not in question
