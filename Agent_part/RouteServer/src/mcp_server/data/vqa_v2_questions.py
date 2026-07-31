"""Deterministic answer-independent question construction for VQA V2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from mcp_server.contracts.direction_profile import direction_label_names
from mcp_server.contracts.vqa_dataset_v2 import (
    VQACapabilityConfigV2,
    VQASourceV2,
)
from mcp_server.contracts.vqa_v2 import (
    VQAQuestionV2,
    get_vqa_task_spec_v2,
)
from mcp_server.data.vqa_v2_eligibility import (
    VQAEligibilityResultV2,
    vqa_task_eligibility_v2,
)


_QUESTION_TEXT = {
    "a1_goal_understanding": (
        "What explicit navigation goal is specified?"
    ),
    "a2_explicit_requirement": (
        "Which explicit navigation requirements are stated?"
    ),
    "a3_task_ambiguity": (
        "Is the prompt-visible navigation task materially ambiguous?"
    ),
    "b1_direction_affordance": (
        "What is the local navigation affordance along the conditioned "
        "direction?"
    ),
    "b2_direction_clearance": (
        "What coarse visible clearance is available along the "
        "conditioned direction?"
    ),
    "b3_direction_surface_effect": (
        "How does the visible support along the conditioned direction "
        "affect navigation?"
    ),
    "b4_conditional_affordance": (
        "Can the robot use the conditioned direction immediately, and "
        "what must it do first?"
    ),
    "b5_direction_comparison": (
        "Which supplied candidate direction is more suitable now?"
    ),
    "b6_unavailable_directions": (
        "Which supplied candidate directions should not be used now?"
    ),
    "c1_goal_to_direction": (
        "Which local direction best approaches the specified visible "
        "goal?"
    ),
    "c2_pixel_goal_to_direction": (
        "Which direction-profile label corresponds to the supplied "
        "pixel goal?"
    ),
    "c2_waypoint_to_direction": (
        "Which direction-profile label corresponds to the supplied "
        "waypoint?"
    ),
    "c3_bev_route_alignment": (
        "Which direction-profile label represents the active validated "
        "BEV route continuation?"
    ),
    "c4_bev_object_anchoring": (
        "Which visible FPV entity matches the supplied validated BEV "
        "element, and in which direction does it lie?"
    ),
    "c5_outside_view_direction": (
        "What validated outside-view information affects a future, not "
        "current, direction?"
    ),
    "c6_bev_fpv_executability": (
        "Is the validated active BEV direction currently executable in "
        "FPV?"
    ),
    "d1_agent_relevance": (
        "Does a visible agent affect movement along the conditioned "
        "direction?"
    ),
    "d2_static_interaction_relation": (
        "What static visible relation affects the conditioned direction?"
    ),
    "d3_temporal_interaction_relation": (
        "What temporally supported interaction affects the conditioned "
        "direction?"
    ),
    "d4_applicable_constraint": (
        "Which navigation constraint applies to the conditioned "
        "direction?"
    ),
    "d5_appropriate_response": (
        "What response is appropriate before taking the conditioned "
        "direction?"
    ),
    "d6_traffic_control_applicability": (
        "Does a visible traffic control govern the conditioned "
        "direction?"
    ),
    "e0_demonstrated_direction_selection": (
        "Which short-horizon direction should the robot take in this "
        "training example?"
    ),
    "e1_candidate_trajectory_preference": (
        "Which prompt-visible candidate trajectory is preferable?"
    ),
    "e2_candidate_direction_classification": (
        "Which direction-profile label best describes the specified "
        "prompt-visible candidate?"
    ),
    "e3_trajectory_compatibility": (
        "Is the specified prompt-visible candidate compatible with the "
        "task and visible scene?"
    ),
    "e4_alternative_direction": (
        "Which supplied direction is the best local alternative?"
    ),
}


@dataclass(frozen=True)
class VQAQuestionBuildResultV2:
    """Return built questions and explicit unavailable tasks.

    Args:
        questions: Eligible answer-independent questions.
        unavailable: Per-task eligibility diagnostics.
    """

    questions: list[VQAQuestionV2]
    unavailable: dict[str, VQAEligibilityResultV2]


def build_vqa_questions_v2(
    source: VQASourceV2,
    capabilities: VQACapabilityConfigV2,
) -> VQAQuestionBuildResultV2:
    """Build all currently export-enabled questions for one source.

    Args:
        source: Frozen VQA source.
        capabilities: Versioned task activation policy.

    Returns:
        Questions plus explicit skip reasons.
    """

    questions: list[VQAQuestionV2] = []
    unavailable: dict[str, VQAEligibilityResultV2] = {}
    for capability in capabilities.tasks:
        task_id = capability.task_id
        eligibility = vqa_task_eligibility_v2(
            source,
            task_id,
            capabilities,
        )
        if not eligibility.eligible:
            unavailable[task_id] = eligibility
            continue
        questions.append(_question_for_task(source, task_id))
    return VQAQuestionBuildResultV2(
        questions=questions,
        unavailable=unavailable,
    )


def _question_for_task(
    source: VQASourceV2,
    task_id: str,
) -> VQAQuestionV2:
    """Build one task-specific question from eligible source context."""

    direction = source.direction_supervision
    profile_id = None
    conditioned = None
    candidates: list[str] = []
    candidate_set_id = None
    visible_candidates: list[dict[str, str]] = []
    target_candidate_id = None
    if task_id in {
        "b1_direction_affordance",
        "b2_direction_clearance",
        "b3_direction_surface_effect",
        "b4_conditional_affordance",
        "d1_agent_relevance",
        "d2_static_interaction_relation",
        "d3_temporal_interaction_relation",
        "d4_applicable_constraint",
        "d5_appropriate_response",
        "d6_traffic_control_applicability",
    }:
        profile_id = direction.profile_id
        conditioned = str(direction.label_name)
    elif task_id in {
        "b5_direction_comparison",
        "b6_unavailable_directions",
        "e4_alternative_direction",
    }:
        profile_id = direction.profile_id
        candidates = _local_candidate_labels(
            source,
            count=2 if task_id != "b6_unavailable_directions" else 3,
            exclude_selected=task_id == "e4_alternative_direction",
        )
    elif task_id in {
        "c1_goal_to_direction",
        "e0_demonstrated_direction_selection",
    }:
        profile_id = direction.profile_id
        candidates = list(direction_label_names(str(profile_id)))

    interface = _interface_for_task(source, task_id)
    if interface is not None:
        candidate_set = getattr(interface, "candidate_set", None)
        if candidate_set is not None:
            candidate_set_id = candidate_set.candidate_set_id
            visible_candidates = [
                {
                    "candidate_id": item.candidate_id,
                    "prompt_visible_representation_ref": (
                        item.prompt_visible_representation_ref
                    ),
                }
                for item in candidate_set.candidates
            ]
            target_candidate_id = getattr(
                interface,
                "target_candidate_id",
                None,
            )
        if profile_id is None and task_id in {
            "c2_pixel_goal_to_direction",
            "c2_waypoint_to_direction",
            "c3_bev_route_alignment",
            "c4_bev_object_anchoring",
            "c5_outside_view_direction",
            "c6_bev_fpv_executability",
            "e2_candidate_direction_classification",
        }:
            interface_direction = (
                getattr(interface, "direction", None)
                or getattr(interface, "bev_direction", None)
                or getattr(interface, "current_direction", None)
            )
            if (
                interface_direction is None
                and candidate_set is not None
                and target_candidate_id is not None
            ):
                interface_direction = next(
                    (
                        item.direction
                        for item in candidate_set.candidates
                        if item.candidate_id == target_candidate_id
                    ),
                    None,
                )
            if interface_direction is not None:
                profile_id = interface_direction.profile_id
                candidates = list(
                    direction_label_names(profile_id)
                )

    evidence_refs = minimal_vqa_evidence_refs_v2(source, task_id)
    target_refs = [
        ref
        for ref in evidence_refs
        if ref.startswith(("goal.", "waypoint.", "bev."))
    ]
    if task_id == "c4_bev_object_anchoring" and interface is not None:
        target_refs = [
            interface.bev_ref,
            interface.fpv_entity_ref,
        ]
    return VQAQuestionV2(
        task_id=task_id,
        template_id=f"{task_id}_question_v2_minimal_evidence",
        question_text=_QUESTION_TEXT[task_id],
        direction_profile_id=profile_id,
        conditioned_direction_label=conditioned,
        candidate_direction_labels=candidates,
        target_refs=target_refs,
        evidence_refs=evidence_refs,
        candidate_set_id=candidate_set_id,
        visible_candidates=visible_candidates,
        target_candidate_id=target_candidate_id,
    )


def minimal_vqa_evidence_refs_v2(
    source: VQASourceV2,
    task_id: str,
) -> list[str]:
    """Return the smallest prompt-visible evidence set for one VQA task.

    Required modalities come from the frozen task registry.  Task-language
    questions additionally require instruction evidence, visible-goal
    grounding requires explicit goal/task evidence, and external-GT tasks
    inherit their interface's exact evidence declaration.
    """

    spec = get_vqa_task_spec_v2(task_id)
    catalog = source.observation.evidence_catalog
    refs: list[str] = []

    def add_matching(predicate) -> None:
        for entry in catalog:
            ref = str(entry.ref)
            if predicate(entry, ref) and ref not in refs:
                refs.append(ref)

    for modality in spec.required_modalities:
        if modality == "fpv":
            add_matching(lambda _entry, ref: ref == "fpv.current")
        elif modality == "history_fpv":
            add_matching(
                lambda _entry, ref: ref.startswith("fpv.history_")
            )
        else:
            add_matching(
                lambda entry, _ref, expected=modality: (
                    str(entry.modality) == expected
                )
            )

    if task_id.startswith("a"):
        add_matching(
            lambda entry, ref: (
                str(entry.modality) == "instruction"
                or ref.startswith("instruction.")
            )
        )
    if task_id == "c1_goal_to_direction":
        add_matching(
            lambda entry, ref: (
                str(entry.modality) in {
                    "goal",
                    "goal_image",
                    "pixel_goal",
                    "instruction",
                }
                or ref.startswith(("goal.", "instruction."))
            )
        )

    interface = _interface_for_task(source, task_id)
    if interface is not None:
        for ref in interface.evidence_refs:
            value = str(ref)
            if value not in refs:
                refs.append(value)
        candidate_set = getattr(interface, "candidate_set", None)
        if candidate_set is not None:
            for candidate in candidate_set.candidates:
                value = str(candidate.prompt_visible_representation_ref)
                if value not in refs:
                    refs.append(value)

    available = {str(entry.ref) for entry in catalog}
    missing = sorted(set(refs) - available)
    if missing:
        raise ValueError(
            f"minimal VQA evidence refs are unavailable: {missing}"
        )
    if not refs:
        raise ValueError(
            f"VQA task has no prompt-visible minimal evidence: {task_id}"
        )
    return refs


def _local_candidate_labels(
    source: VQASourceV2,
    *,
    count: int,
    exclude_selected: bool,
) -> list[str]:
    """Return deterministic shuffled local candidates around expert label."""

    direction = source.direction_supervision
    if direction is None:
        return []
    labels = [
        label
        for label in direction_label_names(direction.profile_id)
        if label != "stop"
    ]
    selected = str(direction.label_name)
    selected_index = (
        labels.index(selected) if selected in labels else len(labels) // 2
    )
    offsets = (-1, 1, -2, 2, -3, 3)
    pool = []
    if not exclude_selected and selected in labels:
        pool.append(selected)
    for offset in offsets:
        candidate_index = min(
            max(selected_index + offset, 0),
            len(labels) - 1,
        )
        candidate = labels[candidate_index]
        if candidate != selected or not exclude_selected:
            if candidate not in pool:
                pool.append(candidate)
        if len(pool) >= count:
            break
    seed = hashlib.sha256(
        (
            source.source_fingerprint
            + ":"
            + str(count)
            + ":"
            + str(exclude_selected)
        ).encode("utf-8")
    ).digest()[0]
    if len(pool) > 1 and seed % 2:
        pool.reverse()
    return pool[:count]


def _interface_for_task(source: VQASourceV2, task_id: str):
    """Return an optional dormant-task source interface."""

    field_name = get_vqa_task_spec_v2(
        task_id
    ).source_interface_field
    return (
        getattr(source.interfaces, field_name)
        if field_name is not None
        else None
    )


__all__ = [
    "VQAQuestionBuildResultV2",
    "build_vqa_questions_v2",
    "minimal_vqa_evidence_refs_v2",
]
