from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_server.agent.reasoning_v4_prompt_policy import get_prompt_profile_v4
from mcp_server.agent.reasoning_v4_text import render_reasoning_target_v4
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionContextV4,
    InputModalitiesV4,
    NavObservationV4,
    REASONING_SOURCE_V4_SCHEMA_VERSION,
    ReasoningSourceV4,
    ReasoningTrainingViewV4,
    reasoning_source_fingerprint_v4,
)
from mcp_server.contracts.reasoning_v4 import (
    DecisionV4,
    DirectionAffordanceV4,
    InteractionConstraintGroundingV4,
    LocalTraversabilityV4,
    NavigationContextGroundingV4,
    ReasoningTargetV4,
    ReasoningTraceV4,
    SceneGroundingV4,
    TaskDirectionV4,
    TaskGroundingV4,
)
from mcp_server.contracts.stage_i_sft import (
    StageIProjectionDropV1,
    parse_stage_i_training_view_v1,
)
from mcp_server.data.stage_i_batch import project_stage_i_batch_v1
from mcp_server.data.stage_i_provenance import (
    build_stage_i_instruction_provenance_v1,
    resolve_stage_i_instruction_role_v1,
)
from mcp_server.data.stage_i_views import (
    project_stage_i_reasoning_view_v1,
    project_stage_i_reasoning_view_with_audit_v1,
    project_stage_i_vqa_view_v1,
    project_stage_i_vqa_view_with_audit_v1,
)
from mcp_server.data.vqa_v2_eligibility import load_vqa_capabilities_v2
from mcp_server.data.vqa_v2_producers import CanonicalDemonstratedDirectionProducerV2
from mcp_server.data.vqa_v2_questions import build_vqa_questions_v2
from mcp_server.data.vqa_v2_source import build_vqa_source_v2


def _direction(label: str = "straight", *, profile_id: str = "fpv_degree10_terminal_centroid_v2") -> dict:
    if profile_id == "ego_degree10_12way_terminal_centroid_v2":
        labels = [
            "stop",
            "straight",
            "left_10",
            "right_10",
            "left_20",
            "right_20",
            "left_30",
            "right_30",
            "left_40",
            "right_40",
            "left_50plus",
            "right_50plus",
        ]
        angles = [None, 0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0, -40.0, 40.0, -50.0, 50.0]
        bin_mode = "degree10_12way_with_stop"
        derivation = "ego_x_right_y_front_terminal_centroid_rule_v2"
    else:
        labels = [
            *[f"left_{angle}" for angle in range(90, 0, -10)],
            "straight",
            *[f"right_{angle}" for angle in range(10, 100, 10)],
            "stop",
        ]
        angles = [*[float(angle) for angle in range(-90, 91, 10)], None]
        bin_mode = "degree10_with_stop"
        derivation = "pixel_trajectory_terminal_centroid_rule_v2"
    index = labels.index(label)
    return {
        "profile_id": profile_id,
        "profile_version": "2.0",
        "label_index": index,
        "label_name": label,
        "choice": chr(ord("A") + index),
        "angle_deg": angles[index],
        "source": "finish_stop_rule" if label == "stop" else derivation,
        "bin_mode": bin_mode,
        "derivation_version": derivation,
    }


def _target(label: str = "straight") -> ReasoningTargetV4:
    stop = label == "stop"
    return ReasoningTargetV4(
        reasoning=ReasoningTraceV4(
            task_grounding=TaskGroundingV4(
                objective="follow_route",
                requirements=["route_adherence"],
            ),
            scene_grounding=SceneGroundingV4(
                scene_type="urban_sidewalk",
                walkable_support="paved_sidewalk",
                key_entities=[],
                traffic_context="none_visible",
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
                    status="not_visible",
                    source="none",
                )
            ),
            interaction_constraint_grounding=InteractionConstraintGroundingV4(),
            decision=DecisionV4(
                selected_action_label=label,
                speed="stop" if stop else "normal",
                response="stop" if stop else "proceed",
            ),
        )
    )


def _formal_reasoning_view(
    label: str = "straight",
    *,
    profile_id: str = "fpv_degree10_terminal_centroid_v2",
) -> ReasoningTrainingViewV4:
    target = _target(label)
    profile = get_prompt_profile_v4("urbannav_reasoning_v4_6_compact_full_cot_sft_v1")
    return ReasoningTrainingViewV4.model_validate(
        {
            "sample_id": f"sidewalk:case-{label}:formal",
            "source_group_id": f"sidewalk:case-{label}",
            "split_group_id": "clip-1",
            "dataset_source": "sidewalk",
            "release_track": "formal_training",
            "audit_id": "b" * 64,
            "review_status": "not_required",
            "artifact_status": "training_ready",
            "artifact_training_eligible": True,
            "formal_training_eligible": True,
            "images": ["/data/current.jpg"],
            "image_roles": ["fpv.current"],
            "instruction": "Follow the route.",
            "model_visible_context": {},
            "evidence_catalog": [
                {"ref": "instruction.active", "modality": "instruction"},
                {"ref": "fpv.current", "modality": "fpv"},
            ],
            "input_modalities": InputModalitiesV4(instruction=True).model_dump(mode="json"),
            "direction_context": DirectionContextV4(
                action_profile_id=profile_id,
                action_profile_version="2.0",
                affordance_probe_profile_id=(
                    "ego_12way_affordance7_v1"
                    if profile_id == "ego_degree10_12way_terminal_centroid_v2"
                    else "fpv_degree10_affordance7_v1"
                ),
                affordance_probe_profile_version="1.0",
            ).model_dump(mode="json"),
            "assistant_text": render_reasoning_target_v4(target),
            "reasoning_target": target.model_dump(mode="json", exclude_none=True),
            "prompt_profile_id": profile.profile_id,
            "prompt_policy_digest": profile.policy_digest,
            "direction_supervision": _direction(label, profile_id=profile_id),
            "metadata": {
                "source_fingerprint": "a" * 64,
                "action_reconciliation_audit_id": "b" * 64,
            },
        }
    )


def _vqa_source():
    observation = NavObservationV4(
        source_group_id="sidewalk:vqa-case",
        dataset_source="sidewalk",
        source_record_id="vqa-case",
        split_group_id="clip-vqa",
        images=["current.jpg"],
        image_roles=["fpv.current"],
        instruction="Continue straight.",
        evidence_catalog=[
            {"ref": "instruction.active", "modality": "instruction"},
            {"ref": "fpv.current", "modality": "fpv"},
            {"ref": "fpv.obstacle_a", "modality": "fpv"},
        ],
        input_modalities=InputModalitiesV4(instruction=True),
        direction_context=DirectionContextV4(
            action_profile_id="fpv_degree10_terminal_centroid_v2",
            action_profile_version="2.0",
            affordance_probe_profile_id="fpv_degree10_affordance7_v1",
            affordance_probe_profile_version="1.0",
        ),
    )
    payload = {
        "schema_version": REASONING_SOURCE_V4_SCHEMA_VERSION,
        "observation": observation.model_dump(mode="json"),
        "direction_supervision": _direction(),
        "trajectory_ego_t": None,
        "trajectory_policy": None,
        "action_mask": None,
        "warnings": [],
    }
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(payload)
    return build_vqa_source_v2(ReasoningSourceV4.model_validate(payload))


def test_full_cot_projection_masks_only_final_action_value() -> None:
    view = project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(),
        task_type="full_cot",
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )

    assert view.answerability.full_cot
    assert view.training_selection.instruction
    assert view.modality_state.bev == "native_missing"
    zero_spans = [span for span in view.loss_spans if span.scale == 0.0]
    assert len(zero_spans) == 1
    span = zero_spans[0]
    assert view.assistant_text[span.start:span.end] == '"straight"'
    assert parse_stage_i_training_view_v1(view.model_dump(mode="json")) == view


def test_full_cot_projection_rejects_native_modality_dropout() -> None:
    with pytest.raises(ValueError, match="dropped_native_instruction"):
        project_stage_i_reasoning_view_v1(
            _formal_reasoning_view(),
            task_type="full_cot",
            modality_profile_id="fpv_current_v1",
            instruction_training_role="unknown",
        )


def test_full_cot_projection_drop_has_structured_reason() -> None:
    outcome = project_stage_i_reasoning_view_with_audit_v1(
        _formal_reasoning_view(),
        task_type="full_cot",
        modality_profile_id="fpv_current_v1",
        instruction_training_role="unknown",
    )

    assert isinstance(outcome, StageIProjectionDropV1)
    assert str(outcome.reason_code) == "full_cot_native_modality_dropout"
    assert len(outcome.drop_digest) == 64


def test_decision_only_projection_has_no_full_cot_prefix_or_system1() -> None:
    view = project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(label="stop"),
        task_type="decision",
        modality_profile_id="fpv_current_v1",
        instruction_training_role="unknown",
        allow_fpv_only_decision=True,
    )
    serialized = view.model_dump(mode="json", exclude_none=True)

    assert view.decision_authority == "fpv_behavior_cloning"
    assert view.assistant_text.startswith("<Decision>")
    assert "<TaskGrounding>" not in view.assistant_text
    assert all(span.scale == 1.0 for span in view.loss_spans)
    assert "trajectory_ego_t" not in serialized
    assert "action_mask" not in serialized


def test_selected_instruction_must_be_primary() -> None:
    with pytest.raises(ValueError, match="trusted primary"):
        project_stage_i_reasoning_view_v1(
            _formal_reasoning_view(),
            task_type="decision",
            modality_profile_id="fpv_instruction_v1",
            instruction_training_role="audit_only",
            allow_fpv_only_decision=True,
        )


def test_decision_projection_drops_bev_route_grounding_with_bev() -> None:
    payload = _formal_reasoning_view().model_dump(mode="json")
    payload["images"].append("/data/bev.jpg")
    payload["image_roles"].append("bev.current")
    payload["evidence_catalog"].append(
        {
            "ref": "bev.current",
            "modality": "bev",
            "attributes": {},
            "bev_route_evidence": {
                "frame": "ego",
                "ego_forward": "up",
                "route_overlay_visible": True,
                "route_direction_authority": "visual_conditional",
                "route_progress_authority": "visual_conditional",
                "route_terminal_marker_visible": False,
                "route_endpoint_semantics": "unknown",
                "local_safety_authority": False,
                "producer_contract_id": "test_bev_overlay",
                "producer_contract_version": "1.0",
            },
        }
    )
    payload["input_modalities"]["bev"] = True
    payload["reasoning_target"]["reasoning"]["navigation_context_grounding"] = {
        "task_direction": {
            "status": "aligned",
            "source": "bev_route",
            "direction": "straight",
        }
    }
    target = ReasoningTargetV4.model_validate(payload["reasoning_target"])
    payload["assistant_text"] = render_reasoning_target_v4(target)
    parent = ReasoningTrainingViewV4.model_validate(payload)

    view = project_stage_i_reasoning_view_v1(
        parent,
        task_type="decision",
        modality_profile_id="fpv_current_v1",
        instruction_training_role="unknown",
        allow_fpv_only_decision=True,
    )
    serialized = view.model_dump(mode="json", exclude_none=True)

    assert view.decision_authority == "fpv_behavior_cloning"
    assert not view.training_selection.bev
    assert all(item.modality != "bev" for item in view.evidence_catalog)
    assert "bev_route" not in str(serialized)


def test_decision_projection_rejects_non_native_mode_selection() -> None:
    with pytest.raises(ValueError, match="not native: bev"):
        project_stage_i_reasoning_view_v1(
            _formal_reasoning_view(),
            task_type="decision",
            modality_profile_id="fpv_bev_v1",
            instruction_training_role="unknown",
            allow_fpv_only_decision=True,
        )


def test_vqa_projection_keeps_annotation_evidence() -> None:
    source = _vqa_source()
    capabilities = load_vqa_capabilities_v2()
    question = next(
        item
        for item in build_vqa_questions_v2(source, capabilities).questions
        if item.task_id == "e0_demonstrated_direction_selection"
    )
    produced = CanonicalDemonstratedDirectionProducerV2().produce(source, question, capabilities)
    assert produced.annotation is not None

    view = project_stage_i_vqa_view_v1(
        source,
        produced.annotation,
        capabilities=capabilities,
        modality_profile_id="fpv_instruction_v1",
        instruction_training_role="primary",
    )

    assert view.answerability.vqa
    assert view.instruction == "Continue straight."
    assert view.modality_state.instruction == "present"
    assert "fpv.obstacle_a" in {str(item.ref) for item in view.evidence_catalog}
    assert set(view.annotation_evidence_refs).issubset({str(item.ref) for item in view.evidence_catalog})


def test_vqa_projection_rejects_removed_annotation_evidence() -> None:
    source = _vqa_source()
    capabilities = load_vqa_capabilities_v2()
    question = next(
        item
        for item in build_vqa_questions_v2(source, capabilities).questions
        if item.task_id == "e0_demonstrated_direction_selection"
    )
    produced = CanonicalDemonstratedDirectionProducerV2().produce(source, question, capabilities)
    assert produced.annotation is not None
    # Legacy annotations cited the complete source envelope.  Preserve this
    # regression fixture to prove the projector still fails closed instead of
    # silently treating a minimal-evidence template as already re-annotated.
    legacy_question = produced.annotation.question.model_copy(
        update={
            "evidence_refs": [
                str(item.ref)
                for item in source.observation.evidence_catalog
            ]
        }
    )
    legacy_annotation = produced.annotation.model_copy(
        update={
            "question": legacy_question,
            "evidence_refs": list(legacy_question.evidence_refs),
        }
    )

    with pytest.raises(ValueError, match="removed annotation evidence"):
        project_stage_i_vqa_view_v1(
            source,
            legacy_annotation,
            capabilities=capabilities,
            modality_profile_id="fpv_current_v1",
            instruction_training_role="unknown",
        )

    outcome = project_stage_i_vqa_view_with_audit_v1(
        source,
        legacy_annotation,
        capabilities=capabilities,
        modality_profile_id="fpv_current_v1",
        instruction_training_role="unknown",
    )
    assert isinstance(outcome, StageIProjectionDropV1)
    assert str(outcome.reason_code) == "vqa_annotation_evidence_removed"


def test_projection_digest_rejects_mutation() -> None:
    view = project_stage_i_reasoning_view_v1(
        _formal_reasoning_view(),
        task_type="decision",
        modality_profile_id="fpv_current_v1",
        instruction_training_role="unknown",
        allow_fpv_only_decision=True,
    )
    payload = copy.deepcopy(view.model_dump(mode="json"))
    payload["assistant_text"] = payload["assistant_text"].replace("straight", "straighx")
    with pytest.raises(ValidationError, match="projection_digest"):
        parse_stage_i_training_view_v1(payload)


def test_instruction_provenance_requires_exact_source_revision_and_text() -> None:
    view = _formal_reasoning_view()
    source_fingerprint = str(view.metadata["source_fingerprint"])
    entry = build_stage_i_instruction_provenance_v1(
        source_group_id=view.source_group_id,
        source_fingerprint=source_fingerprint,
        instruction=view.instruction,
        training_role="primary",
        provenance_source="explicit_source_registry",
        policy_id="stage_i_test_instruction_policy",
        policy_version="1.0",
    )
    registry = {
        (entry.source_group_id, entry.source_fingerprint): entry
    }

    assert resolve_stage_i_instruction_role_v1(
        source_group_id=view.source_group_id,
        source_fingerprint=source_fingerprint,
        instruction=view.instruction,
        registry=registry,
    ) == "primary"
    with pytest.raises(ValueError, match="text digest"):
        resolve_stage_i_instruction_role_v1(
            source_group_id=view.source_group_id,
            source_fingerprint=source_fingerprint,
            instruction="Changed instruction.",
            registry=registry,
        )
    with pytest.raises(ValueError, match="provenance is missing"):
        resolve_stage_i_instruction_role_v1(
            source_group_id=view.source_group_id,
            source_fingerprint="f" * 64,
            instruction=view.instruction,
            registry=registry,
        )


def test_batch_projector_requires_registry_and_emits_drop_audit(tmp_path: Path) -> None:
    view = _formal_reasoning_view()
    parent_path = tmp_path / "reasoning.jsonl"
    parent_path.write_text(
        json.dumps(view.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    empty_registry = tmp_path / "empty-provenance.jsonl"
    empty_registry.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance is missing"):
        project_stage_i_batch_v1(
            output_view_jsonl=tmp_path / "rejected-views.jsonl",
            output_drop_jsonl=tmp_path / "rejected-drops.jsonl",
            instruction_provenance_jsonl=empty_registry,
            reasoning_view_jsonls=[parent_path],
            modality_profile_ids=["fpv_instruction_v1"],
        )

    entry = build_stage_i_instruction_provenance_v1(
        source_group_id=view.source_group_id,
        source_fingerprint=str(view.metadata["source_fingerprint"]),
        instruction=view.instruction,
        training_role="primary",
        provenance_source="explicit_source_registry",
        policy_id="stage_i_test_instruction_policy",
        policy_version="1.0",
    )
    registry_path = tmp_path / "provenance.jsonl"
    registry_path.write_text(
        json.dumps(entry.model_dump(mode="json")) + "\n",
        encoding="utf-8",
    )
    report = project_stage_i_batch_v1(
        output_view_jsonl=tmp_path / "views.jsonl",
        output_drop_jsonl=tmp_path / "drops.jsonl",
        instruction_provenance_jsonl=registry_path,
        reasoning_view_jsonls=[parent_path],
        modality_profile_ids=["fpv_instruction_v1"],
    )

    assert report["counts"]["views"] == 1
    assert report["counts"]["drops"] == 1
    assert report["counts"]["drop:decision_no_visible_authority"] == 1


def test_all_stage_i_task_views_build_ms_swift_physical_rows() -> None:
    ms_root = Path(__file__).resolve().parents[4] / "ms-swift"
    if not ms_root.is_dir():
        pytest.skip("sibling ms-swift checkout is unavailable")
    sys.path.insert(0, str(ms_root))
    try:
        adapter = importlib.import_module(
            "swift.sidewalk_nav.data.stage_i_manifest"
        )
        full_cot = project_stage_i_reasoning_view_v1(
            _formal_reasoning_view(),
            task_type="full_cot",
            modality_profile_id="fpv_instruction_v1",
            instruction_training_role="primary",
        )
        decision = project_stage_i_reasoning_view_v1(
            _formal_reasoning_view(),
            task_type="decision",
            modality_profile_id="fpv_current_v1",
            instruction_training_role="unknown",
            allow_fpv_only_decision=True,
        )
        decision_12way = project_stage_i_reasoning_view_v1(
            _formal_reasoning_view(
                profile_id="ego_degree10_12way_terminal_centroid_v2"
            ),
            task_type="decision",
            modality_profile_id="fpv_current_v1",
            instruction_training_role="unknown",
            allow_fpv_only_decision=True,
        )
        source = _vqa_source()
        capabilities = load_vqa_capabilities_v2()
        question = next(
            item
            for item in build_vqa_questions_v2(source, capabilities).questions
            if item.task_id == "e0_demonstrated_direction_selection"
        )
        produced = CanonicalDemonstratedDirectionProducerV2().produce(
            source,
            question,
            capabilities,
        )
        assert produced.annotation is not None
        vqa = project_stage_i_vqa_view_v1(
            source,
            produced.annotation,
            capabilities=capabilities,
            modality_profile_id="fpv_instruction_v1",
            instruction_training_role="primary",
        )

        rows = [
            adapter.build_stage_i_manifest_row(
                view.model_dump(mode="json")
            )
            for view in (full_cot, vqa, decision, decision_12way)
        ]
        assert [row["task_type"] for row in rows] == [
            "full_cot",
            "vqa",
            "decision",
            "decision",
        ]
        assert rows[-1]["direction_profile_id"] == (
            "ego_degree10_12way_terminal_centroid_v2"
        )
        assert all(row["loss_mask"]["lm"] for row in rows)
    finally:
        sys.path.remove(str(ms_root))
