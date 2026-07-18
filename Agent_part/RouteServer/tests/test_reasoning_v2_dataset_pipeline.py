from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mcp_server.contracts.reasoning import (
    ConstraintApplicabilityV2,
    CrossModalAlignmentV2,
    DecisionV2,
    EvidenceGroundingV2,
    FinishV2,
    GroundedFactV2,
    ReasoningTargetV2,
    ReasoningTraceV2,
    TaskGroundingV2,
)
from mcp_server.contracts.reasoning_dataset import ReasoningAnnotationV2, ReasoningSourceV2
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig, build_reasoning_source_v2
from mcp_server.data.reasoning_views import build_reasoning_training_views
from mcp_server.teacher.reasoning_v2_teacher import (
    ReasoningTeacherConfig,
    TeacherGeneration,
    annotate_reasoning_source,
    build_reasoning_teacher_question,
    label_reasoning_source_manifest,
)


class _SequenceBackend:
    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.questions: list[str] = []

    async def generate(self, source, *, question, system_prompt, image_detail):
        self.questions.append(question)
        answer = self.answers.pop(0)
        return TeacherGeneration(answer=answer, provider="fake", model="fake-vlm")


def test_source_normalization_uses_explicit_roles_and_canonical_system1() -> None:
    row = {
        "id": "clip-a:42",
        "clip_name": "clip-a",
        "instruction": "Follow the mapped route and keep right.",
        "fpv_image_paths": [f"/tmp/fpv-{index}.jpg" for index in range(7)],
        "bev_image_paths": ["/tmp/bev-old.jpg", "/tmp/bev-current.jpg"],
        "route_context": {"desired_direction": "straight", "route_id": "route-7"},
        "independent_route_evidence": True,
        "goal_source": "trajectory_ego_t_lookahead",
        "goal_ego_t": {"x": 0.0, "y": 4.0},
        "route_points_ego_t": [{"x": 0.0, "y": 2.0, "heading": 0.0}],
        "trajectory_ego_t": [[0.0, 1.0, 0.2, 0.5], [1.0, 2.0, 0.4, 0.5]],
    }

    source = build_reasoning_source_v2(
        row,
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk", max_fpv_frames=5),
    )

    assert len(source.images) == 6
    assert source.image_roles == [
        "fpv.history_00",
        "fpv.history_01",
        "fpv.history_02",
        "fpv.history_03",
        "fpv.current",
        "bev.current",
    ]
    assert source.goal_geometry is None
    assert source.route_points_ego_t is None
    assert source.metadata["pseudo_goal_condition_dropped"] is True
    assert source.system1_enabled is True
    assert source.trajectory_ego_t == [[1.0, -0.0, 0.0], [2.0, -1.0, pytest.approx(-0.7853981634)]]
    assert source.trajectory_policy["system1_target_format"] == "ego_pose_xyt_flu_v1"
    assert source.trajectory_policy["normalized"] is False
    assert "route.active" in {item.ref for item in source.evidence_catalog}


def test_source_rejects_ambiguous_multimodal_image_list() -> None:
    with pytest.raises(ValueError, match="explicit image_roles"):
        build_reasoning_source_v2(
            {"id": "ambiguous", "images": ["a.jpg", "b.jpg"]},
            index=0,
            config=ReasoningSourceBuildConfig(dataset_source="test"),
        )


def test_source_drops_route_context_without_independent_provenance() -> None:
    row = {
        "id": "route-derived-from-gt",
        "instruction": "Continue navigation.",
        "fpv_image_path": "/tmp/fpv.jpg",
        "route_context": {
            "intent_gt": "left",
            "route_id": "future-trajectory-derived",
        },
        "route_points_ego_t": [[1.0, 0.0, 0.0]],
    }

    source = build_reasoning_source_v2(
        row,
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="synthetic"),
    )

    assert source.metadata["independent_route_evidence"] is False
    assert source.metadata["unverified_route_context_dropped"] is True
    assert "route" not in source.model_visible_context
    assert source.route_points_ego_t is None
    assert "route" not in source.available_modalities
    assert "route.active" not in {item.ref for item in source.evidence_catalog}


def test_source_keeps_map_provider_route_with_explicit_provenance() -> None:
    row = {
        "id": "amap-route",
        "instruction": "Follow the pedestrian route.",
        "fpv_image_path": "/tmp/fpv.jpg",
        "route_context": {
            "route_id": "route-7",
            "desired_direction": "straight",
            "provider": "amap",
        },
        "route_points_ego_t": [[1.0, 0.0, 0.0]],
    }

    source = build_reasoning_source_v2(
        row,
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="synthetic"),
    )

    assert source.metadata["independent_route_evidence"] is True
    assert source.metadata["unverified_route_context_dropped"] is False
    assert source.model_visible_context["route"]["value"]["provider"] == "amap"
    assert source.route_points_ego_t == [[1.0, 0.0, 0.0]]
    assert "route.active" in {item.ref for item in source.evidence_catalog}


def test_source_fingerprint_detects_post_build_mutation() -> None:
    source = _source(system1_enabled=False)
    payload = source.model_dump(mode="json")
    payload["instruction"] = "mutated"
    with pytest.raises(ValidationError, match="source_fingerprint"):
        ReasoningSourceV2.model_validate(payload)


def test_teacher_repair_validates_refs_and_never_uses_private_context() -> None:
    source = _source(system1_enabled=True)
    target = _target()
    backend = _SequenceBackend(["not json", json.dumps(target.model_dump(mode="json"))])

    annotation = asyncio.run(
        annotate_reasoning_source(
            source,
            backend=backend,
            config=ReasoningTeacherConfig(max_retries=1),
        )
    )

    assert annotation.status == "accepted"
    assert annotation.attempts == 2
    assert annotation.target == target
    assert annotation.metadata["fallback_used"] is False
    assert "teacher_private_context" not in backend.questions[0]
    assert "trajectory_ego_t" not in backend.questions[0]
    assert "VALIDATION_ERRORS" in backend.questions[1]


def test_full_and_qa_views_share_label_but_not_system1_sidecars() -> None:
    source = _source(system1_enabled=True)
    annotation = ReasoningAnnotationV2(
        source_group_id=source.source_group_id,
        source_fingerprint=source.source_fingerprint,
        status="accepted",
        target=_target(),
        attempts=1,
        teacher_provider="fake",
        teacher_model="fake-vlm",
    )

    views = build_reasoning_training_views(source, annotation)
    full = next(view for view in views if view.view_type == "full_cot")
    qa = [view for view in views if view.view_type == "qa"]

    assert "<readout>[SYS1_0]" in full.assistant_text
    assert full.assistant_text.index("</finish>") < full.assistant_text.index("<readout>")
    assert full.trajectory_ego_t == source.trajectory_ego_t
    assert full.loss_mask["system1_action"] is True
    assert {view.task_type for view in qa} >= {
        "instruction_to_map_selection",
        "map_route_state_reading",
        "route_to_fpv_grounding",
        "local_route_executability",
        "social_traversability",
    }
    assert all(view.trajectory_ego_t is None for view in qa)
    assert all("[SYS1_" not in view.assistant_text for view in qa)
    assert all(view.loss_mask["system1_action"] is False for view in qa)


def test_manifest_runner_is_append_only_and_resumable(tmp_path: Path) -> None:
    source = _source(system1_enabled=False)
    source_path = tmp_path / "source.jsonl"
    annotation_path = tmp_path / "annotation.jsonl"
    write_jsonl_atomic(source_path, [source.model_dump(mode="json")])

    def backend_factory(worker_id: int, api_key: str | None):
        return _SequenceBackend([json.dumps(_target().model_dump(mode="json"))])

    first = asyncio.run(
        label_reasoning_source_manifest(
            source_path,
            annotation_path,
            config=ReasoningTeacherConfig(max_retries=0),
            backend_factory=backend_factory,
        )
    )
    second = asyncio.run(
        label_reasoning_source_manifest(
            source_path,
            annotation_path,
            config=ReasoningTeacherConfig(max_retries=0),
            backend_factory=backend_factory,
        )
    )

    assert first["accepted_rows"] == 1
    assert second["resumed_rows"] == 1
    assert len(list(iter_jsonl(annotation_path))) == 1


def test_teacher_prompt_excludes_private_and_continuous_fields() -> None:
    prompt = build_reasoning_teacher_question(_source(system1_enabled=True))
    assert "private_secret" not in prompt
    assert "trajectory_ego_t" not in prompt
    assert "pixel_trajectory" not in prompt


def _source(*, system1_enabled: bool) -> ReasoningSourceV2:
    row = {
        "id": "source-1",
        "instruction": "Continue straight on the mapped pedestrian route.",
        "images": ["/tmp/fpv.jpg", "/tmp/bev.jpg"],
        "image_roles": ["fpv", "bev"],
        "route_context": {"route_id": "route-1", "desired_direction": "straight"},
        "independent_route_evidence": True,
        "teacher_private_context": {"secret": "private_secret"},
    }
    if system1_enabled:
        row["trajectory_ego_t"] = [[0.0, 0.5, 0.2, 0.5], [0.0, 1.0, 0.4, 0.5]]
    config = ReasoningSourceBuildConfig(dataset_source="synthetic", enable_system1=system1_enabled)
    return build_reasoning_source_v2(row, index=0, config=config)


def _target() -> ReasoningTargetV2:
    trace = ReasoningTraceV2(
        task_grounding=TaskGroundingV2(
            route_ref="route.active",
            local_objective="continue_route",
            instruction_constraints=[],
        ),
        evidence_grounding=EvidenceGroundingV2(
            task_evidence_refs=["instruction.active"],
            route_evidence_refs=["route.active", "bev.current"],
            fpv_evidence_refs=["fpv.current", "corridor.center"],
            relevant_facts=[
                GroundedFactV2(
                    fact_code="center_corridor_visible",
                    evidence_refs=["fpv.current", "corridor.center"],
                )
            ],
        ),
        cross_modal_alignment=CrossModalAlignmentV2(
            available=True,
            desired_direction="straight",
            route_to_fpv="center",
            selected_corridor_ref="corridor.center",
            status="aligned",
            evidence_refs=["route.active", "bev.current", "corridor.center"],
        ),
        constraint_applicability=ConstraintApplicabilityV2(),
        decision=DecisionV2(
            motion="go_straight",
            speed="normal",
            social_action="none",
            selected_corridor_ref="corridor.center",
            fallback="slow",
            confidence="high",
            reason_codes=["center_corridor_visible"],
        ),
    )
    return ReasoningTargetV2(
        reasoning=trace,
        finish=FinishV2(
            decision_type="navigate",
            motion="go_straight",
            speed="normal",
            social_action="none",
            confidence_level="high",
            trajectory_type="local_navigation",
            tool_used=False,
        ),
    )
