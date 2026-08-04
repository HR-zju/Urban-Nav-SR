from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mcp_server.agent.vqa_v2_prompt_policy import (
    build_vqa_student_user_prompt_v2,
)
from mcp_server.agent.vqa_v2_text import (
    parse_vqa_answer_v2,
    render_vqa_answer_v2,
)
from mcp_server.contracts.direction_profile import DirectionLabelStamp
from mcp_server.contracts.reasoning_dataset_v4 import (
    DirectionContextV4,
    InputModalitiesV4,
    NavObservationV4,
    REASONING_SOURCE_V4_SCHEMA_VERSION,
    ReasoningSourceV4,
    reasoning_source_fingerprint_v4,
)
from mcp_server.contracts.vqa_dataset_v2 import (
    VQAAnnotationV2,
    VQACapabilityConfigV2,
    VQASourceV2,
    vqa_source_fingerprint_v2,
)
from mcp_server.contracts.vqa_v2 import (
    VQAQuestionV2,
    VQA_V2_TASK_IDS,
    VQA_V2_TASK_SPECS,
    build_vqa_answer_v2,
    get_vqa_task_spec_v2,
    validate_answer_for_question_v2,
)
from mcp_server.data.jsonl_io import write_jsonl_atomic
from mcp_server.data.vqa_v2_eligibility import (
    load_vqa_capabilities_v2,
)
from mcp_server.data.vqa_v2_producers import (
    CanonicalDemonstratedDirectionProducerV2,
    produce_external_interface_annotation_v2,
)
from mcp_server.data.vqa_v2_review import (
    VQAReviewDecisionV2,
    filter_vqa_artifacts_by_review_v2,
)
from mcp_server.data.vqa_v2_questions import (
    build_vqa_questions_v2,
)
from mcp_server.data.vqa_v2_source import build_vqa_source_v2
from mcp_server.data.vqa_v2_migration import (
    migrate_vqa_annotation_payload_v2,
    migrate_vqa_source_payload_v2,
)
from mcp_server.data.vqa_v2_views import (
    build_vqa_training_view_v2,
)
from mcp_server.eval.vqa_v2 import evaluate_vqa_predictions_v2
from mcp_server.teacher.reasoning_v2_teacher import TeacherGeneration
from mcp_server.teacher.vqa_v2_batch import (
    LVLMWeakLabelProducerV2,
    VQABatchConfigV2,
    annotate_vqa_question_v2,
    label_vqa_source_manifest_v2,
)
from mcp_server.teacher.vqa_v2_teacher import (
    parse_vqa_teacher_answer_v2,
)
from tools.model_processing.build_vqa_v2_review_html import (
    build_vqa_v2_review_html,
)


def _source(
    *,
    temporal: bool = False,
    visible_goal: bool = False,
) -> VQASourceV2:
    images = ["current.jpg"]
    roles = ["fpv.current"]
    evidence = [
        {"ref": "instruction.active", "modality": "instruction"},
        {"ref": "fpv.current", "modality": "fpv"},
    ]
    if temporal:
        images.insert(0, "history.jpg")
        roles.insert(0, "fpv.history_01")
        evidence.insert(
            1,
            {"ref": "fpv.history_01", "modality": "fpv"},
        )
    observation = NavObservationV4(
        source_group_id="sidewalk:case-1",
        dataset_source="sidewalk",
        source_record_id="case-1",
        split_group_id="clip-1",
        images=images,
        image_roles=roles,
        instruction="Continue straight on the sidewalk.",
        model_visible_context=(
            {"goal": "visible entrance"} if visible_goal else {}
        ),
        evidence_catalog=evidence,
        input_modalities=InputModalitiesV4(
            history_fpv=temporal,
            instruction=True,
        ),
        direction_context=DirectionContextV4(
            action_profile_id="fpv_degree10_terminal_centroid_v2",
            action_profile_version="2.0",
            affordance_probe_profile_id="fpv_degree10_affordance7_v1",
            affordance_probe_profile_version="1.0",
        ),
        temporal_evidence_eligible=temporal,
    )
    direction = {
        "profile_id": "fpv_degree10_terminal_centroid_v2",
        "profile_version": "2.0",
        "label_index": 9,
        "label_name": "straight",
        "choice": "J",
        "angle_deg": 0.0,
        "source": "pixel_trajectory",
        "bin_mode": "degree10_with_stop",
        "derivation_version": (
            "pixel_trajectory_terminal_centroid_rule_v2"
        ),
    }
    payload = {
        "schema_version": REASONING_SOURCE_V4_SCHEMA_VERSION,
        "observation": observation.model_dump(mode="json"),
        "direction_supervision": direction,
        "trajectory_ego_t": None,
        "trajectory_policy": None,
        "action_mask": None,
        "warnings": [],
    }
    payload["source_fingerprint"] = (
        reasoning_source_fingerprint_v4(payload)
    )
    return build_vqa_source_v2(
        ReasoningSourceV4.model_validate(payload)
    )


def test_capability_registry_defines_active_and_dormant_tasks() -> None:
    capabilities = load_vqa_capabilities_v2()
    expected_dormant = {
        "c2_pixel_goal_to_direction",
        "c2_waypoint_to_direction",
        "c3_bev_route_alignment",
        "c4_bev_object_anchoring",
        "c5_outside_view_direction",
        "c6_bev_fpv_executability",
        "e1_candidate_trajectory_preference",
        "e2_candidate_direction_classification",
        "e3_trajectory_compatibility",
    }
    actual_dormant = {
        item.task_id
        for item in capabilities.tasks
        if not item.training_export_enabled
    }

    assert len(capabilities.tasks) == len(VQA_V2_TASK_IDS) == 27
    assert set(VQA_V2_TASK_SPECS) == set(VQA_V2_TASK_IDS)
    assert actual_dormant == expected_dormant
    assert sum(
        item.training_export_enabled
        for item in capabilities.tasks
    ) == 18
    assert all(
        not item.label_producer_ready
        and not item.evaluation_enabled
        and not item.producer_ids
        for item in capabilities.tasks
        if item.task_id in expected_dormant
    )
    assert capabilities.capability(
        "e0_demonstrated_direction_selection"
    ).training_export_enabled
    dormant = capabilities.capability(
        "c3_bev_route_alignment"
    )
    assert dormant.contract_ready
    assert dormant.source_interface_ready
    assert not dormant.label_producer_ready
    assert not dormant.training_export_enabled
    assert get_vqa_task_spec_v2(
        "c3_bev_route_alignment"
    ).source_interface_field == "bev_route_alignment"


def test_direction_stamp_rejects_route_theta_and_gps_sources() -> None:
    payload = {
        "profile_id": "fpv_degree10_terminal_centroid_v2",
        "profile_version": "2.0",
        "label_index": 9,
        "label_name": "straight",
        "choice": "J",
        "angle_deg": 0.0,
        "bin_mode": "degree10_with_stop",
        "derivation_version": "future_pixel_goal_rule_v1",
    }
    with pytest.raises(ValueError, match="forbidden"):
        DirectionLabelStamp(
            **payload,
            source="route_theta",
        )
    with pytest.raises(ValueError, match="forbidden"):
        DirectionLabelStamp(
            **payload,
            source="gps_route_match",
        )


def test_questions_skip_dormant_tasks_and_gate_temporal_task() -> None:
    capabilities = load_vqa_capabilities_v2()
    build = build_vqa_questions_v2(_source(), capabilities)
    task_ids = {question.task_id for question in build.questions}

    assert all(
        "region" not in question.question_text.lower()
        for question in build.questions
    )
    assert "e0_demonstrated_direction_selection" in task_ids
    assert "c3_bev_route_alignment" not in task_ids
    assert build.unavailable[
        "c3_bev_route_alignment"
    ].reason == "label_producer_unavailable"
    assert "d3_temporal_interaction_relation" not in task_ids
    temporal = build_vqa_questions_v2(
        _source(temporal=True),
        capabilities,
    )
    assert "d3_temporal_interaction_relation" in {
        question.task_id for question in temporal.questions
    }


def test_active_questions_use_task_minimal_evidence() -> None:
    capabilities = load_vqa_capabilities_v2()
    questions = {
        item.task_id: item
        for item in build_vqa_questions_v2(
            _source(temporal=True, visible_goal=True),
            capabilities,
        ).questions
    }

    assert questions["a1_goal_understanding"].evidence_refs == [
        "fpv.current",
        "instruction.active",
    ]
    assert questions["b1_direction_affordance"].evidence_refs == [
        "fpv.current"
    ]
    assert questions["c1_goal_to_direction"].evidence_refs == [
        "fpv.current",
        "instruction.active",
    ]
    assert questions["d3_temporal_interaction_relation"].evidence_refs == [
        "fpv.current",
        "fpv.history_01",
    ]
    assert questions["e0_demonstrated_direction_selection"].evidence_refs == [
        "fpv.current"
    ]


def test_vqa_direction_context_migration_preserves_question_and_answer() -> None:
    source = _source()
    raw_source = source.model_dump(mode="json")
    context = raw_source["observation"].pop("direction_context")
    raw_source["observation"]["direction_output_space"] = {
        "profile_id": context["action_profile_id"],
        "profile_version": context["action_profile_version"],
        "allowed_labels": list(
            source.observation.direction_context.action_labels
        ),
    }
    migrated_source, changed = migrate_vqa_source_payload_v2(raw_source)
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

    migrated_annotation = migrate_vqa_annotation_payload_v2(
        produced.annotation.model_dump(mode="json"),
        migrated_source,
    )

    assert changed
    assert migrated_source.source_fingerprint != source.source_fingerprint
    assert migrated_annotation.question == produced.annotation.question
    assert migrated_annotation.answer == produced.annotation.answer
    assert migrated_annotation.annotation_id != produced.annotation.annotation_id


def test_dormant_interfaces_are_question_ready_when_activated() -> None:
    artifact_root = (
        Path(__file__).resolve().parents[1] / "schemas/vqa_v2"
    )
    interface_cases = json.loads(
        (
            artifact_root / "vqa_v2_interface_golden_valid.json"
        ).read_text(encoding="utf-8")
    )["cases"]
    source_group_id = interface_cases[0]["value"][
        "source_group_id"
    ]
    interfaces = {
        get_vqa_task_spec_v2(case["task_id"]).source_interface_field: (
            case["value"]
        )
        for case in interface_cases
    }
    observation = NavObservationV4(
        source_group_id=source_group_id,
        dataset_source="synthetic_contract_fixture",
        source_record_id="all-dormant-interfaces",
        split_group_id="all-dormant-interfaces",
        images=["history.jpg", "current.jpg", "bev.jpg"],
        image_roles=[
            "fpv.history_01",
            "fpv.current",
            "bev.current",
        ],
        instruction="Approach the visible goal.",
        model_visible_context={"goal": "visible entrance"},
        evidence_catalog=[
            {
                "ref": "instruction.active",
                "modality": "instruction",
            },
            {"ref": "fpv.history_01", "modality": "fpv"},
            {"ref": "fpv.current", "modality": "fpv"},
            {"ref": "bev.current", "modality": "bev"},
            {"ref": "bev.building_a", "modality": "bev"},
            {"ref": "bev.transition_1", "modality": "bev"},
            {"ref": "fpv.entity_e2", "modality": "fpv"},
            {"ref": "goal.pixel_1", "modality": "pixel_goal"},
            {"ref": "waypoint.active", "modality": "waypoint"},
            {"ref": "fpv.candidate_a", "modality": "fpv"},
            {"ref": "fpv.candidate_b", "modality": "fpv"},
        ],
        input_modalities=InputModalitiesV4(
            history_fpv=True,
            instruction=True,
            bev=True,
            pixel_goal=True,
            waypoint=True,
        ),
        direction_context=DirectionContextV4(
            action_profile_id="fpv_degree10_terminal_centroid_v2",
            action_profile_version="2.0",
            affordance_probe_profile_id="fpv_degree10_affordance7_v1",
            affordance_probe_profile_version="1.0",
        ),
        temporal_evidence_eligible=True,
    )
    source_payload = {
        "schema_version": "urbannav_navigation_vqa_source_v2_0",
        "observation": observation.model_dump(mode="json"),
        "direction_supervision": {
            "profile_id": "fpv_degree10_terminal_centroid_v2",
            "profile_version": "2.0",
            "label_index": 9,
            "label_name": "straight",
            "choice": "J",
            "angle_deg": 0.0,
            "source": "pixel_trajectory",
            "bin_mode": "degree10_with_stop",
            "derivation_version": (
                "pixel_trajectory_terminal_centroid_rule_v2"
            ),
        },
        "interfaces": interfaces,
        "warnings": [],
    }
    source_payload["source_fingerprint"] = (
        vqa_source_fingerprint_v2(source_payload)
    )
    source = VQASourceV2.model_validate(source_payload)
    capability_payload = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "config/vqa_v2_stage_a_capabilities.json"
        ).read_text(encoding="utf-8")
    )
    producer_by_task = {
        case["task_id"]: case["value"]["producer_id"]
        for case in interface_cases
    }
    for task in capability_payload["tasks"]:
        producer_id = producer_by_task.get(task["task_id"])
        if producer_id is None:
            continue
        task.update(
            {
                "label_producer_ready": True,
                "training_export_enabled": True,
                "evaluation_enabled": True,
                "producer_ids": [producer_id],
                "disabled_reason": None,
            }
        )
    capabilities = VQACapabilityConfigV2.model_validate(
        capability_payload
    )

    build = build_vqa_questions_v2(source, capabilities)
    questions = {
        question.task_id: question for question in build.questions
    }

    assert set(questions) == set(VQA_V2_TASK_IDS)
    assert not build.unavailable
    assert questions[
        "c5_outside_view_direction"
    ].candidate_direction_labels
    assert questions[
        "e1_candidate_trajectory_preference"
    ].candidate_set_id == "candidate-set-1"
    assert questions[
        "e2_candidate_direction_classification"
    ].target_candidate_id == "candidate_b"
    c4_prompt = build_vqa_student_user_prompt_v2(
        source.observation,
        questions["c4_bev_object_anchoring"],
    )
    assert "bev.building_a" in c4_prompt
    assert "fpv.entity_e2" in c4_prompt
    assert "Prompt-visible evidence catalog" in c4_prompt


def test_e0_producer_and_training_view_are_minimal() -> None:
    source = _source()
    capabilities = load_vqa_capabilities_v2()
    question = next(
        question
        for question in build_vqa_questions_v2(
            source,
            capabilities,
        ).questions
        if question.task_id
        == "e0_demonstrated_direction_selection"
    )
    result = CanonicalDemonstratedDirectionProducerV2().produce(
        source,
        question,
        capabilities,
    )

    assert result.status == "produced"
    assert result.annotation is not None
    view = build_vqa_training_view_v2(
        source,
        result.annotation,
        capabilities,
    )
    assert view.assistant_text == (
        '<VQAAnswer>{"direction_label":"straight"}</VQAAnswer>'
    )
    serialized = view.model_dump(mode="json")
    assert "interfaces" not in serialized
    assert "direction_supervision" not in serialized
    assert "trajectory_ego_t" not in serialized


def test_conditioned_direction_is_not_legal_in_answer_payload() -> None:
    question = VQAQuestionV2(
        task_id="b1_direction_affordance",
        template_id="b1_v1",
        question_text=(
            "What is the local navigation affordance along the "
            "conditioned direction?"
        ),
        direction_profile_id="fpv_degree10_terminal_centroid_v2",
        conditioned_direction_label="right_20",
        evidence_refs=["fpv.current"],
    )
    parsed = parse_vqa_answer_v2(
        (
            '<VQAAnswer>{"direction_label":"right_20",'
            '"affordance":"blocked"}</VQAAnswer>'
        ),
        expected_task_id=question.task_id,
        question=question,
    )

    assert not parsed.valid
    assert "extra_forbidden" in "\n".join(parsed.errors)


def test_teacher_adapter_normalizes_spaced_vqa_wrapper_only() -> None:
    question = VQAQuestionV2(
        task_id="b6_unavailable_directions",
        template_id="b6_v1",
        question_text=(
            "Which supplied directions are visibly unavailable?"
        ),
        direction_profile_id="fpv_degree10_terminal_centroid_v2",
        candidate_direction_labels=["left_10", "right_10"],
        evidence_refs=["fpv.current"],
    )
    answer, errors = parse_vqa_teacher_answer_v2(
        (
            "< VQAAnswer >\n"
            '{"direction_labels":["left_10"],'
            '"reason_codes":["visible_obstruction"]}\n'
            "< /VQAAnswer >"
        ),
        question=question,
    )

    assert not errors
    assert answer is not None
    assert answer.payload.direction_labels == ["left_10"]

    answer, errors = parse_vqa_teacher_answer_v2(
        (
            "prefix < VQAAnswer >"
            '{"direction_labels":[],"reason_codes":[]}'
            "< /VQAAnswer >"
        ),
        question=question,
    )
    assert answer is None
    assert errors == ["vqa_answer_count:0"]


def test_region_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="region terminology"):
        VQAQuestionV2(
            task_id="b1_direction_affordance",
            template_id="b1_v1",
            question_text="Is region_r1 traversable?",
            direction_profile_id=(
                "fpv_degree10_terminal_centroid_v2"
            ),
            conditioned_direction_label="straight",
            evidence_refs=["fpv.current"],
        )


def test_dormant_external_adapter_returns_no_annotation() -> None:
    source = _source()
    capabilities = load_vqa_capabilities_v2()
    question = VQAQuestionV2(
        task_id="c3_bev_route_alignment",
        template_id="c3_v1",
        question_text=(
            "Which direction-profile label represents the active "
            "validated BEV route continuation?"
        ),
        evidence_refs=["fpv.current"],
    )

    result = produce_external_interface_annotation_v2(
        source,
        question,
        capabilities,
    )

    assert result.status == "unavailable"
    assert result.annotation is None
    assert result.reason == "label_producer_unavailable"


class _FakeBackend:
    async def generate(
        self,
        source,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        assert "complete_visible_input" in question
        assert "private" not in question.lower()
        return TeacherGeneration(
            answer='<VQAAnswer>{"affordance":"blocked"}</VQAAnswer>',
            provider="fake",
            model="fake-vlm",
            telemetry={"latency_ms": 1},
        )


class _GoldenFixtureBackend:
    def __init__(self, answers: dict[str, str]):
        self._answers = answers

    async def generate(
        self,
        source,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        request = json.loads(question.split("\n\n", maxsplit=1)[1])
        task_id = request["question"]["task_id"]
        return TeacherGeneration(
            answer=self._answers[task_id],
            provider="synthetic_contract_fixture",
            model="synthetic_contract_fixture",
            telemetry={"semantic_validation": False},
        )


class _EvidenceCaptureBackend:
    def __init__(self) -> None:
        self.image_roles: list[str] = []
        self.request: dict = {}

    async def generate(
        self,
        source,
        *,
        question: str,
        system_prompt: str,
        image_detail: str,
    ) -> TeacherGeneration:
        self.image_roles = list(source.observation.image_roles)
        self.request = json.loads(question.split("\n\n", maxsplit=1)[1])
        return TeacherGeneration(
            answer='<VQAAnswer>{"affordance":"blocked"}</VQAAnswer>',
            provider="fake",
            model="fake-vlm",
        )


def test_weak_teacher_returns_typed_weak_annotation() -> None:
    source = _source()
    capabilities = load_vqa_capabilities_v2()
    question = next(
        question
        for question in build_vqa_questions_v2(
            source,
            capabilities,
        ).questions
        if question.task_id == "b1_direction_affordance"
    )

    annotation, errors = asyncio.run(
        annotate_vqa_question_v2(
            source,
            question,
            backend=_FakeBackend(),
            config=VQABatchConfigV2(max_retries=0),
        )
    )

    assert errors == []
    assert annotation is not None
    assert annotation.label_source == "lvlm_weak"
    assert annotation.source_kind == (
        "lvlm_direction_affordance_weak"
    )
    assert annotation.teacher_model == "fake-vlm"


def test_weak_teacher_receives_only_question_required_evidence() -> None:
    source = _source(temporal=True)
    capabilities = load_vqa_capabilities_v2()
    question = next(
        item
        for item in build_vqa_questions_v2(source, capabilities).questions
        if item.task_id == "b1_direction_affordance"
    )
    backend = _EvidenceCaptureBackend()

    annotation, errors = asyncio.run(
        annotate_vqa_question_v2(
            source,
            question,
            backend=backend,
            config=VQABatchConfigV2(max_retries=0),
        )
    )

    assert errors == []
    assert annotation is not None
    assert question.evidence_refs == ["fpv.current"]
    assert backend.image_roles == ["fpv.current"]
    visible = backend.request["complete_visible_input"]
    assert visible["instruction"] is None
    assert [item["ref"] for item in visible["evidence_catalog"]] == [
        "fpv.current"
    ]


def test_weak_teacher_implements_producer_result_protocol() -> None:
    source = _source()
    capabilities = load_vqa_capabilities_v2()
    question = next(
        question
        for question in build_vqa_questions_v2(
            source,
            capabilities,
        ).questions
        if question.task_id == "b1_direction_affordance"
    )
    producer = LVLMWeakLabelProducerV2(
        _FakeBackend(),
        config=VQABatchConfigV2(max_retries=0),
    )

    result = asyncio.run(
        producer.produce(source, question, capabilities)
    )

    assert result.status == "produced"
    assert result.annotation is not None
    assert result.annotation.label_source == "lvlm_weak"
    invalid = result.annotation.model_dump(mode="json")
    invalid["label_source"] = "external_gt"
    with pytest.raises(
        ValueError,
        match="label_source is not allowed",
    ):
        VQAAnnotationV2.model_validate(invalid)


def test_all_answer_types_render_and_parse_task_specifically() -> None:
    answer = build_vqa_answer_v2(
        "e3_trajectory_compatibility",
        {
            "status": "incompatible",
            "reason_codes": ["blocked_direction"],
        },
    )
    text = render_vqa_answer_v2(answer)
    parsed = parse_vqa_answer_v2(
        text,
        expected_task_id="e3_trajectory_compatibility",
    )

    assert parsed.valid
    assert parsed.answer == answer


def test_direction_answer_must_belong_to_question_candidates() -> None:
    source = _source()
    capabilities = load_vqa_capabilities_v2()
    question = next(
        question
        for question in build_vqa_questions_v2(
            source,
            capabilities,
        ).questions
        if question.task_id
        == "e0_demonstrated_direction_selection"
    )
    answer = build_vqa_answer_v2(
        question.task_id,
        {"direction_label": "not_a_profile_label"},
    )

    with pytest.raises(
        ValueError,
        match="question candidates",
    ):
        validate_answer_for_question_v2(question, answer)


def test_offline_evaluator_and_review_html(
    tmp_path: Path,
) -> None:
    source = _source()
    capabilities = load_vqa_capabilities_v2()
    question = next(
        question
        for question in build_vqa_questions_v2(
            source,
            capabilities,
        ).questions
        if question.task_id
        == "e0_demonstrated_direction_selection"
    )
    result = CanonicalDemonstratedDirectionProducerV2().produce(
        source,
        question,
        capabilities,
    )
    assert result.annotation is not None
    annotation = result.annotation
    view = build_vqa_training_view_v2(
        source,
        annotation,
        capabilities,
    )
    source_path = tmp_path / "sources.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    view_path = tmp_path / "views.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    output_html = tmp_path / "review.html"
    write_jsonl_atomic(
        source_path,
        [source.model_dump(mode="json", exclude_none=True)],
    )
    write_jsonl_atomic(
        annotation_path,
        [annotation.model_dump(mode="json", exclude_none=True)],
    )
    write_jsonl_atomic(
        view_path,
        [view.model_dump(mode="json", exclude_none=True)],
    )
    write_jsonl_atomic(
        prediction_path,
        [
            {
                "annotation_id": annotation.annotation_id,
                "response": view.assistant_text,
            }
        ],
    )

    metrics = evaluate_vqa_predictions_v2(
        annotation_path,
        prediction_path,
        capabilities=capabilities,
    )
    review = build_vqa_v2_review_html(
        source_jsonl=source_path,
        annotation_jsonl=annotation_path,
        views_jsonl=view_path,
        output_html=output_html,
    )

    assert metrics["exact_match_fraction"] == 1.0
    assert metrics["direction_accuracy"] == 1.0
    assert metrics["task_metrics"][
        "c3_bev_route_alignment"
    ]["status"] == "N/A"
    assert review["rendered_annotations"] == 1
    page = output_html.read_text(encoding="utf-8")
    assert "e0_demonstrated_direction_selection" in page
    assert (
        "&quot;direction_label&quot;: &quot;straight&quot;"
        in page
    )
    assert "Export reviewed JSONL" in page
    assert "Human verdict" in page
    assert ".join('\\n')+'\\n'" in page
    assert "split(/\\r?\\n/)" in page
    assert "canonical_reconstructed" not in page
    assert review["review_set_id"].startswith("vqa-v2-")


def test_all_active_tasks_complete_synthetic_e2e_smoke(
    tmp_path: Path,
) -> None:
    artifact_root = (
        Path(__file__).resolve().parents[1] / "schemas/vqa_v2"
    )
    valid_cases = json.loads(
        (artifact_root / "vqa_v2_golden_valid.json").read_text(
            encoding="utf-8"
        )
    )["cases"]
    fixture_answers = {
        case["task_id"]: case["canonical_assistant_text"]
        for case in valid_cases
    }
    source = _source(temporal=True, visible_goal=True)
    capabilities = load_vqa_capabilities_v2()
    source_path = tmp_path / "sources.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    view_path = tmp_path / "views.jsonl"
    review_path = tmp_path / "review.html"
    write_jsonl_atomic(
        source_path,
        [source.model_dump(mode="json", exclude_none=True)],
    )

    report = asyncio.run(
        label_vqa_source_manifest_v2(
            source_path,
            annotation_path,
            capabilities=capabilities,
            config=VQABatchConfigV2(
                max_retries=0,
                concurrency=1,
            ),
            backend_factory=lambda _index, _key: (
                _GoldenFixtureBackend(fixture_answers)
            ),
        )
    )
    annotations = [
        VQAAnnotationV2.model_validate(json.loads(line))
        for line in annotation_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    views = [
        build_vqa_training_view_v2(
            source,
            annotation,
            capabilities,
        )
        for annotation in annotations
    ]
    write_jsonl_atomic(
        view_path,
        [
            view.model_dump(mode="json", exclude_none=True)
            for view in views
        ],
    )
    review = build_vqa_v2_review_html(
        source_jsonl=source_path,
        annotation_jsonl=annotation_path,
        views_jsonl=view_path,
        output_html=review_path,
        title="Synthetic contract smoke; not human reviewed",
    )

    assert report["error_count"] == 0
    assert report["counts"]["accepted_annotations"] == 18
    assert len(annotations) == len(views) == 18
    assert {
        annotation.question.task_id for annotation in annotations
    } == {
        item.task_id
        for item in capabilities.tasks
        if item.training_export_enabled
    }
    assert sum(
        annotation.label_source == "lvlm_weak"
        for annotation in annotations
    ) == 17
    assert sum(
        annotation.label_source == "derived_gt"
        for annotation in annotations
    ) == 1
    assert review["rendered_annotations"] == 18
    review_page = review_path.read_text(encoding="utf-8")
    assert "not human reviewed" in review_page
    assert "question_not_grounded" in review_page

    decision_path = tmp_path / "review_decisions.jsonl"
    filtered_annotation_path = tmp_path / "annotations.filtered.jsonl"
    rejected_annotation_path = tmp_path / "annotations.rejected.jsonl"
    needs_fix_annotation_path = tmp_path / "annotations.needs_fix.jsonl"
    filtered_view_path = tmp_path / "views.filtered.jsonl"
    decisions = [
        VQAReviewDecisionV2(
            review_set_id=review["review_set_id"],
            annotation_id=annotations[0].annotation_id,
            source_group_id=annotations[0].source_group_id,
            source_fingerprint=annotations[0].source_fingerprint,
            task_id=annotations[0].question.task_id,
            label_source=annotations[0].label_source,
            reviewer="unit-test",
            reviewed_at=datetime.now(UTC),
            verdict="keep",
        ),
        VQAReviewDecisionV2(
            review_set_id=review["review_set_id"],
            annotation_id=annotations[1].annotation_id,
            source_group_id=annotations[1].source_group_id,
            source_fingerprint=annotations[1].source_fingerprint,
            task_id=annotations[1].question.task_id,
            label_source=annotations[1].label_source,
            reviewer="unit-test",
            reviewed_at=datetime.now(UTC),
            verdict="reject",
            issue_codes=["answer_incorrect"],
        ),
    ]
    write_jsonl_atomic(
        decision_path,
        [item.model_dump(mode="json") for item in decisions],
    )
    filter_report = filter_vqa_artifacts_by_review_v2(
        annotation_jsonl=annotation_path,
        decision_jsonl=decision_path,
        output_annotation_jsonl=filtered_annotation_path,
        rejected_annotation_jsonl=rejected_annotation_path,
        needs_fix_annotation_jsonl=needs_fix_annotation_path,
        training_view_jsonl=view_path,
        output_training_view_jsonl=filtered_view_path,
        unreviewed_policy="keep",
    )

    assert filter_report["annotation_counts"] == {
        "keep": 1,
        "reject": 1,
        "unreviewed_keep": 16,
    }
    assert len(filtered_annotation_path.read_text().splitlines()) == 17
    assert len(rejected_annotation_path.read_text().splitlines()) == 1
    assert needs_fix_annotation_path.read_text() == ""
    assert len(filtered_view_path.read_text().splitlines()) == 17


def test_legacy_v1_source_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="legacy_vqa_v1_not_supported",
    ):
        build_vqa_source_v2(
            {
                "schema_version": (
                    "sidewalk_navigation_vqa_annotation_v1"
                )
            }
        )
