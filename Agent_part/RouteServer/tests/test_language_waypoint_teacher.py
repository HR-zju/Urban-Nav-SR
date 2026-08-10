"""Focused tests for visible-FPV language-goal annotation V1."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from PIL import Image

from mcp_server.contracts.direction_profile import (
    FPV_DEGREE10_AFFORDANCE7_V1,
    FPV_DEGREE10_TERMINAL_CENTROID_V2,
    resolve_affordance_probe_profile,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.language_waypoint_benchmark import (
    LanguageGoalAnnotationV1,
    LanguageGoalCandidateV1,
)
from mcp_server.data.benchmark.language_waypoint_human_review import (
    LanguageGoalHumanReviewGateConfigV1,
    evaluate_language_goal_human_review_v1,
)
from mcp_server.data.benchmark.language_waypoint_review import (
    language_goal_review_id_v1,
    write_language_waypoint_review_html_v1,
)
from mcp_server.data.benchmark.language_waypoint_annotation_shards import (
    merge_language_goal_annotation_shards_v1,
    partition_language_goal_annotation_inputs_v1,
)
from mcp_server.contracts.reasoning_dataset_v4 import (
    REASONING_SOURCE_V4_SCHEMA_VERSION,
    DirectionContextV4,
    InputModalitiesV4,
    NavObservationV4,
    ReasoningSourceV4,
    reasoning_source_fingerprint_v4,
)
from mcp_server.teacher.language_waypoint_batch import (
    LanguageGoalAnnotationConfigV1,
    annotate_language_goal_candidate_v1,
    label_language_goal_candidates_v1,
)
from mcp_server.teacher.language_waypoint_teacher import (
    build_language_goal_teacher_question_v1,
    language_goal_teacher_profile_v1,
)
from mcp_server.teacher.reasoning_v2_teacher import TeacherGeneration


class _FakeBackend:
    """Return one deterministic valid answer without opening a provider."""

    async def generate(self, source, *, question, system_prompt, image_detail):
        """Return one valid response while retaining only test-visible fields."""

        assert source.image_roles == ["fpv.current"]
        assert "candidate_xy_flu_m" not in question
        return TeacherGeneration(
            answer=json.dumps(
                {
                    "instruction_text": "Proceed along the clear sidewalk toward the entrance ahead.",
                    "instruction_type": "visible_corridor_goal",
                    "target_ref": "fpv.current",
                    "target_bbox_anchor_xyxy": None,
                    "spatial_relation": "ahead along the visible sidewalk",
                    "visibility_at_anchor": True,
                    "evidence_refs": ["fpv.current"],
                    "confidence": "high",
                    "validator_status": "accepted",
                }
            ),
            provider="local",
            model="fake-vlm",
            telemetry={"roundtrip_ms": 1},
        )


def _source(tmp_path: Path) -> ReasoningSourceV4:
    """Build one valid source with one accessible current image path."""

    action = resolve_stage_a_direction_profile(FPV_DEGREE10_TERMINAL_CENTROID_V2)
    probe = resolve_affordance_probe_profile(FPV_DEGREE10_AFFORDANCE7_V1)
    current = tmp_path / "current.jpg"
    current.write_bytes(b"test")
    observation = NavObservationV4(
        source_group_id="sidewalk:language-goal-case",
        dataset_source="sidewalk",
        source_record_id="language-goal-case",
        split_group_id="sidewalk:clip-1",
        images=[str(current)],
        image_roles=["fpv.current"],
        instruction="Walk toward the accessible entrance.",
        input_modalities=InputModalitiesV4(instruction=True),
        evidence_catalog=[{"ref": "fpv.current", "modality": "fpv"}],
        direction_context=DirectionContextV4(
            action_profile_id=FPV_DEGREE10_TERMINAL_CENTROID_V2,
            action_profile_version=action["profile_version"],
            affordance_probe_profile_id=FPV_DEGREE10_AFFORDANCE7_V1,
            affordance_probe_profile_version=probe["profile_version"],
        ),
    )
    payload = {
        "schema_version": REASONING_SOURCE_V4_SCHEMA_VERSION,
        "observation": observation.model_dump(mode="json"),
        "direction_supervision": None,
        "direction_geometry": None,
        "trajectory_ego_t": None,
        "trajectory_policy": None,
        "action_mask": None,
        "warnings": [],
    }
    payload["source_fingerprint"] = reasoning_source_fingerprint_v4(payload)
    return ReasoningSourceV4.model_validate(payload)


def _candidate(source: ReasoningSourceV4) -> LanguageGoalCandidateV1:
    """Build a deterministic teacher-only target candidate."""

    return LanguageGoalCandidateV1(
        candidate_id="candidate-1",
        parent_sample_id=source.observation.source_group_id,
        source_group_id=source.observation.source_group_id,
        split_group_id=source.observation.split_group_id,
        dataset_source=source.observation.dataset_source,
        candidate_time_s=4.0,
        candidate_xy_flu_m={"x_m": 8.0, "y_m": 1.0},
        source_trajectory_sidecar_digest="a" * 64,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write compact JSONL fixture rows."""

    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    """Read one compact test JSONL fixture."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_language_goal_prompt_hides_raw_future_coordinates(tmp_path: Path) -> None:
    """The teacher sees a semantic candidate relation, not raw FLU coordinates."""

    source = _source(tmp_path)
    question = build_language_goal_teacher_question_v1(
        source,
        _candidate(source),
        language_goal_teacher_profile_v1(),
    )
    assert "candidate_xy_flu_m" not in question
    assert '"desired_relation":"ahead"' in question
    assert "Walk toward the accessible entrance." in question


def test_language_goal_annotation_and_resume(tmp_path: Path) -> None:
    """One accepted response writes a typed row and later resumes safely."""

    source = _source(tmp_path)
    candidate = _candidate(source)
    annotation, decision = asyncio.run(
        annotate_language_goal_candidate_v1(
            source,
            candidate,
            backend=_FakeBackend(),
            config=LanguageGoalAnnotationConfigV1(),
        )
    )
    assert annotation is not None
    assert annotation.validator_status == "accepted"
    assert annotation.target_route_distance_m is not None
    assert "trajectory" not in str(annotation.instruction_text).lower()
    assert decision["status"] == "accepted"

    source_path = tmp_path / "source.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    annotation_path = tmp_path / "annotation.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"
    report_path = tmp_path / "report.json"
    _write_jsonl(source_path, [source.model_dump(mode="json")])
    _write_jsonl(candidate_path, [candidate.model_dump(mode="json")])
    report = asyncio.run(
        label_language_goal_candidates_v1(
            source_jsonl=source_path,
            candidates_jsonl=candidate_path,
            annotation_jsonl=annotation_path,
            decisions_jsonl=decisions_path,
            report_json=report_path,
            backend_factory=lambda _index, _key: _FakeBackend(),
        )
    )
    assert report["counts"]["accepted"] == 1
    resumed = asyncio.run(
        label_language_goal_candidates_v1(
            source_jsonl=source_path,
            candidates_jsonl=candidate_path,
            annotation_jsonl=annotation_path,
            decisions_jsonl=decisions_path,
            report_json=report_path,
            backend_factory=lambda _index, _key: _FakeBackend(),
        )
    )
    assert resumed["counts"]["resumed"] == 1


def test_language_goal_shards_are_parent_disjoint_and_merge_in_candidate_order(tmp_path: Path) -> None:
    """Dual-service shards preserve coverage without source or audit mutation."""

    first = _source(tmp_path)
    first_candidate = _candidate(first)
    second_raw = first.model_dump(mode="json")
    second_raw["observation"]["source_group_id"] = "sidewalk:language-goal-case-2"
    second_raw["observation"]["source_record_id"] = "language-goal-case-2"
    second_raw["source_fingerprint"] = reasoning_source_fingerprint_v4(
        {key: value for key, value in second_raw.items() if key != "source_fingerprint"}
    )
    second = ReasoningSourceV4.model_validate(second_raw)
    second_candidate_payload = _candidate(second).model_dump(mode="json")
    second_candidate_payload["candidate_id"] = "candidate-2"
    second_candidate = LanguageGoalCandidateV1(**second_candidate_payload)
    source_path = tmp_path / "source.jsonl"
    candidates_path = tmp_path / "candidates.jsonl"
    _write_jsonl(source_path, [first.model_dump(mode="json"), second.model_dump(mode="json")])
    _write_jsonl(
        candidates_path,
        [first_candidate.model_dump(mode="json"), second_candidate.model_dump(mode="json")],
    )
    partition = partition_language_goal_annotation_inputs_v1(
        source_jsonl=source_path,
        candidates_jsonl=candidates_path,
        output_root=tmp_path / "shards",
        report_json=tmp_path / "partition.json",
        lane_count=2,
    )
    assert partition["candidate_count"] == 2
    shard_parents = set()
    for lane in partition["lanes"]:
        shard_parents.update(
            row["observation"]["source_group_id"] for row in _read_jsonl(Path(lane["source_jsonl"]))
        )
    assert shard_parents == {first_candidate.parent_sample_id, second_candidate.parent_sample_id}

    first_annotation = LanguageGoalAnnotationV1(
        annotation_id="annotation-first",
        parent_sample_id=first_candidate.parent_sample_id,
        instruction_text="Proceed toward the visible entrance ahead.",
        instruction_type="visible_corridor_goal",
        target_ref="fpv.current",
        visibility_at_anchor=True,
        evidence_refs=("fpv.current",),
        teacher_model="test-vlm",
        prompt_version="test",
        confidence="high",
        validator_status="accepted",
        provenance="lvlm_annotation",
    )
    annotation_lane0 = tmp_path / "lane0_annotations.jsonl"
    annotation_lane1 = tmp_path / "lane1_annotations.jsonl"
    decisions_lane0 = tmp_path / "lane0_decisions.jsonl"
    decisions_lane1 = tmp_path / "lane1_decisions.jsonl"
    _write_jsonl(annotation_lane0, [first_annotation.model_dump(mode="json")])
    _write_jsonl(annotation_lane1, [])
    _write_jsonl(
        decisions_lane0,
        [{"candidate_id": first_candidate.candidate_id, "parent_sample_id": first_candidate.parent_sample_id, "status": "accepted"}],
    )
    _write_jsonl(
        decisions_lane1,
        [{"candidate_id": second_candidate.candidate_id, "parent_sample_id": second_candidate.parent_sample_id, "status": "error"}],
    )
    report = merge_language_goal_annotation_shards_v1(
        candidates_jsonl=candidates_path,
        annotation_jsonls=[annotation_lane0, annotation_lane1],
        decisions_jsonls=[decisions_lane0, decisions_lane1],
        output_annotations_jsonl=tmp_path / "merged_annotations.jsonl",
        output_decisions_jsonl=tmp_path / "merged_decisions.jsonl",
        report_json=tmp_path / "merge.json",
    )
    assert report["annotation_count"] == 1
    assert report["decision_count"] == 2
    assert report["status_counts"] == {"accepted": 1, "error": 1}
    assert [row["candidate_id"] for row in _read_jsonl(tmp_path / "merged_decisions.jsonl")] == [
        first_candidate.candidate_id,
        second_candidate.candidate_id,
    ]


def test_language_goal_review_embeds_only_visible_annotation_fields(tmp_path: Path) -> None:
    """The portable review hides teacher-only candidate geometry."""

    source = _source(tmp_path)
    # 生成最小有效 JPEG，使 self-contained 缩略图路径得到真实覆盖。
    Image.new("RGB", (4, 3), color=(32, 64, 96)).save(source.observation.images[0], format="JPEG")
    annotation = LanguageGoalAnnotationV1(
        annotation_id="review-language-1",
        parent_sample_id=source.observation.source_group_id,
        instruction_text="Proceed along the visible sidewalk toward the entrance ahead.",
        instruction_type="visible_corridor_goal",
        target_ref="fpv.current",
        visibility_at_anchor=True,
        evidence_refs=("fpv.current",),
        teacher_model="test-vlm",
        prompt_version="test-profile",
        confidence="high",
        validator_status="accepted",
        provenance="lvlm_annotation",
    )
    source_path = tmp_path / "source.jsonl"
    annotation_path = tmp_path / "annotation.jsonl"
    output_html = tmp_path / "review.html"
    _write_jsonl(source_path, [source.model_dump(mode="json")])
    _write_jsonl(annotation_path, [annotation.model_dump(mode="json")])
    report = write_language_waypoint_review_html_v1(
        source_jsonl=source_path,
        annotation_jsonl=annotation_path,
        output_html=output_html,
    )
    rendered = output_html.read_text(encoding="utf-8")
    assert report["rendered_count"] == 1
    assert "Proceed along the visible sidewalk" in rendered
    assert "candidate_xy_flu_m" not in rendered
    assert "data:image/jpeg;base64," in rendered


def test_human_review_gate_requires_complete_high_acceptance_review(tmp_path: Path) -> None:
    """Human exports become a deterministic full-label promotion decision."""

    instruction_types = (
        "visible_corridor_goal",
        "crossing_entrance_region",
        "landmark_object_goal",
    )
    annotations = [
        LanguageGoalAnnotationV1(
            annotation_id=f"human-gate-{index}",
            parent_sample_id=f"sidewalk:human-gate-{index}",
            instruction_text="Move toward the visible target ahead.",
            instruction_type=instruction_type,
            target_ref="fpv.current",
            visibility_at_anchor=True,
            evidence_refs=("fpv.current",),
            teacher_model="test-vlm",
            prompt_version="test-profile",
            confidence="high",
            validator_status="accepted",
            provenance="lvlm_annotation",
        )
        for index, instruction_type in enumerate(instruction_types)
    ]
    annotation_path = tmp_path / "annotations.jsonl"
    adjudications_path = tmp_path / "adjudications.json"
    _write_jsonl(annotation_path, [item.model_dump(mode="json") for item in annotations])
    adjudications_path.write_text(
        json.dumps(
            [
                {
                    "review_id": language_goal_review_id_v1(annotation),
                    "decision": "accept",
                }
                for annotation in annotations
            ]
        ),
        encoding="utf-8",
    )
    report = evaluate_language_goal_human_review_v1(
        annotation_jsonl=annotation_path,
        adjudications_json=adjudications_path,
        report_json=tmp_path / "gate.json",
        config=LanguageGoalHumanReviewGateConfigV1(
            min_accept_rate=0.9,
            max_uncertain_rate=0.05,
            min_reviewed_per_instruction_type=1,
        ),
    )
    assert report["passed"] is True
    assert report["reviewed_count"] == 3

    adjudications_path.write_text("[]", encoding="utf-8")
    rejected = evaluate_language_goal_human_review_v1(
        annotation_jsonl=annotation_path,
        adjudications_json=adjudications_path,
        report_json=tmp_path / "rejected_gate.json",
    )
    assert rejected["passed"] is False
    assert "eligible_annotation_unreviewed" in rejected["gate_failures"]
