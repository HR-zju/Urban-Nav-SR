"""Focused tests for visible-FPV language-goal annotation V1."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from mcp_server.contracts.direction_profile import (
    FPV_DEGREE10_AFFORDANCE7_V1,
    FPV_DEGREE10_TERMINAL_CENTROID_V2,
    resolve_affordance_probe_profile,
    resolve_stage_a_direction_profile,
)
from mcp_server.contracts.language_waypoint_benchmark import (
    LanguageGoalCandidateV1,
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
