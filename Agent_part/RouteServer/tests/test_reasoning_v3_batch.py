from __future__ import annotations

import asyncio
import json

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningAnnotationV3
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_source import build_reasoning_source_v3
from mcp_server.teacher.reasoning_v2_teacher import TeacherGeneration
from mcp_server.teacher.reasoning_v3_batch import (
    REASONING_V3_TEACHER_PROMPT_VERSION,
    ReasoningV3BatchConfig,
    annotate_reasoning_source_v3,
    label_reasoning_source_manifest_v3,
)
from mcp_server.teacher.reasoning_v3_teacher import parse_reasoning_v3_teacher_observation
from reasoning_v3_test_utils import decision_rationale_payload, observation_payload


class _SequenceBackend:
    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.calls = 0

    async def generate(self, source, **kwargs):
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return TeacherGeneration(answer=answer, provider="fake", model="fake-vlm")


def _source():
    return build_reasoning_source_v3(
        {
            "id": "batch-v3-2",
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


def _answer(source) -> str:
    payload = observation_payload(source, corridor="corridor.right", direction="right")
    return json.dumps(payload)


def _rationale_answer(source, *, wrong_outcome: bool = False) -> str:
    parsed = parse_reasoning_v3_teacher_observation(_answer(source))
    assert parsed.observation is not None, parsed.errors
    payload = decision_rationale_payload(source, parsed.observation)
    if wrong_outcome:
        payload["outcome_motion"] = "go_straight"
    return json.dumps(payload)


def test_v3_batch_repairs_invalid_hindsight_outcome() -> None:
    source = _source()
    backend = _SequenceBackend([
        _answer(source),
        _rationale_answer(source, wrong_outcome=True),
        _rationale_answer(source),
    ])

    annotation = asyncio.run(
        annotate_reasoning_source_v3(
            source,
            backend=backend,
            config=ReasoningV3BatchConfig(max_retries=1),
        )
    )

    assert annotation.status == "accepted"
    assert annotation.attempts == 3
    assert annotation.target is not None
    assert annotation.target.reasoning.decision.motion == "turn_right"
    assert annotation.observation is not None
    assert annotation.label_provenance["demonstrated_direction"].strength == "strong"


def test_v3_batch_rejects_missing_context_source_reference() -> None:
    source = _source()
    payload = json.loads(_answer(source))
    payload["navigation_context_grounding"]["source_groundings"][0]["evidence_refs"] = [
        "fpv.current",
        "corridor.right",
    ]
    annotation = asyncio.run(
        annotate_reasoning_source_v3(
            source,
            backend=_SequenceBackend([json.dumps(payload)]),
            config=ReasoningV3BatchConfig(max_retries=0),
        )
    )

    assert annotation.status == "rejected"
    assert any("must cite source_ref" in error for error in annotation.errors)


def test_v3_manifest_cache_uses_source_fingerprint(tmp_path) -> None:
    source = _source()
    source_jsonl = tmp_path / "source.jsonl"
    annotation_jsonl = tmp_path / "annotations.jsonl"
    source_jsonl.write_text(json.dumps(source.model_dump(mode="json")) + "\n", encoding="utf-8")
    backend = _SequenceBackend([_answer(source), _rationale_answer(source)])

    def factory(worker_id, key):
        return backend

    first = asyncio.run(
        label_reasoning_source_manifest_v3(
            source_jsonl,
            annotation_jsonl,
            config=ReasoningV3BatchConfig(concurrency=2, progress_interval=0),
            backend_factory=factory,
        )
    )
    second = asyncio.run(
        label_reasoning_source_manifest_v3(
            source_jsonl,
            annotation_jsonl,
            config=ReasoningV3BatchConfig(concurrency=2, progress_interval=0),
            backend_factory=factory,
        )
    )

    rows = [ReasoningAnnotationV3.model_validate_json(line) for line in annotation_jsonl.read_text().splitlines()]
    assert first["accepted_rows"] == 1
    assert second["pending_rows"] == 0
    assert second["resume_counts"]["cached_accepted"] == 1
    assert backend.calls == 2
    assert len(rows) == 1


def test_v3_manifest_cache_relabels_accepted_rows_from_old_prompt(tmp_path) -> None:
    source = _source()
    source_jsonl = tmp_path / "source.jsonl"
    annotation_jsonl = tmp_path / "annotations.jsonl"
    source_jsonl.write_text(json.dumps(source.model_dump(mode="json")) + "\n", encoding="utf-8")

    stale = asyncio.run(
        annotate_reasoning_source_v3(
            source,
            backend=_SequenceBackend([_answer(source), _rationale_answer(source)]),
            config=ReasoningV3BatchConfig(max_retries=0),
        )
    )
    stale_metadata = dict(stale.metadata)
    stale_metadata["teacher_prompt_version"] = "urbannav_reasoning_v3_teacher_v1"
    stale = stale.model_copy(update={"metadata": stale_metadata})
    annotation_jsonl.write_text(stale.model_dump_json() + "\n", encoding="utf-8")

    backend = _SequenceBackend([_answer(source), _rationale_answer(source)])
    report = asyncio.run(
        label_reasoning_source_manifest_v3(
            source_jsonl,
            annotation_jsonl,
            config=ReasoningV3BatchConfig(concurrency=1, progress_interval=0),
            backend_factory=lambda worker_id, key: backend,
        )
    )

    rows = [ReasoningAnnotationV3.model_validate_json(line) for line in annotation_jsonl.read_text().splitlines()]
    assert report["accepted_rows"] == 1
    assert report["resume_counts"]["cache_stale_prompt_version"] == 1
    assert rows[-1].metadata["teacher_prompt_version"] == REASONING_V3_TEACHER_PROMPT_VERSION
    assert backend.calls == 2


def test_v3_manifest_cache_relabels_changed_prompt_digest(tmp_path) -> None:
    source = _source()
    source_jsonl = tmp_path / "source.jsonl"
    annotation_jsonl = tmp_path / "annotations.jsonl"
    source_jsonl.write_text(json.dumps(source.model_dump(mode="json")) + "\n", encoding="utf-8")
    stale = asyncio.run(
        annotate_reasoning_source_v3(
            source,
            backend=_SequenceBackend([_answer(source), _rationale_answer(source)]),
            config=ReasoningV3BatchConfig(max_retries=0),
        )
    )
    stale_metadata = dict(stale.metadata)
    stale_metadata["teacher_prompt_policy_digest"] = "0" * 64
    annotation_jsonl.write_text(
        stale.model_copy(update={"metadata": stale_metadata}).model_dump_json() + "\n",
        encoding="utf-8",
    )

    backend = _SequenceBackend([_answer(source), _rationale_answer(source)])
    report = asyncio.run(
        label_reasoning_source_manifest_v3(
            source_jsonl,
            annotation_jsonl,
            config=ReasoningV3BatchConfig(concurrency=1, progress_interval=0),
            backend_factory=lambda worker_id, key: backend,
        )
    )

    assert report["accepted_rows"] == 1
    assert report["resume_counts"]["cache_stale_prompt_policy_digest"] == 1
    assert backend.calls == 2
