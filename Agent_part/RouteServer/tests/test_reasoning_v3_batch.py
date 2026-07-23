from __future__ import annotations

import asyncio
import json

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningAnnotationV3
from mcp_server.teacher.reasoning_v2_teacher import TeacherGeneration
from mcp_server.teacher.reasoning_v3_batch import (
    ReasoningV3BatchConfig,
    annotate_reasoning_source_v3,
    label_reasoning_source_manifest_v3,
)
from mcp_server.teacher.reasoning_v3_teacher import parse_reasoning_v3_teacher_observation
from reasoning_v3_test_utils import (
    build_test_source,
    decision_rationale_payload,
    observation_payload,
)


class _SequenceBackend:
    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.calls = 0

    async def generate(self, source, **kwargs):
        answer = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return TeacherGeneration(
            answer=answer,
            provider="fake",
            model="fake-vlm",
            telemetry={"call": self.calls},
        )


def _observation_answer(source) -> str:
    return json.dumps(observation_payload(source))


def _rationale_answer(source, *, label_index_offset: int = 0) -> str:
    observation = parse_reasoning_v3_teacher_observation(
        _observation_answer(source)
    ).observation
    assert observation is not None
    payload = decision_rationale_payload(source, observation)
    payload["direction_label_index"] += label_index_offset
    return json.dumps(payload)


def test_v3_4_batch_accepted_row_uses_exactly_two_stages() -> None:
    source = build_test_source(sample_id="batch-accepted")
    backend = _SequenceBackend([
        _observation_answer(source),
        _rationale_answer(source),
    ])

    annotation = asyncio.run(
        annotate_reasoning_source_v3(
            source,
            backend=backend,
            config=ReasoningV3BatchConfig(max_retries=0),
        )
    )

    assert annotation.status == "accepted"
    assert annotation.attempts == backend.calls == 2
    assert annotation.observation is not None
    assert annotation.decision_rationale is not None
    assert annotation.target is not None
    assert annotation.section_loss_mask is not None
    assert annotation.metadata["annotation_mode"] == "full_cot"
    assert annotation.target.reasoning.decision.label_index == 12


def test_v3_4_batch_repairs_direction_identity_drift_without_changing_gt() -> None:
    source = build_test_source(sample_id="batch-repair")
    backend = _SequenceBackend([
        _observation_answer(source),
        _rationale_answer(source, label_index_offset=1),
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
    assert annotation.attempts == backend.calls == 3
    assert annotation.target is not None
    assert annotation.target.reasoning.decision.label_index == 12
    assert any(
        "rationale_direction_identity_mismatch" in error
        for attempt in annotation.metadata["repair_history"]
        for error in attempt["errors"]
    )


def test_v3_4_batch_missing_direction_is_observation_only_and_skips_pass2() -> None:
    source = build_test_source(
        sample_id="batch-missing-direction",
        include_direction=False,
    )
    backend = _SequenceBackend([_observation_answer(source)])

    annotation = asyncio.run(
        annotate_reasoning_source_v3(
            source,
            backend=backend,
            config=ReasoningV3BatchConfig(max_retries=0),
        )
    )

    assert annotation.status == "observation_only"
    assert annotation.attempts == backend.calls == 1
    assert annotation.observation is not None
    assert annotation.decision_rationale is None
    assert annotation.target is None
    assert annotation.section_loss_mask is None
    assert annotation.metadata["observation_only_reason"] == "canonical_direction_missing"


def test_v3_4_batch_frodobots_is_observation_only_and_skips_pass2() -> None:
    source = build_test_source(
        sample_id="batch-frodobots",
        dataset_source="frodobots",
        include_direction=True,
        include_trajectory=True,
    )
    backend = _SequenceBackend([_observation_answer(source)])

    annotation = asyncio.run(
        annotate_reasoning_source_v3(
            source,
            backend=backend,
            config=ReasoningV3BatchConfig(max_retries=0),
        )
    )

    assert annotation.status == "observation_only"
    assert annotation.attempts == backend.calls == 1
    assert annotation.decision_rationale is None
    assert annotation.target is None
    assert annotation.section_loss_mask is None
    assert (
        annotation.metadata["observation_only_reason"]
        == "direction_supervision_disabled_by_dataset_policy"
    )


def test_v3_4_manifest_cache_completes_accepted_and_observation_only_rows(
    tmp_path,
) -> None:
    accepted_source = build_test_source(sample_id="cache-accepted")
    observation_source = build_test_source(
        sample_id="cache-observation",
        include_direction=False,
    )
    source_path = tmp_path / "sources.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    source_path.write_text(
        "".join(
            item.model_dump_json() + "\n"
            for item in (accepted_source, observation_source)
        ),
        encoding="utf-8",
    )
    backend = _SequenceBackend(
        [
            _observation_answer(accepted_source),
            _rationale_answer(accepted_source),
            _observation_answer(observation_source),
        ]
    )
    factory = lambda worker_id, key: backend

    first = asyncio.run(
        label_reasoning_source_manifest_v3(
            source_path,
            annotation_path,
            config=ReasoningV3BatchConfig(
                max_retries=0,
                concurrency=1,
                progress_interval=0,
            ),
            backend_factory=factory,
        )
    )
    second = asyncio.run(
        label_reasoning_source_manifest_v3(
            source_path,
            annotation_path,
            config=ReasoningV3BatchConfig(
                max_retries=0,
                concurrency=1,
                progress_interval=0,
            ),
            backend_factory=factory,
        )
    )

    assert first["accepted_rows"] == 1
    assert first["observation_only_rows"] == 1
    assert second["pending_rows"] == 0
    assert second["resume_counts"] == {
        "cached_accepted": 1,
        "cached_observation_only": 1,
    }
    assert backend.calls == 3
    rows = [
        ReasoningAnnotationV3.model_validate_json(line)
        for line in annotation_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row.status for row in rows] == ["accepted", "observation_only"]


def test_v3_4_rejected_mirror_excludes_observation_only_rows(tmp_path) -> None:
    observation_source = build_test_source(
        sample_id="mirror-observation",
        include_direction=False,
    )
    rejected_source = build_test_source(sample_id="mirror-rejected")
    source_path = tmp_path / "sources.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    source_path.write_text(
        observation_source.model_dump_json()
        + "\n"
        + rejected_source.model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    backend = _SequenceBackend([
        _observation_answer(observation_source),
        "not-json",
    ])

    report = asyncio.run(
        label_reasoning_source_manifest_v3(
            source_path,
            annotation_path,
            rejected_jsonl=rejected_path,
            config=ReasoningV3BatchConfig(
                max_retries=0,
                concurrency=1,
                progress_interval=0,
            ),
            backend_factory=lambda worker_id, key: backend,
        )
    )

    mirrored = [
        ReasoningAnnotationV3.model_validate_json(line)
        for line in rejected_path.read_text(encoding="utf-8").splitlines()
    ]
    assert report["observation_only_rows"] == 1
    assert report["rejected_rows"] == 1
    assert [row.status for row in mirrored] == ["rejected"]
