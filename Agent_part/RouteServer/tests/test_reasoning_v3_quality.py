from __future__ import annotations

import json

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningAnnotationV3
from mcp_server.data.reasoning_v3_quality import (
    ReasoningV3QualityThresholds,
    evaluate_reasoning_v3_quality,
)
from mcp_server.teacher.reasoning_v3_teacher import parse_reasoning_v3_teacher_observation
from reasoning_v3_test_utils import (
    accepted_annotation,
    build_test_source,
    observation_payload,
)


def test_v3_4_quality_counts_accepted_and_observation_only_as_valid_artifacts(
    tmp_path,
) -> None:
    accepted_source = build_test_source(sample_id="quality-accepted")
    observation_source = build_test_source(
        sample_id="quality-observation-only",
        include_direction=False,
    )
    accepted = accepted_annotation(accepted_source)
    observation_only = _observation_only_annotation(observation_source)
    source_path, annotation_path = _write_artifacts(
        tmp_path,
        (accepted_source, observation_source),
        (accepted, observation_only),
    )

    report = evaluate_reasoning_v3_quality(
        source_path,
        annotation_path,
        required_dataset_families=("sidewalk",),
    )

    assert report["passed"] is True
    assert report["report_semantics"].startswith("availability_only")
    overall = report["overall"]
    assert overall["fractions"]["valid_annotation_fraction"] == 1.0
    assert overall["fractions"]["accepted_target_fraction"] == 0.5
    assert overall["fractions"]["observation_only_fraction"] == 0.5
    assert overall["fractions"]["failed_annotation_fraction"] == 0.0
    assert overall["counts"]["direction_available_rows"] == 1
    assert overall["counts"]["system1_available_rows"] == 2


def test_v3_4_quality_gates_only_completion_and_failure_fractions(tmp_path) -> None:
    accepted_source = build_test_source(sample_id="quality-gate-accepted")
    failed_source = build_test_source(sample_id="quality-gate-failed")
    accepted = accepted_annotation(accepted_source)
    rejected = ReasoningAnnotationV3(
        source_group_id=failed_source.model_input.source_group_id,
        source_fingerprint=failed_source.source_fingerprint,
        status="rejected",
        errors=["teacher_schema"],
        attempts=1,
    )
    source_path, annotation_path = _write_artifacts(
        tmp_path,
        (accepted_source, failed_source),
        (accepted, rejected),
    )

    report = evaluate_reasoning_v3_quality(
        source_path,
        annotation_path,
        thresholds=ReasoningV3QualityThresholds(
            min_acceptance_fraction=0.75,
            max_reject_fraction=0.20,
            max_conflict_fraction=0.0,
            max_unknown_fraction=0.0,
            max_insufficient_fraction=0.0,
            min_projection_qc_coverage=1.0,
        ),
    )

    assert report["thresholds"] == {
        "min_valid_annotation_fraction": 0.75,
        "max_failed_annotation_fraction": 0.20,
    }
    assert any("valid_annotation_fraction_below_min" in item for item in report["violations"])
    assert any("failed_annotation_fraction_above_max" in item for item in report["violations"])
    assert not any("projection" in item or "conflict" in item for item in report["violations"])


def test_v3_4_quality_rejects_weak_language_loss_provenance(tmp_path) -> None:
    source = build_test_source(sample_id="quality-weak-lm")
    annotation = accepted_annotation(source)
    weak = annotation.label_provenance["task_grounding"].model_copy(
        update={"loss_channel": "weak_lm"}
    )
    annotation = annotation.model_copy(
        update={
            "label_provenance": {
                **annotation.label_provenance,
                "task_grounding": weak,
            }
        }
    )
    source_path, annotation_path = _write_artifacts(
        tmp_path,
        (source,),
        (annotation,),
    )

    report = evaluate_reasoning_v3_quality(source_path, annotation_path)

    assert report["passed"] is False
    assert any(
        item.startswith("annotation_weak_lm_provenance:")
        for item in report["violations"]
    )


def _observation_only_annotation(source) -> ReasoningAnnotationV3:
    observation = parse_reasoning_v3_teacher_observation(
        json.dumps(observation_payload(source))
    ).observation
    assert observation is not None
    return ReasoningAnnotationV3(
        source_group_id=source.model_input.source_group_id,
        source_fingerprint=source.source_fingerprint,
        status="observation_only",
        observation=observation,
        attempts=1,
        metadata={"annotation_mode": "observation_only"},
    )


def _write_artifacts(tmp_path, sources, annotations):
    source_path = tmp_path / "sources.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    source_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in sources),
        encoding="utf-8",
    )
    annotation_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in annotations),
        encoding="utf-8",
    )
    return source_path, annotation_path
