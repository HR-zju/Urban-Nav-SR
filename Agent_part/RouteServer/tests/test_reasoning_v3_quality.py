from __future__ import annotations

import json

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningAnnotationV3
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_quality import (
    ReasoningV3QualityThresholds,
    evaluate_reasoning_v3_quality,
)
from mcp_server.data.reasoning_v3_source import build_reasoning_source_v3
from reasoning_v3_test_utils import accepted_annotation


def test_v3_quality_report_uses_latest_status_and_denominators(tmp_path) -> None:
    accepted_source = _source("quality-accepted", projection_qc=True)
    rejected_source = _source("quality-rejected", projection_qc=False)
    accepted = accepted_annotation(accepted_source, corridor="corridor.right", direction="right")
    rejected = ReasoningAnnotationV3(
        source_group_id=rejected_source.model_input.source_group_id,
        source_fingerprint=rejected_source.source_fingerprint,
        status="rejected",
        errors=["demonstrated_direction_conflicts_with_visual_observation"],
        attempts=2,
    )
    source_path = tmp_path / "sources.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    source_path.write_text(
        "".join(json.dumps(item.model_dump(mode="json")) + "\n" for item in (accepted_source, rejected_source)),
        encoding="utf-8",
    )
    annotation_path.write_text(
        "".join(json.dumps(item.model_dump(mode="json")) + "\n" for item in (accepted, rejected)),
        encoding="utf-8",
    )
    permissive = ReasoningV3QualityThresholds(
        min_acceptance_fraction=0.50,
        max_reject_fraction=0.50,
        max_conflict_fraction=1.0,
        max_unknown_fraction=1.0,
        max_insufficient_fraction=1.0,
        min_projection_qc_coverage=0.50,
    )
    report = evaluate_reasoning_v3_quality(
        source_path,
        annotation_path,
        thresholds=permissive,
        required_dataset_families=("sidewalk",),
    )
    sidewalk = report["per_dataset"]["sidewalk"]
    assert report["passed"] is True
    assert sidewalk["fractions"]["acceptance_fraction"] == 0.5
    assert sidewalk["fractions"]["reject_fraction"] == 0.5
    assert sidewalk["fractions"]["projection_qc_coverage"] == 0.5
    assert sidewalk["counts"]["failed_conflict_rows"] == 1

    strict = evaluate_reasoning_v3_quality(source_path, annotation_path)
    assert strict["passed"] is False
    assert any("acceptance_fraction_below_min" in item for item in strict["violations"])
    assert any("reject_fraction_above_max" in item for item in strict["violations"])


def test_v3_quality_report_counts_unknown_and_insufficient(tmp_path) -> None:
    source = _source("quality-unknown", projection_qc=True)
    annotation = accepted_annotation(source, corridor="corridor.right", direction="right")
    assert annotation.target is not None
    trace = annotation.target.reasoning
    reasoning = trace.model_copy(
        update={
            "local_traversability": trace.local_traversability.model_copy(
                update={"visibility": "unknown"}
            ),
            "navigation_context_grounding": trace.navigation_context_grounding.model_copy(
                update={
                    "source_groundings": [
                        item.model_copy(
                            update={
                                "directive": "unknown",
                                "target_corridor_ref": None,
                                "status": "insufficient_evidence",
                            }
                        )
                        for item in trace.navigation_context_grounding.source_groundings
                    ],
                    "fused_direction": "unknown",
                    "fused_target_corridor_ref": None,
                    "source_consistency": "unknown",
                    "fused_execution_status": "insufficient_evidence",
                }
            ),
        }
    )
    annotation = annotation.model_copy(
        update={"target": annotation.target.model_copy(update={"reasoning": reasoning})}
    )
    source_path = tmp_path / "sources.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    source_path.write_text(json.dumps(source.model_dump(mode="json")) + "\n", encoding="utf-8")
    annotation_path.write_text(json.dumps(annotation.model_dump(mode="json")) + "\n", encoding="utf-8")
    permissive = ReasoningV3QualityThresholds(
        min_acceptance_fraction=1.0,
        max_reject_fraction=0.0,
        max_conflict_fraction=1.0,
        max_unknown_fraction=1.0,
        max_insufficient_fraction=1.0,
        min_projection_qc_coverage=1.0,
    )
    report = evaluate_reasoning_v3_quality(source_path, annotation_path, thresholds=permissive)
    fractions = report["per_dataset"]["sidewalk"]["fractions"]
    assert fractions["critical_unknown_fraction_among_accepted"] == 1.0
    assert fractions["insufficient_evidence_fraction_among_accepted"] == 1.0


def _source(sample_id: str, *, projection_qc: bool):
    row = {
        "id": sample_id,
        "dataset_source": "sidewalk",
        "images": [f"{sample_id}.jpg"],
        "image_roles": ["fpv.current"],
        "instruction": "Turn right toward the entrance.",
        "trajectory_ego_t": [[0.0, 0.0, 0.0], [2.0, -1.0, -0.2]],
        "trajectory_fpv_t": [[0.5, 0.9], [0.8, 0.7]],
        "trajectory_fpv_coord_frame": "normalized_0_1",
    }
    if projection_qc:
        row["fpv_projection_qc_passed"] = True
    return build_reasoning_source_v3(
        row,
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )
