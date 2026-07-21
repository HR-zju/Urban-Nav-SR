from __future__ import annotations

import json

from mcp_server.contracts.reasoning_dataset_v3 import ReasoningAnnotationV3
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_audit import audit_reasoning_v3_artifacts
from mcp_server.data.reasoning_v3_source import build_reasoning_source_v3
from mcp_server.data.reasoning_v3_views import export_reasoning_training_views_v3
from tools.model_processing.build_reasoning_v3_review_html import build_reasoning_v3_review_html
from reasoning_v3_test_utils import accepted_annotation


def test_v3_export_audit_and_self_contained_review(tmp_path) -> None:
    fpv = tmp_path / "fpv.jpg"
    bev = tmp_path / "bev.jpg"
    fpv.write_bytes(b"fake-jpeg-fpv")
    bev.write_bytes(b"fake-jpeg-bev")
    source = build_reasoning_source_v3(
        {
            "id": "artifacts-v3-2",
            "dataset_source": "sidewalk",
            "images": [str(fpv), str(bev)],
            "image_roles": ["fpv.current", "bev.current"],
            "instruction": "Turn right toward the entrance.",
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [2.0, -1.0, -0.2]],
            "trajectory_fpv_t": [[0.5, 0.9], [0.8, 0.7]],
            "trajectory_fpv_coord_frame": "normalized_0_1",
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="sidewalk"),
    )
    annotation = accepted_annotation(source, corridor="corridor.right", direction="right")
    source_path = tmp_path / "sources.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    view_path = tmp_path / "views.jsonl"
    source_path.write_text(json.dumps(source.model_dump(mode="json")) + "\n", encoding="utf-8")
    annotation_path.write_text(json.dumps(annotation.model_dump(mode="json")) + "\n", encoding="utf-8")

    export = export_reasoning_training_views_v3(source_path, annotation_path, view_path)
    audit = audit_reasoning_v3_artifacts(source_path, annotation_path, view_path)
    review_path = tmp_path / "review.html"
    review = build_reasoning_v3_review_html(source_path, annotation_path, review_path, per_stratum=1)

    assert export["counts"]["accepted_sources"] == 1
    assert audit["passed"] is True
    assert audit["trajectory_qc_strong_direction_coverage"] == 1.0
    assert review["missing_images"] == 0
    assert "data:image/jpeg;base64," in review_path.read_text(encoding="utf-8")


def test_v3_review_includes_latest_rejected_annotation(tmp_path) -> None:
    fpv = tmp_path / "fpv.jpg"
    fpv.write_bytes(b"fake-jpeg-fpv")
    source = build_reasoning_source_v3(
        {
            "id": "artifacts-v3-2-rejected",
            "dataset_source": "egowalk",
            "images": [str(fpv)],
            "image_roles": ["fpv.current"],
            "instruction": "Continue along the visible path.",
            "trajectory_ego_t": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        },
        index=0,
        config=ReasoningSourceBuildConfig(dataset_source="egowalk"),
    )
    rejected = ReasoningAnnotationV3(
        source_group_id=source.model_input.source_group_id,
        source_fingerprint=source.source_fingerprint,
        status="rejected",
        errors=["composition_failed:demonstrated_direction_corridor_missing_from_visual_support"],
        attempts=3,
        raw_answer="{}",
        metadata={"repair_history": [{"attempt": 1, "errors": ["teacher_schema"]}]},
    )
    source_path = tmp_path / "sources.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    review_path = tmp_path / "review.html"
    source_path.write_text(json.dumps(source.model_dump(mode="json")) + "\n", encoding="utf-8")
    annotation_path.write_text(json.dumps(rejected.model_dump(mode="json")) + "\n", encoding="utf-8")

    report = build_reasoning_v3_review_html(source_path, annotation_path, review_path, per_stratum=1)

    page = review_path.read_text(encoding="utf-8")
    assert report["selected_rows"] == 1
    assert report["status_counts"] == {"rejected": 1}
    assert report["quality_strata"] == {"rejected_composition": 1}
    assert "Rejected / error audit" in page
    assert "demonstrated_direction_corridor_missing_from_visual_support" in page
