from __future__ import annotations

import json

from mcp_server.contracts.reasoning_dataset_v3 import (
    ReasoningAnnotationV3,
    ReasoningTrainingViewV3,
)
from mcp_server.data.reasoning_v3_audit import audit_reasoning_v3_artifacts
from mcp_server.data.reasoning_v3_views import export_reasoning_training_views_v3
from mcp_server.teacher.reasoning_v3_teacher import parse_reasoning_v3_teacher_observation
from reasoning_v3_test_utils import (
    accepted_annotation,
    build_test_source,
    observation_payload,
)
from tools.model_processing.build_reasoning_v3_review_html import (
    build_reasoning_v3_review_html,
)


def test_v3_4_export_audit_and_review_cover_accepted_and_observation_only(
    tmp_path,
) -> None:
    accepted_image = tmp_path / "accepted.jpg"
    observation_image = tmp_path / "observation.jpg"
    accepted_image.write_bytes(b"fake-jpeg-accepted")
    observation_image.write_bytes(b"fake-jpeg-observation")
    accepted_source = build_test_source(
        sample_id="artifact-accepted",
        include_bev=False,
        extra_row={"images": [str(accepted_image)], "image_roles": ["fpv.current"]},
    )
    observation_source = build_test_source(
        sample_id="artifact-observation",
        include_direction=False,
        include_bev=False,
        extra_row={"images": [str(observation_image)], "image_roles": ["fpv.current"]},
    )
    accepted = accepted_annotation(accepted_source)
    observation_only = _observation_only_annotation(observation_source)
    source_path = tmp_path / "sources.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    view_path = tmp_path / "views.jsonl"
    review_path = tmp_path / "review.html"
    source_path.write_text(
        accepted_source.model_dump_json()
        + "\n"
        + observation_source.model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    annotation_path.write_text(
        accepted.model_dump_json()
        + "\n"
        + observation_only.model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    export = export_reasoning_training_views_v3(
        source_path,
        annotation_path,
        view_path,
    )
    audit = audit_reasoning_v3_artifacts(
        source_path,
        annotation_path,
        view_path,
        required_dataset_families=("sidewalk",),
    )
    review = build_reasoning_v3_review_html(
        source_path,
        annotation_path,
        review_path,
        per_stratum=2,
    )

    assert export["counts"]["annotation:accepted"] == 1
    assert export["counts"]["annotation:observation_only"] == 1
    assert export["counts"]["observation_only_without_view"] == 1
    assert audit["passed"] is True, audit["violations"]
    assert audit["availability"]["accepted_target_fraction"] == 0.5
    assert audit["availability"]["observation_only_fraction"] == 0.5
    assert audit["availability"]["valid_annotation_fraction"] == 1.0
    assert audit["artifact_schema_versions"] == {
        "source": "urbannav_reasoning_source_v3_4",
        "annotation": "urbannav_reasoning_annotation_v3_4",
        "training_view": "urbannav_reasoning_training_view_v3_4",
    }
    assert review["missing_images"] == 0
    assert review["status_counts"] == {"accepted": 1, "observation_only": 1}
    page = review_path.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in page
    assert "observation_only_no_direction_gt" in page


def test_v3_4_audit_rejects_reasoning_lm_loss_even_for_valid_target(tmp_path) -> None:
    source = build_test_source(sample_id="artifact-invalid-loss")
    annotation = accepted_annotation(source)
    source_path = tmp_path / "sources.jsonl"
    annotation_path = tmp_path / "annotations.jsonl"
    view_path = tmp_path / "views.jsonl"
    source_path.write_text(source.model_dump_json() + "\n", encoding="utf-8")
    annotation_path.write_text(annotation.model_dump_json() + "\n", encoding="utf-8")
    export_reasoning_training_views_v3(source_path, annotation_path, view_path)
    view = ReasoningTrainingViewV3.model_validate_json(
        view_path.read_text(encoding="utf-8")
    )
    invalid = view.model_copy(
        update={
            "loss_mask": {
                **view.loss_mask,
                "lm": True,
            }
        }
    )
    view_path.write_text(invalid.model_dump_json() + "\n", encoding="utf-8")

    audit = audit_reasoning_v3_artifacts(source_path, annotation_path, view_path)

    assert audit["passed"] is False
    assert any(
        item["code"] == "training_view_forbidden_loss:lm"
        for item in audit["violations"]
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
        metadata={
            "annotation_mode": "observation_only",
            "observation_only_reason": "canonical_direction_missing",
        },
    )
