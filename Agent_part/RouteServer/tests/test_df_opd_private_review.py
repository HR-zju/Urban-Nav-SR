"""CPU tests for private four-view DF-OPD review bundle rendering."""

from __future__ import annotations

import json

from mcp_server.contracts.df_opd import (
    DFOPDCanonicalSampleV1,
    DFOPDProfileV1,
    DFOPDTeacherViewV1,
    DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION,
    DF_OPD_ORACLE_CUE_ROLE,
    DF_OPD_PROFILE_2S16,
    DF_OPD_TEACHER_CUE_ROLE,
    df_opd_digest_v1,
)
from mcp_server.contracts.df_opd_certification import DFOPDTeacherCertificationScoreV1
from mcp_server.contracts.df_opd_review import DFOPDTeacherReviewPredictionV1
from mcp_server.data.df_opd.review import build_df_opd_private_review_bundle_v1
from mcp_server.data.df_opd.student_review import build_df_opd_student_review_bundle_v1
from tools.evaluation.evaluate_df_opd_student import validate_planned_visualization_manifest
from mcp_server.data.io.jsonl import file_sha256, write_jsonl_atomic


_POLICY_DIGEST = "a" * 64
_DATASET_DIGEST = "b" * 64
_CHECKPOINT_DIGEST = "c" * 64
_PREDICTION_DIGEST = "d" * 64
_CONDITIONS = (
    "same_input_t0",
    "correct_privilege_tc",
    "shuffled_privilege_ts",
    "exact_future_oracle_to",
)


def _canonical(tmp_path) -> DFOPDCanonicalSampleV1:
    """Build one complete four-view canonical sample backed by local files."""

    for name in ("current.jpg", "cue.png", "ts-cue.png", "oracle.png"):
        (tmp_path / name).write_bytes(name.encode("utf-8"))
    profile = DFOPDProfileV1(profile_id=DF_OPD_PROFILE_2S16, target_horizon_steps=16)
    images = ("current.jpg",)
    roles = ("fpv.current",)
    views = (
        DFOPDTeacherViewV1(
            condition="same_input_t0",
            images=images,
            image_roles=roles,
            privilege_policy_digest=_POLICY_DIGEST,
        ),
        DFOPDTeacherViewV1(
            condition="correct_privilege_tc",
            images=(*images, "cue.png"),
            image_roles=(*roles, DF_OPD_TEACHER_CUE_ROLE),
            cue_class="go_straight",
            privilege_policy_digest=_POLICY_DIGEST,
            cue_artifact_sha256=file_sha256(tmp_path / "cue.png"),
        ),
        DFOPDTeacherViewV1(
            condition="shuffled_privilege_ts",
            images=(*images, "ts-cue.png"),
            image_roles=(*roles, DF_OPD_TEACHER_CUE_ROLE),
            cue_class="turn_left",
            privilege_policy_digest=_POLICY_DIGEST,
            cue_artifact_sha256=file_sha256(tmp_path / "ts-cue.png"),
        ),
        DFOPDTeacherViewV1(
            condition="exact_future_oracle_to",
            images=(*images, "oracle.png"),
            image_roles=(*roles, DF_OPD_ORACLE_CUE_ROLE),
            cue_class="oracle",
            privilege_policy_digest=_POLICY_DIGEST,
            cue_artifact_sha256=file_sha256(tmp_path / "oracle.png"),
        ),
    )
    payload = {
        "schema_version": DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION,
        "sample_id": "sample-1",
        "source_group_id": "group-1",
        "split_group_id": "split-1",
        "dataset_source": "sidewalk",
        "profile": profile,
        "images": images,
        "image_roles": roles,
        "decision_target": (
            '<Decision>{"selected_action_label":"straight","speed":"normal",'
            '"response":"proceed"}</Decision>'
        ),
        "trajectory_ego_t": tuple((float(index), 0.0, 0.0) for index in range(16)),
        "action_mask": (True,) * 16,
        "teacher_views": views,
    }
    serialized = {
        key: value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        for key, value in payload.items()
    }
    serialized["teacher_views"] = [view.model_dump(mode="json") for view in views]
    return DFOPDCanonicalSampleV1(**payload, canonical_digest=df_opd_digest_v1(serialized))


def test_private_review_bundle_links_local_media_and_never_becomes_public(tmp_path):
    sample = _canonical(tmp_path)
    score_rows = []
    prediction_rows = []
    for condition in _CONDITIONS:
        score_rows.append(
            DFOPDTeacherCertificationScoreV1(
                sample_id=sample.sample_id,
                source_group_id=sample.source_group_id,
                canonical_digest=sample.canonical_digest,
                profile_id=sample.profile.profile_id,
                profile_digest=sample.profile.profile_digest,
                dataset_fingerprint=_DATASET_DIGEST,
                checkpoint_digest=_CHECKPOINT_DIGEST,
                prediction_manifest_digest=_PREDICTION_DIGEST,
                seed=17,
                condition=condition,
                decision_strict_parse=True,
                decision_correct=True,
                trajectory_valid=True,
                trajectory_ade_m=0.1,
                trajectory_fde_m=0.2,
            )
        )
        prediction_rows.append(
            DFOPDTeacherReviewPredictionV1(
                sample_id=sample.sample_id,
                canonical_digest=sample.canonical_digest,
                profile_id=sample.profile.profile_id,
                profile_digest=sample.profile.profile_digest,
                condition=condition,
                decision_text=sample.decision_target,
                trajectory_ego_t=sample.trajectory_ego_t,
                trajectory_valid=True,
            )
        )
    canonical_path = tmp_path / "canonical.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    html_path = tmp_path / "review.html"
    manifest_path = tmp_path / "review.manifest.json"
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(scores_path, [row.model_dump(mode="json") for row in score_rows])
    write_jsonl_atomic(predictions_path, [row.model_dump(mode="json") for row in prediction_rows])

    receipt = build_df_opd_private_review_bundle_v1(
        canonical_path,
        scores_path,
        predictions_path,
        html_path,
        manifest_path,
    )

    page = html_path.read_text(encoding="utf-8")
    assert receipt["private_artifact"] is True
    assert "file:///" in page
    assert "Private DF-OPD four-view review" in page
    assert "exact oracle" in page
    assert "do not publish" in page


def test_student_review_bundle_renders_gt_and_student_trajectory(tmp_path):
    """Student review keeps the canonical identity and renders both trajectories."""

    sample = _canonical(tmp_path)
    canonical_path = tmp_path / "canonical.jsonl"
    prediction_path = tmp_path / "student-predictions.jsonl"
    html_path = tmp_path / "student-review.html"
    manifest_path = tmp_path / "student-review.manifest.json"
    visualization_index_path = tmp_path / "visualization_index.json"
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(prediction_path, [{
        "sample_id": sample.sample_id,
        "canonical_digest": sample.canonical_digest,
        "df_opd_profile_id": sample.profile.profile_id,
        "df_opd_profile_digest": sample.profile.profile_digest,
        "decision": sample.decision_target,
        "trajectory_ego_t": [list(point) for point in sample.trajectory_ego_t],
        "latency_ms": 10.0,
    }])

    receipt = build_df_opd_student_review_bundle_v1(
        canonical_path, [prediction_path], html_path, manifest_path, media_root=tmp_path,
        visualization_index_json=visualization_index_path, run_id="run-1", run_root=tmp_path,
    )

    page = html_path.read_text(encoding="utf-8")
    visualization_index = json.loads(visualization_index_path.read_text(encoding="utf-8"))
    assert receipt["private_artifact"] is True
    assert "Private DF-OPD Student review" in page
    assert "GT (green) / Student (blue)" in page
    assert "ADE (m)" in page
    assert "FDE (m)" in page
    assert "ADE_T" in page
    assert "file:///" in page
    assert visualization_index["phase"] == "finished"
    assert visualization_index["run_id"] == "run-1"
    assert {item["id"] for item in visualization_index["artifacts"]} == {
        "student_review_html",
    }
    assert receipt["visualization_index"] == str(visualization_index_path.resolve())


def test_student_review_bundle_arguments_are_all_bound_to_finished_index(tmp_path):
    """The evaluation output contract binds scorecard, review, receipt, and index."""

    sample = _canonical(tmp_path)
    canonical_path = tmp_path / "canonical.jsonl"
    prediction_path = tmp_path / "student-predictions.jsonl"
    html_path = tmp_path / "student-review.html"
    receipt_path = tmp_path / "student-review.receipt.json"
    index_path = tmp_path / "visualization_index.json"
    scorecard_path = tmp_path / "scorecard.json"
    scorecard_path.write_text("{}\n", encoding="utf-8")
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(prediction_path, [{
        "sample_id": sample.sample_id,
        "canonical_digest": sample.canonical_digest,
        "df_opd_profile_id": sample.profile.profile_id,
        "df_opd_profile_digest": sample.profile.profile_digest,
        "decision": sample.decision_target,
        "trajectory_ego_t": [list(point) for point in sample.trajectory_ego_t],
    }])

    receipt = build_df_opd_student_review_bundle_v1(
        canonical_path, [prediction_path], html_path, receipt_path,
        visualization_index_json=index_path, scorecard_json=scorecard_path,
        run_id="eval-1", run_root=tmp_path,
    )

    assert html_path.is_file()
    assert receipt_path.is_file()
    assert index_path.is_file()
    assert receipt["visualization_index_sha256"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert {item["id"] for item in index["artifacts"]} == {"scorecard", "student_review_html"}


def test_planned_visualization_manifest_binds_evaluation_identity(tmp_path):
    """The evaluator accepts only the planned manifest for the same run."""

    path = tmp_path / "visualization_manifest.json"
    path.write_text(
        json.dumps({
            "schema_version": "research_visualization_manifest_v1",
            "run_id": "eval-1",
            "run_kind": "evaluation",
            "phase": "planned",
            "run_root": str(tmp_path.resolve()),
            "planned_outputs": [
                {"id": "scorecard", "path": str((tmp_path / "scorecard.json").resolve())},
                {"id": "student_review_html", "path": str((tmp_path / "student_review.html").resolve())},
            ],
        }),
        encoding="utf-8",
    )
    manifest = validate_planned_visualization_manifest(
        path,
        run_id="eval-1",
        run_root=tmp_path,
        required_outputs={
            "scorecard": tmp_path / "scorecard.json",
            "student_review_html": tmp_path / "student_review.html",
        },
    )
    assert manifest["phase"] == "planned"

def test_planned_visualization_manifest_rejects_invalid_json_shapes(tmp_path):
    """Malformed JSON objects fail as contract errors, not incidental tracebacks."""

    path = tmp_path / "invalid_visualization_manifest.json"
    path.write_text(
        json.dumps({
            "schema_version": "research_visualization_manifest_v1",
            "run_id": "eval-1",
            "run_kind": "evaluation",
            "phase": "planned",
            "run_root": str(tmp_path.resolve()),
            "planned_outputs": {"scorecard": "wrong-shape"},
        }),
        encoding="utf-8",
    )
    try:
        validate_planned_visualization_manifest(
            path,
            run_id="eval-1",
            run_root=tmp_path,
            required_outputs={"scorecard": tmp_path / "scorecard.json"},
        )
    except ValueError as exc:
        assert "planned_outputs" in str(exc) or "planned output" in str(exc)
    else:
        raise AssertionError("invalid planned_outputs shape was accepted")

def test_student_review_rejects_unknown_review_selection(tmp_path):
    """Review selection must refer to the same canonical sample set."""

    sample = _canonical(tmp_path)
    canonical_path = tmp_path / "canonical.jsonl"
    prediction_path = tmp_path / "student-predictions.jsonl"
    selection_path = tmp_path / "selection.jsonl"
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(prediction_path, [{
        "sample_id": sample.sample_id,
        "canonical_digest": sample.canonical_digest,
        "df_opd_profile_id": sample.profile.profile_id,
        "df_opd_profile_digest": sample.profile.profile_digest,
        "decision": sample.decision_target,
        "trajectory_ego_t": [list(point) for point in sample.trajectory_ego_t],
    }])
    write_jsonl_atomic(selection_path, [{"sample_id": "missing-sample"}])

    try:
        build_df_opd_student_review_bundle_v1(
            canonical_path, [prediction_path], tmp_path / "review.html",
            tmp_path / "receipt.json", review_selection_jsonl=selection_path,
        )
    except ValueError as exc:
        assert "unknown sample_id" in str(exc)
    else:
        raise AssertionError("unknown review sample_id was accepted")
