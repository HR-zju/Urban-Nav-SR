"""CPU tests for private four-view DF-OPD review bundle rendering."""

from __future__ import annotations

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
