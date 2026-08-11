"""CPU tests for private DF-OPD Teacher prediction-to-evidence scoring."""

from __future__ import annotations

import pytest

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
from mcp_server.data.io.jsonl import iter_jsonl, write_jsonl_atomic
from mcp_server.eval.df_opd_teacher_scoring import build_df_opd_teacher_evidence_v1

_DATASET_DIGEST = "a" * 64
_CHECKPOINT_DIGEST = "b" * 64
_DECISION = '<Decision>{"selected_action_label":"straight","speed":"normal","response":"proceed"}</Decision>'
_CONDITIONS = (
    "same_input_t0",
    "correct_privilege_tc",
    "shuffled_privilege_ts",
    "exact_future_oracle_to",
)


def _canonical() -> DFOPDCanonicalSampleV1:
    """Build one valid four-view canonical fixture without accessing media."""

    profile = DFOPDProfileV1(profile_id=DF_OPD_PROFILE_2S16, target_horizon_steps=16)
    views = (
        DFOPDTeacherViewV1(
            condition="same_input_t0",
            images=("current.jpg",),
            image_roles=("fpv.current",),
            privilege_policy_digest="c" * 64,
        ),
        DFOPDTeacherViewV1(
            condition="correct_privilege_tc",
            images=("current.jpg", "cue.png"),
            image_roles=("fpv.current", DF_OPD_TEACHER_CUE_ROLE),
            cue_class="go_straight",
            privilege_policy_digest="c" * 64,
            cue_artifact_sha256="d" * 64,
        ),
        DFOPDTeacherViewV1(
            condition="shuffled_privilege_ts",
            images=("current.jpg", "ts.png"),
            image_roles=("fpv.current", DF_OPD_TEACHER_CUE_ROLE),
            cue_class="turn_left",
            privilege_policy_digest="c" * 64,
            cue_artifact_sha256="e" * 64,
        ),
        DFOPDTeacherViewV1(
            condition="exact_future_oracle_to",
            images=("current.jpg", "oracle.png"),
            image_roles=("fpv.current", DF_OPD_ORACLE_CUE_ROLE),
            cue_class="oracle",
            privilege_policy_digest="c" * 64,
            cue_artifact_sha256="f" * 64,
        ),
    )
    payload = {
        "schema_version": DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION,
        "sample_id": "sample-1",
        "source_group_id": "group-1",
        "split_group_id": "split-1",
        "dataset_source": "sidewalk",
        "profile": profile,
        "images": ("current.jpg",),
        "image_roles": ("fpv.current",),
        "decision_target": _DECISION,
        "trajectory_ego_t": tuple((float(index), 0.0, 0.0) for index in range(16)),
        "action_mask": (True,) * 16,
        "teacher_views": views,
    }
    serialized = {
        key: value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        for key, value in payload.items()
    }
    serialized["teacher_views"] = [view.model_dump(mode="json") for view in views]
    return DFOPDCanonicalSampleV1(
        **payload,
        canonical_digest=df_opd_digest_v1(serialized),
    )


def _ledger(sample: DFOPDCanonicalSampleV1, condition: str) -> dict:
    """Return one successful private inference ledger row."""

    return {
        "sample_id": sample.sample_id,
        "canonical_digest": sample.canonical_digest,
        "df_opd_profile_id": sample.profile.profile_id,
        "df_opd_profile_digest": sample.profile.profile_digest,
        "df_opd_teacher_condition": condition,
        "private_artifact": True,
        "evaluation_only": True,
        "decision": _DECISION,
        "trajectory_ego_t": [list(point) for point in sample.trajectory_ego_t],
    }


def test_teacher_evidence_bridge_requires_complete_pairing_and_writes_score_review(tmp_path):
    sample = _canonical()
    canonical_path = tmp_path / "canonical.jsonl"
    inference_path = tmp_path / "inference.jsonl"
    score_path = tmp_path / "scores.jsonl"
    review_path = tmp_path / "review_predictions.jsonl"
    receipt_path = tmp_path / "receipt.json"
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(inference_path, [_ledger(sample, condition) for condition in _CONDITIONS])

    receipt = build_df_opd_teacher_evidence_v1(
        canonical_path,
        [inference_path],
        score_path,
        review_path,
        receipt_path,
        dataset_fingerprint=_DATASET_DIGEST,
        checkpoint_digest=_CHECKPOINT_DIGEST,
        seed=17,
    )

    scores = list(iter_jsonl(score_path))
    reviews = list(iter_jsonl(review_path))
    assert len(scores) == len(reviews) == 4
    assert all(row["decision_strict_parse"] and row["decision_correct"] for row in scores)
    assert all(row["trajectory_ade_m"] == 0.0 and row["trajectory_fde_m"] == 0.0 for row in scores)
    assert all(row["trajectory_valid"] is True for row in reviews)
    assert receipt["private_artifact"] is True
    assert "decision_target" not in receipt


def test_teacher_evidence_bridge_rejects_missing_condition_before_writing(tmp_path):
    sample = _canonical()
    canonical_path = tmp_path / "canonical.jsonl"
    inference_path = tmp_path / "inference.jsonl"
    score_path = tmp_path / "scores.jsonl"
    review_path = tmp_path / "review_predictions.jsonl"
    receipt_path = tmp_path / "receipt.json"
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(inference_path, [_ledger(sample, condition) for condition in _CONDITIONS[:-1]])

    with pytest.raises(ValueError, match="complete T0/TC/TS/TO matrix"):
        build_df_opd_teacher_evidence_v1(
            canonical_path,
            [inference_path],
            score_path,
            review_path,
            receipt_path,
            dataset_fingerprint=_DATASET_DIGEST,
            checkpoint_digest=_CHECKPOINT_DIGEST,
            seed=17,
        )
    assert not score_path.exists()
    assert not review_path.exists()


def test_teacher_evidence_bridge_rejects_prompt_or_gt_leak_in_private_ledger(tmp_path):
    sample = _canonical()
    canonical_path = tmp_path / "canonical.jsonl"
    inference_path = tmp_path / "inference.jsonl"
    score_path = tmp_path / "scores.jsonl"
    review_path = tmp_path / "review_predictions.jsonl"
    receipt_path = tmp_path / "receipt.json"
    rows = [_ledger(sample, condition) for condition in _CONDITIONS]
    rows[0]["messages"] = [{"role": "user", "content": "must not persist"}]
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(inference_path, rows)

    with pytest.raises(ValueError, match="leaked forbidden field"):
        build_df_opd_teacher_evidence_v1(
            canonical_path,
            [inference_path],
            score_path,
            review_path,
            receipt_path,
            dataset_fingerprint=_DATASET_DIGEST,
            checkpoint_digest=_CHECKPOINT_DIGEST,
            seed=17,
        )


def test_teacher_evidence_bridge_records_failure_ledger_as_invalid_without_targets(tmp_path):
    sample = _canonical()
    canonical_path = tmp_path / "canonical.jsonl"
    inference_path = tmp_path / "inference.jsonl"
    score_path = tmp_path / "scores.jsonl"
    review_path = tmp_path / "review_predictions.jsonl"
    receipt_path = tmp_path / "receipt.json"
    rows = [_ledger(sample, condition) for condition in _CONDITIONS]
    failed = rows[0]
    failed.pop("decision")
    failed.pop("trajectory_ego_t")
    failed["failure_code"] = "invalid_decision"
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(inference_path, rows)

    build_df_opd_teacher_evidence_v1(
        canonical_path,
        [inference_path],
        score_path,
        review_path,
        receipt_path,
        dataset_fingerprint=_DATASET_DIGEST,
        checkpoint_digest=_CHECKPOINT_DIGEST,
        seed=17,
    )

    scores = {row["condition"]: row for row in iter_jsonl(score_path)}
    reviews = {row["condition"]: row for row in iter_jsonl(review_path)}
    assert scores["same_input_t0"]["decision_strict_parse"] is False
    assert scores["same_input_t0"]["trajectory_valid"] is False
    assert reviews["same_input_t0"]["decision_text"] is None
    assert reviews["same_input_t0"]["trajectory_ego_t"] is None
