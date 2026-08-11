"""CPU tests for Student-only DF-OPD prediction evaluation."""

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
from mcp_server.data.io.jsonl import write_jsonl_atomic
from mcp_server.eval.df_opd_student import evaluate_df_opd_student_predictions_v1

_DATASET_DIGEST = "a" * 64
_CHECKPOINT_DIGEST = "b" * 64
_DECISION = '<Decision>{"selected_action_label":"straight","speed":"normal","response":"proceed"}</Decision>'


def _canonical() -> DFOPDCanonicalSampleV1:
    """Build one minimal one-profile canonical fixture."""

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
            cue_class="straight",
            privilege_policy_digest="c" * 64,
            cue_artifact_sha256="d" * 64,
        ),
        DFOPDTeacherViewV1(
            condition="shuffled_privilege_ts",
            images=("current.jpg", "ts.png"),
            image_roles=("fpv.current", DF_OPD_TEACHER_CUE_ROLE),
            cue_class="left",
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
        "sample_id": "student-1",
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
    serialized["teacher_views"] = [item.model_dump(mode="json") for item in views]
    return DFOPDCanonicalSampleV1(
        **payload,
        canonical_digest=df_opd_digest_v1(serialized),
    )


def _prediction(sample: DFOPDCanonicalSampleV1) -> dict:
    """Return a correct no-leak Student inference output ledger row."""

    return {
        "sample_id": sample.sample_id,
        "canonical_digest": sample.canonical_digest,
        "df_opd_profile_id": sample.profile.profile_id,
        "df_opd_profile_digest": sample.profile.profile_digest,
        "decision": _DECISION,
        "trajectory_ego_t": [list(point) for point in sample.trajectory_ego_t],
        "latency_ms": 12.0,
        "action_metadata": {"latency_ms": {"prefill": 5.0, "total": 10.0}},
    }


def test_student_evaluator_reports_exact_decision_geometry_coverage_and_latency(tmp_path):
    sample = _canonical()
    canonical_path = tmp_path / "canonical.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "report.json"
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(prediction_path, [_prediction(sample)])

    report = evaluate_df_opd_student_predictions_v1(
        canonical_path,
        [prediction_path],
        report_path,
        dataset_fingerprint=_DATASET_DIGEST,
        checkpoint_digest=_CHECKPOINT_DIGEST,
        seed=17,
    )

    assert report["decision"]["strict_parse_rate"] == 1.0
    assert report["decision"]["tuple_accuracy"] == 1.0
    assert report["trajectory"]["2s"]["conditional_ade_m"] == 0.0
    assert report["trajectory"]["2s"]["conditional_fde_m"] == 0.0
    assert report["latency_ms"]["wall_total"]["p50"] == 12.0
    assert "decision_target" not in report


def test_student_evaluator_rejects_private_or_prompt_fields_in_output_ledger(tmp_path):
    sample = _canonical()
    canonical_path = tmp_path / "canonical.jsonl"
    prediction_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "report.json"
    row = _prediction(sample)
    row["messages"] = [{"role": "user", "content": "leak"}]
    write_jsonl_atomic(canonical_path, [sample.model_dump(mode="json")])
    write_jsonl_atomic(prediction_path, [row])

    with pytest.raises(ValueError, match="leaked forbidden field"):
        evaluate_df_opd_student_predictions_v1(
            canonical_path,
            [prediction_path],
            report_path,
            dataset_fingerprint=_DATASET_DIGEST,
            checkpoint_digest=_CHECKPOINT_DIGEST,
            seed=17,
        )

