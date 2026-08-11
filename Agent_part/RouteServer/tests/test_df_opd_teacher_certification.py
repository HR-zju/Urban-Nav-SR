"""CPU tests for paired DF-OPD Teacher-certification evidence gates."""

from __future__ import annotations

import pytest

from mcp_server.contracts.df_opd import DF_OPD_PROFILE_2S16, DF_OPD_PROFILES_V1
from mcp_server.contracts.df_opd_certification import DFOPDTeacherCertificationScoreV1
from mcp_server.data.io.jsonl import write_jsonl_atomic
from mcp_server.eval.df_opd_teacher import (
    certify_df_opd_teacher_jsonl_v1,
    certify_df_opd_teacher_v1,
)


_DATASET_DIGEST = "a" * 64
_CHECKPOINT_DIGEST = "b" * 64
_PREDICTION_DIGEST = "c" * 64


def _score_rows() -> list[DFOPDTeacherCertificationScoreV1]:
    """Build a complete, behaviourally positive four-condition fixture."""

    profile = DF_OPD_PROFILES_V1[DF_OPD_PROFILE_2S16]
    condition_values = {
        "same_input_t0": (False, 3.0, 0.8),
        "correct_privilege_tc": (True, 1.0, 0.1),
        "shuffled_privilege_ts": (False, 2.0, 0.6),
        "exact_future_oracle_to": (True, 0.5, 0.05),
    }
    rows = []
    for index in range(3):
        for condition, (correct, fde, ce) in condition_values.items():
            rows.append(
                DFOPDTeacherCertificationScoreV1(
                    sample_id=f"sample-{index}",
                    source_group_id=f"group-{index}",
                    canonical_digest=f"{index + 1:064x}",
                    profile_id=profile.profile_id,
                    profile_digest=profile.profile_digest,
                    dataset_fingerprint=_DATASET_DIGEST,
                    checkpoint_digest=_CHECKPOINT_DIGEST,
                    prediction_manifest_digest=_PREDICTION_DIGEST,
                    seed=17,
                    condition=condition,
                    decision_strict_parse=correct,
                    decision_correct=correct,
                    decision_ce=ce,
                    trajectory_valid=True,
                    trajectory_ade_m=fde / 2.0,
                    trajectory_fde_m=fde,
                    direction_error_deg=fde * 5.0,
                    is_turn=index == 1,
                    is_stop=index == 2,
                    recoverability_probe_correct=correct,
                    context_conflict_score=float(index),
                )
            )
    return rows


def test_teacher_certification_requires_paired_decision_and_trajectory_evidence():
    report, selection = certify_df_opd_teacher_v1(
        _score_rows(),
        bootstrap_seed=23,
        bootstrap_replicates=200,
        high_conflict_limit=2,
    )

    assert report["promotion_gate"]["status"] == "passed"
    assert report["promotion_gate"]["student_opd_eligible"] is True
    assert report["coverage"]["paired_complete_samples"] == 3
    assert report["paired_comparisons"]["tc_minus_t0_decision_accuracy"]["ci95"]["lower"] > 0.0
    assert report["paired_comparisons"]["tc_minus_t0_fde_improvement_m"]["ci95"]["lower"] > 0.0
    assert [item["sample_id"] for item in selection] == ["sample-2", "sample-1"]


def test_incomplete_conditions_cannot_promote_teacher_to_student_opd():
    rows = [row for row in _score_rows() if row.condition != "exact_future_oracle_to"]
    report, selection = certify_df_opd_teacher_v1(rows, bootstrap_seed=23, bootstrap_replicates=100)

    assert report["promotion_gate"]["status"] == "insufficient_evidence"
    assert report["promotion_gate"]["student_opd_eligible"] is False
    assert selection == []


def test_certification_rejects_cross_checkpoint_or_dataset_identity_drift():
    rows = _score_rows()
    changed = rows[-1].model_copy(update={"checkpoint_digest": "d" * 64})
    with pytest.raises(ValueError, match="run identity drift"):
        certify_df_opd_teacher_v1([*rows[:-1], changed], bootstrap_seed=23, bootstrap_replicates=100)


def test_certification_jsonl_writes_compact_report_and_review_selection(tmp_path):
    scores_path = tmp_path / "scores.jsonl"
    report_path = tmp_path / "report.json"
    selection_path = tmp_path / "review.jsonl"
    write_jsonl_atomic(scores_path, [row.model_dump(mode="json") for row in _score_rows()])

    report = certify_df_opd_teacher_jsonl_v1(
        scores_path,
        report_path,
        high_conflict_jsonl=selection_path,
        bootstrap_seed=23,
        bootstrap_replicates=100,
        high_conflict_limit=1,
    )

    assert report_path.is_file()
    assert selection_path.is_file()
    assert report["high_conflict_jsonl"] == str(selection_path)
    assert len(report["score_jsonl_sha256"]) == 64
    assert len(report["high_conflict_jsonl_sha256"]) == 64
    assert "decision_target" not in report
