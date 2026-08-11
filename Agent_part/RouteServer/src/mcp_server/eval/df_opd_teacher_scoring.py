"""Build private DF-OPD Teacher score and review evidence offline.

This RouteServer-owned module joins frozen canonical GT to completed private
T0/TC/TS/TO inference ledgers.  It never runs a model, opens media, or emits a
Student-deployable artifact.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from mcp_server.contracts.df_opd import (
    DFOPDCanonicalSampleV1,
    DFOPDTeacherConditionV1,
    df_opd_digest_v1,
)
from mcp_server.contracts.df_opd_certification import DFOPDTeacherCertificationScoreV1
from mcp_server.contracts.df_opd_review import DFOPDTeacherReviewPredictionV1
from mcp_server.contracts.reasoning_v4 import DecisionV4
from mcp_server.data.io.jsonl import file_sha256, iter_jsonl, write_jsonl_atomic

DF_OPD_TEACHER_EVIDENCE_RECEIPT_SCHEMA_V1 = "urbannav_df_opd_teacher_evidence_receipt_v1"
_CONDITIONS: tuple[DFOPDTeacherConditionV1, ...] = (
    "same_input_t0",
    "correct_privilege_tc",
    "shuffled_privilege_ts",
    "exact_future_oracle_to",
)
_FORBIDDEN_LEDGER_FIELDS = {
    "messages",
    "images",
    "image_roles",
    "teacher_images",
    "teacher_image_roles",
    "teacher_messages",
    "decision_target",
    "trajectory_target",
    "trajectory_ego_t_target",
    "action_mask",
    "canonical_full_cot",
    "df_opd_privilege_policy_digest",
    "df_opd_teacher_cue_sha256",
}


def build_df_opd_teacher_evidence_v1(
    canonical_jsonl: str | Path,
    inference_jsonls: Sequence[str | Path],
    score_jsonl: str | Path,
    review_prediction_jsonl: str | Path,
    receipt_json: str | Path,
    *,
    dataset_fingerprint: str,
    checkpoint_digest: str,
    seed: int,
) -> dict[str, Any]:
    """Join four private ledgers into certification and review artifacts.

    Args:
        canonical_jsonl: Frozen canonical samples with hidden GT targets.
        inference_jsonls: Private prediction/failure JSONLs whose union has
            exactly one T0/TC/TS/TO result per canonical sample.
        score_jsonl: Destination compact certification score JSONL.
        review_prediction_jsonl: Destination private raw-prediction sidecar.
        receipt_json: Destination non-sensitive private-artifact receipt.
        dataset_fingerprint: Frozen certification-split SHA-256.
        checkpoint_digest: Exact evaluated Teacher checkpoint SHA-256.
        seed: Shared deterministic inference seed.

    Returns:
        Receipt with paths, digests, and aggregate counts only.

    Raises:
        ValueError: If IDs, profile binding, private marking, complete
            condition pairing, or prediction geometry are inconsistent.
    """

    _require_digest(dataset_fingerprint, "dataset_fingerprint")
    _require_digest(checkpoint_digest, "checkpoint_digest")
    canonical_path = Path(canonical_jsonl).expanduser().resolve()
    inference_paths = _resolve_inference_paths(inference_jsonls)
    score_path = Path(score_jsonl).expanduser().resolve()
    review_path = Path(review_prediction_jsonl).expanduser().resolve()
    receipt_path = Path(receipt_json).expanduser().resolve()
    canonical = _read_canonical_rows(canonical_path)
    ledgers = _read_private_ledgers(inference_paths, canonical)
    _require_complete_condition_matrix(canonical, ledgers)
    prediction_manifest_digest = df_opd_digest_v1({
        "inference_jsonl_sha256": sorted(file_sha256(path) for path in inference_paths),
    })

    scores: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for sample_id, sample in canonical.items():
        expected_decision = _parse_strict_decision(sample.decision_target)
        if expected_decision is None:
            raise ValueError(f"canonical sample {sample_id!r} has non-canonical decision_target")
        for condition in _CONDITIONS:
            score, review = _score_one(
                sample,
                condition=condition,
                ledger=ledgers[(sample_id, condition)],
                expected_decision=expected_decision,
                dataset_fingerprint=dataset_fingerprint,
                checkpoint_digest=checkpoint_digest,
                prediction_manifest_digest=prediction_manifest_digest,
                seed=seed,
            )
            scores.append(score.model_dump(mode="json"))
            reviews.append(review.model_dump(mode="json"))

    write_jsonl_atomic(score_path, scores)
    write_jsonl_atomic(review_path, reviews)
    receipt = {
        "schema_version": DF_OPD_TEACHER_EVIDENCE_RECEIPT_SCHEMA_V1,
        "private_artifact": True,
        "canonical_jsonl": str(canonical_path),
        "canonical_jsonl_sha256": file_sha256(canonical_path),
        "inference_jsonls": [str(path) for path in inference_paths],
        "inference_jsonl_sha256": [file_sha256(path) for path in inference_paths],
        "prediction_manifest_digest": prediction_manifest_digest,
        "dataset_fingerprint": dataset_fingerprint,
        "checkpoint_digest": checkpoint_digest,
        "seed": int(seed),
        "score_jsonl": str(score_path),
        "score_jsonl_sha256": file_sha256(score_path),
        "review_prediction_jsonl": str(review_path),
        "review_prediction_jsonl_sha256": file_sha256(review_path),
        "counts": {
            "canonical_samples": len(canonical),
            "inference_rows": len(ledgers),
            "score_rows": len(scores),
            "review_prediction_rows": len(reviews),
            **dict(sorted(Counter(item["condition"] for item in scores).items())),
        },
    }
    _write_json_atomic(receipt_path, receipt)
    return receipt


def _resolve_inference_paths(values: Sequence[str | Path]) -> list[Path]:
    """Resolve one non-empty, unique set of private inference ledgers."""

    paths = [Path(value).expanduser().resolve() for value in values]
    if not paths:
        raise ValueError("DF-OPD Teacher evidence requires inference JSONLs")
    if len(set(paths)) != len(paths):
        raise ValueError("DF-OPD Teacher evidence inference JSONLs must be unique")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _read_canonical_rows(path: Path) -> dict[str, DFOPDCanonicalSampleV1]:
    """Load unique canonical rows while preserving their source order."""

    result: dict[str, DFOPDCanonicalSampleV1] = {}
    for line_number, payload in enumerate(iter_jsonl(path), start=1):
        try:
            sample = DFOPDCanonicalSampleV1.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid canonical DF-OPD row at line {line_number}: {exc}") from exc
        if sample.sample_id in result:
            raise ValueError(f"duplicate canonical DF-OPD sample_id: {sample.sample_id!r}")
        result[sample.sample_id] = sample
    if not result:
        raise ValueError("DF-OPD Teacher evidence requires canonical samples")
    return result


def _read_private_ledgers(
    paths: Sequence[Path],
    canonical: Mapping[str, DFOPDCanonicalSampleV1],
) -> dict[tuple[str, DFOPDTeacherConditionV1], Mapping[str, Any]]:
    """Validate and index private inference rows by sample and condition."""

    result: dict[tuple[str, DFOPDTeacherConditionV1], Mapping[str, Any]] = {}
    for path in paths:
        for line_number, row in enumerate(iter_jsonl(path), start=1):
            if row.get("private_artifact") is not True or row.get("evaluation_only") is not True:
                raise ValueError(f"Teacher inference row is not private/evaluation-only: {path}:{line_number}")
            forbidden = _find_forbidden_ledger_field(row)
            if forbidden is not None:
                raise ValueError(f"Teacher inference ledger leaked forbidden field {forbidden!r}: {path}:{line_number}")
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or sample_id not in canonical:
                raise ValueError(f"Teacher inference row has unknown sample_id: {path}:{line_number}")
            condition = row.get("df_opd_teacher_condition")
            if condition not in _CONDITIONS:
                raise ValueError(f"Teacher inference row has invalid condition: {path}:{line_number}")
            sample = canonical[sample_id]
            if (
                row.get("canonical_digest") != sample.canonical_digest
                or row.get("df_opd_profile_id") != sample.profile.profile_id
                or row.get("df_opd_profile_digest") != sample.profile.profile_digest
            ):
                raise ValueError(f"Teacher inference identity/profile drift: {path}:{line_number}")
            key = (sample_id, condition)
            if key in result:
                raise ValueError(f"duplicate Teacher inference row for {sample_id!r}/{condition}")
            result[key] = row
    return result


def _require_complete_condition_matrix(
    canonical: Mapping[str, DFOPDCanonicalSampleV1],
    ledgers: Mapping[tuple[str, DFOPDTeacherConditionV1], Mapping[str, Any]],
) -> None:
    """Require all four private conditions for every frozen canonical sample."""

    missing = [
        f"{sample_id}/{condition}"
        for sample_id in canonical
        for condition in _CONDITIONS
        if (sample_id, condition) not in ledgers
    ]
    if missing:
        raise ValueError("DF-OPD Teacher evidence lacks complete T0/TC/TS/TO matrix: " + ", ".join(missing[:8]))


def _score_one(
    sample: DFOPDCanonicalSampleV1,
    *,
    condition: DFOPDTeacherConditionV1,
    ledger: Mapping[str, Any],
    expected_decision: DecisionV4,
    dataset_fingerprint: str,
    checkpoint_digest: str,
    prediction_manifest_digest: str,
    seed: int,
) -> tuple[DFOPDTeacherCertificationScoreV1, DFOPDTeacherReviewPredictionV1]:
    """Compute strict tuple and valid-prefix trajectory metrics for one row."""

    raw_decision = ledger.get("decision")
    raw_trajectory = ledger.get("trajectory_ego_t")
    if ledger.get("failure_code") is not None:
        if raw_decision is not None or raw_trajectory is not None:
            raise ValueError("failed Teacher inference ledger must not retain Decision or trajectory output")
        raw_decision = None
        raw_trajectory = None
    predicted_decision = _parse_strict_decision(raw_decision)
    trajectory = _parse_valid_trajectory(raw_trajectory, sample)
    trajectory_valid = trajectory is not None
    metrics = _trajectory_metrics(sample, trajectory) if trajectory is not None else (None, None, None)
    label = str(expected_decision.selected_action_label)
    score = DFOPDTeacherCertificationScoreV1(
        sample_id=sample.sample_id,
        source_group_id=sample.source_group_id,
        canonical_digest=sample.canonical_digest,
        profile_id=sample.profile.profile_id,
        profile_digest=sample.profile.profile_digest,
        dataset_fingerprint=dataset_fingerprint,
        checkpoint_digest=checkpoint_digest,
        prediction_manifest_digest=prediction_manifest_digest,
        seed=int(seed),
        condition=condition,
        decision_strict_parse=predicted_decision is not None,
        decision_correct=predicted_decision == expected_decision,
        trajectory_valid=trajectory_valid,
        trajectory_ade_m=metrics[0],
        trajectory_fde_m=metrics[1],
        direction_error_deg=metrics[2],
        is_turn=_is_turn_label(label),
        is_stop=label == "stop",
    )
    review = DFOPDTeacherReviewPredictionV1(
        sample_id=sample.sample_id,
        canonical_digest=sample.canonical_digest,
        profile_id=sample.profile.profile_id,
        profile_digest=sample.profile.profile_digest,
        condition=condition,
        decision_text=raw_decision if isinstance(raw_decision, str) else None,
        trajectory_ego_t=trajectory,
        trajectory_valid=trajectory_valid,
    )
    return score, review


def _find_forbidden_ledger_field(value: Any) -> str | None:
    """Return the first prompt/media/GT field retained in an evidence ledger."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _FORBIDDEN_LEDGER_FIELDS:
                return str(key)
            nested = _find_forbidden_ledger_field(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _find_forbidden_ledger_field(item)
            if nested is not None:
                return nested
    return None


def _parse_strict_decision(value: Any) -> DecisionV4 | None:
    """Accept only exactly rendered canonical Decision text."""

    if not isinstance(value, str) or not value.startswith("<Decision>") or not value.endswith("</Decision>"):
        return None
    try:
        decision = DecisionV4.model_validate(json.loads(value[len("<Decision>"):-len("</Decision>")]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    canonical = "<Decision>" + json.dumps(
        decision.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    ) + "</Decision>"
    return decision if value == canonical else None


def _parse_valid_trajectory(
    value: Any,
    sample: DFOPDCanonicalSampleV1,
) -> tuple[tuple[float, float, float], ...] | None:
    """Return only complete finite trajectories matching the frozen profile."""

    if not isinstance(value, list) or len(value) != sample.profile.target_horizon_steps:
        return None
    points: list[tuple[float, float, float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != sample.profile.action_dim:
            return None
        try:
            point = tuple(float(item) for item in row)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in point):
            return None
        points.append(point)
    return tuple(points)


def _trajectory_metrics(
    sample: DFOPDCanonicalSampleV1,
    prediction: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Compute metric-FLU valid-prefix ADE, FDE, and terminal yaw error."""

    valid_length = sum(1 for value in sample.action_mask if value)
    if valid_length <= 0:
        raise ValueError("canonical DF-OPD target has no valid trajectory prefix")
    errors = [
        math.hypot(
            prediction[index][0] - sample.trajectory_ego_t[index][0],
            prediction[index][1] - sample.trajectory_ego_t[index][1],
        )
        for index in range(valid_length)
    ]
    yaw_error_deg = math.degrees(abs(_wrap_angle(
        prediction[valid_length - 1][2] - sample.trajectory_ego_t[valid_length - 1][2])))
    return sum(errors) / len(errors), errors[-1], yaw_error_deg


def _wrap_angle(value: float) -> float:
    """Wrap a radian angle to [-pi, pi]."""

    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _is_turn_label(value: str) -> bool:
    """Classify explicit left/right/U-turn labels as the turn subset."""

    return str(value).strip().lower() in {
        "left",
        "right",
        "u_turn",
        "uturn",
        "turn_left",
        "turn_right",
    }


def _require_digest(value: str, field_name: str) -> None:
    """Require one lowercase SHA-256 string."""

    if len(str(value)) != 64 or any(character not in "0123456789abcdef" for character in str(value)):
        raise ValueError(f"{field_name} must be a lowercase SHA256 digest")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist a receipt that excludes raw predictions and GT."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "DF_OPD_TEACHER_EVIDENCE_RECEIPT_SCHEMA_V1",
    "build_df_opd_teacher_evidence_v1",
]
