"""Frozen score-row contract for offline DF-OPD Teacher certification.

Rows are produced after a local Teacher inference run; this contract does not
invoke models or expose private cue media.  It freezes the minimum per-sample
evidence needed to compare T0, TC, TS and audit-only TO fairly.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.df_opd import DF_OPD_PROFILES_V1, DFOPDTeacherConditionV1


DF_OPD_TEACHER_CERTIFICATION_SCORE_SCHEMA_VERSION = "urbannav_df_opd_teacher_certification_score_v1"


class DFOPDTeacherCertificationScoreV1(BaseModel):
    """One condition-specific Teacher score for a frozen canonical sample.

    Args:
        sample_id: Canonical sample identity shared by all four conditions.
        source_group_id: Leakage-control group identity retained for audits.
        canonical_digest: Immutable canonical-sample SHA-256.
        profile_id: Exact frozen DF-OPD profile identity.
        profile_digest: Profile payload SHA-256 matching ``profile_id``.
        dataset_fingerprint: Frozen certification-split identity SHA-256.
        checkpoint_digest: Evaluated Teacher checkpoint SHA-256.
        prediction_manifest_digest: Frozen prediction artifact SHA-256.
        seed: Inference seed shared across all paired conditions.
        condition: T0, TC, TS, or audit-only TO.
        decision_strict_parse: Whether the generated Decision is canonical.
        decision_correct: Exact Decision tuple correctness; false after parse
            rejection.
        decision_ce: Optional per-sample Decision CE for diagnostics only.
        trajectory_valid: Whether trajectory metrics are valid for this row.
        trajectory_ade_m: Optional metric-FLU ADE in metres.
        trajectory_fde_m: Optional metric-FLU FDE in metres.
        direction_error_deg: Optional absolute terminal direction error.
        is_turn: Canonical turn subset membership.
        is_stop: Canonical stop subset membership.
        recoverability_probe_correct: Optional diagnostic probe result.
        context_conflict_score: Optional non-negative conflict score used only
            to select human review candidates.

    Raises:
        ValueError: If profile binding, finite metrics, or invalid-trajectory
            semantics are inconsistent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[DF_OPD_TEACHER_CERTIFICATION_SCORE_SCHEMA_VERSION] = (
        DF_OPD_TEACHER_CERTIFICATION_SCORE_SCHEMA_VERSION
    )
    sample_id: str = Field(min_length=1, max_length=480)
    source_group_id: str = Field(min_length=1, max_length=240)
    canonical_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_id: str = Field(min_length=1, max_length=128)
    profile_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    prediction_manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int
    condition: DFOPDTeacherConditionV1
    decision_strict_parse: bool
    decision_correct: bool
    decision_ce: float | None = Field(default=None, ge=0.0)
    trajectory_valid: bool
    trajectory_ade_m: float | None = Field(default=None, ge=0.0)
    trajectory_fde_m: float | None = Field(default=None, ge=0.0)
    direction_error_deg: float | None = Field(default=None, ge=0.0)
    is_turn: bool = False
    is_stop: bool = False
    recoverability_probe_correct: bool | None = None
    context_conflict_score: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_score(self) -> "DFOPDTeacherCertificationScoreV1":
        """Require exact profile binding and finite valid-trajectory metrics.

        Returns:
            The immutable, validated score row.

        Raises:
            ValueError: If a profile digest drifts, parse correctness is
                inconsistent, or trajectory metrics violate validity rules.
        """

        try:
            profile = DF_OPD_PROFILES_V1[self.profile_id]
        except KeyError as exc:
            raise ValueError(f"unknown DF-OPD profile_id: {self.profile_id!r}") from exc
        if self.profile_digest != profile.profile_digest:
            raise ValueError("DF-OPD certification profile_digest mismatch")
        if not self.decision_strict_parse and self.decision_correct:
            raise ValueError("Decision cannot be correct after strict-parse rejection")
        trajectory_values = (
            self.trajectory_ade_m,
            self.trajectory_fde_m,
            self.direction_error_deg,
        )
        if self.trajectory_valid and any(value is None for value in trajectory_values[:2]):
            raise ValueError("valid trajectory rows require ADE and FDE")
        if not self.trajectory_valid and any(value is not None for value in trajectory_values):
            raise ValueError("invalid trajectory rows cannot report trajectory metrics")
        for name, value in (
            ("decision_ce", self.decision_ce),
            ("trajectory_ade_m", self.trajectory_ade_m),
            ("trajectory_fde_m", self.trajectory_fde_m),
            ("direction_error_deg", self.direction_error_deg),
            ("context_conflict_score", self.context_conflict_score),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when present")
        return self


__all__ = [
    "DFOPDTeacherCertificationScoreV1",
    "DF_OPD_TEACHER_CERTIFICATION_SCORE_SCHEMA_VERSION",
]
