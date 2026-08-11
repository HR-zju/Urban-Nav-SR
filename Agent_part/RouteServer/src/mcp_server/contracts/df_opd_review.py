"""Private prediction-sidecar contract for DF-OPD human review bundles."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.df_opd import (
    DF_OPD_PROFILES_V1,
    DFOPDTeacherConditionV1,
)


DF_OPD_TEACHER_REVIEW_PREDICTION_SCHEMA_VERSION = "urbannav_df_opd_teacher_review_prediction_v1"


class DFOPDTeacherReviewPredictionV1(BaseModel):
    """One private Teacher prediction retained only for local human review.

    Args:
        sample_id: Canonical sample identity.
        canonical_digest: Canonical sample SHA-256.
        profile_id: Frozen target/checkpoint profile identity.
        profile_digest: Exact profile payload SHA-256.
        condition: T0, TC, TS, or TO view that produced this prediction.
        decision_text: Raw generated Decision text, retained for private audit.
        trajectory_ego_t: Predicted cumulative FLU trajectory when valid.
        trajectory_valid: Whether trajectory values are meaningful.

    Raises:
        ValueError: If profile binding, trajectory shape, or finite-value
            constraints are invalid.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[DF_OPD_TEACHER_REVIEW_PREDICTION_SCHEMA_VERSION] = (
        DF_OPD_TEACHER_REVIEW_PREDICTION_SCHEMA_VERSION
    )
    sample_id: str = Field(min_length=1, max_length=480)
    canonical_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_id: str = Field(min_length=1, max_length=128)
    profile_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    condition: DFOPDTeacherConditionV1
    decision_text: str | None = Field(default=None, max_length=4000)
    trajectory_ego_t: tuple[tuple[float, float, float], ...] | None = None
    trajectory_valid: bool

    @model_validator(mode="after")
    def validate_prediction(self) -> "DFOPDTeacherReviewPredictionV1":
        """Require profile-shaped finite trajectories only when marked valid.

        Returns:
            The validated immutable review prediction.

        Raises:
            ValueError: If profile binding or validity semantics drift.
        """

        try:
            profile = DF_OPD_PROFILES_V1[self.profile_id]
        except KeyError as exc:
            raise ValueError(f"unknown DF-OPD profile_id: {self.profile_id!r}") from exc
        if self.profile_digest != profile.profile_digest:
            raise ValueError("DF-OPD review prediction profile_digest mismatch")
        if self.trajectory_valid:
            if self.trajectory_ego_t is None or len(self.trajectory_ego_t) != profile.target_horizon_steps:
                raise ValueError("valid DF-OPD review trajectory must match profile horizon")
            if not all(math.isfinite(value) for point in self.trajectory_ego_t for value in point):
                raise ValueError("DF-OPD review trajectory values must be finite")
        elif self.trajectory_ego_t is not None:
            raise ValueError("invalid DF-OPD review trajectory must be absent")
        return self


__all__ = [
    "DFOPDTeacherReviewPredictionV1",
    "DF_OPD_TEACHER_REVIEW_PREDICTION_SCHEMA_VERSION",
]
