"""Canonical Stage-A direction-profile identities shared by reasoning and VQA."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.reasoning import ReasonCode


FPV_DEGREE10_TERMINAL_CENTROID_V2 = (
    "fpv_degree10_terminal_centroid_v2"
)
EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2 = (
    "ego_degree10_12way_terminal_centroid_v2"
)

_FORBIDDEN_DIRECTION_SOURCE_PARTS = (
    "gps",
    "gnss",
    "route_match",
    "route_theta",
)

_APPROVED_STAGE_A_DIRECTION_PROFILES: dict[str, dict[str, Any]] = {
    FPV_DEGREE10_TERMINAL_CENTROID_V2: {
        "profile_version": "2.0",
        "bin_mode": "degree10_with_stop",
        "derivation_version": (
            "pixel_trajectory_terminal_centroid_rule_v2"
        ),
        "labels": tuple(
            [
                (
                    f"left_{abs(angle)}"
                    if angle < 0
                    else f"right_{angle}"
                    if angle > 0
                    else "straight"
                )
                for angle in range(-90, 91, 10)
            ]
            + ["stop"]
        ),
        "choices": tuple(
            chr(ord("A") + index) for index in range(19)
        )
        + ("T",),
        "angles": tuple(
            float(angle) for angle in range(-90, 91, 10)
        )
        + (None,),
    },
    EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2: {
        "profile_version": "2.0",
        "bin_mode": "degree10_12way_with_stop",
        "derivation_version": (
            "ego_x_right_y_front_terminal_centroid_rule_v2"
        ),
        "labels": (
            "stop",
            "straight",
            "left_10",
            "right_10",
            "left_20",
            "right_20",
            "left_30",
            "right_30",
            "left_40",
            "right_40",
            "left_50plus",
            "right_50plus",
        ),
        "choices": tuple(
            chr(ord("A") + index) for index in range(12)
        ),
        "angles": (
            None,
            0.0,
            -10.0,
            10.0,
            -20.0,
            20.0,
            -30.0,
            30.0,
            -40.0,
            40.0,
            -50.0,
            50.0,
        ),
    },
}


class DirectionLabelStamp(BaseModel):
    """Carry a complete direction-label identity and its derivation source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(min_length=1, max_length=128)
    profile_version: str = Field(min_length=1, max_length=64)
    label_index: int = Field(ge=0)
    label_name: ReasonCode
    choice: str = Field(min_length=1, max_length=32)
    angle_deg: float | None = Field(
        default=None,
        ge=-180.0,
        le=180.0,
    )
    source: ReasonCode
    bin_mode: str = Field(min_length=1, max_length=64)
    derivation_version: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_profile_identity(self) -> "DirectionLabelStamp":
        """Reject unknown profiles, forbidden sources, and label drift."""

        profile = resolve_stage_a_direction_profile(self.profile_id)
        if self.profile_version != profile["profile_version"]:
            raise ValueError("direction profile_version mismatch")
        if self.bin_mode != profile["bin_mode"]:
            raise ValueError("direction bin_mode mismatch")
        normalized_source = str(self.source).lower()
        if any(
            part in normalized_source
            for part in _FORBIDDEN_DIRECTION_SOURCE_PARTS
        ):
            raise ValueError(
                "GPS, route matching, and route_theta direction sources "
                "are forbidden"
            )
        if self.label_index >= len(profile["labels"]):
            raise ValueError(
                "direction label_index is outside approved profile"
            )
        expected = (
            profile["labels"][self.label_index],
            profile["choices"][self.label_index],
            profile["angles"][self.label_index],
        )
        actual = (self.label_name, self.choice, self.angle_deg)
        if actual != expected:
            raise ValueError(
                "direction label identity mismatch: "
                f"expected={expected}, actual={actual}"
            )
        return self


def resolve_stage_a_direction_profile(
    profile_id: str,
) -> dict[str, Any]:
    """Return one approved Stage-A profile.

    Args:
        profile_id: Exact versioned profile identifier.

    Returns:
        Immutable-by-convention profile mapping.

    Raises:
        ValueError: If the profile is outside the Stage-A allowlist.
    """

    try:
        return _APPROVED_STAGE_A_DIRECTION_PROFILES[
            str(profile_id)
        ]
    except KeyError as exc:
        raise ValueError(
            f"unapproved Stage-A direction profile: {profile_id}"
        ) from exc


def approved_stage_a_direction_profiles() -> dict[str, dict[str, Any]]:
    """Return a JSON-compatible copy of the Stage-A profile registry.

    Returns:
        Profile mapping detached from the canonical in-memory registry.
    """

    return json.loads(
        json.dumps(_APPROVED_STAGE_A_DIRECTION_PROFILES)
    )


def direction_label_names(profile_id: str) -> tuple[str, ...]:
    """Return canonical label names for one profile.

    Args:
        profile_id: Exact approved profile identifier.

    Returns:
        Ordered direction-label names.
    """

    profile = resolve_stage_a_direction_profile(profile_id)
    return tuple(str(value) for value in profile["labels"])


def canonical_direction_derivation_version(profile_id: str) -> str:
    """Return the canonical trajectory derivation for one profile.

    Args:
        profile_id: Exact approved profile identifier.

    Returns:
        Frozen trajectory derivation identifier.
    """

    return str(
        resolve_stage_a_direction_profile(profile_id)[
            "derivation_version"
        ]
    )


__all__ = [
    "DirectionLabelStamp",
    "EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2",
    "FPV_DEGREE10_TERMINAL_CENTROID_V2",
    "approved_stage_a_direction_profiles",
    "canonical_direction_derivation_version",
    "direction_label_names",
    "resolve_stage_a_direction_profile",
]
