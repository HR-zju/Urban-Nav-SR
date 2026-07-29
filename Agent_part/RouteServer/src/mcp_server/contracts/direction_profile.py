"""Canonical Stage-A direction-profile identities shared by reasoning and VQA."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mcp_server.contracts.reasoning import ReasonCode


FPV_DEGREE10_TERMINAL_CENTROID_V2 = (
    "fpv_degree10_terminal_centroid_v2"
)
EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2 = (
    "ego_degree10_12way_terminal_centroid_v2"
)
FPV_DEGREE10_AFFORDANCE7_V1 = "fpv_degree10_affordance7_v1"
EGO_12WAY_AFFORDANCE7_V1 = "ego_12way_affordance7_v1"

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
        "intent_compatibility_tolerance_deg": 10.0,
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
        "coarse_motions": tuple(
            "go_straight"
            if abs(angle) <= 10
            else "turn_left"
            if angle < 0
            else "turn_right"
            for angle in range(-90, 91, 10)
        )
        + ("stop",),
    },
    EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2: {
        "profile_version": "2.0",
        "bin_mode": "degree10_12way_with_stop",
        "intent_compatibility_tolerance_deg": 10.0,
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
        "coarse_motions": (
            "stop",
            "go_straight",
            "go_straight",
            "go_straight",
            "turn_left",
            "turn_right",
            "turn_left",
            "turn_right",
            "turn_left",
            "turn_right",
            "turn_left",
            "turn_right",
        ),
    },
}

_APPROVED_AFFORDANCE_PROBE_PROFILES: dict[str, dict[str, Any]] = {
    FPV_DEGREE10_AFFORDANCE7_V1: {
        "profile_version": "1.0",
        "action_profile_id": FPV_DEGREE10_TERMINAL_CENTROID_V2,
        "labels": (
            "left_90",
            "left_30",
            "left_10",
            "straight",
            "right_10",
            "right_30",
            "right_90",
        ),
    },
    EGO_12WAY_AFFORDANCE7_V1: {
        "profile_version": "1.0",
        "action_profile_id": EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2,
        "labels": (
            "left_50plus",
            "left_30",
            "left_10",
            "straight",
            "right_10",
            "right_30",
            "right_50plus",
        ),
    },
}

_ACTION_TO_AFFORDANCE_PROBE_PROFILE = {
    str(value["action_profile_id"]): profile_id
    for profile_id, value in _APPROVED_AFFORDANCE_PROBE_PROFILES.items()
}


@dataclass(frozen=True)
class DirectionComparisonFacts:
    """Carry low-level facts comparing two labels in one direction profile.

    Args:
        angular_delta_deg: Absolute canonical-angle difference, or ``None``
            when at least one label is ``stop``.
        stop_move_mismatch: Whether exactly one compared label is ``stop``.
        sign_conflict: Whether two moving labels turn to opposite sides.
    """

    angular_delta_deg: float | None
    stop_move_mismatch: bool
    sign_conflict: bool


@dataclass(frozen=True)
class AffordanceProbeCandidate:
    """Carry one nearest affordance probe for a fine action.

    Args:
        label: Exact label from the paired affordance-probe profile.
        angle_gap_deg: Absolute angle gap between the action and probe.
    """

    label: str
    angle_gap_deg: float


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


def resolve_affordance_probe_profile(profile_id: str) -> dict[str, Any]:
    """Return one approved local-traversability probe profile.

    Args:
        profile_id: Exact versioned affordance-probe profile identifier.

    Returns:
        Immutable-by-convention profile mapping.

    Raises:
        ValueError: If the profile is outside the Stage-A allowlist.
    """

    try:
        return _APPROVED_AFFORDANCE_PROBE_PROFILES[str(profile_id)]
    except KeyError as exc:
        raise ValueError(
            f"unapproved affordance probe profile: {profile_id}"
        ) from exc


def approved_affordance_probe_profiles() -> dict[str, dict[str, Any]]:
    """Return a JSON-compatible copy of the affordance-probe registry.

    Returns:
        Probe-profile mapping detached from the canonical in-memory registry.
    """

    return json.loads(json.dumps(_APPROVED_AFFORDANCE_PROBE_PROFILES))


def affordance_probe_profile_for_action(action_profile_id: str) -> str:
    """Resolve the unique probe profile owned by one action profile.

    Args:
        action_profile_id: Exact approved Stage-A action-profile identifier.

    Returns:
        Exact affordance-probe profile identifier.

    Raises:
        ValueError: If the action profile has no approved probe profile.
    """

    resolve_stage_a_direction_profile(action_profile_id)
    try:
        return _ACTION_TO_AFFORDANCE_PROBE_PROFILE[str(action_profile_id)]
    except KeyError as exc:
        raise ValueError(
            "action profile has no approved affordance probe profile: "
            f"{action_profile_id}"
        ) from exc


def affordance_probe_label_names(profile_id: str) -> tuple[str, ...]:
    """Return the ordered local-traversability labels for one probe profile.

    Args:
        profile_id: Exact approved affordance-probe profile identifier.

    Returns:
        Ordered probe labels, which never include ``stop``.
    """

    profile = resolve_affordance_probe_profile(profile_id)
    return tuple(str(value) for value in profile["labels"])


def affordance_probe_label_for_action(
    action_profile_id: str,
    label_name: str,
) -> str:
    """Project one fine action label to its nearest public affordance probe.

    The mapping is profile-owned and deterministic. For a non-stop action with
    canonical angle :math:`a` and probe angles :math:`P`, it selects
    :math:`\\operatorname*{argmin}_{p \\in P}|a-p|`; ties follow the frozen probe
    order. This closes Decision support without inventing a second direction
    ontology.

    Args:
        action_profile_id: Exact approved Stage-A action-profile identifier.
        label_name: Exact non-stop action label in that profile.

    Returns:
        Exact label in the paired affordance-probe profile.

    Raises:
        ValueError: If the action is ``stop``, has no canonical angle, or is
            outside the selected profile.
    """

    return affordance_probe_candidates_for_action(
        action_profile_id,
        label_name,
    )[0].label


def affordance_probe_candidates_for_action(
    action_profile_id: str,
    label_name: str,
) -> tuple[AffordanceProbeCandidate, ...]:
    """Return every nearest public affordance probe for one fine action.

    For a non-stop action with canonical angle :math:`a` and probe angles
    :math:`P`, this function returns the complete set
    :math:`\\{p\\in P: |a-p|=\\min_{q\\in P}|a-q|\\}`. Candidate order follows
    the frozen probe profile and therefore remains deterministic without
    discarding equal-distance ties.

    Args:
        action_profile_id: Exact approved Stage-A action-profile identifier.
        label_name: Exact non-stop action label in that profile.

    Returns:
        Ordered nearest candidates with their absolute angle gaps.

    Raises:
        ValueError: If the action is ``stop``, has no canonical angle, or is
            outside the selected profile.
    """

    action = resolve_stage_a_direction_profile(action_profile_id)
    labels = tuple(str(value) for value in action["labels"])
    try:
        action_index = labels.index(str(label_name))
    except ValueError as exc:
        raise ValueError(
            "direction label is outside approved profile: "
            f"profile={action_profile_id}, label={label_name}"
        ) from exc
    action_angle = action["angles"][action_index]
    if action_angle is None:
        raise ValueError("stop has no local traversability probe")

    probe_profile_id = affordance_probe_profile_for_action(
        action_profile_id
    )
    probe_labels = affordance_probe_label_names(probe_profile_id)
    angle_by_label = {
        str(label): angle
        for label, angle in zip(action["labels"], action["angles"])
        if angle is not None
    }
    gaps = tuple(
        abs(float(action_angle) - float(angle_by_label[probe]))
        for probe in probe_labels
    )
    minimum = min(gaps)
    return tuple(
        AffordanceProbeCandidate(
            label=probe,
            angle_gap_deg=gap,
        )
        for probe, gap in zip(probe_labels, gaps)
        if gap == minimum
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


def coarse_motion_for_direction_label(
    profile_id: str,
    label_name: str,
) -> str:
    """Map one approved direction label through its owning profile.

    Args:
        profile_id: Exact approved profile identifier.
        label_name: Exact label in the profile output space.

    Returns:
        One of ``go_straight``, ``turn_left``, ``turn_right``, or ``stop``.

    Raises:
        ValueError: If the label is outside the selected profile.
    """

    profile = resolve_stage_a_direction_profile(profile_id)
    labels = tuple(str(value) for value in profile["labels"])
    try:
        index = labels.index(str(label_name))
    except ValueError as exc:
        raise ValueError(
            "direction label is outside approved profile: "
            f"profile={profile_id}, label={label_name}"
        ) from exc
    return str(profile["coarse_motions"][index])


def direction_labels_are_intent_compatible(
    profile_id: str,
    first_label: str,
    second_label: str,
) -> bool:
    """Compare two labels using the owning profile's angular tolerance.

    Args:
        profile_id: Exact approved profile identifier.
        first_label: First exact label in the profile.
        second_label: Second exact label in the profile.

    Returns:
        Whether both labels are identical stop actions or their canonical
        angles differ by no more than the profile-owned tolerance.

    Raises:
        ValueError: If either label is outside the selected profile.
    """

    return direction_compatibility_class(
        profile_id,
        first_label,
        second_label,
    ) in {"exact", "boundary_compatible"}


def direction_comparison_facts(
    profile_id: str,
    first_label: str,
    second_label: str,
) -> DirectionComparisonFacts:
    """Compute low-level comparison facts without assigning review semantics.

    Args:
        profile_id: Exact approved Stage-A direction-profile identifier.
        first_label: First exact label in the profile.
        second_label: Second exact label in the profile.

    Returns:
        Angle, stop/move, and opposite-turn facts owned by the profile.

    Raises:
        ValueError: If either label is outside the selected profile.
    """

    profile = resolve_stage_a_direction_profile(profile_id)
    labels = tuple(str(value) for value in profile["labels"])
    try:
        first_index = labels.index(str(first_label))
        second_index = labels.index(str(second_label))
    except ValueError as exc:
        raise ValueError(
            "direction label is outside approved profile: "
            f"profile={profile_id}, labels=({first_label}, {second_label})"
        ) from exc
    first_angle = profile["angles"][first_index]
    second_angle = profile["angles"][second_index]
    stop_move_mismatch = (first_angle is None) != (second_angle is None)
    if first_angle is None or second_angle is None:
        return DirectionComparisonFacts(
            angular_delta_deg=None,
            stop_move_mismatch=stop_move_mismatch,
            sign_conflict=False,
        )
    first_value = float(first_angle)
    second_value = float(second_angle)
    return DirectionComparisonFacts(
        angular_delta_deg=abs(first_value - second_value),
        stop_move_mismatch=False,
        sign_conflict=first_value * second_value < 0,
    )


def direction_compatibility_class(
    profile_id: str,
    first_label: str,
    second_label: str,
) -> str:
    """Classify two labels without duplicating profile-owned angle rules.

    Args:
        profile_id: Exact approved Stage-A direction-profile identifier.
        first_label: First exact label in the profile.
        second_label: Second exact label in the profile.

    Returns:
        ``exact``, ``boundary_compatible``, ``warning``, or ``incompatible``.
        Opposite-sign turns and every move/stop comparison are incompatible.

    Raises:
        ValueError: If either label is outside the selected profile.
    """

    if str(first_label) == str(second_label):
        resolve_stage_a_direction_profile(profile_id)
        direction_comparison_facts(
            profile_id,
            first_label,
            second_label,
        )
        return "exact"
    profile = resolve_stage_a_direction_profile(profile_id)
    facts = direction_comparison_facts(
        profile_id,
        first_label,
        second_label,
    )
    if facts.stop_move_mismatch or facts.angular_delta_deg is None:
        return "incompatible"
    if facts.sign_conflict:
        return "incompatible"
    tolerance = float(profile["intent_compatibility_tolerance_deg"])
    if facts.angular_delta_deg <= tolerance:
        return "boundary_compatible"
    if facts.angular_delta_deg <= 2.0 * tolerance:
        return "warning"
    return "incompatible"


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
    "AffordanceProbeCandidate",
    "DirectionComparisonFacts",
    "DirectionLabelStamp",
    "EGO_12WAY_AFFORDANCE7_V1",
    "EGO_DEGREE10_12WAY_TERMINAL_CENTROID_V2",
    "FPV_DEGREE10_AFFORDANCE7_V1",
    "FPV_DEGREE10_TERMINAL_CENTROID_V2",
    "affordance_probe_candidates_for_action",
    "affordance_probe_label_for_action",
    "affordance_probe_label_names",
    "affordance_probe_profile_for_action",
    "approved_affordance_probe_profiles",
    "approved_stage_a_direction_profiles",
    "canonical_direction_derivation_version",
    "coarse_motion_for_direction_label",
    "direction_comparison_facts",
    "direction_compatibility_class",
    "direction_labels_are_intent_compatible",
    "direction_label_names",
    "resolve_affordance_probe_profile",
    "resolve_stage_a_direction_profile",
]
