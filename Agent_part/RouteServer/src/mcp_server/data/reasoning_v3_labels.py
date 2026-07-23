"""Minimal deterministic supervision for Reasoning V3.4.

The module has exactly two responsibilities:

* retain an already validated ``trajectory_ego_t`` for dataset-approved
  System1 supervision; and
* copy a complete, profile-stamped canonical direction record produced by an
  approved adapter.

It never derives direction from trajectory, route/BEV theta, FPV projection,
runtime task phase, goal distance, waypoint state, or route state.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mcp_server.contracts.reasoning_v3 import LabelProvenanceEntryV3, SectionLossMaskV3
from mcp_server.runtime.system1_contract import canonicalize_trajectory_ego_t


REASONING_V3_FIXED_SUPERVISION_VERSION = "reasoning_v3_4_fixed_supervision_v1"

_APPROVED_DIRECTION_PROFILES: dict[str, dict[str, Any]] = {
    "fpv_degree10_terminal_centroid_v2": {
        "profile_version": "2.0",
        "bin_mode": "degree10_with_stop",
        "derivation_version": "pixel_trajectory_terminal_centroid_rule_v2",
        "label_count": 20,
        "candidates": tuple(
            [
                {
                    "name": (
                        f"left_{abs(angle)}"
                        if angle < 0
                        else f"right_{angle}"
                        if angle > 0
                        else "straight"
                    ),
                    "choice": chr(ord("A") + index),
                    "angle_deg": float(angle),
                }
                for index, angle in enumerate(range(-90, 91, 10))
            ]
            + [{"name": "stop", "choice": "T", "angle_deg": None}]
        ),
    },
    "ego_degree10_12way_terminal_centroid_v2": {
        "profile_version": "2.0",
        "bin_mode": "degree10_12way_with_stop",
        "derivation_version": "ego_x_right_y_front_terminal_centroid_rule_v2",
        "label_count": 12,
        "candidates": (
            {"name": "stop", "choice": "A", "angle_deg": None},
            {"name": "straight", "choice": "B", "angle_deg": 0.0},
            {"name": "left_10", "choice": "C", "angle_deg": -10.0},
            {"name": "right_10", "choice": "D", "angle_deg": 10.0},
            {"name": "left_20", "choice": "E", "angle_deg": -20.0},
            {"name": "right_20", "choice": "F", "angle_deg": 20.0},
            {"name": "left_30", "choice": "G", "angle_deg": -30.0},
            {"name": "right_30", "choice": "H", "angle_deg": 30.0},
            {"name": "left_40", "choice": "I", "angle_deg": -40.0},
            {"name": "right_40", "choice": "J", "angle_deg": 40.0},
            {"name": "left_50plus", "choice": "K", "angle_deg": -50.0},
            {"name": "right_50plus", "choice": "L", "angle_deg": 50.0},
        ),
    },
}

_DIRECTION_FLAT_FIELDS = {
    "profile_id": "direction_profile",
    "profile_version": "direction_profile_version",
    "label_index": "direction_choice_label",
    "name": "direction_name",
    "bin_mode": "direction_bin_mode",
    "derivation_version": "direction_derivation_version",
    "source": "direction_source",
    "choice": "direction_choice",
    "angle_deg": "direction_angle_deg",
}


@dataclass(frozen=True)
class DatasetSupervisionPolicyV3:
    """Dataset-specific strong-supervision boundary for V3.4."""

    dataset_source: str
    continuous_trajectory_trusted: bool
    canonical_direction_enabled: bool


@dataclass(frozen=True)
class DeterministicSupervisionV3:
    """Strong labels copied without semantic or geometric inference."""

    dataset_source: str
    canonical_direction: dict[str, Any] | None
    system1_enabled: bool
    teacher_supervision_context: dict[str, Any]
    label_provenance: dict[str, LabelProvenanceEntryV3]
    section_loss_mask: SectionLossMaskV3
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-compatible audit payload."""

        return {
            "dataset_source": self.dataset_source,
            "canonical_direction": self.canonical_direction,
            "system1_enabled": self.system1_enabled,
            "teacher_supervision_context": self.teacher_supervision_context,
            "label_provenance": {
                key: value.model_dump(mode="json") for key, value in self.label_provenance.items()
            },
            "section_loss_mask": self.section_loss_mask.model_dump(mode="json"),
            "warnings": list(self.warnings),
        }


_POLICIES = {
    "sidewalk": DatasetSupervisionPolicyV3("sidewalk", True, True),
    "foxglove": DatasetSupervisionPolicyV3("foxglove", True, True),
    "egowalk": DatasetSupervisionPolicyV3("egowalk", True, True),
    "frodobots": DatasetSupervisionPolicyV3("frodobots", False, False),
}

_DATASET_APPROVED_DIRECTION_PROFILES = {
    "sidewalk": frozenset({"fpv_degree10_terminal_centroid_v2"}),
    "foxglove": frozenset({"fpv_degree10_terminal_centroid_v2"}),
    "egowalk": frozenset({"ego_degree10_12way_terminal_centroid_v2"}),
    "frodobots": frozenset(),
}

_DATASET_ALIASES = {
    "sidewalk-main": "sidewalk",
    "sidewalk-extra": "sidewalk",
    "sidewalk_main": "sidewalk",
    "sidewalk_extra": "sidewalk",
    "foxglove-yaw-p90": "foxglove",
    "foxglove_yaw_p90": "foxglove",
    "egowalk-fpv-only": "egowalk",
    "egowalk_fpv_only": "egowalk",
    "frodobots-qc": "frodobots",
    "frodobots_qc": "frodobots",
}


def dataset_supervision_policy_v3(dataset_source: str) -> DatasetSupervisionPolicyV3:
    """Return the explicit V3.4 trust policy for one dataset family."""

    normalized = _normalize_dataset_source(dataset_source)
    try:
        return _POLICIES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported V3.4 dataset supervision policy: {dataset_source}") from exc


def build_deterministic_supervision_v3(
    row: Mapping[str, Any],
    *,
    dataset_source: str,
) -> DeterministicSupervisionV3:
    """Copy only approved System1 and canonical-direction supervision.

    A row without a direction record remains valid observation-only data.
    A row that advertises a partial, unsupported, or route-theta-derived record
    is rejected rather than repaired or re-binned.
    """

    policy = dataset_supervision_policy_v3(dataset_source)
    warnings: list[str] = []
    provenance: dict[str, LabelProvenanceEntryV3] = {}

    trajectory = _trajectory_value(row)
    system1_enabled = policy.continuous_trajectory_trusted and bool(trajectory)
    if trajectory and not policy.continuous_trajectory_trusted:
        warnings.append(f"{policy.dataset_source}_trajectory_ignored_by_policy")

    canonical_direction: dict[str, Any] | None = None
    if policy.canonical_direction_enabled:
        canonical_direction = _canonical_direction_record_from_row(row)
        if canonical_direction is None:
            warnings.append("canonical_direction_missing_observation_only")
        else:
            approved_profiles = _DATASET_APPROVED_DIRECTION_PROFILES[policy.dataset_source]
            if canonical_direction["profile_id"] not in approved_profiles:
                raise ValueError(
                    "canonical direction profile is not approved for dataset: "
                    f"dataset={policy.dataset_source}, "
                    f"profile={canonical_direction['profile_id']!r}, "
                    f"approved={sorted(approved_profiles)}"
                )
            provenance["decision.direction"] = LabelProvenanceEntryV3(
                source=canonical_direction["source"],
                source_kind="deterministic_adapter",
                supervision="strong",
                student_inferable=True,
                loss_channel="direction_choice",
                generator_version=canonical_direction["derivation_version"],
                details={
                    "profile_id": canonical_direction["profile_id"],
                    "profile_version": canonical_direction["profile_version"],
                    "bin_mode": canonical_direction["bin_mode"],
                },
            )
    elif _row_has_direction_record(row):
        warnings.append(f"{policy.dataset_source}_canonical_direction_ignored_by_policy")

    if system1_enabled:
        provenance["system1.trajectory"] = LabelProvenanceEntryV3(
            source="dataset_trajectory_ego_t",
            source_kind="dataset_gt",
            supervision="strong",
            student_inferable=True,
            loss_channel="system1_action",
            generator_version="system1_trajectory_contract_v1",
            details={
                "dataset_source": policy.dataset_source,
                "target_field": "trajectory_ego_t",
            },
        )

    teacher_context: dict[str, Any] = {
        "dataset_policy": {
            "dataset_source": policy.dataset_source,
            "continuous_trajectory_trusted": policy.continuous_trajectory_trusted,
            "canonical_direction_enabled": policy.canonical_direction_enabled,
            "approved_direction_profiles": sorted(
                _DATASET_APPROVED_DIRECTION_PROFILES[policy.dataset_source]
            ),
        },
        "fixed_supervision_version": REASONING_V3_FIXED_SUPERVISION_VERSION,
        "fixed_labels": {"canonical_direction": canonical_direction},
    }
    return DeterministicSupervisionV3(
        dataset_source=policy.dataset_source,
        canonical_direction=canonical_direction,
        system1_enabled=system1_enabled,
        teacher_supervision_context=teacher_context,
        label_provenance=provenance,
        # Direction uses its own field-level loss channel.  No complete
        # reasoning section is authorized by deterministic supervision.
        section_loss_mask=SectionLossMaskV3(),
        warnings=tuple(warnings),
    )


def _canonical_direction_record_from_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate and copy one complete adapter-produced direction record."""

    raw = _raw_direction_record(row)
    if raw is None:
        return None

    required = (
        "profile_id",
        "profile_version",
        "label_index",
        "name",
        "bin_mode",
        "derivation_version",
        "source",
    )
    missing = [name for name in required if raw.get(name) is None or raw.get(name) == ""]
    if missing:
        raise ValueError(f"canonical direction record is incomplete: missing={missing}")

    profile_id = str(raw["profile_id"]).strip()
    try:
        profile = _APPROVED_DIRECTION_PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"canonical direction profile is not approved: {profile_id!r}") from exc

    profile_version = str(raw["profile_version"]).strip()
    bin_mode = str(raw["bin_mode"]).strip()
    derivation_version = str(raw["derivation_version"]).strip()
    expected_fields = {
        "profile_version": profile["profile_version"],
        "bin_mode": profile["bin_mode"],
        "derivation_version": profile["derivation_version"],
    }
    actual_fields = {
        "profile_version": profile_version,
        "bin_mode": bin_mode,
        "derivation_version": derivation_version,
    }
    mismatches = {
        name: {"expected": expected_fields[name], "actual": actual_fields[name]}
        for name in expected_fields
        if expected_fields[name] != actual_fields[name]
    }
    if mismatches:
        raise ValueError(f"canonical direction profile stamp mismatch: {mismatches}")

    label_index = raw["label_index"]
    if isinstance(label_index, bool) or not isinstance(label_index, int):
        raise ValueError("canonical direction label_index must be an integer")
    if not 0 <= label_index < int(profile["label_count"]):
        raise ValueError(
            "canonical direction label_index is outside the approved profile: "
            f"profile={profile_id}, index={label_index}"
        )

    name = str(raw["name"]).strip()
    source = str(raw["source"]).strip()
    if not name or not source:
        raise ValueError("canonical direction name and source must be non-empty")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,95}", name):
        raise ValueError("canonical direction name must be a lower-snake-case reason code")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,95}", source):
        raise ValueError("canonical direction source must be a lower-snake-case reason code")
    if "route_theta" in source.lower().replace("-", "_"):
        raise ValueError("route_theta direction_source is forbidden in Reasoning V3.4")
    candidate = profile["candidates"][label_index]
    if name != candidate["name"]:
        raise ValueError(
            "canonical direction name does not match profile label_index: "
            f"profile={profile_id}, index={label_index}, "
            f"expected={candidate['name']!r}, actual={name!r}"
        )

    result: dict[str, Any] = {
        "profile_id": profile_id,
        "profile_version": profile_version,
        "label_index": label_index,
        "name": name,
        "bin_mode": bin_mode,
        "derivation_version": derivation_version,
        "source": source,
    }
    choice = raw.get("choice")
    if choice is not None:
        choice = str(choice).strip()
        if not choice:
            raise ValueError("canonical direction choice must be non-empty when provided")
        if len(choice) > 32:
            raise ValueError("canonical direction choice exceeds the 32-character contract")
        if choice != candidate["choice"]:
            raise ValueError(
                "canonical direction choice does not match profile label_index: "
                f"profile={profile_id}, index={label_index}, "
                f"expected={candidate['choice']!r}, actual={choice!r}"
            )
        result["choice"] = choice
    angle = raw.get("angle_deg")
    if angle is not None:
        try:
            angle_value = float(angle)
        except (TypeError, ValueError) as exc:
            raise ValueError("canonical direction angle_deg must be numeric") from exc
        if not math.isfinite(angle_value):
            raise ValueError("canonical direction angle_deg must be finite")
        if not -180.0 <= angle_value <= 180.0:
            raise ValueError("canonical direction angle_deg must lie in [-180, 180]")
        expected_angle = candidate["angle_deg"]
        if expected_angle is None or not math.isclose(
            angle_value,
            float(expected_angle),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "canonical direction angle_deg does not match profile label_index: "
                f"profile={profile_id}, index={label_index}, "
                f"expected={expected_angle!r}, actual={angle_value!r}"
            )
        result["angle_deg"] = angle_value
    elif candidate["angle_deg"] is not None:
        # ``angle_deg`` is optional on the wire for compatibility. The index
        # remains the authoritative target and is already fully validated.
        pass
    return result


def _raw_direction_record(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read a nested canonical record or the canonical flat manifest fields."""

    nested = row.get("canonical_direction")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("canonical_direction must be an object")
        return {
            "profile_id": nested.get("profile_id"),
            "profile_version": nested.get("profile_version", nested.get("version")),
            "label_index": nested.get("label_index", nested.get("index")),
            "name": nested.get("name"),
            "bin_mode": nested.get("bin_mode"),
            "derivation_version": nested.get("derivation_version"),
            "source": nested.get("source"),
            "choice": nested.get("choice"),
            "angle_deg": nested.get("angle_deg", nested.get("angle")),
        }
    if "profile_id" in row and "label_index" in row:
        return {
            "profile_id": row.get("profile_id"),
            "profile_version": row.get("profile_version", row.get("version")),
            "label_index": row.get("label_index"),
            "name": row.get("name"),
            "bin_mode": row.get("bin_mode"),
            "derivation_version": row.get("derivation_version"),
            "source": row.get("source"),
            "choice": row.get("choice"),
            "angle_deg": row.get("angle_deg", row.get("angle")),
        }
    if not any(field in row for field in _DIRECTION_FLAT_FIELDS.values()):
        return None
    return {
        target: row.get(source)
        for target, source in _DIRECTION_FLAT_FIELDS.items()
    }


def _row_has_direction_record(row: Mapping[str, Any]) -> bool:
    """Return whether a row advertises nested or flat canonical direction data."""

    return (
        row.get("canonical_direction") is not None
        or ("profile_id" in row and "label_index" in row)
        or any(field in row for field in _DIRECTION_FLAT_FIELDS.values())
    )


def _trajectory_value(row: Mapping[str, Any]) -> list[list[float]]:
    """Return a validated metric trajectory solely for System1 eligibility."""

    value = row.get("trajectory_ego_t")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    policy = row.get("trajectory_policy") if isinstance(row.get("trajectory_policy"), Mapping) else {}
    source_format = str(
        policy.get("system1_target_format") or row.get("system1_target_format") or ""
    ).strip()
    try:
        return canonicalize_trajectory_ego_t(value, source_format=source_format or None)
    except (TypeError, ValueError):
        return []


def _normalize_dataset_source(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    return _DATASET_ALIASES.get(normalized, normalized)


__all__ = [
    "DatasetSupervisionPolicyV3",
    "DeterministicSupervisionV3",
    "REASONING_V3_FIXED_SUPERVISION_VERSION",
    "build_deterministic_supervision_v3",
    "dataset_supervision_policy_v3",
]
