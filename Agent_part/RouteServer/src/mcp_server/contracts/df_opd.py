"""Canonical UrbanNav contracts for Decision-Flow On-Policy Distillation.

This RouteServer-owned module defines semantic sample identity and privileged
Teacher media.  It does not render cues, tokenize prompts, or train models;
those physical concerns remain in the data exporter and ms-swift respectively.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION = "urbannav_df_opd_canonical_sample_v1"
DF_OPD_PROFILE_SCHEMA_VERSION = "df_opd_profile_v1"
DF_OPD_REPLAY_SCHEMA_VERSION = "urbannav_df_opd_replay_v1"
DF_OPD_PROFILE_2S16 = "df_opd_2s16_q16_v1"
DF_OPD_PROFILE_4S32 = "df_opd_4s32_q16_v1"
DF_OPD_TEACHER_CUE_ROLE = "teacher.future_motion_cue"
DF_OPD_ORACLE_CUE_ROLE = "teacher.exact_future_trajectory_oracle"

DFOPDTeacherConditionV1 = Literal[
    "same_input_t0",
    "correct_privilege_tc",
    "shuffled_privilege_ts",
    "exact_future_oracle_to",
]
_DF_OPD_DECISION_FIELDS = (
    "selected_action_label",
    "speed",
    "response",
)
_DF_OPD_SPEED_VALUES = {"normal", "slow", "stop"}
_DF_OPD_RESPONSE_VALUES = {"proceed", "slow", "yield", "detour", "stop"}


class DFOPDProfileV1(BaseModel):
    """Freeze one independent DF-OPD horizon/query checkpoint profile.

    Args:
        profile_id: Public, versioned profile identity.
        target_horizon_steps: Trajectory/action target length ``T``.
        query_count: Forced trajectory-query count ``Q``.
        dt_sec: Time step in seconds.
        action_dim: FLU pose dimensions, fixed to three in V1.
        flow_solver: Deployment ODE solver identity.
        default_flow_steps: Fixed default Euler step count.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[DF_OPD_PROFILE_SCHEMA_VERSION] = DF_OPD_PROFILE_SCHEMA_VERSION
    profile_id: Literal[DF_OPD_PROFILE_2S16, DF_OPD_PROFILE_4S32]
    target_horizon_steps: Literal[16, 32]
    query_count: Literal[16] = 16
    dt_sec: Literal[0.125] = 0.125
    action_dim: Literal[3] = 3
    flow_solver: Literal["euler_v1"] = "euler_v1"
    default_flow_steps: Literal[5] = 5

    @model_validator(mode="after")
    def validate_profile(self) -> "DFOPDProfileV1":
        """Require each named profile to carry its one permitted horizon.

        Returns:
            The validated immutable profile.

        Raises:
            ValueError: If the profile ID and horizon do not match.
        """

        expected_horizon = {
            DF_OPD_PROFILE_2S16: 16,
            DF_OPD_PROFILE_4S32: 32,
        }[self.profile_id]
        if self.target_horizon_steps != expected_horizon:
            raise ValueError("DF-OPD profile_id and target_horizon_steps mismatch")
        return self

    @property
    def profile_digest(self) -> str:
        """Return the immutable SHA-256 binding for manifests and checkpoints."""

        return df_opd_digest_v1(self.model_dump(mode="json"))


DF_OPD_PROFILES_V1: dict[str, DFOPDProfileV1] = {
    DF_OPD_PROFILE_2S16: DFOPDProfileV1(
        profile_id=DF_OPD_PROFILE_2S16,
        target_horizon_steps=16,
    ),
    DF_OPD_PROFILE_4S32: DFOPDProfileV1(
        profile_id=DF_OPD_PROFILE_4S32,
        target_horizon_steps=32,
    ),
}


class DFOPDTeacherViewV1(BaseModel):
    """Canonical paired Teacher media, separated from Student-visible media.

    Args:
        condition: T0, TC, TS, or audit-only TO privilege condition.
        images: Full ordered Teacher media sequence.
    image_roles: Roles aligned one-to-one with ``images``.
    cue_class: Coarse cue category; absent only for T0.
    privilege_policy_digest: Exact cue-rendering policy SHA-256.
    cue_artifact_sha256: SHA-256 of the rendered private cue; absent only for
        T0 because no cue image exists in that condition.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition: DFOPDTeacherConditionV1
    images: tuple[str, ...] = Field(min_length=1)
    image_roles: tuple[str, ...] = Field(min_length=1)
    cue_class: str | None = Field(default=None, min_length=1, max_length=128)
    privilege_policy_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    cue_artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_internal_shape(self) -> "DFOPDTeacherViewV1":
        """Require aligned, unique media roles and condition-specific cue fields."""

        if len(self.images) != len(self.image_roles):
            raise ValueError("Teacher images/image_roles length mismatch")
        if len(set(self.image_roles)) != len(self.image_roles):
            raise ValueError("Teacher image_roles must be unique")
        if self.condition == "same_input_t0":
            if self.cue_class is not None or self.cue_artifact_sha256 is not None:
                raise ValueError("T0 cannot carry a private cue artifact")
        elif self.cue_class is None or self.cue_artifact_sha256 is None:
            raise ValueError("privileged Teacher views require a cue class and artifact digest")
        return self

    def validate_against_student(
        self,
        student_images: Sequence[str],
        student_image_roles: Sequence[str],
        *,
        trainable: bool,
    ) -> None:
        """Verify the Student prefix and reject exact-future oracle export.

        Args:
            student_images: Ordered Student-visible media paths.
            student_image_roles: Roles paired with Student media.
            trainable: Whether the target is a Student/Teacher training view.

        Raises:
            ValueError: If media differ, a role is invalid, or TO is trainable.
        """

        images = tuple(student_images)
        roles = tuple(student_image_roles)
        if not images or len(images) != len(roles):
            raise ValueError("Student images/image_roles must be non-empty and aligned")
        if self.condition == "same_input_t0":
            if self.images != images or self.image_roles != roles:
                raise ValueError("T0 must exactly equal Student media")
            return
        if self.images[:-1] != images or self.image_roles[:-1] != roles:
            raise ValueError("privileged Teacher view must retain the complete Student media prefix")
        expected_role = (
            DF_OPD_ORACLE_CUE_ROLE
            if self.condition == "exact_future_oracle_to"
            else DF_OPD_TEACHER_CUE_ROLE
        )
        if self.image_roles[-1] != expected_role:
            raise ValueError("Teacher privilege cue role mismatch")
        if trainable and self.condition in {"shuffled_privilege_ts", "exact_future_oracle_to"}:
            raise ValueError("TS and TO Teacher views are evaluation-only and cannot be exported for training")


class DFOPDCanonicalSampleV1(BaseModel):
    """One immutable canonical DF-OPD sample before physical training export.

    Args:
        sample_id: Stable canonical sample identity.
        source_group_id: Leakage-control group identity.
        split_group_id: Frozen train/eval split identity.
        dataset_source: Dataset family.
        profile: Target/checkpoint profile.
        images: Student-visible media only.
        image_roles: Roles paired with Student media.
        decision_target: Canonical GT Decision retained for replay only.
        trajectory_ego_t: Metric FLU cumulative future pose target.
        action_mask: Valid prefix mask for target points.
        teacher_views: All four T0/TC/TS/TO Teacher media views, keyed by
            their explicit condition and kept separate from Student media.
        canonical_digest: SHA-256 over every other field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION] = DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION
    sample_id: str = Field(min_length=1, max_length=480)
    source_group_id: str = Field(min_length=1, max_length=240)
    split_group_id: str = Field(min_length=1, max_length=240)
    dataset_source: str = Field(min_length=1, max_length=128)
    profile: DFOPDProfileV1
    images: tuple[str, ...] = Field(min_length=1)
    image_roles: tuple[str, ...] = Field(min_length=1)
    decision_target: str = Field(min_length=1, max_length=2000)
    trajectory_ego_t: tuple[tuple[float, float, float], ...] = Field(min_length=16, max_length=32)
    action_mask: tuple[bool, ...] = Field(min_length=16, max_length=32)
    teacher_views: tuple[DFOPDTeacherViewV1, ...] = Field(min_length=4, max_length=4)
    canonical_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_canonical_sample(self) -> "DFOPDCanonicalSampleV1":
        """Bind target shape, media separation and immutable sample digest.

        Returns:
            The validated canonical sample.

        Raises:
            ValueError: If any profile/media/target/digest invariant fails.
        """

        horizon = self.profile.target_horizon_steps
        _validate_canonical_decision_target(self.decision_target)
        if len(self.images) != len(self.image_roles):
            raise ValueError("Student images/image_roles length mismatch")
        if len(set(self.image_roles)) != len(self.image_roles):
            raise ValueError("Student image_roles must be unique")
        if len(self.trajectory_ego_t) != horizon or len(self.action_mask) != horizon:
            raise ValueError("DF-OPD target/action_mask length must equal profile horizon")
        if not any(self.action_mask):
            raise ValueError("DF-OPD action_mask requires at least one valid target step")
        if any((not isinstance(value, bool)) for value in self.action_mask):
            raise ValueError("DF-OPD action_mask must contain booleans")
        if any((not self.action_mask[index] and self.action_mask[index + 1])
               for index in range(len(self.action_mask) - 1)):
            raise ValueError("DF-OPD action_mask must be a valid-prefix mask")
        views_by_condition = {view.condition: view for view in self.teacher_views}
        if len(views_by_condition) != len(self.teacher_views) or set(views_by_condition) != {
            "same_input_t0",
            "correct_privilege_tc",
            "shuffled_privilege_ts",
            "exact_future_oracle_to",
        }:
            raise ValueError("DF-OPD canonical samples require exactly one T0/TC/TS/TO Teacher view")
        for view in self.teacher_views:
            view.validate_against_student(self.images, self.image_roles, trainable=False)
        if (
            views_by_condition["correct_privilege_tc"].cue_class
            == views_by_condition["shuffled_privilege_ts"].cue_class
        ):
            raise ValueError("TS cue_class must differ from TC cue_class in one canonical sample")
        expected = df_opd_digest_v1(self.model_dump(mode="json", exclude={"canonical_digest"}))
        if self.canonical_digest != expected:
            raise ValueError("DF-OPD canonical_digest mismatch")
        return self

    def to_trainable_physical_row(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        teacher_condition: Literal["same_input_t0", "correct_privilege_tc"],
    ) -> dict[str, Any]:
        """Export Student/Teacher physical fields while quarantining TO.

        Args:
            messages: Student-visible chat messages; copied unchanged to T0/TC.
            teacher_condition: Explicit trainable Teacher view selector.  TS
                and TO are evaluation-only and cannot enter this exporter.

        Returns:
            A JSON-safe ms-swift physical row with separate Teacher fields.

        Raises:
            ValueError: If a non-trainable view is requested or messages leak
                a private oracle key.
        """

        view = self.teacher_view_for_condition(teacher_condition)
        view.validate_against_student(self.images, self.image_roles, trainable=True)
        safe_messages = [dict(item) for item in messages]
        _reject_private_keys(safe_messages)
        return {
            "sample_id": self.sample_id,
            "source_group_id": self.source_group_id,
            "split_group_id": self.split_group_id,
            "dataset_source": self.dataset_source,
            "images": list(self.images),
            "image_roles": list(self.image_roles),
            "messages": safe_messages,
            "teacher_messages": [dict(item) for item in safe_messages],
            "teacher_images": list(view.images),
            "teacher_image_roles": list(view.image_roles),
            "df_opd_profile_id": self.profile.profile_id,
            "df_opd_profile_digest": self.profile.profile_digest,
            "df_opd_teacher_condition": view.condition,
            "df_opd_privilege_policy_digest": view.privilege_policy_digest,
            "df_opd_teacher_cue_sha256": view.cue_artifact_sha256,
            "decision_target": self.decision_target,
            "trajectory_ego_t": [list(point) for point in self.trajectory_ego_t],
            "action_mask": list(self.action_mask),
            "canonical_digest": self.canonical_digest,
        }

    def to_teacher_evaluation_physical_row(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        teacher_condition: DFOPDTeacherConditionV1,
    ) -> dict[str, Any]:
        """Export one private Teacher-only inference row without GT targets.

        Unlike :meth:`to_trainable_physical_row`, this method allows all four
        frozen T0/TC/TS/TO conditions because TS and TO are required to audit
        a trained Teacher.  The output deliberately uses ordinary
        ``messages``/``images`` keys so the registered multimodal template
        receives exactly the selected Teacher view; it never serializes the
        Student view, replay targets, or a privileged target text channel.

        Args:
            messages: Student-visible chat messages copied unchanged to the
                selected Teacher view.
            teacher_condition: Explicit T0, TC, TS, or audit-only TO view.

        Returns:
            Private evaluation-only row with Teacher media and immutable
            identity/profile/cue bindings, but no Decision or trajectory GT.

        Raises:
            ValueError: If messages contain a forbidden private text field or
                the requested Teacher view is missing or malformed.
        """

        view = self.teacher_view_for_condition(teacher_condition)
        view.validate_against_student(self.images, self.image_roles, trainable=False)
        safe_messages = [dict(item) for item in messages]
        _reject_private_keys(safe_messages)
        return {
            "sample_id": self.sample_id,
            "source_group_id": self.source_group_id,
            "split_group_id": self.split_group_id,
            "dataset_source": self.dataset_source,
            "messages": safe_messages,
            "images": list(view.images),
            "image_roles": list(view.image_roles),
            "df_opd_profile_id": self.profile.profile_id,
            "df_opd_profile_digest": self.profile.profile_digest,
            "df_opd_teacher_condition": view.condition,
            "df_opd_privilege_policy_digest": view.privilege_policy_digest,
            "df_opd_teacher_cue_sha256": view.cue_artifact_sha256,
            "canonical_digest": self.canonical_digest,
            "df_opd_evaluation_only": True,
            "private_artifact": True,
        }

    def teacher_view_for_condition(self, condition: DFOPDTeacherConditionV1) -> DFOPDTeacherViewV1:
        """Return one canonical Teacher view by its explicit condition.

        Args:
            condition: T0, TC, TS, or TO selector.

        Returns:
            The selected immutable Teacher media view.

        Raises:
            ValueError: If the canonical view set lacks the requested condition.
        """

        for view in self.teacher_views:
            if view.condition == condition:
                return view
        raise ValueError(f"canonical DF-OPD sample lacks Teacher view {condition!r}")


def choose_ts_donor_v1(
    *,
    sample_id: str,
    own_cue_class: str,
    candidates: Mapping[str, str],
    seed: int,
) -> str:
    """Choose a deterministic shuffled-cue donor with a distinct class.

    Args:
        sample_id: Recipient canonical identity.
        own_cue_class: Recipient's correct coarse cue class.
        candidates: Mapping from donor ID to donor coarse cue class.
        seed: Frozen derangement seed.

    Returns:
        A donor ID whose cue class differs from ``own_cue_class``.

    Raises:
        ValueError: If no valid different-class donor exists.
    """

    eligible = sorted(
        donor_id
        for donor_id, cue_class in candidates.items()
        if donor_id != sample_id and cue_class != own_cue_class
    )
    if not eligible:
        raise ValueError("TS derangement requires a donor with a different coarse cue class")
    payload = f"{seed}:{sample_id}:{own_cue_class}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(eligible)
    return eligible[index]


def df_opd_digest_v1(value: Mapping[str, Any]) -> str:
    """Return the canonical SHA-256 digest for one JSON-safe DF-OPD payload."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_canonical_decision_target(value: str) -> None:
    """Require the standalone Decision wire form accepted by DF-OPD training.

    The canonical contract is mirrored into ms-swift without RouteServer
    imports, so this intentionally encodes only the stable V4 Decision
    surface: exact wrapper/key order, lowercase ReasonCode shape, speed and
    response domains, and their shared stop/slow invariants.  The generated
    Reasoning V4 codec remains the downstream tokenizer/span authority.
    """

    if not isinstance(value, str) or not value.startswith("<Decision>") or not value.endswith("</Decision>"):
        raise ValueError("DF-OPD decision_target must be a standalone canonical <Decision> section")
    try:
        payload = json.loads(value[len("<Decision>"):-len("</Decision>")])
    except json.JSONDecodeError as exc:
        raise ValueError("DF-OPD decision_target contains invalid JSON") from exc
    if not isinstance(payload, Mapping) or tuple(payload.keys()) != _DF_OPD_DECISION_FIELDS:
        raise ValueError("DF-OPD decision_target fields must be canonical and ordered")
    action = payload["selected_action_label"]
    speed = payload["speed"]
    response = payload["response"]
    if not _is_reason_code(action):
        raise ValueError("DF-OPD decision_target selected_action_label must be a ReasonCode")
    if speed not in _DF_OPD_SPEED_VALUES or response not in _DF_OPD_RESPONSE_VALUES:
        raise ValueError("DF-OPD decision_target speed or response is outside the V4 Decision domain")
    is_stop = action == "stop"
    if is_stop and (speed != "stop" or response != "stop"):
        raise ValueError("DF-OPD stop Decision requires speed=response=stop")
    if not is_stop and (speed == "stop" or response == "stop"):
        raise ValueError("DF-OPD moving Decision cannot use stop speed or response")
    if response == "slow" and speed != "slow":
        raise ValueError("DF-OPD Decision response=slow requires speed=slow")
    if response in {"yield", "detour"} and speed != "slow":
        raise ValueError("DF-OPD Decision yield/detour requires speed=slow")
    canonical = "<Decision>" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "</Decision>"
    if value != canonical:
        raise ValueError("DF-OPD decision_target must use exact canonical JSON rendering")


def _is_reason_code(value: Any) -> bool:
    """Check the package-independent ReasonCode lexical surface."""

    if not isinstance(value, str) or not 1 <= len(value) <= 96:
        return False
    if value[0] not in "abcdefghijklmnopqrstuvwxyz":
        return False
    return all(character in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in value)


def _reject_private_keys(value: Any) -> None:
    """Reject known oracle fields recursively from Student-visible messages."""

    forbidden = {"exact_future_trajectory", "teacher.exact_future_trajectory_oracle", "oracle_images"}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in forbidden:
                raise ValueError(f"Student messages contain private DF-OPD field: {key}")
            _reject_private_keys(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_private_keys(item)


__all__ = [
    "DFOPDCanonicalSampleV1",
    "DFOPDProfileV1",
    "DFOPDTeacherViewV1",
    "DF_OPD_CANONICAL_SAMPLE_SCHEMA_VERSION",
    "DF_OPD_ORACLE_CUE_ROLE",
    "DF_OPD_PROFILE_2S16",
    "DF_OPD_PROFILE_4S32",
    "DF_OPD_PROFILES_V1",
    "DF_OPD_REPLAY_SCHEMA_VERSION",
    "DF_OPD_TEACHER_CUE_ROLE",
    "choose_ts_donor_v1",
    "df_opd_digest_v1",
]
