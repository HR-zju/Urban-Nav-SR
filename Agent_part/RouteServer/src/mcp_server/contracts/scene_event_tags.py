"""Typed CityWalker-compatible scene-event slice contracts.

This schema records a deterministic, evidence-preserving projection from
UrbanNav quality sidecars.  It is deliberately not a replacement for the
human-authored CityWalker categories: ``action_target_mismatch`` remains
unavailable unless an independent action/target audit is provided.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCENE_EVENT_TAGS_SCHEMA_VERSION_V1 = "urbannav_scene_event_tags_v1"
CITYWALKER_COMPATIBLE_SCENE_EVENT_TAXONOMY_ID_V1 = "citywalker_compatible_scene_events_v1"
CITYWALKER_COMPATIBLE_SCENE_EVENT_TAG_ORDER_V1 = (
    "crowd",
    "person_close_by",
    "turn",
    "action_target_mismatch",
    "crossing",
    "other",
)

SceneEventTagNameV1 = Literal[
    "crowd",
    "person_close_by",
    "turn",
    "action_target_mismatch",
    "crossing",
    "other",
]
SceneEventTagStateV1 = Literal["present", "absent", "unavailable"]
SceneEventTagProvenanceV1 = Literal[
    "motion_episode_summary",
    "motion_scene_assessment",
    "scene_suitability_assessment",
    "local_traversability_geometry",
    "direction_supervision",
    "independent_action_target_audit_unavailable",
]


class SceneEventTagV1(BaseModel):
    """One overlap-permitted audit tag with explicit evidence availability.

    Args:
        name: Stable CityWalker-compatible tag name.
        state: ``present`` only when the listed sidecar evidence supports it;
            ``absent`` when applicable evidence is available but does not
            trigger the event; ``unavailable`` otherwise.
        provenance: Sidecars or source-supervision channels consulted by the
            deterministic projection.
        evidence_refs: Prompt-addressable visual references, when available.
        reason_codes: Stable projection reasons, never a free-text rationale.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: SceneEventTagNameV1
    state: SceneEventTagStateV1
    provenance: tuple[SceneEventTagProvenanceV1, ...] = Field(min_length=1, max_length=4)
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=12)
    reason_codes: tuple[str, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def validate_evidence_availability(self) -> "SceneEventTagV1":
        """Prevent unavailable tags from masquerading as grounded negatives."""

        if self.state == "unavailable" and self.evidence_refs:
            raise ValueError("unavailable_scene_event_tag_must_not_claim_evidence_refs")
        if self.state != "unavailable" and not self.reason_codes:
            raise ValueError("available_scene_event_tag_requires_reason_code")
        return self


class SceneEventTagsV1(BaseModel):
    """One immutable multi-label scene-event projection sidecar.

    Args:
        projection_id: Content-addressed identity for this projection policy.
        parent_sample_id: Stable physical source parent.
        source_fingerprint: Exact canonical source revision.
        dataset_source: Dataset family associated with the source.
        filter_decision_id: Exact quality-filter sidecar used, when current.
        geometry_id: Exact traversability sidecar used, when current.
        tags: One ordered state for every taxonomy member.
        warnings: Missing/stale sidecar findings retained for audit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[SCENE_EVENT_TAGS_SCHEMA_VERSION_V1] = SCENE_EVENT_TAGS_SCHEMA_VERSION_V1
    taxonomy_id: Literal[CITYWALKER_COMPATIBLE_SCENE_EVENT_TAXONOMY_ID_V1] = (
        CITYWALKER_COMPATIBLE_SCENE_EVENT_TAXONOMY_ID_V1
    )
    projection_id: str = Field(min_length=1, max_length=160)
    parent_sample_id: str = Field(min_length=1, max_length=480)
    source_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    dataset_source: str = Field(min_length=1, max_length=128)
    filter_decision_id: str | None = Field(default=None, max_length=160)
    geometry_id: str | None = Field(default=None, max_length=160)
    tags: tuple[SceneEventTagV1, ...] = Field(min_length=6, max_length=6)
    warnings: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_complete_taxonomy(self) -> "SceneEventTagsV1":
        """Require one ordered state per multi-label taxonomy member."""

        names = tuple(tag.name for tag in self.tags)
        if names != CITYWALKER_COMPATIBLE_SCENE_EVENT_TAG_ORDER_V1:
            raise ValueError("scene_event_tags_must_match_canonical_taxonomy_order")
        return self


def scene_event_projection_id_v1(
    *,
    parent_sample_id: str,
    source_fingerprint: str,
    filter_decision_id: str | None,
    geometry_id: str | None,
) -> str:
    """Return a cache-safe identity for one deterministic event projection.

    Args:
        parent_sample_id: Stable physical source parent identity.
        source_fingerprint: Exact canonical source revision.
        filter_decision_id: Current trainability sidecar identity, if any.
        geometry_id: Current geometry sidecar identity, if any.

    Returns:
        A content-addressed projection identifier.  Sidecar identities are in
        the digest so repaired geometry/filter results cannot reuse old tags.
    """

    payload = {
        "parent_sample_id": parent_sample_id,
        "source_fingerprint": source_fingerprint,
        "filter_decision_id": filter_decision_id,
        "geometry_id": geometry_id,
        "taxonomy_id": CITYWALKER_COMPATIBLE_SCENE_EVENT_TAXONOMY_ID_V1,
        "schema_version": SCENE_EVENT_TAGS_SCHEMA_VERSION_V1,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"scene_event_tags:{digest}"


__all__ = [
    "CITYWALKER_COMPATIBLE_SCENE_EVENT_TAG_ORDER_V1",
    "CITYWALKER_COMPATIBLE_SCENE_EVENT_TAXONOMY_ID_V1",
    "SCENE_EVENT_TAGS_SCHEMA_VERSION_V1",
    "SceneEventTagNameV1",
    "SceneEventTagProvenanceV1",
    "SceneEventTagStateV1",
    "SceneEventTagV1",
    "SceneEventTagsV1",
    "scene_event_projection_id_v1",
]
