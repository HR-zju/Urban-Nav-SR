"""Typed image-plane geometry for local pedestrian traversability.

This contract is intentionally separate from language-goal anchors.  A language
anchor can be a landmark bounding box, whereas a traversability geometry is a
locally safe or unsafe support region in the current FPV image.  The first
version uses only normalized image coordinates and never claims metric ground
projection without a dataset-specific calibrated sidecar.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


LOCAL_TRAVERSABILITY_GEOMETRY_SCHEMA_VERSION_V1 = "urbannav_local_traversability_geometry_v1"
LOCAL_TRAVERSABILITY_GEOMETRY_COORDINATE_FRAME_V1 = "image_uv_norm_v1"

TraversabilityRegionRoleV1 = Literal[
    "direct_traversable",
    "conditional_traversable",
    "blocked",
    "forbidden",
]
TraversabilitySurfaceTypeV1 = Literal[
    "sidewalk",
    "crosswalk",
    "ramp",
    "tactile_paving",
    "shared_path",
    "road_edge",
    "grass",
    "vegetation",
    "stairs",
    "curb",
    "vehicle",
    "pedestrian_cluster",
    "wall_or_fence",
    "construction",
    "unknown",
]
TraversabilityGeometryStatusV1 = Literal["accepted", "rejected", "pending_human"]
GeometryConfidenceV1 = Literal["high", "medium", "low"]


class ImagePointUVV1(BaseModel):
    """One finite point in normalized image coordinates.

    Args:
        u: Horizontal coordinate in ``[0, 1]`` from left to right.
        v: Vertical coordinate in ``[0, 1]`` from top to bottom.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    u: float = Field(ge=0.0, le=1.0)
    v: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_finite(self) -> "ImagePointUVV1":
        """Reject non-finite coordinates even when provider JSON parses."""

        if not math.isfinite(self.u) or not math.isfinite(self.v):
            raise ValueError("image point must be finite")
        return self


class PerspectiveQuadrilateralUVV1(BaseModel):
    """A perspective-aware local corridor quadrilateral.

    Vertices are explicitly ordered as ``near_left``, ``near_right``,
    ``far_right``, ``far_left``.  This avoids treating an axis-aligned bounding
    box as a corridor and makes the near/far image geometry auditable.

    Args:
        near_left: Lower-left boundary of the locally reachable corridor.
        near_right: Lower-right boundary of the locally reachable corridor.
        far_right: Far-right boundary of the same corridor.
        far_left: Far-left boundary of the same corridor.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    near_left: ImagePointUVV1
    near_right: ImagePointUVV1
    far_right: ImagePointUVV1
    far_left: ImagePointUVV1

    @model_validator(mode="after")
    def validate_perspective_quadrilateral(self) -> "PerspectiveQuadrilateralUVV1":
        r"""Require a non-self-intersecting, non-degenerate near-to-far quad.

        The signed shoelace area is evaluated in normalized image space:

        \[
        A = \frac{1}{2}\left|\sum_i u_i v_{i+1} - v_i u_{i+1}\right|.
        \]

        A valid local corridor must be convex, have a visible near-to-far
        ordering, and occupy a non-trivial but not almost full-image area.  It
        is deliberately not required to be rectangular.
        """

        points = self.points
        area = abs(_shoelace_area(points))
        if area < 0.0015:
            raise ValueError("quadrilateral_area_too_small")
        if area > 0.82:
            raise ValueError("quadrilateral_area_too_large")
        if not _is_strictly_convex(points):
            raise ValueError("quadrilateral_must_be_convex_and_non_self_intersecting")
        near_v = (self.near_left.v + self.near_right.v) / 2.0
        far_v = (self.far_left.v + self.far_right.v) / 2.0
        # 近边必须在图像中明显低于远边，否则不能代表当前可执行走廊。
        if near_v - far_v < 0.015:
            raise ValueError("quadrilateral_near_edge_must_be_below_far_edge")
        if self.near_left.u >= self.near_right.u:
            raise ValueError("quadrilateral_near_left_must_precede_near_right")
        if self.far_left.u >= self.far_right.u:
            raise ValueError("quadrilateral_far_left_must_precede_far_right")
        return self

    @property
    def points(self) -> tuple[ImagePointUVV1, ImagePointUVV1, ImagePointUVV1, ImagePointUVV1]:
        """Return vertices in canonical perimeter order."""

        return self.near_left, self.near_right, self.far_right, self.far_left

    @property
    def area_ratio(self) -> float:
        """Return normalized image-plane area enclosed by the quadrilateral."""

        return abs(_shoelace_area(self.points))


class TraversabilityRegionV1(BaseModel):
    """One local FPV region classified by physical walking support.

    Args:
        region_id: Stable row-local region identifier.
        role: Physical support state, never a route decision.
        surface_type: Dominant visible support or obstacle category.
        quadrilateral_uv: Explicit perspective-aware image support geometry.
        limiting_factors: Compact visible reasons limiting use of the region.
        evidence_refs: Current-image evidence references supporting this region.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    region_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    role: TraversabilityRegionRoleV1
    surface_type: TraversabilitySurfaceTypeV1
    quadrilateral_uv: PerspectiveQuadrilateralUVV1
    limiting_factors: tuple[str, ...] = Field(default=(), max_length=8)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_region_evidence(self) -> "TraversabilityRegionV1":
        """Require every geometry label to be grounded in the current FPV."""

        if "fpv.current" not in self.evidence_refs:
            raise ValueError("traversability_region_requires_fpv_current_evidence")
        if self.role in {"direct_traversable", "conditional_traversable"} and self.surface_type in {
            "grass",
            "vegetation",
            "stairs",
            "vehicle",
            "pedestrian_cluster",
            "wall_or_fence",
            "construction",
        }:
            raise ValueError("traversable_region_has_non_walkable_surface_type")
        if self.role == "conditional_traversable" and not self.limiting_factors:
            raise ValueError("conditional_traversable_region_requires_visible_limiting_factor")
        return self


class LocalTraversabilityGeometryV1(BaseModel):
    """One versioned direct-LVLM local-traversability sidecar.

    This is a sidecar rather than a replacement for ``LocalTraversabilityV4``.
    Existing V4.6 FullCoT manifests therefore remain readable while a future
    training projection can opt into geometry supervision explicitly.

    Args:
        geometry_id: Content-addressed annotation identity.
        parent_sample_id: Stable source parent identity.
        source_fingerprint: Exact canonical source revision.
        dataset_source: Dataset family of the source observation.
        regions: Visible support and obstacle quadrilaterals.
        selected_corridor_region_id: Region selected as the local corridor, if
            a safe or conditionally safe corridor is visible.
        validator_status: Teacher/validator outcome for this sidecar.
        teacher_model: Provider-reported model identity.
        prompt_profile_id: Frozen direct-LVLM profile identity.
        prompt_profile_digest: Digest of prompt and output-schema contract.
        confidence: Overall geometry confidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[LOCAL_TRAVERSABILITY_GEOMETRY_SCHEMA_VERSION_V1] = (
        LOCAL_TRAVERSABILITY_GEOMETRY_SCHEMA_VERSION_V1
    )
    coordinate_frame: Literal[LOCAL_TRAVERSABILITY_GEOMETRY_COORDINATE_FRAME_V1] = (
        LOCAL_TRAVERSABILITY_GEOMETRY_COORDINATE_FRAME_V1
    )
    geometry_id: str = Field(min_length=1, max_length=160)
    parent_sample_id: str = Field(min_length=1, max_length=480)
    source_fingerprint: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    dataset_source: str = Field(min_length=1, max_length=128)
    regions: tuple[TraversabilityRegionV1, ...] = Field(default=(), max_length=12)
    selected_corridor_region_id: str | None = Field(default=None, max_length=64)
    validator_status: TraversabilityGeometryStatusV1
    teacher_model: str = Field(min_length=1, max_length=256)
    prompt_profile_id: str = Field(min_length=1, max_length=160)
    prompt_profile_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    confidence: GeometryConfidenceV1 | None = None
    warnings: tuple[str, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def validate_annotation_state(self) -> "LocalTraversabilityGeometryV1":
        """Keep accepted geometry fail-closed and reviewable."""

        ids = [region.region_id for region in self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError("traversability_geometry_region_ids_must_be_unique")
        selected = None
        if self.selected_corridor_region_id is not None:
            selected = next(
                (region for region in self.regions if region.region_id == self.selected_corridor_region_id),
                None,
            )
            if selected is None:
                raise ValueError("selected_corridor_region_id_not_found")
            if selected.role not in {"direct_traversable", "conditional_traversable"}:
                raise ValueError("selected_corridor_must_be_traversable_or_conditional")
        if self.validator_status == "accepted":
            if selected is None:
                raise ValueError("accepted_geometry_requires_selected_corridor")
            if self.confidence not in {"high", "medium"}:
                raise ValueError("accepted_geometry_requires_high_or_medium_confidence")
        elif self.selected_corridor_region_id is not None:
            raise ValueError("nonaccepted_geometry_must_not_select_corridor")
        return self


def local_traversability_geometry_id_v1(
    *,
    parent_sample_id: str,
    source_fingerprint: str,
    prompt_profile_id: str,
    prompt_profile_digest: str,
) -> str:
    """Return a stable geometry sidecar identity.

    Args:
        parent_sample_id: Stable physical parent identity.
        source_fingerprint: Exact source revision.
        prompt_profile_id: Teacher prompt profile identifier.
        prompt_profile_digest: Exact profile digest.

    Returns:
        Content-addressed sidecar identity.  The profile is part of the key so
        changed prompts never silently reuse stale cached quadrilaterals.
    """

    payload = {
        "parent_sample_id": parent_sample_id,
        "source_fingerprint": source_fingerprint,
        "prompt_profile_id": prompt_profile_id,
        "prompt_profile_digest": prompt_profile_digest,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    return f"local_traversability_geometry:{digest}"


def _shoelace_area(points: tuple[ImagePointUVV1, ...]) -> float:
    """Return signed polygon area in normalized image coordinates."""

    total = 0.0
    for current, following in zip(points, points[1:] + points[:1]):
        total += current.u * following.v - current.v * following.u
    return total / 2.0


def _is_strictly_convex(points: tuple[ImagePointUVV1, ...]) -> bool:
    """Check one ordered quadrilateral for a single non-zero turn orientation."""

    turn_signs: list[float] = []
    for first, second, third in zip(points, points[1:] + points[:1], points[2:] + points[:2]):
        cross = (second.u - first.u) * (third.v - second.v) - (second.v - first.v) * (third.u - second.u)
        if abs(cross) <= 1e-7:
            return False
        turn_signs.append(math.copysign(1.0, cross))
    return all(sign == turn_signs[0] for sign in turn_signs[1:])


__all__ = [
    "GeometryConfidenceV1",
    "ImagePointUVV1",
    "LOCAL_TRAVERSABILITY_GEOMETRY_COORDINATE_FRAME_V1",
    "LOCAL_TRAVERSABILITY_GEOMETRY_SCHEMA_VERSION_V1",
    "LocalTraversabilityGeometryV1",
    "PerspectiveQuadrilateralUVV1",
    "TraversabilityGeometryStatusV1",
    "TraversabilityRegionRoleV1",
    "TraversabilityRegionV1",
    "TraversabilitySurfaceTypeV1",
    "local_traversability_geometry_id_v1",
]
