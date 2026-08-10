"""Direct-LVLM quadrilateral annotation for local FPV traversability."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from mcp_server.contracts.local_traversability_geometry import (
    LOCAL_TRAVERSABILITY_GEOMETRY_SCHEMA_VERSION_V1,
    LocalTraversabilityGeometryV1,
    local_traversability_geometry_id_v1,
)
from mcp_server.contracts.reasoning_dataset_v4 import ReasoningSourceV4


TRAVERSABILITY_GEOMETRY_LVLM_PROFILE_ID_V1 = "urbannav_local_traversability_quad_direct_v1"
TRAVERSABILITY_GEOMETRY_LVLM_PRODUCER_VERSION_V1 = "urbannav_local_traversability_quad_teacher_v1_2"

# 仅处理与受控 ontology 一一对应的词表别名；绝不修改 teacher 的几何坐标或角色判断。
_SURFACE_TYPE_ALIASES_V1 = {
    "road": "road_edge",
}

TRAVERSABILITY_GEOMETRY_LVLM_SYSTEM_PROMPT_V1 = """\
You are an offline annotation teacher for local pedestrian traversability in an
urban first-person image. You receive exactly one current FPV image. Identify
only visible local support regions; FPV is authoritative for immediate surface
safety. Do not infer free space behind occlusion or outside the image.

Each output region is an arbitrary perspective quadrilateral in normalized
image coordinates, ordered near_left, near_right, far_right, far_left. The
near edge is close to the camera and the far edge is deeper in the scene. Do
not output a generic axis-aligned rectangle over the lower image. Use compact
quadrilaterals that follow the visible walking support and stop before an
occlusion, curb, obstacle, crowd, vehicle, grass, vegetation, stairs, wall,
fence, or construction area.

Every emitted quadrilateral must contain four different vertices. Both the
near edge and far edge must have visible non-zero width, and the four vertices
must form one convex perimeter in exactly this order. A valid shape resembles
near_left=(0.20,0.90), near_right=(0.80,0.90), far_right=(0.60,0.45),
far_left=(0.40,0.45); this is only a geometric ordering example, not a region
to copy. Never repeat a vertex, create a zero-width strip, or trace a vertical
object silhouette. Omit a thin, vertical, or heavily occluded obstacle when it
cannot be represented as a non-degenerate local ground-region quadrilateral.

Classify visible regions as direct_traversable, conditional_traversable,
blocked, or forbidden. Grass, vegetation, stairs, vehicle bodies, dense
pedestrian clusters, walls/fences, and construction are never direct walking
support. Conditional support is allowed only when a concrete visible condition
or connector is present; list it in limiting_factors. A selected corridor must
refer to one direct or conditional region. If no reliable locally reachable
corridor is visible, return rejected or pending_human with no selected corridor.

Emit at most five regions: prioritize one selected walkable corridor and only
the largest directly relevant constraints. The allowed surface_type values are
sidewalk, crosswalk, ramp, tactile_paving, shared_path, road_edge, grass,
vegetation, stairs, curb, vehicle, pedestrian_cluster, wall_or_fence,
construction, and unknown. Use road_edge for a visible road edge or carriageway;
never output the unsupported token road.

This is image-plane geometry only. Never claim metres, camera calibration,
homography, trajectory, waypoint, future motion, ground truth, or an action.
Return exactly one JSON object matching the supplied schema and no markdown."""


@dataclass(frozen=True)
class TraversabilityGeometryTeacherProfileV1:
    """Freeze direct-LVLM geometry prompt and response-schema identity.

    Args:
        profile_id: Stable prompt profile identifier.
        producer_version: Exact offline teacher implementation version.
        system_prompt: Shared geometry and safety policy.
        output_json_schema: Strict provider response schema.
        policy_digest: SHA-256 digest of all frozen profile inputs.
    """

    profile_id: str
    producer_version: str
    system_prompt: str
    output_json_schema: dict[str, Any]
    policy_digest: str


def traversability_geometry_teacher_profile_v1() -> TraversabilityGeometryTeacherProfileV1:
    """Return the immutable V1 direct-quadrilateral teacher profile.

    Returns:
        Frozen prompt, strict output schema, and cache-safe digest.
    """

    point = {
        "type": "object",
        "additionalProperties": False,
        "required": ["u", "v"],
        "properties": {
            "u": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "v": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    }
    quadrilateral = {
        "type": "object",
        "additionalProperties": False,
        "required": ["near_left", "near_right", "far_right", "far_left"],
        "properties": {
            "near_left": point,
            "near_right": point,
            "far_right": point,
            "far_left": point,
        },
    }
    region = {
        "type": "object",
        "additionalProperties": False,
        "required": ["region_id", "role", "surface_type", "quadrilateral_uv", "limiting_factors", "evidence_refs"],
        "properties": {
            "region_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$", "maxLength": 64},
            "role": {
                "type": "string",
                "enum": ["direct_traversable", "conditional_traversable", "blocked", "forbidden"],
            },
            "surface_type": {
                "type": "string",
                "enum": [
                    "sidewalk", "crosswalk", "ramp", "tactile_paving", "shared_path", "road_edge",
                    "grass", "vegetation", "stairs", "curb", "vehicle", "pedestrian_cluster",
                    "wall_or_fence", "construction", "unknown",
                ],
            },
            "quadrilateral_uv": quadrilateral,
            "limiting_factors": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                "maxItems": 8,
            },
            "evidence_refs": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 4,
            },
        },
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["regions", "selected_corridor_region_id", "confidence", "validator_status"],
        "properties": {
            "regions": {"type": "array", "items": region, "maxItems": 5},
            "selected_corridor_region_id": {"type": ["string", "null"], "maxLength": 64},
            "confidence": {"type": ["string", "null"], "enum": ["high", "medium", "low", None]},
            "validator_status": {"type": "string", "enum": ["accepted", "rejected", "pending_human"]},
        },
    }
    payload = {
        "profile_id": TRAVERSABILITY_GEOMETRY_LVLM_PROFILE_ID_V1,
        "producer_version": TRAVERSABILITY_GEOMETRY_LVLM_PRODUCER_VERSION_V1,
        "schema_version": LOCAL_TRAVERSABILITY_GEOMETRY_SCHEMA_VERSION_V1,
        "system_prompt": TRAVERSABILITY_GEOMETRY_LVLM_SYSTEM_PROMPT_V1,
        "output_json_schema": schema,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return TraversabilityGeometryTeacherProfileV1(
        profile_id=TRAVERSABILITY_GEOMETRY_LVLM_PROFILE_ID_V1,
        producer_version=TRAVERSABILITY_GEOMETRY_LVLM_PRODUCER_VERSION_V1,
        system_prompt=TRAVERSABILITY_GEOMETRY_LVLM_SYSTEM_PROMPT_V1,
        output_json_schema=schema,
        policy_digest=digest,
    )


def build_traversability_geometry_question_v1(
    source: ReasoningSourceV4,
    profile: TraversabilityGeometryTeacherProfileV1,
    *,
    previous_answer: str | None = None,
    previous_errors: Iterable[str] = (),
) -> str:
    """Build one current-FPV direct-quadrilateral annotation request.

    Args:
        source: Frozen source row containing the current FPV image.
        profile: Frozen direct-geometry profile.
        previous_answer: Optional prior invalid provider response for repair.
        previous_errors: Deterministic validator failures for repair.

    Returns:
        JSON-only user request with no trajectory or other private sidecars.

    Raises:
        ValueError: If ``fpv.current`` is unavailable in the source.
    """

    if "fpv.current" not in source.observation.image_roles:
        raise ValueError("source_has_no_fpv_current")
    visible_input = {
        "source_group_id": source.observation.source_group_id,
        "image_order": [{"ordinal": 1, "role": "fpv.current"}],
        "visible_evidence_catalog": [{"ref": "fpv.current", "modality": "fpv"}],
        "output_json_schema": profile.output_json_schema,
    }
    parts = [
        "Annotate only the current FPV image as local traversability quadrilaterals.",
        "Return compact perspective quadrilaterals, never a generic image-bottom rectangle.",
        (
            "Emit no more than five regions. surface_type must use the supplied enum exactly: "
            "use road_edge rather than road."
        ),
        (
            "GEOMETRY_VALIDITY_CONTRACT: each emitted region has four distinct vertices in the exact "
            "perimeter order near_left -> near_right -> far_right -> far_left; both near and far edges "
            "have non-zero width; the polygon is convex. Omit any thin or vertical obstacle that cannot "
            "satisfy this ground-region geometry."
        ),
        "FROZEN_VISIBLE_INPUT=" + json.dumps(visible_input, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    ]
    errors = [str(value)[:500] for value in previous_errors if str(value)]
    if previous_answer is not None or errors:
        parts.extend(
            [
                "The previous response was invalid. Repair geometry without inventing unobserved free space.",
                "VALIDATION_ERRORS=" + json.dumps(errors[:16], ensure_ascii=False, separators=(",", ":")),
                "PREVIOUS_RESPONSE=" + str(previous_answer or "")[:12000],
            ]
        )
    return "\n\n".join(parts)


def parse_traversability_geometry_answer_v1(
    text: str,
    *,
    source: ReasoningSourceV4,
    teacher_model: str | None,
    profile: TraversabilityGeometryTeacherProfileV1,
) -> tuple[LocalTraversabilityGeometryV1 | None, list[str]]:
    """Parse and validate one direct-LVLM quadrilateral response.

    Args:
        text: Raw provider response.
        source: Exact source revision used by the teacher.
        teacher_model: Provider-reported model identity.
        profile: Frozen prompt/schema profile.

    Returns:
        Typed geometry sidecar or deterministic validation errors.
    """

    body = _strip_json_fence_v1(str(text or "").strip())
    if not body:
        return None, ["empty_teacher_answer"]
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, [f"invalid_json:{exc.msg}"]
    if not isinstance(payload, dict):
        return None, ["teacher_payload_must_be_object"]
    allowed = set(profile.output_json_schema["properties"])
    missing = sorted(allowed - set(payload))
    extras = sorted(set(payload) - allowed)
    if missing or extras:
        errors: list[str] = []
        if missing:
            errors.append("teacher_payload_missing:" + ",".join(missing))
        if extras:
            errors.append("teacher_payload_extra:" + ",".join(extras))
        return None, errors
    status = payload.get("validator_status")
    if status not in {"accepted", "rejected", "pending_human"}:
        return None, ["invalid_validator_status"]
    if status != "accepted" and (payload.get("regions") or payload.get("selected_corridor_region_id") is not None):
        return None, ["nonaccepted_geometry_must_not_emit_regions_or_selected_corridor"]
    normalized_payload, normalization_warnings = _normalize_geometry_payload_v1(payload)
    try:
        annotation = LocalTraversabilityGeometryV1(
            geometry_id=local_traversability_geometry_id_v1(
                parent_sample_id=source.observation.source_group_id,
                source_fingerprint=source.source_fingerprint,
                prompt_profile_id=profile.profile_id,
                prompt_profile_digest=profile.policy_digest,
            ),
            parent_sample_id=source.observation.source_group_id,
            source_fingerprint=source.source_fingerprint,
            dataset_source=source.observation.dataset_source,
            regions=tuple(normalized_payload.get("regions") or ()),
            selected_corridor_region_id=normalized_payload.get("selected_corridor_region_id"),
            validator_status=status,
            teacher_model=teacher_model or "unknown_teacher_model",
            prompt_profile_id=profile.profile_id,
            prompt_profile_digest=profile.policy_digest,
            confidence=normalized_payload.get("confidence"),
            warnings=tuple(normalization_warnings),
        )
    except (TypeError, ValueError) as exc:
        return None, [f"geometry_schema:{exc}"]
    return annotation, []


def _normalize_geometry_payload_v1(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Normalize only lossless surface-type aliases before typed validation.

    ``road`` is an old teacher spelling for the schema's non-walkable
    ``road_edge`` category.  This deterministic mapping is recorded in the
    sidecar warning so it is inspectable and never masquerades as new visual
    reasoning.  Geometry, role, and evidence fields are intentionally left
    untouched for the strict validator.
    """

    regions = payload.get("regions")
    if not isinstance(regions, list):
        return payload, []
    normalized = dict(payload)
    normalized_regions: list[Any] = []
    warnings: list[str] = []
    for index, value in enumerate(regions):
        if not isinstance(value, dict):
            normalized_regions.append(value)
            continue
        region = dict(value)
        surface_type = region.get("surface_type")
        if isinstance(surface_type, str):
            alias = _SURFACE_TYPE_ALIASES_V1.get(surface_type.strip().lower())
            if alias is not None:
                region["surface_type"] = alias
                warnings.append(f"surface_type_alias_normalized:{index}:{surface_type}->{alias}")
        normalized_regions.append(region)
    normalized["regions"] = normalized_regions
    return normalized, warnings


def _strip_json_fence_v1(value: str) -> str:
    """Strip one whole-response JSON fence when present."""

    match = re.fullmatch(r"```(?:json)?\s*(?P<body>.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    return match.group("body").strip() if match else value


__all__ = [
    "TRAVERSABILITY_GEOMETRY_LVLM_PRODUCER_VERSION_V1",
    "TRAVERSABILITY_GEOMETRY_LVLM_PROFILE_ID_V1",
    "TRAVERSABILITY_GEOMETRY_LVLM_SYSTEM_PROMPT_V1",
    "TraversabilityGeometryTeacherProfileV1",
    "build_traversability_geometry_question_v1",
    "parse_traversability_geometry_answer_v1",
    "traversability_geometry_teacher_profile_v1",
]
