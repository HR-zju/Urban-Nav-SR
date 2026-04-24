from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from mcp_server.brain.fpv_segmentation import validate_rendered_prompt_against_segmentation

from .angles import bev_theta_deg, fpv_theta_from_u, wrap_angle_deg
from .schema import PointUV, RangeM


@dataclass
class BEVPromptTarget:
    target_id: str
    source: str
    theta_bev_deg: float
    range_m: RangeM | None
    span_deg: float | None
    semantic_tag: str
    priority: float
    debug: dict[str, Any]


@dataclass
class FPVPromptCandidate:
    prompt_id: str
    source_bev_target_id: str | None
    prompt_type: str
    center_uv: PointUV | None
    theta_fpv_deg: float | None
    span_px: float | None
    path_uv: list[PointUV]
    coverage_polygon_uv: list[PointUV]
    semantic_tag: str
    route_alignment_score: float
    fpv_support_score: float
    segmentation_valid: bool | None
    reject_reason: str | None
    debug: dict[str, Any]


@dataclass
class PoVPromptCandidateBundle:
    image_width: int
    image_height: int
    theta_max_deg: float
    bev_targets: list[BEVPromptTarget]
    fpv_candidates: list[FPVPromptCandidate]
    selected_prompt_id: str | None
    reject_all: bool
    rejection_reasons: list[str]
    debug: dict[str, Any]


def _point_from_raw(raw: Any) -> PointUV | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        return PointUV(u=float(raw[0]), v=float(raw[1]))
    except (TypeError, ValueError):
        return None


def _bev_point(raw: Any) -> tuple[float, float] | None:
    if isinstance(raw, dict):
        try:
            return float(raw["x_m"]), float(raw["y_m"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _radial_range(points: list[dict[str, Any]]) -> RangeM | None:
    distances: list[float] = []
    for item in points:
        point = _bev_point(item)
        if point is None:
            continue
        x_m, y_m = point
        distances.append(math.hypot(x_m, y_m))
    if not distances:
        return None
    return RangeM(min_m=round(min(distances), 3), max_m=round(max(distances), 3))


def _target_from_points(
    *,
    target_id: str,
    source: str,
    points: list[dict[str, Any]],
    semantic_tag: str,
    priority: float,
    span_deg: float | None = None,
) -> BEVPromptTarget | None:
    usable = [_bev_point(item) for item in points]
    usable = [item for item in usable if item is not None]
    if not usable:
        return None
    x_m = sum(item[0] for item in usable) / len(usable)
    y_m = sum(item[1] for item in usable) / len(usable)
    return BEVPromptTarget(
        target_id=target_id,
        source=source,
        theta_bev_deg=bev_theta_deg(x_m, y_m),
        range_m=_radial_range(points),
        span_deg=span_deg,
        semantic_tag=semantic_tag,
        priority=float(priority),
        debug={"point_count": len(usable), "centroid_bev": {"x_m": round(x_m, 3), "y_m": round(y_m, 3)}},
    )


def _target_from_sector(
    *,
    target_id: str,
    source: str,
    sector: dict[str, Any] | None,
    semantic_tag: str,
    priority: float,
) -> BEVPromptTarget | None:
    if not isinstance(sector, dict):
        return None
    try:
        theta = float(sector.get("center_deg", 0.0) or 0.0)
        span = float(sector.get("span_deg", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    return BEVPromptTarget(
        target_id=target_id,
        source=source,
        theta_bev_deg=theta,
        range_m=None,
        span_deg=span,
        semantic_tag=semantic_tag,
        priority=float(priority),
        debug={"sector": dict(sector)},
    )


def _extract_path_points(rendered_prompt: dict[str, Any]) -> list[PointUV]:
    path_samples = rendered_prompt.get("path_samples")
    if isinstance(path_samples, list):
        points = [_point_from_raw(sample.get("center_uv")) for sample in path_samples if isinstance(sample, dict)]
        points = [point for point in points if point is not None]
        if points:
            return points
    path_uv = rendered_prompt.get("path_uv")
    if isinstance(path_uv, list):
        points = [_point_from_raw(item) for item in path_uv]
        return [point for point in points if point is not None]
    center = _point_from_raw(rendered_prompt.get("center_uv"))
    return [center] if center is not None else []


def _extract_polygon_points(rendered_prompt: dict[str, Any]) -> list[PointUV]:
    polygon = rendered_prompt.get("coverage_polygon_uv")
    if not isinstance(polygon, list):
        return []
    points = [_point_from_raw(item) for item in polygon]
    return [point for point in points if point is not None]


def _route_alignment(theta_fpv: float | None, bev_targets: list[BEVPromptTarget]) -> tuple[float, str | None]:
    if theta_fpv is None or not bev_targets:
        return 0.0, None
    best_target = max(
        bev_targets,
        key=lambda target: max(0.0, 1.0 - abs(wrap_angle_deg(theta_fpv - target.theta_bev_deg)) / 90.0) * target.priority,
    )
    diff = abs(wrap_angle_deg(theta_fpv - best_target.theta_bev_deg))
    return round(max(0.0, 1.0 - diff / 90.0), 4), best_target.target_id


def _build_bev_targets(local_bev_context: dict[str, Any]) -> list[BEVPromptTarget]:
    targets: list[BEVPromptTarget] = []
    local_segment = local_bev_context.get("local_segment_bev") if isinstance(local_bev_context, dict) else []
    connector = local_bev_context.get("connector_bev") if isinstance(local_bev_context, dict) else []

    for item in [
        _target_from_points(
            target_id="bev_route_continuation",
            source="local_segment_bev",
            points=local_segment or [],
            semantic_tag="route_continuation",
            priority=1.0,
        ),
        _target_from_points(
            target_id="bev_connector",
            source="connector_bev",
            points=connector or [],
            semantic_tag="connector",
            priority=0.85,
        ),
        _target_from_sector(
            target_id="bev_corridor_sector",
            source="corridor_sector_deg",
            sector=local_bev_context.get("corridor_sector_deg") if isinstance(local_bev_context, dict) else None,
            semantic_tag="corridor_sector",
            priority=0.55,
        ),
        _target_from_sector(
            target_id="bev_turn_sector",
            source="turn_sector_deg",
            sector=local_bev_context.get("turn_sector_deg") if isinstance(local_bev_context, dict) else None,
            semantic_tag="turn_sector",
            priority=0.65,
        ),
    ]:
        if item is not None:
            targets.append(item)

    next_turn = local_bev_context.get("next_turn_point_bev") if isinstance(local_bev_context, dict) else None
    if isinstance(next_turn, dict):
        target = _target_from_points(
            target_id="bev_next_turn_point",
            source="next_turn_point_bev",
            points=[next_turn],
            semantic_tag="next_turn_point",
            priority=0.75,
            span_deg=20.0,
        )
        if target is not None:
            targets.append(target)
    return targets


def build_pov_prompt_candidate_bundle(
    *,
    route_context: dict[str, Any],
    local_bev_context: dict[str, Any],
    image_width: int,
    image_height: int,
    segmentation_context: dict[str, Any] | None,
    rendered_prompt: dict[str, Any] | None = None,
    theta_max_deg: float = 50.0,
) -> PoVPromptCandidateBundle:
    del route_context
    bev_targets = _build_bev_targets(local_bev_context)
    fpv_candidates: list[FPVPromptCandidate] = []
    rejection_reasons: list[str] = []

    if isinstance(rendered_prompt, dict):
        path_points = _extract_path_points(rendered_prompt)
        center = path_points[-1] if path_points else _point_from_raw(rendered_prompt.get("center_uv"))
        theta_fpv = fpv_theta_from_u(center.u, image_width, theta_max_deg) if center is not None else None
        alignment_score, source_target_id = _route_alignment(theta_fpv, bev_targets)
        validation = validate_rendered_prompt_against_segmentation(rendered_prompt, segmentation_context)
        segmentation_valid = None if validation is None else bool(validation.get("valid"))
        if validation is not None and not segmentation_valid:
            rejection_reasons.append(str(validation.get("reason") or "segmentation_invalid"))
        support_score = 1.0
        if validation is not None:
            support_score = float(validation.get("center_valid_ratio", 0.0) or 0.0)
        fpv_candidates.append(
            FPVPromptCandidate(
                prompt_id="final_rendered_prompt",
                source_bev_target_id=source_target_id,
                prompt_type=str(rendered_prompt.get("type") or "unknown"),
                center_uv=center,
                theta_fpv_deg=round(theta_fpv, 3) if theta_fpv is not None else None,
                span_px=float(rendered_prompt.get("span_px")) if rendered_prompt.get("span_px") is not None else None,
                path_uv=path_points,
                coverage_polygon_uv=_extract_polygon_points(rendered_prompt),
                semantic_tag=str(rendered_prompt.get("semantic_tag") or rendered_prompt.get("type") or "prompt"),
                route_alignment_score=alignment_score,
                fpv_support_score=round(support_score, 4),
                segmentation_valid=segmentation_valid,
                reject_reason=None if segmentation_valid is not False else str((validation or {}).get("reason") or "segmentation_invalid"),
                debug={"segmentation_validation": validation},
            )
        )
    else:
        rejection_reasons.append("no_rendered_prompt")

    selected = None
    if fpv_candidates:
        selected_candidate = max(fpv_candidates, key=lambda item: (item.route_alignment_score + item.fpv_support_score) * 0.5)
        selected = selected_candidate.prompt_id
    return PoVPromptCandidateBundle(
        image_width=int(image_width),
        image_height=int(image_height),
        theta_max_deg=float(theta_max_deg),
        bev_targets=bev_targets,
        fpv_candidates=fpv_candidates,
        selected_prompt_id=selected,
        reject_all=not bool(fpv_candidates) or any(item.segmentation_valid is False for item in fpv_candidates),
        rejection_reasons=rejection_reasons,
        debug={"candidate_source": "final_rendered_prompt_v1"},
    )


def pov_candidate_bundle_to_dict(bundle: PoVPromptCandidateBundle | None) -> dict[str, Any] | None:
    if bundle is None:
        return None
    return asdict(bundle)
