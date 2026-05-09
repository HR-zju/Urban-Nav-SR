"""Deterministic teachers for visual-prompt QA samples.

The QA bundle records small, inspectable supervision targets next to each
visual-prompt run. These teachers are not learned models; they distill route,
BEV, segmentation, and rendered-prompt artifacts into labels that can be used
for regression tests, dataset auditing, and model-evaluation dashboards.

Core angular formulas:

    theta_bev = atan2(x_m, y_m)

    theta_fpv = ((u / image_width) - 0.5) * 2 * theta_max

    confidence_prompt =
        mean(route_confidence, prompt_present, segmentation_score, candidate_score)

The teachers prefer explicit artifact-derived labels. Fallback labels are
marked through their ``source`` field so downstream analysis can separate strong
geometric supervision from weak priors.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

from .angles import bev_theta_deg, coarse_bin_from_theta, fpv_theta_from_u, theta_bin_residual
from .schema import PointUV, RangeM, ThetaLabel


def make_theta_label(theta_deg: float, *, confidence: float, source: str) -> ThetaLabel:
    """Normalize one angle into the common QA theta-label schema."""
    bin_index, residual = theta_bin_residual(theta_deg)
    clamped = max(-90.0, min(90.0, float(theta_deg)))
    return ThetaLabel(
        theta_deg=round(clamped, 3),
        coarse_bin=coarse_bin_from_theta(clamped),
        bin_index=bin_index,
        residual_deg=round(residual, 3),
        confidence=round(max(0.0, min(1.0, float(confidence))), 4),
        source=source,
    )


def _bev_points(raw_points: Any) -> list[tuple[float, float]]:
    if not isinstance(raw_points, list):
        return []
    points: list[tuple[float, float]] = []
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        try:
            points.append((float(item["x_m"]), float(item["y_m"])))
        except (KeyError, TypeError, ValueError):
            continue
    return points


def _point_uv(raw: Any) -> PointUV | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        return PointUV(u=float(raw[0]), v=float(raw[1]))
    except (TypeError, ValueError):
        return None


def _path_points(rendered_prompt: dict[str, Any] | None) -> list[PointUV]:
    if not isinstance(rendered_prompt, dict):
        return []
    path_samples = rendered_prompt.get("path_samples")
    if isinstance(path_samples, list):
        points = [_point_uv(sample.get("center_uv")) for sample in path_samples if isinstance(sample, dict)]
        points = [point for point in points if point is not None]
        if points:
            return points
    path_uv = rendered_prompt.get("path_uv")
    if isinstance(path_uv, list):
        points = [_point_uv(point) for point in path_uv]
        points = [point for point in points if point is not None]
        if points:
            return points
    center = _point_uv(rendered_prompt.get("center_uv"))
    return [center] if center is not None else []


def teacher_bev_route_theta(local_bev_context: dict[str, Any]) -> dict[str, Any]:
    """Estimate the route lookahead direction in BEV degrees."""
    local_points = _bev_points(local_bev_context.get("local_segment_bev"))
    route_points = _bev_points(local_bev_context.get("route_polyline_local"))
    points = local_points or route_points
    if len(points) >= 2:
        start = points[0]
        lookahead = [point for point in points[1:] if 0.5 <= math.hypot(point[0] - start[0], point[1] - start[1]) <= 10.0]
        if not lookahead:
            lookahead = points[1: min(len(points), 4)]
        dx = sum(point[0] - start[0] for point in lookahead) / max(len(lookahead), 1)
        dy = sum(point[1] - start[1] for point in lookahead) / max(len(lookahead), 1)
        theta = bev_theta_deg(dx, dy)
        return {"theta": asdict(make_theta_label(theta, confidence=0.9, source="local_segment_bev"))}
    sector = local_bev_context.get("corridor_sector_deg") or {}
    theta = float(sector.get("center_deg", 0.0) or 0.0)
    return {"theta": asdict(make_theta_label(theta, confidence=0.55, source="corridor_sector_fallback"))}


def teacher_bev_component_exists(local_bev_context: dict[str, Any], component_type: str) -> dict[str, Any]:
    semantic_layers = local_bev_context.get("semantic_bev_layers") or {}
    source = "semantic_bev_layers"
    exists = False
    confidence = 0.35
    if component_type == "connector":
        exists = len(_bev_points(local_bev_context.get("connector_bev"))) >= 2
        source = "connector_bev"
        confidence = 0.85 if exists else 0.75
    elif component_type == "crossing":
        exists = bool(semantic_layers.get("crossing_mask"))
    elif component_type == "entrance":
        exists = bool(semantic_layers.get("entrance_candidates_bev"))
    return {"exists": exists, "confidence": confidence, "source": source}


def teacher_bev_component_theta_range(local_bev_context: dict[str, Any], component_type: str) -> dict[str, Any]:
    semantic_layers = local_bev_context.get("semantic_bev_layers") or {}
    raw_points: Any = []
    source = "semantic_bev_layers"
    if component_type == "connector":
        raw_points = local_bev_context.get("connector_bev")
        source = "connector_bev"
    elif component_type == "entrance":
        raw_points = semantic_layers.get("entrance_candidates_bev")
    points = _bev_points(raw_points)
    if not points:
        return {"exists": False, "theta": None, "range_m": None, "span_deg": None}
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    distances = [math.hypot(point[0], point[1]) for point in points]
    thetas = [bev_theta_deg(point[0], point[1]) for point in points]
    span = max(thetas) - min(thetas) if len(thetas) > 1 else 10.0
    return {
        "exists": True,
        "theta": asdict(make_theta_label(bev_theta_deg(cx, cy), confidence=0.82, source=source)),
        "range_m": asdict(RangeM(min_m=round(min(distances), 3), max_m=round(max(distances), 3))),
        "span_deg": round(float(span), 3),
    }


def teacher_bev_navigation_stage(route_context: dict[str, Any]) -> dict[str, Any]:
    progress = route_context.get("progress") or {}
    local = route_context.get("local_semantics") or {}
    turn_type = str(progress.get("next_turn_type") or local.get("turn_type") or "go_straight")
    turn_semantics = str(local.get("turn_semantics") or "")
    distance = progress.get("distance_to_next_turn_m")
    try:
        distance_m = float(distance)
    except (TypeError, ValueError):
        distance_m = 999999.0
    if "arrive" in turn_type or "arrive" in turn_semantics:
        stage = "approach_entrance"
    elif distance_m <= 12.0 and "cross" in turn_semantics:
        stage = "approach_crossing"
    elif distance_m <= 15.0 and turn_type not in {"", "go_straight"}:
        stage = "approach_turn"
    else:
        stage = "straight"
    return {"stage": stage, "confidence": 0.82, "source": "route_progress_local_semantics"}


def teacher_point_prompt(rendered_prompt: dict[str, Any] | None, image_width: int) -> dict[str, Any]:
    points = _path_points(rendered_prompt)
    if not points:
        return {"target_point_uv": None, "theta": None, "reject_all": True, "source": "no_rendered_prompt"}
    target = points[-1]
    theta = make_theta_label(fpv_theta_from_u(target.u, image_width), confidence=0.86, source="rendered_prompt_target")
    return {"target_point_uv": asdict(target), "theta": asdict(theta), "reject_all": False, "source": "rendered_prompt"}


def _present_class_names(segmentation_summary: dict[str, Any]) -> set[str]:
    present_names: set[str] = set()
    for row in segmentation_summary.get("present_classes") or []:
        if not isinstance(row, dict):
            continue
        try:
            pixel_ratio = float(row.get("pixel_ratio", 0.0) or 0.0)
        except (TypeError, ValueError):
            pixel_ratio = 0.0
        if pixel_ratio > 0.0:
            present_names.add(str(row.get("class_name") or "").strip().lower())
    return {name for name in present_names if name}


def teacher_fpv_scene_stage(segmentation_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(segmentation_summary, dict):
        return {"scene_stage": "unknown", "confidence": 0.25, "source": "no_segmentation_summary"}
    present_names = _present_class_names(segmentation_summary)
    traversable_names = {
        str(name).strip().lower()
        for name in (
            segmentation_summary.get("traversable_classes")
            or segmentation_summary.get("present_traversable_classes")
            or segmentation_summary.get("traversable_class_names")
            or []
        )
    }
    forbidden_names = {
        str(name).strip().lower()
        for name in (
            segmentation_summary.get("forbidden_classes")
            or segmentation_summary.get("present_forbidden_classes")
            or segmentation_summary.get("forbidden_class_names")
            or []
        )
    }
    traversable = sorted(present_names & traversable_names) if present_names else sorted(traversable_names)
    forbidden = sorted(present_names & forbidden_names) if present_names else sorted(forbidden_names)
    if traversable and forbidden:
        stage = "mixed_corridor_obstacles"
    elif traversable:
        stage = "open_corridor"
    else:
        stage = "uncertain"
    return {"scene_stage": stage, "confidence": 0.55, "source": "segmentation_summary"}


def teacher_step_instruction(route_context: dict[str, Any], rendered_prompt: dict[str, Any] | None) -> dict[str, Any]:
    local = route_context.get("local_semantics") or {}
    instruction = str(local.get("instruction_hint") or local.get("short_action_summary") or "continue")
    termination = rendered_prompt.get("termination_semantics") if isinstance(rendered_prompt, dict) else None
    if termination:
        instruction = f"{instruction}; {termination}"
    stage = teacher_bev_navigation_stage(route_context)["stage"]
    return {"instruction": instruction, "stage": stage, "confidence": 0.72}


def teacher_prompt_confidence(
    route_context: dict[str, Any],
    rendered_prompt: dict[str, Any] | None,
    segmentation_validation: dict[str, Any] | None,
    candidate_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Aggregate independent reliability signals into a scalar prompt score."""
    pose = route_context.get("pose_match") or {}
    route_confidence = float(pose.get("route_confidence", 0.0) or 0.0)
    prompt_present = 1.0 if isinstance(rendered_prompt, dict) else 0.0
    segmentation_score = 0.5
    if isinstance(segmentation_validation, dict):
        segmentation_score = 1.0 if segmentation_validation.get("valid") else 0.0
    candidate_score = 0.5
    if isinstance(candidate_bundle, dict) and candidate_bundle.get("fpv_candidates"):
        selected_id = candidate_bundle.get("selected_prompt_id")
        candidates = candidate_bundle.get("fpv_candidates") or []
        selected = next((item for item in candidates if item.get("prompt_id") == selected_id), candidates[0])
        candidate_score = (float(selected.get("route_alignment_score", 0.0) or 0.0) + float(selected.get("fpv_support_score", 0.0) or 0.0)) * 0.5
    factors = {
        "route_confidence": round(route_confidence, 4),
        "prompt_present": prompt_present,
        "segmentation_score": round(segmentation_score, 4),
        "candidate_score": round(candidate_score, 4),
    }
    confidence = sum(factors.values()) / len(factors)
    return {"confidence": round(confidence, 4), "factors": factors}


def teacher_prompt_validity(
    rendered_prompt: dict[str, Any] | None,
    segmentation_validation: dict[str, Any] | None,
    arbiter_decision: dict[str, Any],
    candidate_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_present = isinstance(rendered_prompt, dict)
    arbiter_reject = bool(arbiter_decision.get("reject_all")) if isinstance(arbiter_decision, dict) else False
    segmentation_valid = None
    if isinstance(segmentation_validation, dict):
        segmentation_valid = bool(segmentation_validation.get("valid"))
    candidate_reject = None
    candidate_reasons: list[str] = []
    if isinstance(candidate_bundle, dict):
        candidate_reject = bool(candidate_bundle.get("reject_all"))
        candidate_reasons = [str(item) for item in (candidate_bundle.get("rejection_reasons") or [])]

    checks = {
        "grounded": {
            "arbiter_reject_all": arbiter_reject,
            "prompt_present": prompt_present,
        },
        "fpv": {
            "segmentation_checked": segmentation_valid is not None,
            "segmentation_valid": segmentation_valid,
        },
        "crossview": {
            "candidate_bundle_present": isinstance(candidate_bundle, dict),
            "candidate_reject_all": candidate_reject,
            "rejection_reasons": candidate_reasons,
        },
    }
    if arbiter_reject:
        reason = str(arbiter_decision.get("reason") or "arbiter_reject_all")
        return {"reject_all": True, "decision": "reject", "reason": reason, "confidence": 0.92, "checks": checks}
    if not prompt_present:
        return {"reject_all": True, "decision": "reject", "reason": "no_rendered_prompt", "confidence": 0.9, "checks": checks}
    if segmentation_valid is False:
        reason = str(segmentation_validation.get("reason") or "segmentation_invalid")
        return {"reject_all": True, "decision": "reject", "reason": reason, "confidence": 0.9, "checks": checks}
    if candidate_reject:
        reason = "; ".join(candidate_reasons) if candidate_reasons else "candidate_bundle_reject_all"
        return {"reject_all": True, "decision": "reject", "reason": reason, "confidence": 0.82, "checks": checks}
    return {
        "reject_all": False,
        "decision": "accept",
        "reason": "prompt_valid",
        "confidence": 0.74,
        "checks": checks,
    }
