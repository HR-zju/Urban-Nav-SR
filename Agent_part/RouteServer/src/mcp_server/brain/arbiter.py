from __future__ import annotations

import json
import math
from typing import Any

from mcp_server.route_context.schema import VisualPromptDecision


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _compact_route_context(route_context: dict[str, Any]) -> dict[str, Any]:
    pose_match = route_context.get("pose_match", {}) or {}
    progress = route_context.get("progress", {}) or {}
    local = route_context.get("local_semantics", {}) or {}
    return {
        "off_route": bool(pose_match.get("off_route")),
        "route_confidence": round(float(pose_match.get("route_confidence", 0.0) or 0.0), 4),
        "cross_track_error_m": pose_match.get("cross_track_error_m"),
        "heading_delta_deg": pose_match.get("heading_delta_deg"),
        "distance_to_next_turn_m": progress.get("distance_to_next_turn_m"),
        "next_turn_type": progress.get("next_turn_type"),
        "turn_type": local.get("turn_type"),
        "turn_semantics": local.get("turn_semantics"),
        "corridor_type": local.get("corridor_type"),
        "short_action_summary": local.get("short_action_summary"),
        "instruction_hint": local.get("instruction_hint"),
    }


def _navigation_mode_note(navigation_mode: str) -> str:
    mode = str(navigation_mode or "").strip().lower()
    if mode == "foot":
        return (
            "Pedestrian mode: prefer sidewalks, pedestrian plazas, curb-cut entries, and legal crossings. "
            "Avoid suggesting motor-vehicle lanes or roadway space unless the image clearly shows a pedestrian crossing."
        )
    if mode == "bike":
        return (
            "Cycling mode: prefer bike lanes, mixed-use paths, and edge-of-road rideable space. "
            "Avoid sidewalks unless the scene clearly indicates shared cycling access."
        )
    return (
        "Driving mode: prefer legal drivable road space and avoid sidewalks, plazas, and pedestrian-only areas."
    )


def _candidate_compact(candidate: dict[str, Any]) -> dict[str, Any]:
    debug = candidate.get("debug", {}) or {}
    return {
        "proposal_id": candidate.get("proposal_id"),
        "source_bev_proposal_id": candidate.get("source_bev_proposal_id"),
        "prompt_type": candidate.get("prompt_type"),
        "priority": candidate.get("priority"),
        "projection_confidence": candidate.get("projection_confidence"),
        "visual_confidence": candidate.get("visual_confidence"),
        "center_uv": candidate.get("center_uv"),
        "span_px": candidate.get("span_px"),
        "semantic_tag": candidate.get("semantic_tag"),
        "proposal_type": debug.get("proposal_type"),
        "matched_visual_region_id": debug.get("matched_visual_region_id"),
        "matched_visual_region_score": debug.get("matched_visual_region_score"),
        "region_overlap": debug.get("region_overlap"),
    }


def build_direct_prompt_system_prompt() -> str:
    return (
        "You are a mode-aware first-person navigation visual prompt generator. "
        "Your job is to generate one conservative image-plane guidance prompt for the next 1 to 3 seconds. "
        "Use route context only as route intent; it is not permission to ignore visible scene safety. "
        "For pedestrian mode, treat sidewalks, pedestrian plazas, curb cuts, entrances, and marked crossings as preferred space. "
        "For pedestrian mode, treat motor-vehicle lanes, open roadway, sky, walls, fences, parked cars, and freestanding obstacles such as signboards as forbidden unless a legal crossing is visually obvious. "
        "For pedestrian mode, keep the prompt inside the visible sidewalk or pedestrian-safe corridor; do not spill outside curb boundaries when a sidewalk exists. "
        "Treat grass, shrubs, planters, tree bases, decorative green belts, and other non-paved landscaping as non-traversable support surfaces for pedestrian prompts. "
        "Avoid trees, poles, bollards, stairs, planters, building facades, and people. "
        "Prefer a structured path corridor over a wide sector. "
        "Prefer the nearest safe sidewalk-side opening over a farther but wider drivable area. "
        "The prompt must start near the lower image area and every point of the trajectory must lie on the visible mode-valid traversable surface. "
        "For pedestrian mode, every path point and every part of the support corridor must be on visible sidewalk, pedestrian paving, curb cut, plaza, or legal crossing surface. "
        "If fpv_segmentation_summary is provided, treat it as an explicit traversability prior and keep the prompt inside its traversable classes. "
        "If fpv_segmentation_summary is provided, prefer the widest bottom opening and keep the center path near the middle of the traversable corridor (maximize clearance from landscaping and obstacles). "
        "If the traversable corridor becomes ambiguous ahead (e.g., mixed with landscaping), keep the corridor short and terminate early with stop_before_occlusion rather than drifting into uncertain regions. "
        "Represent the prompt as a center path plus left and right support boundaries sampled along the path and an explicit termination semantic. "
        "The termination semantic must explain whether the prompt stops because it reaches visible corridor depth, reaches a turn entry, or should stop before occlusion. "
        "Route/BEV/PoV candidates, when provided, are advisory priors only; reject or override them if they land on obstacles or non-walkable regions. "
        "If the scene is uncertain, be conservative and set reject_all=true. "
        "Return JSON only with keys: selected_prompt_type, instruction, confidence, reason, reject_all, rendered_prompt. "
        "For type=path_corridor, rendered_prompt must include path_samples and termination_semantics. "
        "Each path sample must contain center_uv, left_edge_uv, right_edge_uv, and support_label. "
        "Do not freehand a wide polygon first; define the corridor by boundary samples anchored to visible traversable edges. "
        "Prefer type=path_corridor. Do not output prose outside JSON."
    )


def build_direct_prompt_verifier_system_prompt() -> str:
    return (
        "You are a first-person navigation visual prompt verifier and refiner. "
        "You receive a proposed prompt and must verify whether its center path and support corridor are truly grounded on visible traversable surface for the current navigation mode. "
        "You must inspect the image, not trust the proposal text. "
        "For pedestrian mode, visible paved sidewalk, pedestrian path, curb cut, crosswalk, and plaza surfaces are valid support. "
        "Grass, shrubs, decorative green belts, planters, tree bases, exposed soil, walls, fences, and motor lanes are not valid support. "
        "If fpv_segmentation_summary is provided, use it as an explicit traversability prior and reject prompts that leave those traversable classes. "
        "If the center path is reasonable but the support corridor is too wide, tighten the left/right support boundaries while preserving the path. "
        "If the path itself enters non-traversable support, revise the path and boundary samples together. "
        "If no safe and visible support corridor can be verified, set reject_all=true. "
        "Return JSON only with keys: selected_prompt_type, instruction, confidence, reason, reject_all, rendered_prompt. "
        "For path_corridor, rendered_prompt must include path_samples, termination_semantics, and support_confidence."
    )


def build_direct_prompt_question(
    *,
    route_context: dict[str, Any],
    prompt_proposals: list[dict[str, Any]],
    visual_regions: dict[str, Any],
    image_width: int,
    image_height: int,
    navigation_mode: str = "foot",
    pov_prompt_candidates: list[dict[str, Any]] | None = None,
    fpv_segmentation_summary: dict[str, Any] | None = None,
    alignment_report: dict[str, Any] | None = None,
    overlay_legend: dict[str, Any] | None = None,
    summary_source: str | None = None,
) -> str:
    compact_route = _compact_route_context(route_context)
    top_prior = None
    if prompt_proposals:
        first = prompt_proposals[0]
        top_prior = {
            "proposal_type": first.get("proposal_type"),
            "semantic_tag": first.get("semantic_tag"),
            "priority": first.get("priority"),
            "sector_center_deg": first.get("sector_center_deg"),
            "sector_span_deg": first.get("sector_span_deg"),
            "range_m": first.get("range_m"),
        }
    compact_regions = [
        {
            "region_id": item.get("region_id"),
            "region_type": item.get("region_type"),
            "bbox_uv": item.get("bbox_uv"),
            "score": item.get("score"),
            "obstacle_penalty": (item.get("debug", {}) or {}).get("obstacle_penalty"),
            "boundary_penalty": (item.get("debug", {}) or {}).get("boundary_penalty"),
            "corridor_target_uv": (item.get("debug", {}) or {}).get("corridor_target_uv"),
        }
        for item in ((visual_regions or {}).get("candidate_regions", []) or [])[:4]
    ]
    compact_candidates = [_candidate_compact(item) for item in (pov_prompt_candidates or [])[:5]]
    return json.dumps(
        {
            "task": (
                "Generate one conservative image-plane navigation trajectory prompt. "
                "Use the image first, then route context, then the top route prior. "
                "Represent the result as a center path with left/right traversable boundary samples along that path. "
                "Prefer a single path corridor through a visible mode-valid opening and avoid blocked central obstacles. "
                "Candidate hints are advisory and must not override visible walkability."
            ),
            "navigation_mode": str(navigation_mode or "foot"),
            "navigation_constraints": _navigation_mode_note(navigation_mode),
            "image_size": {"width": image_width, "height": image_height},
            "route_context": compact_route,
            "top_route_prior": top_prior,
            "visual_regions_hint": compact_regions,
            "optional_candidate_hints": compact_candidates,
            "fpv_segmentation_summary": fpv_segmentation_summary,
            "alignment_report": alignment_report,
            "overlay_legend": overlay_legend,
            "summary_source": str(summary_source or "from_same_mask_entity"),
            "hard_rules": [
                "For foot mode, prioritize sidewalk or pedestrian-space openings.",
                "Do not place the trajectory on apparent motor lanes or open roadway if sidewalk-like space exists.",
                "Do not run the trajectory through a central obstacle such as a signboard.",
                "Keep both the center path and the left/right support boundaries inside the visible pedestrian-safe corridor and away from curb edges.",
                "If fpv_segmentation_summary.bottom_openings is present, start from the widest bottom opening and keep the center path in the middle of that opening (avoid hugging left/right edges).",
                "If fpv_segmentation_summary.traversable_bbox_uv is present, keep the path corridor inside that bbox and avoid terminating at its extreme boundary; prefer a shorter corridor over drifting into uncertain regions.",
                "Do not use grass, shrubs, planters, or decorative green belts as support boundaries in foot mode.",
                "Prefer a path_corridor that follows the sidewalk centerline.",
                "The path should begin near the lower image region and every point must stay on the visible traversable surface for the current mode.",
                "Every path sample must provide the visible left and right traversable edges around the center path.",
                "Boundary samples must touch the visible traversable support around the center path, not sky, building facade, or landscaping.",
                "If you cannot see a continuous traversable support surface for the whole path corridor, set reject_all=true instead of drawing.",
                "Prefer the shortest safe path that stays on pedestrian space.",
                "Set termination_semantics to one of: continue_visible_corridor, reach_turn_entry, stop_before_occlusion.",
                "If route prior and visible mode-valid space conflict, prefer visible mode-valid space and explain the conflict.",
                "If fpv_segmentation_summary is present, keep the center path inside its traversable classes and keep support boundaries away from its forbidden classes.",
            ],
            "output_requirements": {
                "selected_prompt_type": "path_corridor|arrow_path|arrow|sector|region",
                "instruction": "short English instruction",
                "confidence": "number in [0,1]",
                "reason": "one short sentence",
                "reject_all": "boolean",
                "rendered_prompt": {
                    "type": "path_corridor|arrow_path|arrow|sector|region",
                    "center_uv": "optional for path_corridor; otherwise [u, v] inside image bounds",
                    "direction_vec": "optional [dx, dy] unit-like vector or null",
                    "span_px": "optional positive number",
                    "path_uv": "optional for path_corridor if path_samples are provided; required for arrow_path",
                    "path_samples": [
                        {
                            "center_uv": "[u, v] on visible mode-valid traversable surface",
                            "left_edge_uv": "[u, v] left traversable boundary corresponding to this center point",
                            "right_edge_uv": "[u, v] right traversable boundary corresponding to this center point",
                            "support_label": "sidewalk|pedestrian_path|crosswalk|plaza|bike_lane|road_lane"
                        }
                    ],
                    "coverage_polygon_uv": "optional for path_corridor; if omitted it will be derived from path_samples",
                    "termination_semantics": "required for path_corridor; continue_visible_corridor | reach_turn_entry | stop_before_occlusion",
                    "mask": "optional polygon list or null",
                },
            },
        },
        ensure_ascii=False,
    )


def build_direct_prompt_verifier_question(
    *,
    route_context: dict[str, Any],
    rendered_prompt: dict[str, Any],
    image_width: int,
    image_height: int,
    navigation_mode: str = "foot",
    fpv_segmentation_summary: dict[str, Any] | None = None,
    alignment_report: dict[str, Any] | None = None,
    overlay_legend: dict[str, Any] | None = None,
    summary_source: str | None = None,
) -> str:
    compact_route = _compact_route_context(route_context)
    return json.dumps(
        {
            "task": (
                "Verify and, if needed, refine the proposed visual prompt so that its center path and left/right support boundaries are fully supported by visible traversable surface."
            ),
            "navigation_mode": str(navigation_mode or "foot"),
            "navigation_constraints": _navigation_mode_note(navigation_mode),
            "image_size": {"width": image_width, "height": image_height},
            "route_context": compact_route,
            "proposed_prompt": rendered_prompt,
            "fpv_segmentation_summary": fpv_segmentation_summary,
            "alignment_report": alignment_report,
            "overlay_legend": overlay_legend,
            "summary_source": str(summary_source or "from_same_mask_entity"),
            "verification_objectives": [
                "Check whether the path centerline stays on visible mode-valid traversable surface.",
                "Check whether the left and right support boundaries tightly match the visible support of the path rather than spilling into road, grass, shrubs, or obstacles.",
                "Check whether the termination semantic matches the actual visible corridor depth or turn entry.",
                "If fpv_segmentation_summary is present, verify that the path stays inside its traversable classes and does not cross its forbidden classes.",
                "If the proposal is partially wrong, revise it instead of only commenting on it.",
                "If the proposal cannot be grounded in visible support, reject it."
            ],
            "output_requirements": {
                "selected_prompt_type": "path_corridor|arrow_path|arrow|sector|region",
                "instruction": "short English instruction",
                "confidence": "number in [0,1]",
                "reason": "one short sentence explaining verification or correction",
                "reject_all": "boolean",
                "rendered_prompt": {
                    "type": "path_corridor|arrow_path|arrow|sector|region",
                    "center_uv": "optional for path_corridor; otherwise [u, v] inside image bounds",
                    "direction_vec": "optional [dx, dy] unit-like vector or null",
                    "span_px": "optional positive number",
                    "path_uv": "optional for path_corridor if path_samples are provided; required for arrow_path",
                    "path_samples": [
                        {
                            "center_uv": "[u, v] on visible mode-valid traversable surface",
                            "left_edge_uv": "[u, v] left traversable boundary corresponding to this center point",
                            "right_edge_uv": "[u, v] right traversable boundary corresponding to this center point",
                            "support_label": "sidewalk|pedestrian_path|crosswalk|plaza|bike_lane|road_lane"
                        }
                    ],
                    "coverage_polygon_uv": "optional for path_corridor; if omitted it will be derived from path_samples",
                    "termination_semantics": "required for path_corridor; continue_visible_corridor | reach_turn_entry | stop_before_occlusion",
                    "support_confidence": "required for path_corridor; number in [0,1]",
                    "mask": "optional polygon list or null"
                }
            },
        },
        ensure_ascii=False,
    )


def _coerce_polygon(
    raw_polygon: Any,
    *,
    image_width: int,
    image_height: int,
    min_points: int = 3,
) -> list[list[float]] | None:
    if not isinstance(raw_polygon, list):
        return None
    parsed: list[list[float]] = []
    for point in raw_polygon:
        if not (isinstance(point, (list, tuple)) and len(point) == 2):
            continue
        try:
            pu = _clamp(float(point[0]), 0.0, float(image_width))
            pv = _clamp(float(point[1]), 0.0, float(image_height))
        except Exception:
            continue
        parsed.append([round(pu, 2), round(pv, 2)])
    if len(parsed) < min_points:
        return None
    return parsed


def _coerce_point(
    raw_point: Any,
    *,
    image_width: int,
    image_height: int,
) -> list[float] | None:
    if not (isinstance(raw_point, (list, tuple)) and len(raw_point) == 2):
        return None
    try:
        pu = _clamp(float(raw_point[0]), 0.0, float(image_width))
        pv = _clamp(float(raw_point[1]), 0.0, float(image_height))
    except Exception:
        return None
    return [round(pu, 2), round(pv, 2)]


def _coerce_path(
    raw_path: Any,
    *,
    image_width: int,
    image_height: int,
) -> list[list[float]] | None:
    if not isinstance(raw_path, list):
        return None
    parsed_path: list[list[float]] = []
    for point in raw_path:
        if not (isinstance(point, (list, tuple)) and len(point) == 2):
            continue
        try:
            pu = _clamp(float(point[0]), 0.0, float(image_width))
            pv = _clamp(float(point[1]), 0.0, float(image_height))
        except Exception:
            continue
        parsed_path.append([round(pu, 2), round(pv, 2)])
    if len(parsed_path) < 2:
        return None
    return parsed_path


def _coerce_path_samples(
    raw_samples: Any,
    *,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]] | None:
    if not isinstance(raw_samples, list):
        return None
    parsed_samples: list[dict[str, Any]] = []
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, dict):
            continue
        center = _coerce_point(raw_sample.get("center_uv"), image_width=image_width, image_height=image_height)
        left_edge = _coerce_point(
            raw_sample.get("left_edge_uv") or raw_sample.get("left_support_edge_uv"),
            image_width=image_width,
            image_height=image_height,
        )
        right_edge = _coerce_point(
            raw_sample.get("right_edge_uv") or raw_sample.get("right_support_edge_uv"),
            image_width=image_width,
            image_height=image_height,
        )
        if center is None or left_edge is None or right_edge is None:
            continue
        support_label = str(raw_sample.get("support_label", "") or "").strip().lower() or None
        parsed_samples.append(
            {
                "center_uv": center,
                "left_edge_uv": left_edge,
                "right_edge_uv": right_edge,
                "support_label": support_label,
            }
        )
    if len(parsed_samples) < 2:
        return None
    return parsed_samples


def _derive_path_uv_from_samples(path_samples: list[dict[str, Any]] | None) -> list[list[float]] | None:
    if not path_samples:
        return None
    return [sample["center_uv"] for sample in path_samples if isinstance(sample.get("center_uv"), list)]


def _derive_corridor_polygon_from_samples(path_samples: list[dict[str, Any]] | None) -> list[list[float]] | None:
    if not path_samples:
        return None
    left_chain = [sample["left_edge_uv"] for sample in path_samples if isinstance(sample.get("left_edge_uv"), list)]
    right_chain = [sample["right_edge_uv"] for sample in path_samples if isinstance(sample.get("right_edge_uv"), list)]
    if len(left_chain) < 2 or len(right_chain) < 2:
        return None
    polygon = left_chain + list(reversed(right_chain))
    if polygon and polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    return polygon if len(polygon) >= 4 else None


def _derive_span_from_samples(path_samples: list[dict[str, Any]] | None) -> float | None:
    if not path_samples:
        return None
    widths: list[float] = []
    for sample in path_samples:
        left_edge = sample.get("left_edge_uv")
        right_edge = sample.get("right_edge_uv")
        if not (isinstance(left_edge, list) and isinstance(right_edge, list)):
            continue
        widths.append(math.hypot(right_edge[0] - left_edge[0], right_edge[1] - left_edge[1]))
    if not widths:
        return None
    return round(sum(widths) / len(widths), 2)


def _derive_direction_from_path(path_uv: list[list[float]] | None) -> list[float] | None:
    if not path_uv or len(path_uv) < 2:
        return None
    start = path_uv[-2]
    end = path_uv[-1]
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return None
    return [round(dx / norm, 4), round(dy / norm, 4)]


def _coerce_rendered_prompt(raw_prompt: Any, *, image_width: int, image_height: int) -> dict[str, Any] | None:
    if not isinstance(raw_prompt, dict):
        return None
    prompt_type = str(raw_prompt.get("type", "") or "").strip().lower()
    if prompt_type not in {"arrow", "sector", "region", "arrow_path", "path_corridor"}:
        return None
    center = _coerce_point(raw_prompt.get("center_uv"), image_width=image_width, image_height=image_height)
    u = center[0] if center else None
    v = center[1] if center else None
    direction_vec = raw_prompt.get("direction_vec")
    if isinstance(direction_vec, (list, tuple)) and len(direction_vec) == 2:
        try:
            direction = [float(direction_vec[0]), float(direction_vec[1])]
        except Exception:
            direction = None
    else:
        direction = None
    try:
        span_px = max(float(raw_prompt.get("span_px", 0.0) or 0.0), 12.0)
    except Exception:
        span_px = 64.0
    mask = _coerce_polygon(raw_prompt.get("mask"), image_width=image_width, image_height=image_height)
    path_uv = _coerce_path(raw_prompt.get("path_uv"), image_width=image_width, image_height=image_height)
    path_samples = _coerce_path_samples(
        raw_prompt.get("path_samples"),
        image_width=image_width,
        image_height=image_height,
    )
    if path_uv is None:
        path_uv = _derive_path_uv_from_samples(path_samples)
    coverage_polygon_uv = _coerce_polygon(
        raw_prompt.get("coverage_polygon_uv"),
        image_width=image_width,
        image_height=image_height,
    )
    if coverage_polygon_uv is None:
        coverage_polygon_uv = _derive_corridor_polygon_from_samples(path_samples)
    if coverage_polygon_uv is None and mask is not None:
        coverage_polygon_uv = mask
    termination_semantics = str(raw_prompt.get("termination_semantics", "") or "").strip().lower() or None
    if termination_semantics not in {None, "continue_visible_corridor", "reach_turn_entry", "stop_before_occlusion"}:
        termination_semantics = None
    support_confidence = raw_prompt.get("support_confidence")
    try:
        support_confidence = (
            round(_clamp(float(support_confidence), 0.0, 1.0), 4)
            if support_confidence is not None
            else None
        )
    except Exception:
        support_confidence = None
    if prompt_type == "arrow_path" and path_uv is None:
        if u is None or v is None:
            return None
        path_uv = [[round(image_width * 0.5, 2), round(image_height * 0.94, 2)], [round(u, 2), round(v, 2)]]
    if prompt_type == "path_corridor":
        if path_samples is None or path_uv is None or coverage_polygon_uv is None or termination_semantics is None:
            return None
        u, v = path_uv[-1]
        if direction is None:
            direction = _derive_direction_from_path(path_uv)
        derived_span = _derive_span_from_samples(path_samples)
        if derived_span is not None:
            span_px = derived_span
    if u is None or v is None:
        return None
    return {
        "type": prompt_type,
        "center_uv": [round(u, 2), round(v, 2)],
        "direction_vec": direction,
        "span_px": round(span_px, 2),
        "path_uv": path_uv,
        "path_samples": path_samples,
        "coverage_polygon_uv": coverage_polygon_uv,
        "termination_semantics": termination_semantics,
        "support_confidence": support_confidence,
        "mask": mask,
    }


def coerce_direct_prompt_decision(
    *,
    raw: dict[str, Any] | None,
    route_context: dict[str, Any],
    image_width: int,
    image_height: int,
) -> VisualPromptDecision | None:
    if not isinstance(raw, dict):
        return None
    reject_all = bool(raw.get("reject_all"))
    rendered_prompt = _coerce_rendered_prompt(
        raw.get("rendered_prompt"),
        image_width=image_width,
        image_height=image_height,
    )
    if rendered_prompt is None and not reject_all:
        return None
    local = route_context.get("local_semantics", {}) or {}
    try:
        confidence = _clamp(float(raw.get("confidence", 0.0) or 0.0), 0.0, 1.0)
    except Exception:
        confidence = 0.0
    prompt_type = str(raw.get("selected_prompt_type", "") or (rendered_prompt or {}).get("type") or "").strip() or None
    return VisualPromptDecision(
        selected_candidate_id=None,
        selected_prompt_type=prompt_type,
        instruction=str(raw.get("instruction", "") or local.get("short_action_summary", "") or "Follow the selected visual prompt.").strip(),
        confidence=round(confidence, 4),
        reason=str(raw.get("reason", "") or "LLM directly generated a visual prompt.").strip(),
        reject_all=reject_all,
        fallback_used=False,
        arbiter_source="llm_direct",
        rendered_prompt=rendered_prompt,
        debug={"raw": raw},
    )
