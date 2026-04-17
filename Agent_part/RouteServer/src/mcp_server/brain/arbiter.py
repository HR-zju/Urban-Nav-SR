from __future__ import annotations

import json
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
        "Your job is to draw one conservative image-plane guidance prompt for the next 1 to 3 seconds. "
        "Use route context only as route intent; it is not permission to ignore visible scene safety. "
        "For pedestrian mode, treat sidewalks, pedestrian plazas, curb cuts, entrances, and marked crossings as preferred space. "
        "For pedestrian mode, treat motor-vehicle lanes, open roadway, sky, walls, fences, parked cars, and freestanding obstacles such as signboards as forbidden unless a legal crossing is visually obvious. "
        "For pedestrian mode, keep the prompt inside the visible sidewalk or pedestrian-safe corridor; do not spill outside curb boundaries when a sidewalk exists. "
        "Avoid trees, poles, bollards, stairs, planters, building facades, and people. "
        "Prefer a short sidewalk-centerline arrow over a wide sector. "
        "Do not send the user into the middle of a road when pedestrian space exists, even if the route prior points forward or right. "
        "Prefer the nearest safe sidewalk-side opening over a farther but wider drivable area. "
        "The prompt must start near the lower image area and every point of the trajectory must lie on the visible mode-valid traversable surface. "
        "For pedestrian mode, every arrow_path point must be on visible sidewalk, pedestrian paving, curb cut, plaza, or legal crossing surface. "
        "Route/BEV/PoV candidates, when provided, are advisory priors only; reject or override them if they land on obstacles or non-walkable regions. "
        "If the scene is uncertain, be conservative and set reject_all=true. "
        "Return JSON only with keys: selected_prompt_type, instruction, confidence, reason, reject_all, rendered_prompt. "
        "rendered_prompt must include: type, center_uv, direction_vec, span_px, path_uv, mask. "
        "Prefer type=arrow_path. Do not output prose outside JSON."
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
                "Prefer a single arrow trajectory through a visible mode-valid opening and avoid blocked central obstacles. "
                "Candidate hints are advisory and must not override visible walkability."
            ),
            "navigation_mode": str(navigation_mode or "foot"),
            "navigation_constraints": _navigation_mode_note(navigation_mode),
            "image_size": {"width": image_width, "height": image_height},
            "route_context": compact_route,
            "top_route_prior": top_prior,
            "visual_regions_hint": compact_regions,
            "optional_candidate_hints": compact_candidates,
            "hard_rules": [
                "For foot mode, prioritize sidewalk or pedestrian-space openings.",
                "Do not place the trajectory on apparent motor lanes or open roadway if sidewalk-like space exists.",
                "Do not run the trajectory through a central obstacle such as a signboard.",
                "Keep the trajectory inside the visible pedestrian-safe corridor and away from curb edges.",
                "Prefer a short arrow_path that follows the sidewalk centerline.",
                "The path should begin near the lower image region and every point must stay on the visible traversable surface for the current mode.",
                "If you cannot see a continuous traversable surface for all arrow_path points, set reject_all=true instead of drawing.",
                "Prefer the shortest safe path that stays on pedestrian space.",
                "If route prior and visible mode-valid space conflict, prefer visible mode-valid space and explain the conflict.",
            ],
            "output_requirements": {
                "selected_prompt_type": "arrow_path|arrow|sector|region",
                "instruction": "short English instruction",
                "confidence": "number in [0,1]",
                "reason": "one short sentence",
                "reject_all": "boolean",
                "rendered_prompt": {
                    "type": "arrow_path|arrow|sector|region",
                    "center_uv": "[u, v] inside image bounds",
                    "direction_vec": "[dx, dy] unit-like vector or null",
                    "span_px": "positive number",
                    "path_uv": "optional polyline [[u,v], ...] from lower image to target",
                    "mask": "optional polygon list or null",
                },
            },
        },
        ensure_ascii=False,
    )


def _coerce_rendered_prompt(raw_prompt: Any, *, image_width: int, image_height: int) -> dict[str, Any] | None:
    if not isinstance(raw_prompt, dict):
        return None
    prompt_type = str(raw_prompt.get("type", "") or "").strip().lower()
    if prompt_type not in {"arrow", "sector", "region", "arrow_path"}:
        return None
    center = raw_prompt.get("center_uv")
    if not (isinstance(center, (list, tuple)) and len(center) == 2):
        return None
    try:
        u = _clamp(float(center[0]), 0.0, float(image_width))
        v = _clamp(float(center[1]), 0.0, float(image_height))
    except Exception:
        return None
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
    mask = raw_prompt.get("mask")
    if not isinstance(mask, list):
        mask = None
    raw_path = raw_prompt.get("path_uv")
    path_uv = None
    if isinstance(raw_path, list):
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
        if len(parsed_path) >= 2:
            path_uv = parsed_path
    if prompt_type == "arrow_path" and path_uv is None:
        path_uv = [[round(image_width * 0.5, 2), round(image_height * 0.94, 2)], [round(u, 2), round(v, 2)]]
    return {
        "type": prompt_type,
        "center_uv": [round(u, 2), round(v, 2)],
        "direction_vec": direction,
        "span_px": round(span_px, 2),
        "path_uv": path_uv,
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
