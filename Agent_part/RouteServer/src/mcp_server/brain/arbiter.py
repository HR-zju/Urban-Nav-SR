from __future__ import annotations

import json
from typing import Any

from mcp_server.route_context.schema import VisualPromptDecision


TURN_FOCUSED_TYPES = {"turn_sector", "entrance_candidate_sector", "crossing_candidate_sector"}


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


def candidate_to_rendered_prompt(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    prompt_type = str(candidate.get("prompt_type") or "").strip() or "arrow"
    rendered = {
        "type": prompt_type,
        "center_uv": candidate.get("center_uv"),
        "direction_vec": candidate.get("direction_vec"),
        "span_px": candidate.get("span_px"),
        "mask": candidate.get("mask"),
        "semantic_tag": candidate.get("semantic_tag"),
    }
    return rendered


def build_arbiter_system_prompt() -> str:
    return (
        "You are a navigation visual prompt arbiter. "
        "Given one FPV image, a compact route context summary, and a small list of prompt candidates, "
        "select the single candidate that is most visually traversable and route-consistent. "
        "Never invent candidate ids. Prefer candidates aligned with visible open corridor regions and the route intent. "
        "If all candidates are clearly unsafe or irrelevant, set reject_all=true. "
        "Return JSON only with keys: selected_candidate_id, selected_prompt_type, instruction, confidence, reason, reject_all."
    )


def build_direct_prompt_system_prompt() -> str:
    return (
        "You are a pedestrian navigation visual prompt generator. "
        "Your job is to draw one conservative image-plane guidance prompt for the next 1 to 3 seconds of walking. "
        "Treat sidewalks, pedestrian plazas, curb cuts, and marked pedestrian crossings as preferred space. "
        "Treat motor-vehicle lanes, open roadway, sky, walls, fences, parked cars, and freestanding obstacles such as signboards as forbidden unless a legal crossing is visually obvious. "
        "Do not send the user into the middle of a road when pedestrian space exists. "
        "Prefer the nearest safe sidewalk-side opening over a farther but wider drivable area. "
        "The prompt must start near the lower image area and end on a visible walkable opening. "
        "If the scene is uncertain, be conservative and set reject_all=true. "
        "Return JSON only with keys: selected_prompt_type, instruction, confidence, reason, reject_all, rendered_prompt. "
        "rendered_prompt must include: type, center_uv, direction_vec, span_px, path_uv, mask. "
        "Prefer type=arrow_path. Do not output prose outside JSON."
    )


def build_arbiter_question(
    *,
    route_context: dict[str, Any],
    prompt_proposals: list[dict[str, Any]],
    pov_prompt_candidates: list[dict[str, Any]],
    navigation_mode: str = "foot",
) -> str:
    compact_route = _compact_route_context(route_context)
    compact_proposals = [
        {
            "proposal_id": item.get("proposal_id"),
            "proposal_type": item.get("proposal_type"),
            "semantic_tag": item.get("semantic_tag"),
            "priority": item.get("priority"),
        }
        for item in (prompt_proposals or [])[:6]
    ]
    compact_candidates = [_candidate_compact(item) for item in (pov_prompt_candidates or [])[:6]]
    return json.dumps(
        {
            "task": (
                "Select the best prompt candidate for this FPV frame. "
                "Use the image first, then route context, then candidate scores. "
                "The instruction should be short and action-oriented."
            ),
            "navigation_mode": str(navigation_mode or "foot"),
            "navigation_constraints": _navigation_mode_note(navigation_mode),
            "route_context": compact_route,
            "bev_prompt_proposals": compact_proposals,
            "pov_prompt_candidates": compact_candidates,
            "output_requirements": {
                "selected_candidate_id": "must be one of the provided proposal_id values, unless reject_all=true",
                "selected_prompt_type": "copy the selected candidate prompt_type",
                "instruction": "short English instruction",
                "confidence": "number in [0,1]",
                "reason": "one short sentence",
                "reject_all": "boolean",
            },
        },
        ensure_ascii=False,
    )


def build_direct_prompt_question(
    *,
    route_context: dict[str, Any],
    prompt_proposals: list[dict[str, Any]],
    visual_regions: dict[str, Any],
    image_width: int,
    image_height: int,
    navigation_mode: str = "foot",
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
        }
        for item in ((visual_regions or {}).get("candidate_regions", []) or [])[:4]
    ]
    return json.dumps(
        {
            "task": (
                "Generate one conservative image-plane navigation trajectory prompt. "
                "Use the image first, then route context, then the top route prior. "
                "Prefer a single arrow trajectory through a visible sidewalk-side opening and avoid blocked central obstacles."
            ),
            "navigation_mode": str(navigation_mode or "foot"),
            "navigation_constraints": _navigation_mode_note(navigation_mode),
            "image_size": {"width": image_width, "height": image_height},
            "route_context": compact_route,
            "top_route_prior": top_prior,
            "visual_regions_hint": compact_regions,
            "hard_rules": [
                "For foot mode, prioritize sidewalk or pedestrian-space openings.",
                "Do not place the trajectory on apparent motor lanes or open roadway if sidewalk-like space exists.",
                "Do not run the trajectory through a central obstacle such as a signboard.",
                "The path should begin near the lower image region and terminate on a visible walkable opening.",
                "Prefer the shortest safe path that stays on pedestrian space.",
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


def _candidate_type_bonus(proposal_type: str, *, off_route: bool, distance_to_next_turn_m: float | None) -> float:
    bonus = 0.0
    if off_route and proposal_type == "connector_merge_sector":
        bonus += 0.18
    if proposal_type in TURN_FOCUSED_TYPES and distance_to_next_turn_m is not None and distance_to_next_turn_m <= 12.0:
        bonus += 0.16
    if proposal_type == "corridor_sector" and (distance_to_next_turn_m is None or distance_to_next_turn_m > 18.0):
        bonus += 0.12
    return bonus


def _rule_score(candidate: dict[str, Any], route_context: dict[str, Any]) -> float:
    pose_match = route_context.get("pose_match", {}) or {}
    progress = route_context.get("progress", {}) or {}
    debug = candidate.get("debug", {}) or {}
    proposal_type = str(debug.get("proposal_type", ""))
    priority = float(candidate.get("priority", 0.0) or 0.0)
    proj = float(candidate.get("projection_confidence", 0.0) or 0.0)
    visual = float(candidate.get("visual_confidence", 0.0) or 0.0)
    overlap = float(debug.get("region_overlap", 0.0) or 0.0)
    route_conf = float(pose_match.get("route_confidence", 0.0) or 0.0)
    distance_to_turn = progress.get("distance_to_next_turn_m")
    if distance_to_turn is not None:
        try:
            distance_to_turn = float(distance_to_turn)
        except Exception:
            distance_to_turn = None
    score = (
        0.40 * priority
        + 0.20 * proj
        + 0.20 * visual
        + 0.10 * route_conf
        + 0.10 * overlap
        + _candidate_type_bonus(
            proposal_type,
            off_route=bool(pose_match.get("off_route")),
            distance_to_next_turn_m=distance_to_turn,
        )
    )
    return _clamp(score, 0.0, 1.2)


def rule_based_visual_prompt_decision(
    *,
    route_context: dict[str, Any],
    pov_prompt_candidates: list[dict[str, Any]],
) -> VisualPromptDecision:
    local = route_context.get("local_semantics", {}) or {}
    if not pov_prompt_candidates:
        return VisualPromptDecision(
            selected_candidate_id=None,
            selected_prompt_type=None,
            instruction=str(local.get("short_action_summary", "")).strip() or "No visual prompt candidate is available.",
            confidence=0.0,
            reason="No prompt candidates were available, so the arbiter could not select a visual prompt.",
            reject_all=True,
            fallback_used=True,
            arbiter_source="rule_fallback",
            rendered_prompt=None,
            debug={},
        )

    scored = sorted(
        (
            {
                "candidate": candidate,
                "score": _rule_score(candidate, route_context),
            }
            for candidate in pov_prompt_candidates
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    chosen = scored[0]["candidate"]
    score = float(scored[0]["score"])
    debug = chosen.get("debug", {}) or {}
    reason = (
        f"Rule fallback selected {debug.get('proposal_type') or chosen.get('prompt_type')} "
        "based on route prior, visual confidence, and turn urgency."
    )
    return VisualPromptDecision(
        selected_candidate_id=str(chosen.get("proposal_id") or ""),
        selected_prompt_type=str(chosen.get("prompt_type") or debug.get("proposal_type") or ""),
        instruction=str(local.get("short_action_summary", "")).strip() or "Follow the selected forward visual prompt.",
        confidence=round(_clamp(score, 0.0, 1.0), 4),
        reason=reason,
        reject_all=False,
        fallback_used=True,
        arbiter_source="rule_fallback",
        rendered_prompt=candidate_to_rendered_prompt(chosen),
        debug={
            "ranked_scores": [
                {
                    "proposal_id": item["candidate"].get("proposal_id"),
                    "score": round(float(item["score"]), 4),
                }
                for item in scored[:5]
            ]
        },
    )


def coerce_arbiter_decision(
    *,
    raw: dict[str, Any] | None,
    route_context: dict[str, Any],
    pov_prompt_candidates: list[dict[str, Any]],
) -> VisualPromptDecision | None:
    if not isinstance(raw, dict):
        return None
    allowed_ids = {str(item.get("proposal_id") or "") for item in pov_prompt_candidates if str(item.get("proposal_id") or "")}
    selected_candidate_id = str(raw.get("selected_candidate_id", "") or "").strip()
    reject_all = bool(raw.get("reject_all"))
    if selected_candidate_id and selected_candidate_id not in allowed_ids:
        return None
    if not selected_candidate_id and not reject_all:
        return None

    chosen = next((item for item in pov_prompt_candidates if str(item.get("proposal_id") or "") == selected_candidate_id), None)
    local = route_context.get("local_semantics", {}) or {}
    prompt_type = str(raw.get("selected_prompt_type", "") or (chosen or {}).get("prompt_type") or "").strip() or None
    instruction = str(raw.get("instruction", "") or local.get("short_action_summary", "") or "").strip()
    reason = str(raw.get("reason", "")).strip()
    try:
        confidence = _clamp(float(raw.get("confidence", 0.0) or 0.0), 0.0, 1.0)
    except Exception:
        confidence = 0.0

    return VisualPromptDecision(
        selected_candidate_id=selected_candidate_id or None,
        selected_prompt_type=prompt_type,
        instruction=instruction or "Follow the selected visual prompt.",
        confidence=round(confidence, 4),
        reason=reason or "LLM selected the prompt candidate from the provided shortlist.",
        reject_all=reject_all,
        fallback_used=False,
        arbiter_source="llm",
        rendered_prompt=candidate_to_rendered_prompt(chosen),
        debug={
            "raw": raw,
            "selected_candidate_exists": chosen is not None if selected_candidate_id else False,
        },
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


def rule_based_direct_prompt_decision(
    *,
    route_context: dict[str, Any],
    fallback_candidate: dict[str, Any] | None,
) -> VisualPromptDecision:
    local = route_context.get("local_semantics", {}) or {}
    rendered_prompt = candidate_to_rendered_prompt(fallback_candidate)
    if rendered_prompt is None:
        return VisualPromptDecision(
            selected_candidate_id=None,
            selected_prompt_type=None,
            instruction=str(local.get("short_action_summary", "") or "Hold the current heading conservatively.").strip(),
            confidence=0.22,
            reason="No valid direct prompt was available, so the system fell back to a conservative no-prompt response.",
            reject_all=True,
            fallback_used=True,
            arbiter_source="rule_direct_fallback",
            rendered_prompt=None,
            debug={},
        )
    return VisualPromptDecision(
        selected_candidate_id=None,
        selected_prompt_type="arrow_path",
        instruction=str(local.get("short_action_summary", "") or "Follow the conservative fallback prompt.").strip(),
        confidence=0.46,
        reason="The direct visual prompt baseline fell back to the top structured candidate for stability.",
        reject_all=False,
        fallback_used=True,
        arbiter_source="rule_direct_fallback",
        rendered_prompt={
            "type": "arrow_path",
            "center_uv": rendered_prompt.get("center_uv"),
            "direction_vec": rendered_prompt.get("direction_vec"),
            "span_px": rendered_prompt.get("span_px"),
            "path_uv": [
                [
                    float((rendered_prompt.get("center_uv") or [128.0, 96.0])[0]),
                    float((rendered_prompt.get("center_uv") or [128.0, 96.0])[1]) + max(float(rendered_prompt.get("span_px") or 72.0), 48.0),
                ],
                rendered_prompt.get("center_uv") or [128.0, 96.0],
            ],
            "mask": rendered_prompt.get("mask"),
            "semantic_tag": rendered_prompt.get("semantic_tag"),
        },
        debug={},
    )
