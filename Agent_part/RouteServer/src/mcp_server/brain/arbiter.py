"""LVLM prompt contracts and output coercion for visual navigation prompts.

This module defines the language-level A/V contract:

    A(generator): produce one conservative image-plane prompt from FPV images,
                  route intent, segmentation summaries, SoM anchors, and
                  safe-corridor teacher priors.

    V(verifier): inspect and optionally refine a proposed prompt before the
                 deterministic mask validator accepts or rejects it.

The module does not perform segmentation math itself. It makes the contract
explicit, compact, and replayable so production runs can audit exactly what the
LVLM saw and why a prompt was accepted, repaired, or rejected.
"""

from __future__ import annotations

import json
import math
from typing import Any

from mcp_server.route_context.schema import VisualPromptDecision


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _compact_route_context(route_context: dict[str, Any]) -> dict[str, Any]:
    """Reduce RouteContext to the intent fields the LVLM needs.

    Route context is intent only. It should influence which visible support to
    choose, but it must never override image-visible safety constraints.
    """
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


def _strip_empty(value: Any) -> Any:
    """Remove None/empty fields before serializing prompt context for the LVLM."""
    if isinstance(value, dict):
        cleaned = {
            key: _strip_empty(child)
            for key, child in value.items()
            if child is not None
        }
        return {
            key: child
            for key, child in cleaned.items()
            if child is not None and child != {} and child != []
        }
    if isinstance(value, list):
        cleaned_items = [_strip_empty(item) for item in value]
        return [item for item in cleaned_items if item is not None and item != {} and item != []]
    return value


def _compact_json_dumps(payload: dict[str, Any]) -> str:
    """Serialize prompt payloads with minimal whitespace and no null clutter."""
    return json.dumps(_strip_empty(payload), ensure_ascii=False, separators=(",", ":"))


def _join_dsl_clauses(clauses: list[str]) -> str:
    """Encode repeated natural-language rules as compact semicolon DSL."""
    return ";".join(str(item).strip().rstrip(".") for item in clauses if str(item or "").strip())


def _compact_spatial_constraints_for_llm(constraints: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compress verbose SoM/segmentation guardrails into a compact DSL payload.

    The deterministic validator still receives the full source-space structure.
    This compact view is only for reducing LVLM prompt tokens.
    """
    if not isinstance(constraints, dict):
        return None
    start = constraints.get("start_constraint") if isinstance(constraints.get("start_constraint"), dict) else {}
    width = constraints.get("corridor_width_constraint") if isinstance(constraints.get("corridor_width_constraint"), dict) else {}
    center = constraints.get("centerline_constraint") if isinstance(constraints.get("centerline_constraint"), dict) else {}
    edge = constraints.get("edge_margin_constraint") if isinstance(constraints.get("edge_margin_constraint"), dict) else {}
    def _pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except Exception:
                return default
        return default
    dsl_parts = [
        (
            "start(bottom,v>={:.2f},u={:.2f}..{:.2f},pref_u={:.2f}..{:.2f})".format(
                float(start.get("required_v_min_ratio", 0.82) or 0.82),
                _pair(start.get("required_u_range_ratio"), (0.4, 0.6))[0],
                _pair(start.get("required_u_range_ratio"), (0.4, 0.6))[1],
                _pair(start.get("preferred_u_range_ratio"), (0.45, 0.55))[0],
                _pair(start.get("preferred_u_range_ratio"), (0.45, 0.55))[1],
            )
        ),
        "corridor(width<={:.2f},pref<={:.2f})".format(
            float(width.get("max_width_ratio", 0.40) or 0.40),
            float(width.get("preferred_width_ratio", 0.30) or 0.30),
        ),
        "center(mid_run,off<={:.2f})".format(float(center.get("max_offset_ratio", 0.25) or 0.25)),
        "edge(inset>={:.3f}W)".format(float(edge.get("min_margin_ratio", 0.025) or 0.025)),
        "support(pref>cond,forbid=red/obstacle)",
    ]
    safe_prior = constraints.get("safe_corridor_prior")
    compact_prior = None
    if isinstance(safe_prior, dict):
        compact_prior = {
            "src": safe_prior.get("source"),
            "path": safe_prior.get("path_uv"),
            "samples": [
                {
                    "c": sample.get("center_uv"),
                    "l": sample.get("left_edge_uv"),
                    "r": sample.get("right_edge_uv"),
                    "s": sample.get("support_label"),
                }
                for sample in list(safe_prior.get("path_samples") or [])[:6]
                if isinstance(sample, dict)
            ],
            "poly": safe_prior.get("coverage_polygon_uv"),
            "term": safe_prior.get("termination_semantics"),
        }
    return {
        "schema": constraints.get("schema_version"),
        "space": constraints.get("coordinate_space"),
        "img": constraints.get("image_size"),
        "dsl": "|".join(dsl_parts),
        "bottom_openings": start.get("bottom_openings"),
        "preferred_regions": constraints.get("preferred_regions"),
        "conditional_regions": constraints.get("conditional_connector_regions"),
        "forbidden_regions": constraints.get("forbidden_regions"),
        "label_mapping": constraints.get("label_mapping"),
        "safe_prior": compact_prior,
    }


def build_direct_prompt_system_prompt() -> str:
    """Build the A-stage system prompt for direct visual-prompt regression."""
    return (
        "Role: first-person navigation visual prompt generator.\n"
        "Goal: directly regress ONE conservative image-plane prompt for the next visible navigation action.\n"
        "Inputs: image (primary), numbered SoM overlay image when present, route_context (intent only), fpv_segmentation_summary, and som_summary.\n"
        "Output: JSON ONLY with keys: selected_prompt_type, instruction, confidence, reason, reject_all, rendered_prompt.\n"
        "\n"
        "Keywords: traversable, forbidden, bottom_openings, corridor_centerline, clearance, occlusion, reject_all.\n"
        "\n"
        "Rules:\n"
        "- Image-first: never ignore visible safety to satisfy route intent.\n"
        "- Direct regression: choose your own image-plane path from the images and structured context; do not copy or expect a precomputed geometry corridor.\n"
        "- Prefer selected_prompt_type=path_corridor.\n"
        "- The first path sample must start from the bottom-center of the image, close to the viewer's current position, not from a distant mid-image region.\n"
        "- For path_corridor: provide 3–6 smooth, forward-extending path_samples; each sample has center_uv, left_edge_uv, right_edge_uv, support_label.\n"
        "- Corridor width must be realistic and compact: do not outline the whole free-space mask; boundaries should show the safe walking channel with obstacle clearance.\n"
        "- Avoid people, vehicles, poles, walls, fences, stairs, planters, trees, and any obvious obstacles.\n"
        "\n"
        "Foot-mode semantics:\n"
        "- Prefer sidewalk / pedestrian paving / curb-cuts / marked crossings.\n"
        "- Treat grass/natural/landscaping as non-traversable.\n"
        "- Road/curb/gutter may be used as conditional connector surfaces when needed, but prefer pedestrian surfaces and never compromise obstacle clearance.\n"
        "\n"
        "If fpv_segmentation_summary is present:\n"
        "- Keep the CENTER path inside allowed_traversable classes; prefer traversable classes over conditional_traversable classes.\n"
        "- Place the CENTER path near the middle of the visible preferred traversable band, not along either edge.\n"
        "- Keep LEFT/RIGHT boundaries inside boundary support classes and away from forbidden.\n"
        "- Keep LEFT/RIGHT boundaries inset from the traversable/free-space edge; do not trace the exact sidewalk edge, road edge, grass edge, curb, or obstacle boundary.\n"
        "- Use bottom_openings: start from the widest opening and keep the center path near its middle (maximize clearance).\n"
        "- If safe_corridor_prior is present, treat it as a deterministic teacher guardrail for centered path_samples and inset boundaries.\n"
        "- If traversable becomes ambiguous ahead, shorten the corridor and set termination_semantics=stop_before_occlusion.\n"
        "\n"
        "If som_summary is present:\n"
        "- Treat region_id and numeric label as stable visual anchors over the same image plane.\n"
        "- If som_overlay_legend is present, the next image after the original FPV frame is a numbered SoM overlay.\n"
        "- Green/blue SoM labels are preferred pedestrian support; yellow/orange labels are conditional connector surfaces; red labels are forbidden obstacle/non-traversable regions.\n"
        "- Prefer path centerlines that remain inside top-ranked preferred traversable regions; use conditional labels only to connect safely, and avoid red forbidden labels.\n"
        "- Mention the chosen SoM label/region_id in reason when it helps explain the prompt.\n"
        "- If prompt and region anchors disagree, trust visible safety and explain why in reason.\n"
        "\n"
        "If bev_map_overlay_legend is present:\n"
        "- One extra image is a real OpenStreetMap raster screenshot rotated into an ego-aligned BEV view where forward is up.\n"
        "- Use it only as cross-view route/topology context; final prompt coordinates must still be placed on the FPV image plane.\n"
        "- Do not copy BEV geometry into image pixels; use it to disambiguate sidewalk, crossing, entrance, and route intent.\n"
        "\n"
        "If you cannot draw a corridor that stays on visible traversable support, set reject_all=true."
    )


def build_direct_prompt_verifier_system_prompt() -> str:
    """Build the V-stage LVLM verifier prompt."""
    return (
        "You are a first-person navigation visual prompt verifier and refiner. "
        "You receive a proposed prompt and must verify whether its center path and support corridor are truly grounded on visible traversable surface for the current navigation mode. "
        "You must inspect the image, not trust the proposal text. "
        "For pedestrian mode, visible paved sidewalk, pedestrian path, curb cut, crosswalk, and plaza surfaces are preferred support. "
        "Road, curb, and gutter pixels may be conditional connector support when needed, but should not dominate the prompt if pedestrian support is visible. "
        "Grass, shrubs, decorative green belts, planters, tree bases, exposed soil, walls, and fences are not valid support. "
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
    som_summary: dict[str, Any] | None = None,
    som_candidate_hints: list[dict[str, Any]] | None = None,
    som_overlay_legend: dict[str, Any] | None = None,
    bev_map_overlay_legend: dict[str, Any] | None = None,
    som_spatial_constraints: dict[str, Any] | None = None,
    alignment_report: dict[str, Any] | None = None,
    overlay_legend: dict[str, Any] | None = None,
    summary_source: str | None = None,
) -> str:
    """Serialize the A-stage user question as auditable JSON.

    Key contract fields:
        route_context: high-level navigation intent
        fpv_segmentation_summary: semantic traversability prior
        som_summary / som_candidate_hints: numbered visual anchors
        som_spatial_constraints: hard spatial rules and safe_corridor_prior

    The LVLM returns prompt geometry in the coordinate space described by
    image_size. The service later maps that geometry back to source pixels.
    """
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
    return _compact_json_dumps(
        {
            "prompt_encoding": "compact_json_with_constraint_dsl",
            "task": (
                "Generate ONE conservative image-plane navigation prompt (prefer path_corridor). "
                "Use the image first. route_context is intent only. "
                "Directly regress smooth image coordinates using the numbered SoM mask as visual grounding when present."
            ),
            "navigation_mode": str(navigation_mode or "foot"),
            "navigation_constraints": _navigation_mode_note(navigation_mode),
            "image_size": {"width": image_width, "height": image_height},
            "route_context": compact_route,
            "top_route_prior": top_prior,
            "visual_regions_hint": compact_regions,
            "optional_candidate_hints": compact_candidates,
            "fpv_segmentation_summary": fpv_segmentation_summary,
            "som_summary": som_summary,
            "som_candidate_hints": som_candidate_hints,
            "som_overlay_legend": som_overlay_legend,
            "bev_map_overlay_legend": bev_map_overlay_legend,
            "som_spatial_constraints": _compact_spatial_constraints_for_llm(som_spatial_constraints),
            "alignment_report": alignment_report,
            "overlay_legend": overlay_legend,
            "summary_source": str(summary_source or "from_same_mask_entity"),
            "hard_rules_dsl": _join_dsl_clauses([
                "Return JSON only (no prose).",
                "Prefer selected_prompt_type=path_corridor.",
                "The first path sample must satisfy som_spatial_constraints.start_constraint: bottom-center image band, with center_uv.v at or below the required bottom-band threshold and center_uv.u inside the narrow center range.",
                "Provide 3–6 smooth, forward-extending path_samples (center/left/right) with realistic compact corridor width.",
                "Avoid obstacles and people; do not place center or boundaries on them.",
                "If segmentation summary is present: keep center inside allowed_traversable; prefer traversable over conditional_traversable; keep boundaries away from forbidden.",
                "If som_overlay_legend is present: use the numbered SoM overlay image to align region labels with visible pixels.",
                "If bev_map_overlay_legend is present: use the real OSM screenshot as cross-view topology context only; final prompt coordinates are still FPV image coordinates.",
                "If som_summary is present: use preferred/conditional/forbidden SoM labels as grounding context, not as fixed candidate paths.",
                "Favor top-ranked preferred traversable SoM region anchors, use conditional/yellow connectors only when needed, and avoid forbidden/red labels.",
                "If som_spatial_constraints is present: obey corridor_width_constraint; do not place center or left/right boundary points inside forbidden_regions; road/curb/gutter regions may only be connector support.",
                "If som_spatial_constraints has centerline_constraint: keep each center_uv near the midpoint of the preferred traversable same-row band, similar to the old geometric fallback.",
                "If som_spatial_constraints has edge_margin_constraint: keep left_edge_uv/right_edge_uv inset from the traversable-region edges with the requested clearance.",
                "If som_spatial_constraints has safe_corridor_prior: use its path_samples as concrete teacher coordinates for the centered safe walking lane; adjust only for visible obstacle/occlusion or route intent.",
                "Left/right edges must avoid obstacles as strictly as the center path and should stay near the usable walking lane, not the far outer edge of every traversable pixel.",
                "When using a SoM anchor, include its numeric label and region_id in reason.",
                "If bottom_openings is empty: do not invent a wide bottom-starting corridor; draw a short grounded prompt or reject.",
                "If bottom_openings is present: start from the widest opening and keep center near opening midline (maximize clearance).",
                "If traversable becomes ambiguous: shorten corridor and use stop_before_occlusion.",
                "If you cannot stay on allowed support without hitting forbidden obstacle classes: set reject_all=true.",
            ]),
            "output_dsl": "json_only selected_prompt_type,instruction,confidence,reason,reject_all,rendered_prompt; path_corridor samples=[{center_uv,left_edge_uv,right_edge_uv,support_label}] term=continue_visible_corridor|reach_turn_entry|stop_before_occlusion",
        }
    )


def build_direct_prompt_verifier_question(
    *,
    route_context: dict[str, Any],
    rendered_prompt: dict[str, Any],
    image_width: int,
    image_height: int,
    navigation_mode: str = "foot",
    fpv_segmentation_summary: dict[str, Any] | None = None,
    som_summary: dict[str, Any] | None = None,
    som_candidate_hints: list[dict[str, Any]] | None = None,
    som_overlay_legend: dict[str, Any] | None = None,
    bev_map_overlay_legend: dict[str, Any] | None = None,
    som_spatial_constraints: dict[str, Any] | None = None,
    segmentation_validation: dict[str, Any] | None = None,
    alignment_report: dict[str, Any] | None = None,
    overlay_legend: dict[str, Any] | None = None,
    summary_source: str | None = None,
) -> str:
    """Serialize the V-stage verifier/refiner task as auditable JSON."""
    compact_route = _compact_route_context(route_context)
    return _compact_json_dumps(
        {
            "prompt_encoding": "compact_json_with_constraint_dsl",
            "task": (
                "Verify and, if needed, refine the proposed visual prompt so that its center path and left/right support boundaries are fully supported by visible traversable surface."
            ),
            "navigation_mode": str(navigation_mode or "foot"),
            "navigation_constraints": _navigation_mode_note(navigation_mode),
            "image_size": {"width": image_width, "height": image_height},
            "route_context": compact_route,
            "proposed_prompt": rendered_prompt,
            "fpv_segmentation_summary": fpv_segmentation_summary,
            "som_summary": som_summary,
            "som_candidate_hints": som_candidate_hints,
            "som_overlay_legend": som_overlay_legend,
            "bev_map_overlay_legend": bev_map_overlay_legend,
            "som_spatial_constraints": _compact_spatial_constraints_for_llm(som_spatial_constraints),
            "segmentation_validation": segmentation_validation,
            "alignment_report": alignment_report,
            "overlay_legend": overlay_legend,
            "summary_source": str(summary_source or "from_same_mask_entity"),
            "verification_objectives_dsl": _join_dsl_clauses([
                "Check whether the path centerline stays on visible mode-valid traversable surface.",
                "Check whether the left and right support boundaries tightly match visible support; road/curb may be connector support, but grass, shrubs, vehicles, trees, people, and occlusion are hard failures.",
                "Check whether corridor width is compact and obstacle-aware; overly wide corridors that span most of the image must be narrowed or rejected.",
                "Check whether the termination semantic matches the actual visible corridor depth or turn entry.",
                "If fpv_segmentation_summary is present, verify that the path stays inside allowed_traversable classes, prefers traversable classes, and does not cross forbidden classes.",
                "If segmentation_validation is present, explicitly repair the listed forbidden hit points or reject if repair is impossible.",
                "If som_overlay_legend is present, use the numbered SoM overlay image to verify the path against the labeled region pixels and avoid red forbidden labels.",
                "If bev_map_overlay_legend is present, use the real OSM screenshot only for cross-view route/topology consistency; keep all revised prompt coordinates in FPV image space.",
                "If som_summary is present, verify that the selected path remains geometrically consistent with its top-ranked region anchors.",
                "If som_spatial_constraints is present, enforce its bottom-center start constraint, corridor_width_constraint, and forbidden/conditional region rules.",
                "If centerline_constraint or edge_margin_constraint is present, repair prompts whose centerline hugs a traversable edge or whose boundaries trace the free-space/sidewalk edge.",
                "If safe_corridor_prior is present, compare the proposal to that teacher corridor and prefer a centered/inset revision unless the image visibly contradicts it.",
                "If the proposal is partially wrong, revise it instead of only commenting on it.",
                "If the proposal cannot be grounded in visible support, reject it."
            ]),
            "output_dsl": "json_only selected_prompt_type,instruction,confidence,reason,reject_all,rendered_prompt; path_corridor samples=[{center_uv,left_edge_uv,right_edge_uv,support_label}] term=continue_visible_corridor|reach_turn_entry|stop_before_occlusion support_confidence=0..1",
        }
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
