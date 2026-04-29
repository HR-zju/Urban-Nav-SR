from __future__ import annotations

import base64
import io
import os
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional
from urllib.parse import quote
from uuid import uuid4

from PIL import Image, ImageOps

from mcp_server.brain import (
    build_direct_prompt_question,
    build_direct_prompt_system_prompt,
    build_direct_prompt_verifier_question,
    build_direct_prompt_verifier_system_prompt,
    coerce_direct_prompt_decision,
)
from mcp_server.brain.fpv_segmentation import (
    build_segmentation_teacher_bundle,
    load_fpv_segmentation_context,
    segmentation_summary_for_llm,
    validate_rendered_prompt_against_segmentation,
)
from mcp_server.services.fpv_segmentation_service import segment_fpv_frame
from mcp_server.llm import (
    arbitrate_local_pov_candidates_with_llm,
    arbitrate_pov_candidates_with_llm,
    local_image_path_to_data_url,
)
from mcp_server.qa import (
    build_pov_prompt_candidate_bundle,
    build_visual_prompt_qa_bundle,
    pov_candidate_bundle_to_dict,
)
from mcp_server.route_context import (
    build_route_state,
    local_bev_context_to_dict,
    route_context_to_dict,
    visual_prompt_decision_to_dict,
)
from mcp_server.route_context.schema import VisualPromptDecision


def _image_href_for_overlay(*, image_path: Optional[str], image_url: Optional[str]) -> str:
    if str(image_path or "").strip():
        return local_image_path_to_data_url(str(image_path))
    return str(image_url or "").strip()


def _load_prompt_image(*, image_path: Optional[str], image_url: Optional[str]) -> Image.Image:
    if str(image_path or "").strip():
        return ImageOps.exif_transpose(Image.open(Path(str(image_path)).expanduser())).convert("RGB")
    raw = str(image_url or "").strip()
    if raw.startswith("data:image/") and "," in raw:
        _, encoded = raw.split(",", 1)
        return ImageOps.exif_transpose(Image.open(io.BytesIO(base64.b64decode(encoded)))).convert("RGB")
    raise RuntimeError("Visual prompt rendering currently supports image_path or data:image URL inputs")


def _prompt_image_size(*, image_path: Optional[str], image_url: Optional[str]) -> tuple[int, int]:
    image = _load_prompt_image(image_path=image_path, image_url=image_url)
    return int(image.size[0]), int(image.size[1])


def _render_prompt_overlay_svg(
    *,
    image_href: str,
    image_width: int,
    image_height: int,
    rendered_prompt: Dict[str, Any] | None,
    title: str = "Visual Prompt Overlay",
) -> str:
    width = max(int(image_width), 32)
    height = max(int(image_height), 32)
    max_display_width = max(int(os.getenv("VISUAL_PROMPT_SVG_DISPLAY_WIDTH", "1200") or 1200), 320)
    display_width = min(width, max_display_width)
    display_height = max(int(round(display_width * height / max(width, 1))), 1)
    min_dim = min(width, height)
    title_font = max(min_dim * 0.018, 18.0)
    label_font = max(min_dim * 0.014, 14.0)
    line_stroke = max(min_dim * 0.009, 5.0)
    outline_stroke = max(min_dim * 0.005, 2.0)
    marker_radius = max(min_dim * 0.011, 6.0)
    bg = (
        f'<image href="{image_href}" x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="xMidYMid slice" />'
        if image_href
        else f'<rect x="0" y="0" width="{width}" height="{height}" fill="#0f172a" />'
    )
    if not isinstance(rendered_prompt, dict):
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{display_width}" height="{display_height}" viewBox="0 0 {width} {height}">'
            f"{bg}"
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="rgba(2,6,23,0.10)" />'
            f'<text x="{title_font:.1f}" y="{title_font * 1.7:.1f}" fill="#e2e8f0" font-size="{title_font:.1f}">{escape(title)}</text>'
            f'<text x="{width/2:.1f}" y="{height/2:.1f}" text-anchor="middle" fill="#fca5a5" font-size="{title_font:.1f}">No rendered prompt</text>'
            "</svg>"
        )

    prompt_type = str(rendered_prompt.get("type") or "arrow").strip().lower()
    center = rendered_prompt.get("center_uv") or [width * 0.5, height * 0.55]
    try:
        cx = float(center[0])
        cy = float(center[1])
    except Exception:
        cx = width * 0.5
        cy = height * 0.55
    span = max(float(rendered_prompt.get("span_px", 72.0) or 72.0), 16.0)
    direction = rendered_prompt.get("direction_vec") or [0.0, -1.0]
    try:
        dx = float(direction[0])
        dy = float(direction[1])
    except Exception:
        dx, dy = 0.0, -1.0
    coverage_polygon = rendered_prompt.get("coverage_polygon_uv")
    path_samples = rendered_prompt.get("path_samples")

    anchor_x = width * 0.5
    anchor_y = height * 0.94
    label = str(rendered_prompt.get("semantic_tag") or prompt_type)
    termination_semantics = str(rendered_prompt.get("termination_semantics") or "").strip()
    support_confidence = rendered_prompt.get("support_confidence")

    overlay_parts = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="rgba(2,6,23,0.06)" />',
        f'<rect x="{title_font * 0.6:.1f}" y="{title_font * 0.45:.1f}" width="{min(width * 0.72, title_font * 24):.1f}" height="{title_font * 1.65:.1f}" rx="{title_font * 0.35:.1f}" fill="rgba(2,6,23,0.58)" />',
        f'<text x="{title_font:.1f}" y="{title_font * 1.55:.1f}" fill="#e2e8f0" font-size="{title_font:.1f}" font-family="monospace">{escape(title)}</text>',
    ]

    if isinstance(coverage_polygon, list) and len(coverage_polygon) >= 3:
        points = " ".join(
            f"{float(point[0]):.1f},{float(point[1]):.1f}"
            for point in coverage_polygon
            if isinstance(point, (list, tuple)) and len(point) == 2
        )
        if points:
            overlay_parts.append(
                f'<polygon points="{points}" fill="rgba(34,197,94,0.12)" stroke="#22c55e" stroke-width="{outline_stroke:.1f}" />'
            )

    if isinstance(path_samples, list):
        left_points = []
        right_points = []
        center_points = []
        for sample in path_samples:
            if not isinstance(sample, dict):
                continue
            left = sample.get("left_edge_uv")
            right = sample.get("right_edge_uv")
            center_sample = sample.get("center_uv")
            if isinstance(left, (list, tuple)) and len(left) == 2:
                left_points.append([float(left[0]), float(left[1])])
            if isinstance(right, (list, tuple)) and len(right) == 2:
                right_points.append([float(right[0]), float(right[1])])
            if isinstance(center_sample, (list, tuple)) and len(center_sample) == 2:
                center_points.append([float(center_sample[0]), float(center_sample[1])])
        if len(left_points) >= 2:
            left_str = " ".join(f"{point[0]:.1f},{point[1]:.1f}" for point in left_points)
            overlay_parts.append(
                f'<polyline points="{left_str}" fill="none" stroke="#22c55e" stroke-width="{outline_stroke * 1.2:.1f}" stroke-linecap="round" />'
            )
        if len(right_points) >= 2:
            right_str = " ".join(f"{point[0]:.1f},{point[1]:.1f}" for point in right_points)
            overlay_parts.append(
                f'<polyline points="{right_str}" fill="none" stroke="#22c55e" stroke-width="{outline_stroke * 1.2:.1f}" stroke-linecap="round" />'
            )

        if center_points:
            cx = float(center_points[-1][0])
            cy = float(center_points[-1][1])

    path_uv = rendered_prompt.get("path_uv")
    valid_path = (
        isinstance(path_uv, list)
        and len(path_uv) >= 2
        and all(isinstance(point, (list, tuple)) and len(point) == 2 for point in path_uv)
    )

    if prompt_type in {"arrival", "arrival_marker", "destination_marker"}:
        overlay_parts.extend(
            [
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{marker_radius * 2.4:.1f}" fill="rgba(34,197,94,0.24)" stroke="#22c55e" stroke-width="{outline_stroke * 1.8:.1f}" />',
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{marker_radius * 0.8:.1f}" fill="#22c55e" />',
                f'<path d="M {cx - marker_radius * 1.0:.1f} {cy + marker_radius * 0.05:.1f} L {cx - marker_radius * 0.28:.1f} {cy + marker_radius * 0.75:.1f} L {cx + marker_radius * 1.15:.1f} {cy - marker_radius * 0.9:.1f}" fill="none" stroke="#ecfdf5" stroke-width="{outline_stroke * 1.7:.1f}" stroke-linecap="round" stroke-linejoin="round" />',
                f'<text x="{cx:.1f}" y="{cy + marker_radius * 3.45:.1f}" text-anchor="middle" fill="#dcfce7" font-size="{title_font:.1f}" font-family="monospace">ARRIVED</text>',
            ]
        )
    elif prompt_type in {"arrow", "arrow_path", "path_corridor"} and valid_path:
        points = " ".join(f"{float(point[0]):.1f},{float(point[1]):.1f}" for point in path_uv)
        overlay_parts.append(
            f'<polyline points="{points}" fill="none" stroke="#f97316" stroke-width="{line_stroke:.1f}" stroke-linecap="round" stroke-linejoin="round" />'
        )
        end_x = float(path_uv[-1][0])
        end_y = float(path_uv[-1][1])
        prev_x = float(path_uv[-2][0])
        prev_y = float(path_uv[-2][1])
        vx = end_x - prev_x
        vy = end_y - prev_y
        norm = max((vx * vx + vy * vy) ** 0.5, 1e-6)
        ux, uy = vx / norm, vy / norm
        px, py = -uy, ux
        ah = max(span * 0.16, marker_radius * 2.2)
        ax1 = end_x - ux * ah - px * ah * 0.55
        ay1 = end_y - uy * ah - py * ah * 0.55
        ax2 = end_x - ux * ah + px * ah * 0.55
        ay2 = end_y - uy * ah + py * ah * 0.55
        overlay_parts.append(
            f'<polygon points="{end_x:.1f},{end_y:.1f} {ax1:.1f},{ay1:.1f} {ax2:.1f},{ay2:.1f}" fill="#f97316" />'
        )
    elif prompt_type == "arrow":
        overlay_parts.append(
            f'<line x1="{anchor_x:.1f}" y1="{anchor_y:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" stroke="#f97316" stroke-width="{line_stroke:.1f}" stroke-linecap="round" />'
        )
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        ux, uy = dx / norm, dy / norm
        px, py = -uy, ux
        ah = max(span * 0.16, marker_radius * 2.2)
        ax1 = cx - ux * ah - px * ah * 0.55
        ay1 = cy - uy * ah - py * ah * 0.55
        ax2 = cx - ux * ah + px * ah * 0.55
        ay2 = cy - uy * ah + py * ah * 0.55
        overlay_parts.append(
            f'<polygon points="{cx:.1f},{cy:.1f} {ax1:.1f},{ay1:.1f} {ax2:.1f},{ay2:.1f}" fill="#f97316" />'
        )
    elif prompt_type == "region":
        overlay_parts.append(
            f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{span * 0.55:.1f}" ry="{span * 0.32:.1f}" fill="rgba(168,85,247,0.18)" stroke="#a855f7" stroke-width="{outline_stroke * 1.5:.1f}" />'
        )
    else:
        left_x = cx - span * 0.6
        right_x = cx + span * 0.6
        top_y = cy - span * 0.52
        bottom_y = cy + span * 0.52
        overlay_parts.append(
            f'<polygon points="{anchor_x:.1f},{anchor_y:.1f} {left_x:.1f},{bottom_y:.1f} {cx:.1f},{top_y:.1f} {right_x:.1f},{bottom_y:.1f}" fill="rgba(56,189,248,0.22)" stroke="#38bdf8" stroke-width="{outline_stroke * 1.5:.1f}" />'
        )

    if prompt_type not in {"arrival", "arrival_marker", "destination_marker"}:
        termination_label = termination_semantics[:32] if termination_semantics else ""
        support_label = ""
        if support_confidence is not None:
            try:
                support_label = f"support={float(support_confidence):.2f}"
            except Exception:
                support_label = ""
        overlay_parts.extend(
            [
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{marker_radius:.1f}" fill="#e2e8f0" />',
                f'<rect x="{cx + marker_radius * 1.4:.1f}" y="{cy - label_font * 1.45:.1f}" width="{min(width - cx - marker_radius * 2.0, max(label_font * 11.0, 1.0)):.1f}" height="{label_font * 1.7:.1f}" rx="{label_font * 0.35:.1f}" fill="rgba(2,6,23,0.58)" />',
                f'<text x="{cx + marker_radius * 1.9:.1f}" y="{cy - label_font * 0.22:.1f}" fill="#e2e8f0" font-size="{label_font:.1f}" font-family="monospace">{escape(label[:42])}</text>',
            ]
        )
        if termination_label or support_label:
            meta_text = " | ".join(item for item in [termination_label, support_label] if item)
            overlay_parts.extend(
                [
                    f'<rect x="{title_font * 0.6:.1f}" y="{title_font * 2.3:.1f}" width="{min(width * 0.78, max(len(meta_text) * label_font * 0.72, title_font * 8.0)):.1f}" height="{label_font * 1.65:.1f}" rx="{label_font * 0.35:.1f}" fill="rgba(2,6,23,0.52)" />',
                    f'<text x="{title_font:.1f}" y="{title_font * 3.0:.1f}" fill="#bbf7d0" font-size="{label_font:.1f}" font-family="monospace">{escape(meta_text)}</text>',
                ]
            )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{display_width}" height="{display_height}" viewBox="0 0 {width} {height}">'
        f"{bg}{''.join(overlay_parts)}</svg>"
    )


def _svg_to_data_url(svg: str) -> str:
    return "data:image/svg+xml;utf8," + quote(svg, safe=":/?&=,+-_.!~*'()#% ")


def _is_arrival_state(route_context: Dict[str, Any], *, threshold_m: float | None = None) -> bool:
    enable_arrival_marker = str(os.getenv("VISUAL_PROMPT_ENABLE_ARRIVAL_MARKER", "")).strip().lower()
    if enable_arrival_marker not in {"1", "true", "yes", "on"}:
        return False
    progress = route_context.get("progress", {}) or {}
    pose_match = route_context.get("pose_match", {}) or {}
    if bool(pose_match.get("off_route")):
        return False
    try:
        remaining = float(progress.get("remaining_distance_m"))
    except Exception:
        return False
    if threshold_m is None:
        threshold_m = float(os.getenv("VISUAL_PROMPT_ARRIVAL_THRESHOLD_M", "2.0") or 2.0)
    return remaining <= max(float(threshold_m), 0.0)


def _arrival_visual_prompt_decision(
    *,
    route_context: Dict[str, Any],
    image_width: int,
    image_height: int,
) -> VisualPromptDecision:
    progress = route_context.get("progress", {}) or {}
    try:
        remaining = float(progress.get("remaining_distance_m"))
    except Exception:
        remaining = 0.0
    return VisualPromptDecision(
        selected_candidate_id=None,
        selected_prompt_type="arrival_marker",
        instruction="You have arrived at the destination.",
        confidence=0.95,
        reason=f"RouteContext reports remaining_distance_m={remaining:.2f}, within the arrival threshold.",
        reject_all=False,
        fallback_used=False,
        arbiter_source="route_arrival_state",
        rendered_prompt={
            "type": "arrival_marker",
            "center_uv": [float(image_width) * 0.5, float(image_height) * 0.68],
            "direction_vec": [0.0, 0.0],
            "span_px": max(min(float(image_width), float(image_height)) * 0.12, 48.0),
            "path_uv": None,
            "mask": None,
            "semantic_tag": "destination_arrived",
        },
        debug={
            "remaining_distance_m": remaining,
            "arrival_threshold_m": float(os.getenv("VISUAL_PROMPT_ARRIVAL_THRESHOLD_M", "2.0") or 2.0),
            "arrival_marker_enabled": True,
        },
    )


def _reject_visual_prompt_decision(
    *,
    reason: str,
    arbiter_source: str,
    route_context: Dict[str, Any],
) -> VisualPromptDecision:
    local = route_context.get("local_semantics", {}) or {}
    return VisualPromptDecision(
        selected_candidate_id=None,
        selected_prompt_type=None,
        instruction=str(local.get("short_action_summary") or "Visual prompt unavailable for this frame.").strip(),
        confidence=0.0,
        reason=str(reason).strip() or "Visual prompt generation failed.",
        reject_all=True,
        fallback_used=False,
        arbiter_source=arbiter_source,
        rendered_prompt=None,
        debug={},
    )


def _apply_segmentation_constraint(
    *,
    decision: VisualPromptDecision | None,
    route_context: Dict[str, Any],
    segmentation_context: Dict[str, Any] | None,
) -> tuple[VisualPromptDecision | None, Dict[str, Any] | None]:
    if decision is None or not isinstance(decision.rendered_prompt, dict) or not isinstance(segmentation_context, dict):
        return decision, None
    validation = validate_rendered_prompt_against_segmentation(decision.rendered_prompt, segmentation_context)
    if validation is None:
        return decision, None
    decision.debug = dict(decision.debug or {})
    decision.debug["fpv_segmentation_validation"] = validation
    if validation.get("valid"):
        return decision, validation
    reason = (
        "FPV segmentation traversability constraint rejected the rendered prompt: "
        f"{validation.get('reason')}"
    )
    rejected = _reject_visual_prompt_decision(
        reason=reason,
        arbiter_source="fpv_segmentation_reject",
        route_context=route_context,
    )
    rejected.debug["fpv_segmentation_validation"] = validation
    rejected.debug["rejected_rendered_prompt"] = decision.rendered_prompt
    rejected.debug["rejected_instruction"] = decision.instruction
    rejected.debug["rejected_confidence"] = decision.confidence
    return rejected, validation


def _arbiter_extra_image_urls(
    *,
    fpv_segmentation: Dict[str, Any] | None,
    enable_segmentation_overlay_input: bool,
) -> tuple[list[str], dict[str, Any]]:
    debug: dict[str, Any] = {
        "enabled": bool(enable_segmentation_overlay_input),
        "used": False,
        "source": None,
        "error": None,
    }
    if not enable_segmentation_overlay_input or not isinstance(fpv_segmentation, dict):
        return [], debug

    visualization_url = str(
        fpv_segmentation.get("visualization_url")
        or fpv_segmentation.get("visualization_image_url")
        or ""
    ).strip()
    if visualization_url:
        debug["used"] = True
        debug["source"] = "visualization_url"
        return [visualization_url], debug

    visualization_path = str(fpv_segmentation.get("visualization_path") or "").strip()
    if not visualization_path:
        debug["error"] = "visualization_path_missing"
        return [], debug

    try:
        overlay_data_url = local_image_path_to_data_url(visualization_path)
        debug["used"] = True
        debug["source"] = "visualization_path"
        debug["visualization_path"] = str(Path(visualization_path).expanduser())
        return [overlay_data_url], debug
    except Exception as exc:
        debug["error"] = str(exc)
        debug["visualization_path"] = str(Path(visualization_path).expanduser())
        return [], debug


def _segmentation_overlay_legend(fpv_segmentation_summary: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(fpv_segmentation_summary, dict):
        return None
    return {
        "summary_source": "from_same_mask_entity",
        "color_semantics": {
            "traversable": "green",
            "boundary": "teal",
            "forbidden": "red",
        },
        "traversable_class_names": list(fpv_segmentation_summary.get("traversable_class_names") or []),
        "boundary_class_names": list(fpv_segmentation_summary.get("boundary_class_names") or []),
        "forbidden_class_names": list(fpv_segmentation_summary.get("forbidden_class_names") or []),
    }


def _segmentation_reject_code(validation: Dict[str, Any] | None) -> str | None:
    if not isinstance(validation, dict):
        return None
    reason = str(validation.get("reason") or "")
    if "low_center_traversable_ratio" in reason:
        return "low_center_traversable_ratio"
    if "center_path_hits_forbidden_surface" in reason:
        return "center_hits_forbidden"
    if "boundary_hits_forbidden_surface" in reason:
        return "boundary_hits_forbidden"
    if "no_center_path_samples" in reason:
        return "no_center_path_samples"
    return None


def _build_inputs_for_A(
    *,
    route_context: Dict[str, Any],
    image_width: int,
    image_height: int,
    navigation_mode: str,
    fpv_segmentation_summary: Dict[str, Any] | None,
) -> Dict[str, Any]:
    teacher_alignment_report = None
    overlay_legend = _segmentation_overlay_legend(fpv_segmentation_summary)
    if isinstance(fpv_segmentation_summary, dict):
        candidate_alignment = fpv_segmentation_summary.get("alignment_report")
        if isinstance(candidate_alignment, dict):
            teacher_alignment_report = candidate_alignment
    return {
        "summary_source": "from_same_mask_entity" if isinstance(fpv_segmentation_summary, dict) else "not_available",
        "alignment_report": teacher_alignment_report,
        "overlay_legend": overlay_legend,
        "question": build_direct_prompt_question(
            route_context=route_context,
            prompt_proposals=[],
            visual_regions={},
            image_width=image_width,
            image_height=image_height,
            navigation_mode=navigation_mode,
            fpv_segmentation_summary=fpv_segmentation_summary,
            alignment_report=teacher_alignment_report,
            overlay_legend=overlay_legend,
            summary_source=(
                "from_same_mask_entity" if isinstance(fpv_segmentation_summary, dict) else "not_available"
            ),
        ),
        "system_prompt": build_direct_prompt_system_prompt(),
    }


async def _run_A_generator(
    *,
    image_path: Optional[str],
    image_url: Optional[str],
    question: str,
    system_prompt: str,
    llm_detail: str,
    extra_image_urls: list[str],
    route_context: Dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[Dict[str, Any] | None, VisualPromptDecision | None]:
    if str(image_path or "").strip():
        llm_raw = await arbitrate_local_pov_candidates_with_llm(
            image_path=str(image_path),
            question=question,
            detail=llm_detail,
            system_prompt=system_prompt,
            extra_image_urls=extra_image_urls,
        )
    else:
        llm_raw = await arbitrate_pov_candidates_with_llm(
            image_url=str(image_url),
            question=question,
            detail=llm_detail,
            system_prompt=system_prompt,
            extra_image_urls=extra_image_urls,
        )
    decision = coerce_direct_prompt_decision(
        raw=llm_raw,
        route_context=route_context,
        image_width=image_width,
        image_height=image_height,
    )
    return llm_raw, decision


def _run_V_constraint(
    *,
    decision: VisualPromptDecision | None,
    route_context: Dict[str, Any],
    segmentation_context: Dict[str, Any] | None,
    constraint_stage: str,
) -> tuple[VisualPromptDecision | None, Dict[str, Any] | None]:
    constrained_decision, validation = _apply_segmentation_constraint(
        decision=decision,
        route_context=route_context,
        segmentation_context=segmentation_context,
    )
    if isinstance(constrained_decision, VisualPromptDecision):
        constrained_decision.debug = dict(constrained_decision.debug or {})
        constrained_decision.debug["constraint_stage"] = str(constraint_stage)
        reject_code = _segmentation_reject_code(validation)
        if reject_code:
            constrained_decision.debug["segmentation_reject_code"] = reject_code
    return constrained_decision, validation


async def _run_V_verifier(
    *,
    image_path: Optional[str],
    image_url: Optional[str],
    route_context: Dict[str, Any],
    rendered_prompt: Dict[str, Any],
    image_width: int,
    image_height: int,
    navigation_mode: str,
    fpv_segmentation_summary: Dict[str, Any] | None,
    alignment_report: Dict[str, Any] | None,
    overlay_legend: Dict[str, Any] | None,
    summary_source: str,
    llm_detail: str,
    extra_image_urls: list[str],
) -> tuple[Dict[str, Any] | None, str, VisualPromptDecision | None]:
    verifier_question = build_direct_prompt_verifier_question(
        route_context=route_context,
        rendered_prompt=rendered_prompt,
        image_width=image_width,
        image_height=image_height,
        navigation_mode=navigation_mode,
        fpv_segmentation_summary=fpv_segmentation_summary,
        alignment_report=alignment_report,
        overlay_legend=overlay_legend,
        summary_source=summary_source,
    )
    verifier_system_prompt = build_direct_prompt_verifier_system_prompt()
    if str(image_path or "").strip():
        verifier_raw = await arbitrate_local_pov_candidates_with_llm(
            image_path=str(image_path),
            question=verifier_question,
            detail=llm_detail,
            system_prompt=verifier_system_prompt,
            extra_image_urls=extra_image_urls,
        )
    else:
        verifier_raw = await arbitrate_pov_candidates_with_llm(
            image_url=str(image_url),
            question=verifier_question,
            detail=llm_detail,
            system_prompt=verifier_system_prompt,
            extra_image_urls=extra_image_urls,
        )
    verified_decision = coerce_direct_prompt_decision(
        raw=verifier_raw,
        route_context=route_context,
        image_width=image_width,
        image_height=image_height,
    )
    return verifier_raw, verifier_question, verified_decision


async def run_zero_shot_visual_prompt_pipeline(
    *,
    gps: Dict[str, float],
    route_polyline: List[Dict[str, Any]] | List[List[float]],
    coarse_waypoints: List[Dict[str, Any]],
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    fpv_segmentation: Optional[Dict[str, Any]] = None,
    sam3_request: Optional[Dict[str, Any]] = None,
    heading_deg: Optional[float] = None,
    lookahead_m: float = 20.0,
    off_route_threshold_m: float = 20.0,
    front_extent_m: float = 30.0,
    back_extent_m: float = 10.0,
    side_extent_m: float = 15.0,
    visual_top_k: int = 5,
    candidate_top_k: int = 5,
    llm_detail: str = "auto",
    navigation_mode: str = "foot",
    enable_verifier: bool = True,
    enable_segmentation_overlay_input: bool = False,
    enforce_alignment: bool = True,
    emit_qa_bundle: bool = False,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    del visual_top_k, candidate_top_k
    started_at = perf_counter()
    resolved_request_id = str(request_id or uuid4())
    telemetry: Dict[str, Any] = {
        "request_id": resolved_request_id,
        "elapsed_ms": 0,
        "error_code": None,
        "stages": {},
    }

    def _mark_stage(
        stage: str,
        stage_started_at: float,
        *,
        status: str,
        error_code: str | None = None,
    ) -> None:
        telemetry["stages"][stage] = {
            "status": str(status),
            "elapsed_ms": int(round((perf_counter() - stage_started_at) * 1000.0)),
            "error_code": str(error_code) if error_code else None,
        }
        if error_code and telemetry.get("error_code") is None:
            telemetry["error_code"] = str(error_code)

    if not str(image_path or "").strip() and not str(image_url or "").strip():
        raise RuntimeError("run_zero_shot_visual_prompt_agent requires image_path or image_url")

    route_stage_started_at = perf_counter()
    state = build_route_state(
        gps=gps,
        heading_deg=heading_deg,
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
        lookahead_m=lookahead_m,
        off_route_threshold_m=off_route_threshold_m,
        front_extent_m=front_extent_m,
        back_extent_m=back_extent_m,
        side_extent_m=side_extent_m,
    )
    _mark_stage("route_context", route_stage_started_at, status="ok")
    route_context_dict = route_context_to_dict(state.route_context)
    local_bev_dict = local_bev_context_to_dict(state.local_bev_context)
    image_width, image_height = _prompt_image_size(image_path=image_path, image_url=image_url)
    qa_bundle: Dict[str, Any] | None = None
    candidate_bundle = None
    segmentation_validation: Dict[str, Any] | None = None
    segmentation_context = None
    segmentation_summary = None
    segmentation_error: str | None = None
    segmentation_error_code: str | None = None
    segmentation_teacher_bundle: Dict[str, Any] | None = None
    extra_image_urls: list[str] = []
    multi_image_debug: dict[str, Any] = {
        "enabled": bool(enable_segmentation_overlay_input),
        "used": False,
    }
    sam3_result: Dict[str, Any] | None = None
    normalized_sam3: Dict[str, Any] | None = None
    segmentation_stage_started_at = perf_counter()
    if not isinstance(fpv_segmentation, dict) and isinstance(sam3_request, dict):
        try:
            sam3_payload = dict(sam3_request)
            sam3_payload.setdefault("request_id", resolved_request_id)
            sam3_result = await segment_fpv_frame(**sam3_payload)
            normalized_sam3 = {
                "status": sam3_result.get("status"),
                "provider": sam3_result.get("provider"),
                "schema_version": sam3_result.get("schema_version"),
                "request_id": sam3_result.get("request_id"),
                "timing_ms": sam3_result.get("timing_ms"),
                "artifacts": sam3_result.get("artifacts"),
                "error": sam3_result.get("error"),
            }
            generated_fpv_segmentation = sam3_result.get("fpv_segmentation")
            if not isinstance(generated_fpv_segmentation, dict):
                artifacts = sam3_result.get("artifacts")
                if isinstance(artifacts, dict):
                    seg_npy = str(artifacts.get("segmentation_npy_path") or "").strip()
                    class_map = str(artifacts.get("class_mapping_path") or "").strip()
                    vis_path = str(artifacts.get("visualization_path") or "").strip()
                    if seg_npy and class_map:
                        generated_fpv_segmentation = {
                            "segmentation_npy_path": seg_npy,
                            "class_mapping_path": class_map,
                        }
                        if vis_path:
                            generated_fpv_segmentation["visualization_path"] = vis_path
            if str(sam3_result.get("status") or "").lower() != "ok":
                error_payload = sam3_result.get("error") or {}
                segmentation_error_code = str(error_payload.get("code") or "SEGMENTATION_PAYLOAD_INVALID")
                segmentation_error = str(
                    error_payload.get("message")
                    or "segment_fpv_frame returned status=error"
                )
                generated_fpv_segmentation = None
            if isinstance(generated_fpv_segmentation, dict):
                fpv_segmentation = generated_fpv_segmentation
            else:
                if segmentation_error is None:
                    segmentation_error = "segment_fpv_frame returned an invalid fpv_segmentation payload"
                    segmentation_error_code = "SEGMENTATION_PAYLOAD_INVALID"
        except Exception as exc:
            segmentation_error = str(exc)
            segmentation_error_code = "SEGMENTATION_COMMAND_FAILED"
    if isinstance(fpv_segmentation, dict):
        try:
            segmentation_context = load_fpv_segmentation_context(
                fpv_segmentation,
                navigation_mode=navigation_mode,
                image_width=image_width,
                image_height=image_height,
            )
            segmentation_summary = segmentation_summary_for_llm(segmentation_context)
        except Exception as exc:
            segmentation_error = str(exc)
            segmentation_error_code = "SEGMENTATION_PAYLOAD_INVALID"
            if not enforce_alignment:
                segmentation_error = None
                segmentation_error_code = None
    if isinstance(segmentation_context, dict):
        segmentation_teacher_bundle = build_segmentation_teacher_bundle(segmentation_context)
        _mark_stage("segmentation", segmentation_stage_started_at, status="ok")
    elif segmentation_error:
        _mark_stage(
            "segmentation",
            segmentation_stage_started_at,
            status="error",
            error_code=segmentation_error_code or "SEGMENTATION_PAYLOAD_INVALID",
        )
    else:
        _mark_stage("segmentation", segmentation_stage_started_at, status="skipped")

    extra_image_urls, multi_image_debug = _arbiter_extra_image_urls(
        fpv_segmentation=fpv_segmentation,
        enable_segmentation_overlay_input=enable_segmentation_overlay_input,
    )

    llm_raw: Dict[str, Any] | None = None
    verifier_raw: Dict[str, Any] | None = None
    llm_error: str | None = None
    verifier_error: str | None = None
    decision = None
    generator_decision = None
    question = ""
    system_prompt = ""
    verifier_question = ""
    generator_stage_started_at = perf_counter()
    verifier_stage_started_at = perf_counter()
    av_input_stage_started_at = perf_counter()
    v_constraint_generator_started_at = perf_counter()
    v_constraint_verifier_started_at = perf_counter()
    if _is_arrival_state(route_context_dict):
        decision = _arrival_visual_prompt_decision(
            route_context=route_context_dict,
            image_width=image_width,
            image_height=image_height,
        )
        _mark_stage("A_build_inputs", av_input_stage_started_at, status="skipped")
        _mark_stage("A_generator", generator_stage_started_at, status="skipped")
        _mark_stage("V_constraint_generator", v_constraint_generator_started_at, status="skipped")
        _mark_stage("V_verifier", verifier_stage_started_at, status="skipped")
        _mark_stage("V_constraint_verifier", v_constraint_verifier_started_at, status="skipped")
        _mark_stage("generator_llm", generator_stage_started_at, status="skipped")
        _mark_stage("verifier_llm", verifier_stage_started_at, status="skipped")
    else:
        try:
            av_inputs = _build_inputs_for_A(
                route_context=route_context_dict,
                image_width=image_width,
                image_height=image_height,
                navigation_mode=navigation_mode,
                fpv_segmentation_summary=segmentation_summary,
            )
            question = str(av_inputs.get("question") or "")
            system_prompt = str(av_inputs.get("system_prompt") or "")
            _mark_stage("A_build_inputs", av_input_stage_started_at, status="ok")
            llm_raw, decision = await _run_A_generator(
                image_path=image_path,
                image_url=image_url,
                question=question,
                system_prompt=system_prompt,
                llm_detail=llm_detail,
                extra_image_urls=extra_image_urls,
                route_context=route_context_dict,
                image_width=image_width,
                image_height=image_height,
            )
            _mark_stage("A_generator", generator_stage_started_at, status="ok")
            _mark_stage("generator_llm", generator_stage_started_at, status="ok")
            decision, segmentation_validation = _run_V_constraint(
                decision=decision,
                route_context=route_context_dict,
                segmentation_context=segmentation_context,
                constraint_stage="generator",
            )
            _mark_stage("V_constraint_generator", v_constraint_generator_started_at, status="ok")
            if (
                enable_verifier
                and decision is not None
                and not decision.reject_all
                and isinstance(decision.rendered_prompt, dict)
            ):
                try:
                    verifier_raw, verifier_question, verified_decision = await _run_V_verifier(
                        image_path=image_path,
                        image_url=image_url,
                        route_context=route_context_dict,
                        rendered_prompt=decision.rendered_prompt,
                        image_width=image_width,
                        image_height=image_height,
                        navigation_mode=navigation_mode,
                        fpv_segmentation_summary=segmentation_summary,
                        alignment_report=av_inputs.get("alignment_report") if isinstance(av_inputs.get("alignment_report"), dict) else None,
                        overlay_legend=av_inputs.get("overlay_legend") if isinstance(av_inputs.get("overlay_legend"), dict) else None,
                        summary_source=str(av_inputs.get("summary_source") or "from_same_mask_entity"),
                        llm_detail=llm_detail,
                        extra_image_urls=extra_image_urls,
                    )
                    generator_decision = decision
                    _mark_stage("V_verifier", verifier_stage_started_at, status="ok")
                    _mark_stage("verifier_llm", verifier_stage_started_at, status="ok")
                    verified_decision, segmentation_validation = _run_V_constraint(
                        decision=verified_decision,
                        route_context=route_context_dict,
                        segmentation_context=segmentation_context,
                        constraint_stage="verifier",
                    )
                    _mark_stage("V_constraint_verifier", v_constraint_verifier_started_at, status="ok")
                    if verified_decision is not None:
                        verified_decision.arbiter_source = "llm_direct_verified"
                        verified_decision.debug = {
                            "generator_raw": llm_raw,
                            "generator_decision": visual_prompt_decision_to_dict(generator_decision),
                            "verifier_raw": verifier_raw,
                            "verifier_decision": visual_prompt_decision_to_dict(verified_decision),
                            "av_alignment_report": av_inputs.get("alignment_report"),
                            "av_summary_source": av_inputs.get("summary_source"),
                        }
                        decision = verified_decision
                except Exception as exc:
                    verifier_error = str(exc)
                    _mark_stage("V_verifier", verifier_stage_started_at, status="error", error_code="LLM_VERIFIER_FAILED")
                    _mark_stage(
                        "V_constraint_verifier",
                        v_constraint_verifier_started_at,
                        status="skipped",
                    )
                    _mark_stage(
                        "verifier_llm",
                        verifier_stage_started_at,
                        status="error",
                        error_code="LLM_VERIFIER_FAILED",
                    )
            else:
                _mark_stage("V_verifier", verifier_stage_started_at, status="skipped")
                _mark_stage("V_constraint_verifier", v_constraint_verifier_started_at, status="skipped")
                _mark_stage("verifier_llm", verifier_stage_started_at, status="skipped")
        except Exception as exc:
            llm_error = str(exc)
            _mark_stage("A_build_inputs", av_input_stage_started_at, status="error", error_code="LLM_GENERATOR_FAILED")
            _mark_stage("A_generator", generator_stage_started_at, status="error", error_code="LLM_GENERATOR_FAILED")
            _mark_stage("V_constraint_generator", v_constraint_generator_started_at, status="skipped")
            _mark_stage("V_verifier", verifier_stage_started_at, status="skipped")
            _mark_stage("V_constraint_verifier", v_constraint_verifier_started_at, status="skipped")
            _mark_stage(
                "generator_llm",
                generator_stage_started_at,
                status="error",
                error_code="LLM_GENERATOR_FAILED",
            )
            _mark_stage("verifier_llm", verifier_stage_started_at, status="skipped")

    if (
        enable_verifier
        and decision is not None
        and not decision.reject_all
        and isinstance(decision.rendered_prompt, dict)
        and verifier_raw is None
        and not verifier_error
        and question
    ):
        verifier_error = "visual prompt verifier did not run"

    if decision is None:
        decision = _reject_visual_prompt_decision(
            reason=(
                llm_error
                or "The RouteContext-driven LVLM agent did not return a valid rendered prompt."
            ),
            arbiter_source="llm_route_context_reject",
            route_context=route_context_dict,
        )

    decision_dict = visual_prompt_decision_to_dict(decision)
    decision_dict.setdefault("debug", {})
    if llm_error:
        decision_dict["debug"]["llm_error"] = llm_error
    if verifier_error:
        decision_dict["debug"]["verifier_error"] = verifier_error
    if segmentation_error:
        decision_dict["debug"]["fpv_segmentation_error"] = segmentation_error
    if segmentation_error_code:
        decision_dict["debug"]["fpv_segmentation_error_code"] = segmentation_error_code
    if segmentation_summary:
        decision_dict["debug"]["fpv_segmentation_summary"] = segmentation_summary
    if segmentation_teacher_bundle:
        decision_dict["debug"]["segmentation_teacher_bundle"] = segmentation_teacher_bundle
    if sam3_result:
        decision_dict["debug"]["sam3_result"] = sam3_result
    if normalized_sam3:
        decision_dict["debug"]["normalized_sam3"] = normalized_sam3
    if not enable_verifier:
        decision_dict["debug"]["verifier_skipped"] = True
    decision_dict["debug"]["multi_image_arbiter"] = {
        **multi_image_debug,
        "extra_image_count": len(extra_image_urls),
    }
    # Export the exact LVLM prompt payload for audit/debug. This is offline/debug-only
    # and can be large, but it avoids guessing whether the right constraints were sent.
    try:
        question_json = json.loads(question) if question else None
    except Exception:
        question_json = None
    decision_dict["debug"]["arbiter_prompt"] = {
        "generator": {
            "system_prompt": system_prompt,
            "question": question,
            "question_json": question_json,
        },
        "verifier": {
            "question": verifier_question,
        },
    }

    qa_stage_started_at = perf_counter()
    if emit_qa_bundle:
        candidate_bundle = build_pov_prompt_candidate_bundle(
            route_context=route_context_dict,
            local_bev_context=local_bev_dict,
            image_width=image_width,
            image_height=image_height,
            segmentation_context=segmentation_context,
            rendered_prompt=decision.rendered_prompt,
        )
        qa_bundle = build_visual_prompt_qa_bundle(
            route_context=route_context_dict,
            local_bev_context=local_bev_dict,
            image_width=image_width,
            image_height=image_height,
            fpv_segmentation_summary=segmentation_summary,
            segmentation_validation=segmentation_validation,
            rendered_prompt=decision.rendered_prompt,
            arbiter_decision=decision_dict,
            candidate_bundle=candidate_bundle,
        )
        decision_dict["debug"]["qa_bundle_summary"] = {
            "schema_version": qa_bundle.get("schema_version"),
            "task_counts": qa_bundle.get("task_counts", {}),
            "family_counts": qa_bundle.get("family_counts", {}),
        }
        _mark_stage("qa_bundle", qa_stage_started_at, status="ok")
    else:
        _mark_stage("qa_bundle", qa_stage_started_at, status="skipped")

    render_stage_started_at = perf_counter()
    image_href = _image_href_for_overlay(image_path=image_path, image_url=image_url)
    overlay_svg = _render_prompt_overlay_svg(
        image_href=image_href,
        image_width=image_width,
        image_height=image_height,
        rendered_prompt=decision.rendered_prompt,
        title="Mode-Aware Visual Prompt",
    )

    generator_overlay_svg: str | None = None
    generator_overlay_svg_data_url: str | None = None
    if generator_decision is not None and isinstance(generator_decision.rendered_prompt, dict):
        generator_overlay_svg = _render_prompt_overlay_svg(
            image_href=image_href,
            image_width=image_width,
            image_height=image_height,
            rendered_prompt=generator_decision.rendered_prompt,
            title="Generator Visual Prompt (Pre-Verifier)",
        )
        generator_overlay_svg_data_url = _svg_to_data_url(generator_overlay_svg)
    _mark_stage("render", render_stage_started_at, status="ok")

    telemetry["elapsed_ms"] = int(round((perf_counter() - started_at) * 1000.0))
    decision_dict["debug"]["telemetry"] = telemetry

    return {
        "request_id": resolved_request_id,
        "route_context": route_context_dict,
        "local_bev_context": local_bev_dict,
        "arbiter_decision": decision_dict,
        "overlay_svg": overlay_svg,
        "overlay_svg_data_url": _svg_to_data_url(overlay_svg),
        "generator_overlay_svg": generator_overlay_svg,
        "generator_overlay_svg_data_url": generator_overlay_svg_data_url,
        "sam3": sam3_result,
        "fpv_segmentation_summary": segmentation_summary,
        "qa_bundle": qa_bundle,
        "pov_prompt_candidate_bundle": pov_candidate_bundle_to_dict(candidate_bundle) if emit_qa_bundle else None,
        "telemetry": telemetry,
    }
