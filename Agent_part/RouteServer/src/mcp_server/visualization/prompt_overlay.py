"""Historical image-plane prompt rendering without model or segmentation execution."""
from __future__ import annotations
import os
from html import escape
from typing import Any, Dict
from urllib.parse import quote


def _rendered_prompt_path_uv(rendered_prompt: Dict[str, Any] | None) -> list[list[float]]:
    """Return the center trajectory for baseline evaluation/export.

    RouteServer historically stores path-corridor prompts as center/left/right
    samples.  SideWalk Phase1 evaluates only the center trajectory, so this
    helper makes the export path explicit and avoids treating support
    boundaries or coverage polygons as model predictions.
    """
    if not isinstance(rendered_prompt, dict):
        return []
    raw_path = rendered_prompt.get("path_uv")
    if isinstance(raw_path, list):
        path: list[list[float]] = []
        for point in raw_path:
            if not (isinstance(point, (list, tuple)) and len(point) == 2):
                continue
            try:
                path.append([round(float(point[0]), 2), round(float(point[1]), 2)])
            except Exception:
                continue
        if len(path) >= 2:
            return path
    path_samples = rendered_prompt.get("path_samples")
    if isinstance(path_samples, list):
        path = []
        for sample in path_samples:
            if not isinstance(sample, dict):
                continue
            center = sample.get("center_uv")
            if not (isinstance(center, (list, tuple)) and len(center) == 2):
                continue
            try:
                path.append([round(float(center[0]), 2), round(float(center[1]), 2)])
            except Exception:
                continue
        if len(path) >= 2:
            return path
    return []


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
    path_uv = _rendered_prompt_path_uv(rendered_prompt)
    if path_uv:
        cx = float(path_uv[-1][0])
        cy = float(path_uv[-1][1])

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
