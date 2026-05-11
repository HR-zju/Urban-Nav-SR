"""FPV semantic-mask grounding for visual navigation prompts.

This module is the deterministic geometry layer behind the LVLM visual prompt
pipeline. It owns three responsibilities:

1. Convert raw FPV segmentation artifacts into navigation-mode class sets.
2. Build a safe-corridor teacher prior from the semantic mask.
3. Validate and, when possible, repair rendered LVLM prompts against that mask.

Core geometry:

    preferred run midpoint:
        u_center = (u_start + u_end) / 2

    inset corridor boundaries:
        u_left  = u_center - min(w_available / 2, w_max / 2, w_pref / 2)
        u_right = u_center + min(w_available / 2, w_max / 2, w_pref / 2)

    centerline offset:
        offset = |u_prompt - u_center| / ((u_end - u_start + 1) / 2)

    edge clearance:
        margin_ratio = min(u_left - u_allowed_start,
                           u_allowed_end - u_right) / image_width

    corridor width:
        width_ratio = ||right_edge_uv - left_edge_uv||_2 / image_width

The LVLM may still directly regress prompt geometry, but this module provides a
production-grade deterministic guardrail so final prompt geometry remains
centered, compact, and inside traversable support whenever the mask allows it.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy import ndimage as _ndimage  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    _ndimage = None

DEFAULT_FOOT_CORRIDOR_WIDTH_RATIO = 0.40
DEFAULT_NON_FOOT_CORRIDOR_WIDTH_RATIO = 0.75
CORRIDOR_WIDTH_RATIO_ENV = "VISUAL_PROMPT_CORRIDOR_WIDTH_RATIO"
CENTERLINE_OFFSET_RATIO_ENV = "VISUAL_PROMPT_CENTERLINE_OFFSET_RATIO"
EDGE_MARGIN_RATIO_ENV = "VISUAL_PROMPT_EDGE_MARGIN_RATIO"


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def corridor_width_ratio_limits(navigation_mode: str) -> tuple[float, float]:
    """Return hard and preferred width ratios for path-corridor rendering.

    Formula:
        max_width_px       = max_width_ratio * image_width
        preferred_width_px = preferred_width_ratio * image_width

    Foot navigation defaults to a narrower affordance because the prompt should
    describe a walkable lane, not the full free-space/sidewalk mask.
    """
    mode = str(navigation_mode or "foot").strip().lower()
    default_max = DEFAULT_FOOT_CORRIDOR_WIDTH_RATIO if mode == "foot" else DEFAULT_NON_FOOT_CORRIDOR_WIDTH_RATIO
    raw = os.getenv(CORRIDOR_WIDTH_RATIO_ENV, "").strip()
    max_ratio = default_max
    if raw:
        try:
            max_ratio = _clamp_float(float(raw), 0.10, 0.95)
        except Exception:
            max_ratio = default_max
    preferred_ratio = round(max_ratio * 0.75, 4)
    return float(max_ratio), float(preferred_ratio)


def centerline_offset_ratio_limit() -> float:
    """Maximum normalized deviation from the same-row traversable midpoint."""
    raw = os.getenv(CENTERLINE_OFFSET_RATIO_ENV, "").strip()
    if raw:
        try:
            return _clamp_float(float(raw), 0.10, 1.00)
        except Exception:
            pass
    return 0.55


def edge_margin_ratio_limit() -> float:
    """Minimum normalized clearance from allowed-run edges for corridor borders."""
    raw = os.getenv(EDGE_MARGIN_RATIO_ENV, "").strip()
    if raw:
        try:
            return _clamp_float(float(raw), 0.00, 0.20)
        except Exception:
            pass
    return 0.025


def _coerce_path(path_value: Any) -> Path | None:
    raw = str(path_value or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _load_class_mapping(mapping_path: Path) -> tuple[dict[int, str], dict[str, int]]:
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    raw_mapping = payload.get("class_to_idx") or {}
    id_to_name: dict[int, str] = {}
    name_to_id: dict[str, int] = {}
    for name, class_id in raw_mapping.items():
        try:
            numeric_id = int(class_id)
        except Exception:
            continue
        normalized = str(name).strip().lower()
        id_to_name[numeric_id] = normalized
        name_to_id[normalized] = numeric_id
    return id_to_name, name_to_id


def _mode_rule_names(navigation_mode: str) -> dict[str, set[str]]:
    """Map navigation mode to semantic class policy.

    The sets intentionally distinguish preferred support from conditional
    connector support. For foot mode, sidewalk/crosswalk/driveway are preferred;
    road/curb/gutter are only conditional connector surfaces.
    """
    mode = str(navigation_mode or "foot").strip().lower()
    conditional: set[str] = set()
    if mode == "car":
        traversable = {"road", "driveway"}
        boundary = {"road", "driveway", "curb", "gutter"}
    elif mode == "bike":
        traversable = {"road", "driveway", "crosswalk", "sidewalk"}
        boundary = {"road", "driveway", "crosswalk", "sidewalk", "curb", "gutter"}
    else:
        traversable = {"sidewalk", "crosswalk", "driveway"}
        conditional = {"road", "curb", "gutter"}
        boundary = {"sidewalk", "crosswalk", "driveway", "road", "curb", "gutter"}
    forbidden = {
        "natural",
        "grass",
        "lawn",
        "wall",
        "building",
        "fence",
        "trees",
        "tree",
        "vehicle",
        "cyclist",
        "scooter",
        "animal",
        "person",
        "pedestrian",
        "rider",
        "occlusion",
        "generic_obstacle",
    }
    return {
        "traversable": traversable,
        "conditional_traversable": conditional,
        "allowed_traversable": traversable | conditional,
        "boundary": boundary,
        "forbidden": forbidden,
    }


def _point_to_pixel(mask: np.ndarray, point_uv: list[float] | tuple[float, float]) -> tuple[int, int]:
    height, width = mask.shape[:2]
    u = _clamp_int(round(float(point_uv[0])), 0, width - 1)
    v = _clamp_int(round(float(point_uv[1])), 0, height - 1)
    return u, v


def _class_id_near(
    mask: np.ndarray,
    point_uv: list[float] | tuple[float, float],
    *,
    radius_px: int = 4,
) -> set[int]:
    u, v = _point_to_pixel(mask, point_uv)
    height, width = mask.shape[:2]
    x0 = _clamp_int(u - radius_px, 0, width - 1)
    x1 = _clamp_int(u + radius_px, 0, width - 1)
    y0 = _clamp_int(v - radius_px, 0, height - 1)
    y1 = _clamp_int(v + radius_px, 0, height - 1)
    patch = mask[y0 : y1 + 1, x0 : x1 + 1]
    return {int(class_id) for class_id in np.unique(patch)}


def _interpolate_points(path_uv: list[list[float]], step_px: float = 16.0) -> list[list[float]]:
    if len(path_uv) < 2:
        return list(path_uv)
    sampled: list[list[float]] = [list(path_uv[0])]
    for start, end in zip(path_uv[:-1], path_uv[1:]):
        dx = float(end[0] - start[0])
        dy = float(end[1] - start[1])
        distance = math.hypot(dx, dy)
        if distance <= 1e-6:
            continue
        steps = max(int(math.ceil(distance / max(step_px, 1.0))), 1)
        for step in range(1, steps + 1):
            t = step / steps
            sampled.append([round(start[0] + dx * t, 2), round(start[1] + dy * t, 2)])
    return sampled


def _point_list(raw_points: Any) -> list[list[float]]:
    if not isinstance(raw_points, list):
        return []
    points: list[list[float]] = []
    for point in raw_points:
        if not (isinstance(point, (list, tuple)) and len(point) == 2):
            continue
        try:
            points.append([float(point[0]), float(point[1])])
        except Exception:
            continue
    return points


def _class_names_for_ids(id_to_name: dict[int, str], class_ids: set[int]) -> list[str]:
    return [id_to_name.get(int(class_id), f"unknown_{class_id}") for class_id in sorted(class_ids)]


def _ids_for_names(id_to_name: dict[int, str], names: set[str]) -> set[int]:
    normalized = {str(name).strip().lower() for name in names}
    return {int(class_id) for class_id, class_name in id_to_name.items() if str(class_name).strip().lower() in normalized}


def _polygon_mask(mask_shape_hw: tuple[int, int], polygon_uv: list[list[float]]) -> np.ndarray | None:
    if len(polygon_uv) < 3:
        return None
    height, width = mask_shape_hw
    image = Image.new("L", (int(width), int(height)), 0)
    try:
        ImageDraw.Draw(image).polygon([(float(point[0]), float(point[1])) for point in polygon_uv], fill=1)
    except Exception:
        return None
    polygon = np.asarray(image, dtype=bool)
    if not bool(polygon.any()):
        return None
    return polygon


def _class_ratio(mask: np.ndarray, region_mask: np.ndarray, class_ids: set[int]) -> float:
    total = int(region_mask.sum())
    if total <= 0 or not class_ids:
        return 0.0
    return float(np.isin(mask[region_mask], list(class_ids)).sum()) / float(total)


def _top_class_ratios(mask: np.ndarray, region_mask: np.ndarray, id_to_name: dict[int, str], *, limit: int = 8) -> list[dict[str, Any]]:
    total = int(region_mask.sum())
    if total <= 0:
        return []
    class_ids, counts = np.unique(mask[region_mask], return_counts=True)
    rows: list[dict[str, Any]] = []
    for class_id, count in sorted(zip(class_ids.tolist(), counts.tolist()), key=lambda item: item[1], reverse=True)[:limit]:
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": id_to_name.get(int(class_id), f"unknown_{class_id}"),
                "pixel_count": int(count),
                "pixel_ratio": round(float(count) / float(total), 4),
            }
        )
    return rows


def _largest_bottom_run(mask: np.ndarray, valid_ids: set[int], row_ratio: float) -> dict[str, Any] | None:
    if not valid_ids:
        return None
    height, width = mask.shape[:2]
    row = _clamp_int(round((height - 1) * row_ratio), 0, height - 1)
    row_mask = np.isin(mask[row], list(valid_ids))
    if not row_mask.any():
        return None
    best_start = None
    best_end = None
    run_start = None
    for index, value in enumerate(row_mask.tolist() + [False]):
        if value and run_start is None:
            run_start = index
        if not value and run_start is not None:
            run_end = index - 1
            if best_start is None or (run_end - run_start) > (best_end - best_start):
                best_start = run_start
                best_end = run_end
            run_start = None
    if best_start is None or best_end is None:
        return None
    return {
        "row_v": row,
        "u_min": int(best_start),
        "u_max": int(best_end),
        "width_px": int(best_end - best_start + 1),
    }


def _class_runs_at_row(mask: np.ndarray, valid_ids: set[int], row_v: float) -> list[tuple[int, int]]:
    """Return contiguous valid-pixel intervals [u_start, u_end] on one image row."""
    if not valid_ids:
        return []
    height, width = mask.shape[:2]
    row = _clamp_int(round(float(row_v)), 0, height - 1)
    row_mask = np.isin(mask[row], list(valid_ids))
    runs: list[tuple[int, int]] = []
    run_start = None
    for index, value in enumerate(row_mask.tolist() + [False]):
        if value and run_start is None:
            run_start = index
        if not value and run_start is not None:
            runs.append((int(run_start), int(index - 1)))
            run_start = None
    return runs


def _run_containing_or_nearest(runs: list[tuple[int, int]], point_u: float) -> tuple[int, int] | None:
    if not runs:
        return None
    u = float(point_u)
    for run_start, run_end in runs:
        if float(run_start) <= u <= float(run_end):
            return run_start, run_end
    return min(runs, key=lambda item: min(abs(u - float(item[0])), abs(u - float(item[1]))))


def _run_midpoint(run: tuple[int, int]) -> float:
    return (float(run[0]) + float(run[1])) / 2.0


def _is_conditional_connector_label(label: Any) -> bool:
    text = str(label or "").strip().lower()
    return any(token in text for token in ("conditional", "connector", "road", "curb", "gutter"))


def _allowed_clearance_map(mask: np.ndarray, allowed_ids: set[int]) -> np.ndarray | None:
    """Return the per-pixel clearance to the nearest non-allowed pixel.

    The map is a geometric proxy for a safe walking corridor:

        clearance(u, v) = dist((u, v), complement(allowed_mask))

    This is the quantity we maximize when choosing a centerline sample, because
    it naturally prefers the medial axis of the traversable region and pushes
    the prompt away from obstacles, edges, and thin bottlenecks.
    """
    if not allowed_ids:
        return None
    allowed_mask = np.isin(mask, list(allowed_ids))
    if not bool(np.any(allowed_mask)):
        return None
    if _ndimage is None:
        return None
    return _ndimage.distance_transform_edt(allowed_mask).astype(np.float32)


def _corridor_sample_from_runs(
    *,
    center_run: tuple[int, int],
    edge_run: tuple[int, int],
    row_v: float,
    image_width: int,
    navigation_mode: str,
    support_label: str,
    clearance_row: np.ndarray | None = None,
    target_u: float | None = None,
    target_weight: float = 0.25,
    max_lateral_step_px: float | None = None,
) -> dict[str, Any] | None:
    """Derive one centered, inset corridor sample from preferred/allowed runs.

    Let R_p = [p0, p1] be the preferred traversable run and R_a = [a0, a1] be
    the allowed run at the same row. The sample is:

        center_u = midpoint(R_p)
        left_u   = center_u - half_width
        right_u  = center_u + half_width

    where:

        half_width = min((center_u - (a0 + margin_px)),
                         ((a1 - margin_px) - center_u),
                         max_width_px / 2,
                         preferred_width_px / 2)
    """
    center_u = _run_midpoint(center_run)
    center_clearance_px = None
    if isinstance(clearance_row, np.ndarray) and clearance_row.ndim == 1:
        run_start = _clamp_int(center_run[0], 0, max(int(clearance_row.shape[0]) - 1, 0))
        run_end = _clamp_int(center_run[1], 0, max(int(clearance_row.shape[0]) - 1, 0))
        if run_end >= run_start:
            candidate_u = np.arange(run_start, run_end + 1, dtype=np.float32)
            candidate_clearance = clearance_row[run_start : run_end + 1].astype(np.float32)
            if target_u is not None and max_lateral_step_px is not None and max_lateral_step_px > 0:
                # Enforce a hard curvature envelope in image space:
                #
                #     |u_i - u_{i-1}| <= delta_u_max
                #
                # The clearance objective still chooses the safest pixel inside
                # that envelope, but a single obstacle row can no longer pull
                # the corridor into a visually abrupt side jump.
                lower_u = float(target_u) - float(max_lateral_step_px)
                upper_u = float(target_u) + float(max_lateral_step_px)
                smooth_mask = (candidate_u >= lower_u) & (candidate_u <= upper_u)
                if bool(np.any(smooth_mask)):
                    candidate_u = candidate_u[smooth_mask]
                    candidate_clearance = candidate_clearance[smooth_mask]
            if candidate_clearance.size > 0:
                smooth_target = float(target_u) if target_u is not None else center_u
                smooth_penalty_px = np.abs(candidate_u - smooth_target)
                candidate_score = candidate_clearance - float(target_weight) * smooth_penalty_px
                best_index = int(np.argmax(candidate_score))
                center_u = float(candidate_u[best_index])
                center_clearance_px = float(candidate_clearance[best_index])
    max_width_ratio, preferred_width_ratio = corridor_width_ratio_limits(navigation_mode)
    # Generate with more clearance than the validator requires. Validation
    # samples a small neighborhood around each edge point, so using the exact
    # threshold can still touch an obstacle after interpolation or resizing.
    required_margin_px = max(float(image_width) * edge_margin_ratio_limit(), 1.0)
    margin_px = max(required_margin_px * 1.5, required_margin_px + 2.0)
    edge_start = float(edge_run[0])
    edge_end = float(edge_run[1])
    left_limit = edge_start + margin_px
    right_limit = edge_end - margin_px
    if right_limit <= left_limit:
        left_limit = edge_start
        right_limit = edge_end
    center_u = _clamp_float(center_u, left_limit, right_limit)
    if center_clearance_px is None and isinstance(clearance_row, np.ndarray) and clearance_row.ndim == 1:
        center_index = _clamp_int(round(float(center_u)), 0, max(int(clearance_row.shape[0]) - 1, 0))
        center_clearance_px = float(clearance_row[center_index]) if clearance_row.size else None
    available_half_width = min(center_u - left_limit, right_limit - center_u)
    if center_clearance_px is not None:
        available_half_width = min(available_half_width, max(float(center_clearance_px) - float(margin_px), 0.0))
    max_half_width = max(float(image_width) * float(max_width_ratio) * 0.5, 1.0)
    preferred_half_width = max(float(image_width) * float(preferred_width_ratio) * 0.5, 1.0)
    half_width = min(max_half_width, preferred_half_width, max(available_half_width, 0.0))
    if half_width <= 0.5:
        return None
    left_u = center_u - half_width
    right_u = center_u + half_width
    return {
        "center_uv": [round(float(center_u), 2), round(float(row_v), 2)],
        "left_edge_uv": [round(float(left_u), 2), round(float(row_v), 2)],
        "right_edge_uv": [round(float(right_u), 2), round(float(row_v), 2)],
        "support_label": support_label,
        "debug": {
            "preferred_run_u": [int(center_run[0]), int(center_run[1])],
            "support_run_u": [int(center_run[0]), int(center_run[1])],
            "support_run_role": str(support_label),
            "allowed_run_u": [int(edge_run[0]), int(edge_run[1])],
            "edge_margin_px": round(float(margin_px), 2),
            "required_edge_margin_px": round(float(required_margin_px), 2),
            "corridor_width_px": round(float(right_u - left_u), 2),
            "center_clearance_px": round(float(center_clearance_px), 2) if center_clearance_px is not None else None,
            "target_u": round(float(target_u), 2) if target_u is not None else None,
            "target_weight": round(float(target_weight), 4),
            "max_lateral_step_px": round(float(max_lateral_step_px), 2) if max_lateral_step_px is not None else None,
            "max_width_ratio": round(float(max_width_ratio), 4),
            "preferred_width_ratio": round(float(preferred_width_ratio), 4),
        },
    }


def _corridor_polygon_from_samples(path_samples: list[dict[str, Any]]) -> list[list[float]] | None:
    left_chain = [
        list(sample["left_edge_uv"])
        for sample in path_samples
        if isinstance(sample.get("left_edge_uv"), list)
    ]
    right_chain = [
        list(sample["right_edge_uv"])
        for sample in path_samples
        if isinstance(sample.get("right_edge_uv"), list)
    ]
    if len(left_chain) < 2 or len(right_chain) < 2:
        return None
    polygon = left_chain + list(reversed(right_chain))
    if polygon and polygon[0] != polygon[-1]:
        polygon.append(list(polygon[0]))
    return polygon


def _direction_from_path(path_uv: list[list[float]]) -> list[float] | None:
    if len(path_uv) < 2:
        return None
    start = path_uv[-2]
    end = path_uv[-1]
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return None
    return [round(dx / norm, 4), round(dy / norm, 4)]


def build_safe_corridor_prior(
    segmentation_context: dict[str, Any] | None,
    *,
    sample_count: int = 5,
) -> dict[str, Any] | None:
    """Build a deterministic, obstacle-aware corridor prior from the mask.

    This is a teacher/guardrail, not the final visual prompt. It gives the LVLM
    concrete coordinates for the desired geometric behavior: choose a smooth
    centerline that maximizes clearance inside the traversable support, then
    inset left/right boundaries from the allowed run.

    Sampling model:
        rows = round((H - 1) * r), r in [0.92, 0.85, 0.78, ...]

    Per-row objective:
        score(u_i) = clearance(u_i, v_i)
                   - lambda_smooth * |u_i - u_{i-1}| / image_width

        subject to |u_i - u_{i-1}| <= 0.20 * image_width

    This is a lightweight dynamic-programming approximation of a medial-axis
    path: clearance pulls samples toward the middle of obstacle-free support,
    while the smoothness term prevents unrealistic zig-zags.
    """
    if not isinstance(segmentation_context, dict):
        return None
    mask = segmentation_context.get("mask")
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return None
    traversable_ids = set(segmentation_context.get("traversable_ids") or [])
    conditional_ids = set(segmentation_context.get("conditional_traversable_ids") or [])
    allowed_ids = set(segmentation_context.get("allowed_traversable_ids") or (traversable_ids | conditional_ids))
    if not allowed_ids:
        return None
    navigation_mode = str(segmentation_context.get("navigation_mode") or "foot").strip().lower()
    height, width = int(mask.shape[0]), int(mask.shape[1])
    clearance_map = _allowed_clearance_map(mask, allowed_ids)
    row_ratios = [0.92, 0.85, 0.78, 0.70, 0.62, 0.54]
    row_ratios = row_ratios[: max(2, min(int(sample_count), len(row_ratios)))]
    preferred_u = float(width) * 0.5
    max_lateral_step_px = max(float(width) * 0.20, 1.0)
    samples: list[dict[str, Any]] = []
    rows_debug: list[dict[str, Any]] = []
    for row_index, ratio in enumerate(row_ratios):
        row_v = _clamp_int(round((height - 1) * float(ratio)), 0, height - 1)
        preferred_runs = _class_runs_at_row(mask, traversable_ids, row_v) if traversable_ids else []
        allowed_runs = _class_runs_at_row(mask, allowed_ids, row_v)
        center_run = _run_containing_or_nearest(preferred_runs, preferred_u)
        support_label = "preferred_traversable"
        if center_run is None:
            center_run = _run_containing_or_nearest(allowed_runs, preferred_u)
            support_label = "conditional_connector"
        if center_run is None:
            rows_debug.append({"row_v": int(row_v), "status": "no_traversable_run"})
            continue
        edge_run = _run_containing_or_nearest(allowed_runs, _run_midpoint(center_run)) or center_run
        clearance_row = clearance_map[row_v] if isinstance(clearance_map, np.ndarray) else None
        sample = _corridor_sample_from_runs(
            center_run=center_run,
            edge_run=edge_run,
            row_v=float(row_v),
            image_width=width,
            navigation_mode=navigation_mode,
            support_label=support_label,
            clearance_row=clearance_row,
            target_u=preferred_u,
            target_weight=1.0 if row_index == 0 else 0.40,
            max_lateral_step_px=None if row_index == 0 else max_lateral_step_px,
        )
        if sample is None:
            rows_debug.append({"row_v": int(row_v), "status": "too_narrow", "center_run_u": list(center_run)})
            continue
        preferred_u = float(sample["center_uv"][0])
        rows_debug.append({"row_v": int(row_v), "status": "ok", **sample.get("debug", {})})
        samples.append(sample)
    if len(samples) < 2:
        return None
    path_uv = [list(sample["center_uv"]) for sample in samples]
    coverage_polygon_uv = _corridor_polygon_from_samples(samples)
    widths = [
        math.hypot(
            float(sample["right_edge_uv"][0]) - float(sample["left_edge_uv"][0]),
            float(sample["right_edge_uv"][1]) - float(sample["left_edge_uv"][1]),
        )
        for sample in samples
    ]
    return {
        "schema_version": 1,
        "coordinate_space": "image_uv",
        "source": "segmentation_safe_corridor_prior_v1",
        "description": "Deterministic mask-derived teacher prior; use as a geometric guardrail, not as a forced final prompt.",
        "path_samples": samples,
        "path_uv": path_uv,
        "coverage_polygon_uv": coverage_polygon_uv,
        "direction_vec": _direction_from_path(path_uv),
        "span_px": round(float(sum(widths)) / float(len(widths)), 2) if widths else None,
        "termination_semantics": "continue_visible_corridor",
        "support_confidence": 0.82,
        "debug": {
            "row_samples": rows_debug,
            "sample_count": len(samples),
            "navigation_mode": navigation_mode,
            "image_size": {"width": width, "height": height},
            "max_lateral_step_ratio": 0.20,
            "clearance_source": "distance_transform_edt" if isinstance(clearance_map, np.ndarray) else "row_run_midpoint_fallback",
        },
    }


def repair_rendered_prompt_with_safe_corridor(
    rendered_prompt: dict[str, Any] | None,
    segmentation_context: dict[str, Any] | None,
    *,
    validation: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Deterministically recenter and inset an LVLM path_corridor using the mask.

    For local width/offset errors, the repair preserves the LVLM's sampled rows
    and replaces each sample with same-row safe-corridor geometry. For obstacle
    failures, it regenerates the full path from the obstacle-aware prior:

        fatal_replan_reasons =
            center/boundary forbidden hits,
            polygon forbidden hits,
            missing preferred run,
            bottom-start failure.

    The caller must re-run segmentation validation before accepting the repair.
    """
    if not isinstance(rendered_prompt, dict) or not isinstance(segmentation_context, dict):
        return rendered_prompt, None
    if str(rendered_prompt.get("type") or "").strip().lower() != "path_corridor":
        return rendered_prompt, {"applied": False, "reason": "not_path_corridor"}
    raw_samples = rendered_prompt.get("path_samples")
    if not isinstance(raw_samples, list) or len(raw_samples) < 2:
        return rendered_prompt, {"applied": False, "reason": "missing_path_samples"}
    mask = segmentation_context.get("mask")
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return rendered_prompt, {"applied": False, "reason": "missing_mask"}
    traversable_ids = set(segmentation_context.get("traversable_ids") or [])
    conditional_ids = set(segmentation_context.get("conditional_traversable_ids") or [])
    allowed_ids = set(segmentation_context.get("allowed_traversable_ids") or (traversable_ids | conditional_ids))
    if not allowed_ids:
        return rendered_prompt, {"applied": False, "reason": "missing_allowed_ids"}
    navigation_mode = str(segmentation_context.get("navigation_mode") or "foot").strip().lower()
    height, width = int(mask.shape[0]), int(mask.shape[1])
    reason_text = str((validation or {}).get("reason") or "")
    should_replan = any(
        token in reason_text
        for token in [
            "center_path_hits_forbidden_surface",
            "boundary_hits_forbidden_surface",
            "polygon_hits_forbidden_ratio",
            "centerline_preferred_run_missing",
            "start_point_not_in_lower_middle_band",
        ]
    )
    if should_replan:
        prior = build_safe_corridor_prior(segmentation_context, sample_count=max(len(raw_samples), 3))
        if isinstance(prior, dict) and isinstance(prior.get("path_samples"), list):
            repaired = dict(rendered_prompt)
            repaired["path_samples"] = prior["path_samples"]
            repaired["path_uv"] = prior.get("path_uv")
            repaired["coverage_polygon_uv"] = prior.get("coverage_polygon_uv")
            if isinstance(prior.get("path_uv"), list) and prior["path_uv"]:
                repaired["center_uv"] = list(prior["path_uv"][-1])
            repaired["direction_vec"] = prior.get("direction_vec")
            repaired["span_px"] = prior.get("span_px")
            repaired["termination_semantics"] = prior.get("termination_semantics") or "continue_visible_corridor"
            repaired["support_confidence"] = max(float(repaired.get("support_confidence") or 0.0), 0.80)
            return repaired, {
                "applied": True,
                "source": "segmentation_obstacle_aware_replan_v1",
                "reason": "validation_failure_requires_full_path_replan",
                "validation_reason": reason_text,
                "safe_corridor_prior_debug": prior.get("debug"),
            }

    clearance_map = _allowed_clearance_map(mask, allowed_ids)
    max_lateral_step_px = max(float(width) * 0.20, 1.0)
    repaired_samples: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, dict):
            continue
        center = raw_sample.get("center_uv")
        if not (isinstance(center, (list, tuple)) and len(center) >= 2):
            continue
        row_v = _clamp_int(round(float(center[1])), 0, height - 1)
        original_u = float(center[0])
        preferred_runs = _class_runs_at_row(mask, traversable_ids, row_v) if traversable_ids else []
        allowed_runs = _class_runs_at_row(mask, allowed_ids, row_v)
        center_run = _run_containing_or_nearest(preferred_runs, original_u)
        support_label = str(raw_sample.get("support_label") or "preferred_traversable")
        if center_run is None:
            center_run = _run_containing_or_nearest(allowed_runs, original_u)
            support_label = "conditional_connector"
        if center_run is None:
            repairs.append({"sample_index": index, "status": "no_run", "row_v": int(row_v)})
            continue
        edge_run = _run_containing_or_nearest(allowed_runs, _run_midpoint(center_run)) or center_run
        clearance_row = clearance_map[row_v] if isinstance(clearance_map, np.ndarray) else None
        target_u = original_u
        if index == 0 and not (float(width) * 0.40 <= original_u <= float(width) * 0.60):
            target_u = float(width) * 0.5
        repaired_sample = _corridor_sample_from_runs(
            center_run=center_run,
            edge_run=edge_run,
            row_v=float(row_v),
            image_width=width,
            navigation_mode=navigation_mode,
            support_label=support_label,
            clearance_row=clearance_row,
            target_u=target_u,
            target_weight=1.0 if index == 0 else 0.40,
            max_lateral_step_px=None if index == 0 else max_lateral_step_px,
        )
        if repaired_sample is None:
            repairs.append({"sample_index": index, "status": "too_narrow", "row_v": int(row_v)})
            continue
        repairs.append(
            {
                "sample_index": index,
                "status": "repaired",
                "original_center_uv": list(center[:2]),
                "repaired_center_uv": repaired_sample["center_uv"],
                **repaired_sample.get("debug", {}),
            }
        )
        repaired_samples.append(repaired_sample)

    if len(repaired_samples) < 2:
        return rendered_prompt, {"applied": False, "reason": "insufficient_repaired_samples", "sample_repairs": repairs}

    repaired = dict(rendered_prompt)
    path_uv = [list(sample["center_uv"]) for sample in repaired_samples]
    coverage_polygon_uv = _corridor_polygon_from_samples(repaired_samples)
    widths = [
        math.hypot(
            float(sample["right_edge_uv"][0]) - float(sample["left_edge_uv"][0]),
            float(sample["right_edge_uv"][1]) - float(sample["left_edge_uv"][1]),
        )
        for sample in repaired_samples
    ]
    repaired["path_samples"] = repaired_samples
    repaired["path_uv"] = path_uv
    repaired["coverage_polygon_uv"] = coverage_polygon_uv
    repaired["center_uv"] = list(path_uv[-1])
    repaired["direction_vec"] = _direction_from_path(path_uv)
    repaired["span_px"] = round(float(sum(widths)) / float(len(widths)), 2) if widths else repaired.get("span_px")
    repaired.setdefault("termination_semantics", "continue_visible_corridor")
    repaired["support_confidence"] = max(float(repaired.get("support_confidence") or 0.0), 0.78)
    return repaired, {
        "applied": True,
        "source": "segmentation_safe_corridor_repair_v1",
        "sample_repairs": repairs,
        "clearance_source": "distance_transform_edt" if isinstance(clearance_map, np.ndarray) else "row_run_midpoint_fallback",
    }


def _present_classes(mask: np.ndarray, id_to_name: dict[int, str]) -> list[dict[str, Any]]:
    unique_ids, counts = np.unique(mask, return_counts=True)
    total = float(mask.size) if mask.size else 1.0
    present: list[dict[str, Any]] = []
    for class_id, count in sorted(zip(unique_ids.tolist(), counts.tolist()), key=lambda item: item[1], reverse=True):
        present.append(
            {
                "class_id": int(class_id),
                "class_name": id_to_name.get(int(class_id), f"unknown_{class_id}"),
                "pixel_count": int(count),
                "pixel_ratio": round(float(count) / total, 6),
            }
        )
    return present


def validate_fpv_segmentation_alignment(
    fpv_segmentation: dict[str, Any] | None,
    *,
    image_width: int,
    image_height: int,
    sequence_name: str | None = None,
    camera_view: str | None = None,
    frame_name: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "valid": False,
        "reasons": [],
        "checks": {},
    }
    if not isinstance(fpv_segmentation, dict):
        report["reasons"].append("fpv_segmentation_missing")
        return report

    mask_path = _coerce_path(
        fpv_segmentation.get("segmentation_npy_path")
        or fpv_segmentation.get("mask_path")
    )
    mapping_path = _coerce_path(
        fpv_segmentation.get("class_mapping_path")
        or ((fpv_segmentation.get("summary") or {}).get("class_mapping_path") if isinstance(fpv_segmentation.get("summary"), dict) else None)
    )
    contract = fpv_segmentation.get("alignment_contract") if isinstance(fpv_segmentation.get("alignment_contract"), dict) else {}

    report["checks"]["mask_exists"] = bool(mask_path and mask_path.exists())
    report["checks"]["class_mapping_exists"] = bool(mapping_path and mapping_path.exists())
    if not report["checks"]["mask_exists"]:
        report["reasons"].append("mask_missing")
    if not report["checks"]["class_mapping_exists"]:
        report["reasons"].append("class_mapping_missing")
    if report["reasons"]:
        return report

    mask = np.load(mask_path, mmap_mode="r")
    if mask.ndim != 2:
        report["checks"]["mask_ndim_hxw"] = False
        report["reasons"].append(f"mask_not_hxw:{tuple(mask.shape)}")
        return report

    mask_h = int(mask.shape[0])
    mask_w = int(mask.shape[1])
    report["checks"]["mask_ndim_hxw"] = True
    report["checks"]["mask_matches_image_size"] = bool(mask_h == int(image_height) and mask_w == int(image_width))
    report["mask_shape_hw"] = [mask_h, mask_w]
    report["image_shape_hw"] = [int(image_height), int(image_width)]
    if not report["checks"]["mask_matches_image_size"]:
        report["reasons"].append("mask_image_size_mismatch")

    identity = contract.get("frame_identity") if isinstance(contract, dict) else {}
    if sequence_name is not None:
        report["checks"]["sequence_name_matches_contract"] = bool(not identity or str(identity.get("sequence_name")) == str(sequence_name))
        if not report["checks"]["sequence_name_matches_contract"]:
            report["reasons"].append("sequence_name_mismatch")
    if camera_view is not None:
        report["checks"]["camera_view_matches_contract"] = bool(not identity or str(identity.get("camera_view")) == str(camera_view))
        if not report["checks"]["camera_view_matches_contract"]:
            report["reasons"].append("camera_view_mismatch")
    if frame_name is not None:
        report["checks"]["frame_name_matches_contract"] = bool(not identity or str(identity.get("frame_name")) == str(frame_name))
        if not report["checks"]["frame_name_matches_contract"]:
            report["reasons"].append("frame_name_mismatch")

    report["valid"] = not report["reasons"]
    return report


def build_segmentation_teacher_bundle(segmentation_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(segmentation_context, dict):
        return None
    summary = segmentation_context.get("summary") if isinstance(segmentation_context.get("summary"), dict) else None
    alignment = segmentation_context.get("alignment_report") if isinstance(segmentation_context.get("alignment_report"), dict) else None
    if summary is None:
        return None
    return {
        "summary": summary,
        "alignment_report": alignment,
    }


def load_fpv_segmentation_context(
    fpv_segmentation: dict[str, Any] | None,
    *,
    navigation_mode: str,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """Load an aligned HxW FPV semantic mask and derive mode-specific classes.

    The returned context is the canonical source-image visual prior: downstream
    SoM summaries, LVLM prompt constraints, and post-generation V checks all use
    these class-id sets instead of duplicating navigation-mode policy.
    """
    if not isinstance(fpv_segmentation, dict):
        return None

    mask_path = _coerce_path(
        fpv_segmentation.get("segmentation_npy_path")
        or fpv_segmentation.get("mask_path")
    )
    if mask_path is None or not mask_path.exists():
        return None

    summary = fpv_segmentation.get("summary") if isinstance(fpv_segmentation.get("summary"), dict) else {}
    mapping_path = _coerce_path(
        fpv_segmentation.get("class_mapping_path")
        or summary.get("class_mapping_path")
        or (mask_path.parent / "class_mapping.json")
    )
    if mapping_path is None or not mapping_path.exists():
        return None

    alignment_report = validate_fpv_segmentation_alignment(
        fpv_segmentation,
        image_width=image_width,
        image_height=image_height,
    )
    if not alignment_report.get("valid"):
        raise RuntimeError(
            "fpv_segmentation alignment validation failed: "
            + "; ".join(str(item) for item in alignment_report.get("reasons") or ["unknown_reason"])
        )

    mask = np.load(mask_path)
    if mask.ndim != 2:
        raise RuntimeError(f"fpv_segmentation mask must be HxW, got shape={mask.shape}")
    if mask.shape[1] != int(image_width) or mask.shape[0] != int(image_height):
        raise RuntimeError(
            f"fpv_segmentation mask/image size mismatch: mask={mask.shape[::-1]} image={(image_width, image_height)}"
        )

    id_to_name, name_to_id = _load_class_mapping(mapping_path)
    rules = _mode_rule_names(navigation_mode)
    traversable_ids = {name_to_id[name] for name in rules["traversable"] if name in name_to_id}
    conditional_traversable_ids = {name_to_id[name] for name in rules["conditional_traversable"] if name in name_to_id}
    allowed_traversable_ids = traversable_ids | conditional_traversable_ids
    boundary_ids = {name_to_id[name] for name in rules["boundary"] if name in name_to_id}
    forbidden_ids = {name_to_id[name] for name in rules["forbidden"] if name in name_to_id}
    mode = str(navigation_mode or "foot").strip().lower()
    disallowed_surface_names: set[str] = set()
    disallowed_surface_ids = {name_to_id[name] for name in disallowed_surface_names if name in name_to_id}
    curb_ids = {name_to_id[name] for name in {"curb"} if name in name_to_id}

    traversable_mask = np.isin(mask, list(traversable_ids)) if traversable_ids else np.zeros_like(mask, dtype=bool)
    traversable_pixels = int(traversable_mask.sum())
    traversable_bbox = None
    if traversable_pixels > 0:
        ys, xs = np.where(traversable_mask)
        traversable_bbox = {
            "u_min": int(xs.min()),
            "u_max": int(xs.max()),
            "v_min": int(ys.min()),
            "v_max": int(ys.max()),
        }

    compact_summary = {
        "mask_path": str(mask_path),
        "class_mapping_path": str(mapping_path),
        "alignment_report": alignment_report,
        "shape_hw": [int(mask.shape[0]), int(mask.shape[1])],
        "present_classes": _present_classes(mask, id_to_name)[:12],
        "traversable_class_ids": sorted(int(class_id) for class_id in traversable_ids),
        "traversable_class_names": sorted(rules["traversable"]),
        "conditional_traversable_class_ids": sorted(int(class_id) for class_id in conditional_traversable_ids),
        "conditional_traversable_class_names": sorted(rules["conditional_traversable"]),
        "allowed_traversable_class_ids": sorted(int(class_id) for class_id in allowed_traversable_ids),
        "allowed_traversable_class_names": sorted(rules["allowed_traversable"]),
        "boundary_class_ids": sorted(int(class_id) for class_id in boundary_ids),
        "boundary_class_names": sorted(rules["boundary"]),
        "forbidden_class_ids": sorted(int(class_id) for class_id in forbidden_ids),
        "forbidden_class_names": sorted(rules["forbidden"]),
        "disallowed_surface_class_ids": sorted(int(class_id) for class_id in disallowed_surface_ids),
        "disallowed_surface_class_names": sorted(disallowed_surface_names),
        "traversable_pixel_ratio": round(float(traversable_pixels) / float(mask.size), 6) if mask.size else 0.0,
        "traversable_bbox_uv": traversable_bbox,
        "bottom_openings": [
            opening
            for opening in [
                _largest_bottom_run(mask, traversable_ids, 0.75),
                _largest_bottom_run(mask, traversable_ids, 0.85),
                _largest_bottom_run(mask, traversable_ids, 0.92),
            ]
            if opening is not None
        ],
        "allowed_bottom_openings": [
            opening
            for opening in [
                _largest_bottom_run(mask, allowed_traversable_ids, 0.75),
                _largest_bottom_run(mask, allowed_traversable_ids, 0.85),
                _largest_bottom_run(mask, allowed_traversable_ids, 0.92),
            ]
            if opening is not None
        ],
    }
    return {
        "mask": mask,
        "id_to_name": id_to_name,
        "name_to_id": name_to_id,
        "traversable_ids": traversable_ids,
        "conditional_traversable_ids": conditional_traversable_ids,
        "allowed_traversable_ids": allowed_traversable_ids,
        "boundary_ids": boundary_ids,
        "forbidden_ids": forbidden_ids,
        "disallowed_surface_ids": disallowed_surface_ids,
        "curb_ids": curb_ids,
        "navigation_mode": mode,
        "alignment_report": alignment_report,
        "summary": compact_summary,
    }


def segmentation_summary_for_llm(segmentation_context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(segmentation_context, dict):
        return None
    summary = segmentation_context.get("summary")
    if not isinstance(summary, dict):
        return None
    return summary


def validate_rendered_prompt_against_segmentation(
    rendered_prompt: dict[str, Any] | None,
    segmentation_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate a rendered prompt against the FPV semantic mask.

    Centerline samples must stay on allowed traversable support, left/right
    boundary samples must avoid forbidden surfaces, and corridor polygons must
    be compact enough to represent a navigation affordance rather than the full
    visible free-space mask.

    Acceptance ratios:
        center_valid_ratio   = valid_center_samples / center_samples
        boundary_valid_ratio = valid_boundary_samples / boundary_samples
        polygon_allowed_ratio = allowed_polygon_pixels / polygon_pixels

    The final validity predicate is intentionally conservative: start point,
    centerline support, boundary support, forbidden hits, corridor width,
    centerline offset, edge margin, and polygon support must all pass together.
    """
    if not isinstance(rendered_prompt, dict) or not isinstance(segmentation_context, dict):
        return None

    mask = segmentation_context["mask"]
    id_to_name = segmentation_context["id_to_name"]
    traversable_ids: set[int] = set(segmentation_context["traversable_ids"])
    conditional_traversable_ids: set[int] = set(segmentation_context.get("conditional_traversable_ids") or [])
    allowed_traversable_ids: set[int] = set(segmentation_context.get("allowed_traversable_ids") or (traversable_ids | conditional_traversable_ids))
    boundary_ids: set[int] = set(segmentation_context["boundary_ids"])
    forbidden_ids: set[int] = set(segmentation_context["forbidden_ids"])
    disallowed_surface_ids: set[int] = set(segmentation_context.get("disallowed_surface_ids") or [])
    curb_ids: set[int] = set(segmentation_context.get("curb_ids") or _ids_for_names(id_to_name, {"curb"}))
    navigation_mode = str(segmentation_context.get("navigation_mode") or "foot").strip().lower()

    path_samples = rendered_prompt.get("path_samples") if isinstance(rendered_prompt.get("path_samples"), list) else []
    path_uv = _point_list(rendered_prompt.get("path_uv"))
    coverage_polygon_uv = _point_list(rendered_prompt.get("coverage_polygon_uv"))

    center_points = _point_list([sample.get("center_uv") for sample in path_samples if isinstance(sample, dict)])
    left_points = _point_list([sample.get("left_edge_uv") for sample in path_samples if isinstance(sample, dict)])
    right_points = _point_list([sample.get("right_edge_uv") for sample in path_samples if isinstance(sample, dict)])
    if not center_points and path_uv:
        center_points = [point for point in path_uv if isinstance(point, list) and len(point) == 2]
    center_track = _interpolate_points(center_points)
    height, width = int(mask.shape[0]), int(mask.shape[1])
    start_point_uv = center_points[0] if center_points else None
    start_min_v_ratio = 0.82
    start_u_min_ratio = 0.40
    start_u_max_ratio = 0.60
    start_relaxed_u_min_ratio = 0.35
    start_relaxed_u_max_ratio = 0.75
    start_min_v = float(height) * start_min_v_ratio
    start_u_min = float(width) * start_u_min_ratio
    start_u_max = float(width) * start_u_max_ratio
    start_relaxed_u_min = float(width) * start_relaxed_u_min_ratio
    start_relaxed_u_max = float(width) * start_relaxed_u_max_ratio
    start_strict_valid = False
    start_relaxed_valid = False
    start_valid = False
    start_relaxed_reason: str | None = None
    start_neighbor_ids: set[int] = set()
    if isinstance(start_point_uv, list) and len(start_point_uv) == 2:
        try:
            start_u = float(start_point_uv[0])
            start_v = float(start_point_uv[1])
            start_strict_valid = (
                start_v >= start_min_v
                and start_u >= start_u_min
                and start_u <= start_u_max
            )
            start_neighbor_ids = _class_id_near(mask, [start_u, start_v], radius_px=5)
        except Exception:
            start_strict_valid = False

    center_total = 0
    center_valid = 0
    center_preferred = 0
    center_conditional = 0
    center_forbidden_hits: list[dict[str, Any]] = []
    center_disallowed_hits: list[dict[str, Any]] = []
    for point in center_track:
        neighbor_ids = _class_id_near(mask, point, radius_px=5)
        center_total += 1
        if neighbor_ids & allowed_traversable_ids:
            center_valid += 1
        if neighbor_ids & traversable_ids:
            center_preferred += 1
        elif neighbor_ids & conditional_traversable_ids:
            center_conditional += 1
        if neighbor_ids & forbidden_ids:
            forbidden_class_ids = sorted(int(class_id) for class_id in (neighbor_ids & forbidden_ids))
            center_forbidden_hits.append(
                {
                    "point_uv": [round(float(point[0]), 2), round(float(point[1]), 2)],
                    "class_ids": forbidden_class_ids,
                    "class_names": [id_to_name.get(class_id, f"unknown_{class_id}") for class_id in forbidden_class_ids],
                }
            )
        if neighbor_ids & disallowed_surface_ids:
            disallowed_class_ids = sorted(int(class_id) for class_id in (neighbor_ids & disallowed_surface_ids))
            center_disallowed_hits.append(
                {
                    "point_uv": [round(float(point[0]), 2), round(float(point[1]), 2)],
                    "class_ids": disallowed_class_ids,
                    "class_names": [id_to_name.get(class_id, f"unknown_{class_id}") for class_id in disallowed_class_ids],
                }
            )

    boundary_total = 0
    boundary_valid = 0
    boundary_preferred = 0
    boundary_conditional = 0
    boundary_forbidden_hits: list[dict[str, Any]] = []
    boundary_disallowed_hits: list[dict[str, Any]] = []
    boundary_track = _interpolate_points(left_points) + _interpolate_points(right_points)
    for point in boundary_track:
        neighbor_ids = _class_id_near(mask, point, radius_px=4)
        boundary_total += 1
        if neighbor_ids & (boundary_ids | allowed_traversable_ids):
            boundary_valid += 1
        if neighbor_ids & traversable_ids:
            boundary_preferred += 1
        elif neighbor_ids & conditional_traversable_ids:
            boundary_conditional += 1
        if neighbor_ids & forbidden_ids:
            forbidden_class_ids = sorted(int(class_id) for class_id in (neighbor_ids & forbidden_ids))
            boundary_forbidden_hits.append(
                {
                    "point_uv": [round(float(point[0]), 2), round(float(point[1]), 2)],
                    "class_ids": forbidden_class_ids,
                    "class_names": [id_to_name.get(class_id, f"unknown_{class_id}") for class_id in forbidden_class_ids],
                }
            )
        if neighbor_ids & disallowed_surface_ids:
            disallowed_class_ids = sorted(int(class_id) for class_id in (neighbor_ids & disallowed_surface_ids))
            boundary_disallowed_hits.append(
                {
                    "point_uv": [round(float(point[0]), 2), round(float(point[1]), 2)],
                    "class_ids": disallowed_class_ids,
                    "class_names": [id_to_name.get(class_id, f"unknown_{class_id}") for class_id in disallowed_class_ids],
                }
            )

    corridor_width_samples: list[float] = []
    paired_edge_points = min(len(left_points), len(right_points))
    for index in range(paired_edge_points):
        left = left_points[index]
        right = right_points[index]
        try:
            width_px = math.hypot(float(right[0]) - float(left[0]), float(right[1]) - float(left[1]))
        except Exception:
            continue
        corridor_width_samples.append(float(width_px))
    corridor_width_ratios = [width_px / max(float(width), 1.0) for width_px in corridor_width_samples]
    max_corridor_width_ratio_observed = max(corridor_width_ratios) if corridor_width_ratios else 0.0
    mean_corridor_width_ratio = (
        float(sum(corridor_width_ratios)) / float(len(corridor_width_ratios)) if corridor_width_ratios else 0.0
    )
    # The prompt is a navigational affordance, not a full free-space segmentation mask.
    # For foot mode, keep the rendered corridor compact enough that its boundaries still
    # communicate obstacle clearance instead of spanning the whole visible pavement.
    max_corridor_width_ratio, preferred_corridor_width_ratio = corridor_width_ratio_limits(navigation_mode)
    corridor_width_valid = (
        not corridor_width_ratios
        or (
            max_corridor_width_ratio_observed <= max_corridor_width_ratio
            and mean_corridor_width_ratio <= max_corridor_width_ratio
        )
    )

    centerline_offset_limit = centerline_offset_ratio_limit()
    edge_margin_ratio = edge_margin_ratio_limit()
    centerline_offsets: list[float] = []
    centerline_clearances: list[float] = []
    edge_margin_ratios: list[float] = []
    low_margin_samples: list[dict[str, Any]] = []
    off_center_samples: list[dict[str, Any]] = []
    preferred_run_missing_samples: list[dict[str, Any]] = []
    conditional_connector_samples: list[dict[str, Any]] = []
    paired_samples = min(len(center_points), len(left_points), len(right_points))
    min_edge_margin_px = float(width) * float(edge_margin_ratio)
    clearance_map = _allowed_clearance_map(mask, allowed_traversable_ids)
    sample_support_labels = [
        str(sample.get("support_label") or "").strip().lower()
        for sample in path_samples[:paired_samples]
        if isinstance(sample, dict)
    ]
    preferred_runs_by_sample: list[tuple[int, int] | None] = []
    allowed_runs_by_sample: list[tuple[int, int] | None] = []
    for index in range(paired_samples):
        center = center_points[index]
        preferred_runs = _class_runs_at_row(mask, traversable_ids, center[1])
        allowed_runs = _class_runs_at_row(mask, allowed_traversable_ids, center[1])
        preferred_runs_by_sample.append(_run_containing_or_nearest(preferred_runs, center[0]))
        allowed_runs_by_sample.append(_run_containing_or_nearest(allowed_runs, center[0]))
    first_preferred_sample_index = next(
        (index for index, run in enumerate(preferred_runs_by_sample) if run is not None),
        None,
    )
    for index in range(paired_samples):
        center = center_points[index]
        left = left_points[index]
        right = right_points[index]
        center_run = preferred_runs_by_sample[index]
        edge_run = allowed_runs_by_sample[index]
        support_label = sample_support_labels[index] if index < len(sample_support_labels) else ""
        implicit_connector_prefix = bool(
            center_run is None
            and edge_run is not None
            and first_preferred_sample_index is not None
            and index < first_preferred_sample_index
        )
        connector_prefix_allowed = bool(
            implicit_connector_prefix
            and (
                _is_conditional_connector_label(support_label)
                or not support_label
                or support_label in {"sidewalk", "pedestrian_path", "crosswalk", "plaza"}
            )
        )
        if center_run is None:
            if connector_prefix_allowed:
                conditional_connector_samples.append(
                    {
                        "sample_index": int(index),
                        "center_uv": [round(float(center[0]), 2), round(float(center[1]), 2)],
                        "support_label": support_label,
                        "inferred_support_role": "conditional_connector_prefix",
                        "allowed_run_u": [int(edge_run[0]), int(edge_run[1])] if edge_run is not None else None,
                        "connects_to_preferred_sample_index": int(first_preferred_sample_index),
                    }
                )
                center_run = edge_run
            else:
                preferred_run_missing_samples.append(
                    {
                        "sample_index": int(index),
                        "center_uv": [round(float(center[0]), 2), round(float(center[1]), 2)],
                        "support_label": support_label,
                        "allowed_run_u": [int(edge_run[0]), int(edge_run[1])] if edge_run is not None else None,
                    }
                )
                continue
        run_start, run_end = center_run
        run_width = max(float(run_end - run_start + 1), 1.0)
        run_mid = (float(run_start) + float(run_end)) / 2.0
        half_width = max(run_width / 2.0, 1.0)
        center_offset = abs(float(center[0]) - run_mid) / half_width
        center_clearance = 0.0
        if isinstance(clearance_map, np.ndarray):
            center_u_px, center_v_px = _point_to_pixel(mask, center)
            center_clearance = float(clearance_map[center_v_px, center_u_px])
        centerline_clearances.append(float(center_clearance))
        centerline_offsets.append(float(center_offset))
        edge_start, edge_end = edge_run if edge_run is not None else center_run
        left_margin = float(left[0]) - float(edge_start)
        right_margin = float(edge_end) - float(right[0])
        margin_ratio = min(left_margin, right_margin) / max(float(width), 1.0)
        edge_margin_ratios.append(float(margin_ratio))
        clearance_centered = center_clearance >= max(float(min_edge_margin_px) * 2.0, 6.0)
        if connector_prefix_allowed:
            # Conditional connector samples are judged primarily by allowed
            # support, forbidden avoidance, and edge clearance. They often sit
            # in road/curb/gutter support before joining the preferred sidewalk,
            # so a preferred-run midpoint offset is not meaningful for them.
            clearance_centered = True
        if center_offset > centerline_offset_limit and not clearance_centered:
            off_center_samples.append(
                {
                    "sample_index": int(index),
                    "center_uv": [round(float(center[0]), 2), round(float(center[1]), 2)],
                    "preferred_run_u": [int(run_start), int(run_end)],
                    "centerline_offset_ratio": round(float(center_offset), 4),
                    "center_clearance_px": round(float(center_clearance), 2),
                }
            )
        if left_margin < min_edge_margin_px or right_margin < min_edge_margin_px:
            low_margin_samples.append(
                {
                    "sample_index": int(index),
                    "center_uv": [round(float(center[0]), 2), round(float(center[1]), 2)],
                    "left_edge_uv": [round(float(left[0]), 2), round(float(left[1]), 2)],
                    "right_edge_uv": [round(float(right[0]), 2), round(float(right[1]), 2)],
                    "allowed_run_u": [int(edge_start), int(edge_end)],
                    "left_margin_px": round(float(left_margin), 2),
                    "right_margin_px": round(float(right_margin), 2),
                    "min_required_margin_px": round(float(min_edge_margin_px), 2),
                }
            )
    max_centerline_offset_ratio = max(centerline_offsets) if centerline_offsets else 0.0
    mean_centerline_offset_ratio = (
        float(sum(centerline_offsets)) / float(len(centerline_offsets)) if centerline_offsets else 0.0
    )
    min_edge_margin_ratio_observed = min(edge_margin_ratios) if edge_margin_ratios else 1.0
    centerline_valid = bool(
        centerline_offsets
        and not off_center_samples
        and not preferred_run_missing_samples
    )
    edge_margin_valid = bool(edge_margin_ratios and min_edge_margin_ratio_observed >= edge_margin_ratio)

    center_valid_ratio = float(center_valid) / float(center_total) if center_total else 0.0
    center_preferred_ratio = float(center_preferred) / float(center_total) if center_total else 0.0
    center_conditional_ratio = float(center_conditional) / float(center_total) if center_total else 0.0
    boundary_valid_ratio = float(boundary_valid) / float(boundary_total) if boundary_total else 1.0
    boundary_preferred_ratio = float(boundary_preferred) / float(boundary_total) if boundary_total else 0.0
    boundary_conditional_ratio = float(boundary_conditional) / float(boundary_total) if boundary_total else 0.0
    polygon_total = 0
    polygon_allowed_ratio = 0.0
    polygon_traversable_ratio = 0.0
    polygon_conditional_ratio = 0.0
    polygon_boundary_ratio = 0.0
    polygon_forbidden_ratio = 0.0
    polygon_disallowed_ratio = 0.0
    polygon_curb_ratio = 0.0
    polygon_top_classes: list[dict[str, Any]] = []
    polygon_mask = _polygon_mask(mask.shape[:2], coverage_polygon_uv)
    if polygon_mask is not None:
        polygon_total = int(polygon_mask.sum())
        polygon_allowed_ratio = _class_ratio(mask, polygon_mask, allowed_traversable_ids)
        polygon_traversable_ratio = _class_ratio(mask, polygon_mask, traversable_ids)
        polygon_conditional_ratio = _class_ratio(mask, polygon_mask, conditional_traversable_ids)
        polygon_boundary_ratio = _class_ratio(mask, polygon_mask, boundary_ids)
        polygon_forbidden_ratio = _class_ratio(mask, polygon_mask, forbidden_ids)
        polygon_disallowed_ratio = _class_ratio(mask, polygon_mask, disallowed_surface_ids)
        polygon_curb_ratio = _class_ratio(mask, polygon_mask, curb_ids)
        polygon_top_classes = _top_class_ratios(mask, polygon_mask, id_to_name)

    has_polygon = polygon_total > 0
    min_polygon_allowed_ratio = 0.80 if navigation_mode == "foot" else 0.80
    max_polygon_forbidden_ratio = 0.002
    max_polygon_disallowed_ratio = 0.02
    start_relaxed_conditions: dict[str, Any] = {
        "enabled": False,
        "rule": "allow an off-center bottom start only when it is supported by a continuous allowed connector and reaches preferred traversable support",
    }
    if isinstance(start_point_uv, list) and len(start_point_uv) == 2 and not start_strict_valid:
        try:
            start_u = float(start_point_uv[0])
            start_v = float(start_point_uv[1])
            start_in_relaxed_band = (
                start_v >= start_min_v
                and start_u >= start_relaxed_u_min
                and start_u <= start_relaxed_u_max
            )
            start_has_allowed_support = bool(start_neighbor_ids & allowed_traversable_ids)
            connector_prefix_complete = bool(
                first_preferred_sample_index is not None
                and (
                    first_preferred_sample_index == 0
                    or len(conditional_connector_samples) >= int(first_preferred_sample_index)
                )
            )
            support_is_safe = bool(
                center_total > 0
                and boundary_total > 0
                and center_valid_ratio >= 0.8
                and boundary_valid_ratio >= 0.92
                and not center_forbidden_hits
                and not boundary_forbidden_hits
                and polygon_forbidden_ratio <= max_polygon_forbidden_ratio
                and polygon_disallowed_ratio <= max_polygon_disallowed_ratio
            )
            start_relaxed_valid = bool(
                start_in_relaxed_band
                and start_has_allowed_support
                and connector_prefix_complete
                and support_is_safe
            )
            if start_relaxed_valid:
                start_relaxed_reason = "off_center_allowed_connector_to_preferred_traversable"
            start_relaxed_conditions = {
                "enabled": True,
                "rule": start_relaxed_conditions["rule"],
                "u_range_ratio": [round(float(start_relaxed_u_min_ratio), 4), round(float(start_relaxed_u_max_ratio), 4)],
                "u_range_px": [round(float(start_relaxed_u_min), 2), round(float(start_relaxed_u_max), 2)],
                "start_in_relaxed_band": bool(start_in_relaxed_band),
                "start_has_allowed_support": bool(start_has_allowed_support),
                "connector_prefix_complete": bool(connector_prefix_complete),
                "first_preferred_sample_index": (
                    int(first_preferred_sample_index) if first_preferred_sample_index is not None else None
                ),
                "support_is_safe": bool(support_is_safe),
                "accepted": bool(start_relaxed_valid),
                "reason": start_relaxed_reason,
            }
        except Exception as exc:
            start_relaxed_conditions = {
                "enabled": True,
                "rule": start_relaxed_conditions["rule"],
                "accepted": False,
                "error": str(exc),
            }
    start_valid = bool(start_strict_valid or start_relaxed_valid)
    polygon_valid = (
        not has_polygon
        or (
            polygon_allowed_ratio >= min_polygon_allowed_ratio
            and polygon_forbidden_ratio <= max_polygon_forbidden_ratio
            and polygon_disallowed_ratio <= max_polygon_disallowed_ratio
        )
    )
    valid = (
        # V is intentionally conservative: one prompt is useful only if its
        # start, centerline, support boundaries, width, and polygon all agree.
        center_total > 0
        and start_valid
        and center_valid_ratio >= 0.8
        and boundary_valid_ratio >= 0.92
        and not center_forbidden_hits
        and not boundary_forbidden_hits
        and not center_disallowed_hits
        and not boundary_disallowed_hits
        and corridor_width_valid
        and centerline_valid
        and edge_margin_valid
        and polygon_valid
    )
    reason_parts: list[str] = []
    if center_total <= 0:
        reason_parts.append("no_center_path_samples")
    if center_total > 0 and not start_valid:
        reason_parts.append("start_point_not_in_lower_middle_band")
    if center_valid_ratio < 0.8:
        reason_parts.append(f"low_center_traversable_ratio={center_valid_ratio:.3f}")
    if boundary_valid_ratio < 0.92:
        reason_parts.append(f"low_boundary_support_ratio={boundary_valid_ratio:.3f}")
    if center_forbidden_hits:
        reason_parts.append("center_path_hits_forbidden_surface")
    if boundary_forbidden_hits:
        reason_parts.append("boundary_hits_forbidden_surface")
    if center_disallowed_hits:
        reason_parts.append("center_path_hits_mode_disallowed_surface")
    if boundary_disallowed_hits:
        reason_parts.append("boundary_hits_mode_disallowed_surface")
    if not corridor_width_valid:
        reason_parts.append(f"corridor_width_too_wide={max_corridor_width_ratio_observed:.3f}")
    if not centerline_valid:
        if preferred_run_missing_samples:
            reason_parts.append("centerline_preferred_run_missing")
        else:
            reason_parts.append(f"centerline_offset_too_high={max_centerline_offset_ratio:.3f}")
    if not edge_margin_valid:
        if not edge_margin_ratios:
            reason_parts.append("edge_margin_samples_missing")
        else:
            reason_parts.append(f"edge_margin_too_low={min_edge_margin_ratio_observed:.3f}")
    if has_polygon:
        if polygon_allowed_ratio < min_polygon_allowed_ratio:
            reason_parts.append(f"low_polygon_allowed_ratio={polygon_allowed_ratio:.3f}")
        if polygon_forbidden_ratio > max_polygon_forbidden_ratio:
            reason_parts.append(f"polygon_hits_forbidden_ratio={polygon_forbidden_ratio:.3f}")
        if polygon_disallowed_ratio > max_polygon_disallowed_ratio:
            reason_parts.append(f"polygon_hits_mode_disallowed_ratio={polygon_disallowed_ratio:.3f}")
    if not reason_parts:
        reason_parts.append("segmentation_support_ok")

    return {
        "valid": valid,
        "center_sample_count": int(center_total),
        "start_point_uv": (
            [round(float(start_point_uv[0]), 2), round(float(start_point_uv[1]), 2)]
            if isinstance(start_point_uv, list) and len(start_point_uv) == 2
            else None
        ),
        "start_valid": bool(start_valid),
        "start_strict_valid": bool(start_strict_valid),
        "start_relaxed_valid": bool(start_relaxed_valid),
        "start_constraint": {
            "rule": "first path sample must start from the bottom-center image band unless a safe off-center connector is the only visible route support",
            "v_min_ratio": round(float(start_min_v_ratio), 4),
            "u_range_ratio": [round(float(start_u_min_ratio), 4), round(float(start_u_max_ratio), 4)],
            "v_min_px": round(float(start_min_v), 2),
            "u_range_px": [round(float(start_u_min), 2), round(float(start_u_max), 2)],
            "relaxed_connector_rule": start_relaxed_conditions,
        },
        "center_valid_ratio": round(center_valid_ratio, 4),
        "center_preferred_ratio": round(center_preferred_ratio, 4),
        "center_conditional_ratio": round(center_conditional_ratio, 4),
        "boundary_sample_count": int(boundary_total),
        "boundary_valid_ratio": round(boundary_valid_ratio, 4),
        "boundary_preferred_ratio": round(boundary_preferred_ratio, 4),
        "boundary_conditional_ratio": round(boundary_conditional_ratio, 4),
        "corridor_width_valid": bool(corridor_width_valid),
        "corridor_width_sample_count": int(len(corridor_width_ratios)),
        "corridor_width_ratios": [round(float(value), 4) for value in corridor_width_ratios[:8]],
        "max_corridor_width_ratio_observed": round(float(max_corridor_width_ratio_observed), 4),
        "mean_corridor_width_ratio": round(float(mean_corridor_width_ratio), 4),
        "corridor_width_constraint": {
            "rule": "path_corridor boundaries must stay compact and obstacle-aware; do not span the whole visible free-space mask",
            "max_width_ratio": round(float(max_corridor_width_ratio), 4),
            "preferred_width_ratio": round(float(preferred_corridor_width_ratio), 4),
            "config_env": CORRIDOR_WIDTH_RATIO_ENV,
        },
        "centerline_valid": bool(centerline_valid),
        "centerline_offset_ratios": [round(float(value), 4) for value in centerline_offsets[:8]],
        "centerline_clearance_px": [round(float(value), 2) for value in centerline_clearances[:8]],
        "max_centerline_offset_ratio": round(float(max_centerline_offset_ratio), 4),
        "mean_centerline_offset_ratio": round(float(mean_centerline_offset_ratio), 4),
        "centerline_constraint": {
            "rule": "path center samples should stay near the centerline of the preferred traversable run at the same image row",
            "max_offset_ratio": round(float(centerline_offset_limit), 4),
            "config_env": CENTERLINE_OFFSET_RATIO_ENV,
        },
        "edge_margin_valid": bool(edge_margin_valid),
        "edge_margin_ratios": [round(float(value), 4) for value in edge_margin_ratios[:8]],
        "min_edge_margin_ratio_observed": round(float(min_edge_margin_ratio_observed), 4),
        "edge_margin_constraint": {
            "rule": "left/right corridor boundaries must stay inside the allowed traversable run with positive clearance from its edges",
            "min_margin_ratio": round(float(edge_margin_ratio), 4),
            "min_margin_px": round(float(min_edge_margin_px), 2),
            "config_env": EDGE_MARGIN_RATIO_ENV,
        },
        "polygon_pixel_count": int(polygon_total),
        "polygon_allowed_ratio": round(polygon_allowed_ratio, 4),
        "polygon_traversable_ratio": round(polygon_traversable_ratio, 4),
        "polygon_conditional_ratio": round(polygon_conditional_ratio, 4),
        "polygon_boundary_ratio": round(polygon_boundary_ratio, 4),
        "polygon_forbidden_ratio": round(polygon_forbidden_ratio, 4),
        "polygon_mode_disallowed_ratio": round(polygon_disallowed_ratio, 4),
        "polygon_curb_ratio": round(polygon_curb_ratio, 4),
        "polygon_top_classes": polygon_top_classes,
        "center_forbidden_hits": center_forbidden_hits[:8],
        "boundary_forbidden_hits": boundary_forbidden_hits[:8],
        "center_mode_disallowed_hits": center_disallowed_hits[:8],
        "boundary_mode_disallowed_hits": boundary_disallowed_hits[:8],
        "off_center_samples": off_center_samples[:8],
        "low_edge_margin_samples": low_margin_samples[:8],
        "preferred_run_missing_samples": preferred_run_missing_samples[:8],
        "conditional_connector_samples": conditional_connector_samples[:8],
        "first_preferred_sample_index": (
            int(first_preferred_sample_index) if first_preferred_sample_index is not None else None
        ),
        "navigation_mode": navigation_mode,
        "reason": "; ".join(reason_parts),
    }
