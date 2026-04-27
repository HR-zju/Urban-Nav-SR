from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


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
    mode = str(navigation_mode or "foot").strip().lower()
    if mode == "car":
        traversable = {"road", "driveway"}
        boundary = {"road", "driveway", "curb", "gutter"}
    elif mode == "bike":
        traversable = {"road", "driveway", "crosswalk", "sidewalk"}
        boundary = {"road", "driveway", "crosswalk", "sidewalk", "curb", "gutter"}
    else:
        traversable = {"sidewalk", "crosswalk", "driveway"}
        boundary = {"sidewalk", "crosswalk", "driveway", "curb", "gutter"}
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
    boundary_ids = {name_to_id[name] for name in rules["boundary"] if name in name_to_id}
    forbidden_ids = {name_to_id[name] for name in rules["forbidden"] if name in name_to_id}

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
        "boundary_class_ids": sorted(int(class_id) for class_id in boundary_ids),
        "boundary_class_names": sorted(rules["boundary"]),
        "forbidden_class_ids": sorted(int(class_id) for class_id in forbidden_ids),
        "forbidden_class_names": sorted(rules["forbidden"]),
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
    }
    return {
        "mask": mask,
        "id_to_name": id_to_name,
        "name_to_id": name_to_id,
        "traversable_ids": traversable_ids,
        "boundary_ids": boundary_ids,
        "forbidden_ids": forbidden_ids,
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
    if not isinstance(rendered_prompt, dict) or not isinstance(segmentation_context, dict):
        return None

    mask = segmentation_context["mask"]
    id_to_name = segmentation_context["id_to_name"]
    traversable_ids: set[int] = set(segmentation_context["traversable_ids"])
    boundary_ids: set[int] = set(segmentation_context["boundary_ids"])
    forbidden_ids: set[int] = set(segmentation_context["forbidden_ids"])

    path_samples = rendered_prompt.get("path_samples") if isinstance(rendered_prompt.get("path_samples"), list) else []
    path_uv = rendered_prompt.get("path_uv") if isinstance(rendered_prompt.get("path_uv"), list) else []

    center_points = [sample.get("center_uv") for sample in path_samples if isinstance(sample, dict) and isinstance(sample.get("center_uv"), list)]
    left_points = [sample.get("left_edge_uv") for sample in path_samples if isinstance(sample, dict) and isinstance(sample.get("left_edge_uv"), list)]
    right_points = [sample.get("right_edge_uv") for sample in path_samples if isinstance(sample, dict) and isinstance(sample.get("right_edge_uv"), list)]
    if not center_points and path_uv:
        center_points = [point for point in path_uv if isinstance(point, list) and len(point) == 2]
    center_track = _interpolate_points(center_points)

    center_total = 0
    center_valid = 0
    center_forbidden_hits: list[dict[str, Any]] = []
    for point in center_track:
        neighbor_ids = _class_id_near(mask, point, radius_px=5)
        center_total += 1
        if neighbor_ids & traversable_ids:
            center_valid += 1
        if neighbor_ids & forbidden_ids:
            forbidden_class_ids = sorted(int(class_id) for class_id in (neighbor_ids & forbidden_ids))
            center_forbidden_hits.append(
                {
                    "point_uv": [round(float(point[0]), 2), round(float(point[1]), 2)],
                    "class_ids": forbidden_class_ids,
                    "class_names": [id_to_name.get(class_id, f"unknown_{class_id}") for class_id in forbidden_class_ids],
                }
            )

    boundary_total = 0
    boundary_valid = 0
    boundary_forbidden_hits: list[dict[str, Any]] = []
    for point in left_points + right_points:
        neighbor_ids = _class_id_near(mask, point, radius_px=4)
        boundary_total += 1
        if neighbor_ids & boundary_ids:
            boundary_valid += 1
        if neighbor_ids & forbidden_ids:
            forbidden_class_ids = sorted(int(class_id) for class_id in (neighbor_ids & forbidden_ids))
            boundary_forbidden_hits.append(
                {
                    "point_uv": [round(float(point[0]), 2), round(float(point[1]), 2)],
                    "class_ids": forbidden_class_ids,
                    "class_names": [id_to_name.get(class_id, f"unknown_{class_id}") for class_id in forbidden_class_ids],
                }
            )

    center_valid_ratio = float(center_valid) / float(center_total) if center_total else 0.0
    boundary_valid_ratio = float(boundary_valid) / float(boundary_total) if boundary_total else 1.0
    valid = center_total > 0 and center_valid_ratio >= 0.8 and not center_forbidden_hits and not boundary_forbidden_hits
    reason_parts: list[str] = []
    if center_total <= 0:
        reason_parts.append("no_center_path_samples")
    if center_valid_ratio < 0.8:
        reason_parts.append(f"low_center_traversable_ratio={center_valid_ratio:.3f}")
    if center_forbidden_hits:
        reason_parts.append("center_path_hits_forbidden_surface")
    if boundary_forbidden_hits:
        reason_parts.append("boundary_hits_forbidden_surface")
    if not reason_parts:
        reason_parts.append("segmentation_support_ok")

    return {
        "valid": valid,
        "center_sample_count": int(center_total),
        "center_valid_ratio": round(center_valid_ratio, 4),
        "boundary_sample_count": int(boundary_total),
        "boundary_valid_ratio": round(boundary_valid_ratio, 4),
        "center_forbidden_hits": center_forbidden_hits[:8],
        "boundary_forbidden_hits": boundary_forbidden_hits[:8],
        "reason": "; ".join(reason_parts),
    }
