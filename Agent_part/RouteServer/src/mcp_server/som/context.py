from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

try:
    from scipy import ndimage as _ndimage  # type: ignore
except Exception:  # pragma: no cover - fallback path
    _ndimage = None


def _connected_components(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    if binary_mask.ndim != 2:
        raise RuntimeError(f"Expected a 2D mask, got shape={tuple(binary_mask.shape)}")
    mask = np.asarray(binary_mask, dtype=bool)
    if _ndimage is not None:
        structure = np.ones((3, 3), dtype=np.int8)
        labels, count = _ndimage.label(mask, structure=structure)
        return labels.astype(np.int32, copy=False), int(count)

    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    count = 0
    for v in range(height):
        for u in range(width):
            if not mask[v, u] or labels[v, u] != 0:
                continue
            count += 1
            queue: deque[tuple[int, int]] = deque([(v, u)])
            labels[v, u] = count
            while queue:
                cv, cu = queue.popleft()
                for dv in (-1, 0, 1):
                    for du in (-1, 0, 1):
                        if dv == 0 and du == 0:
                            continue
                        nv = cv + dv
                        nu = cu + du
                        if nv < 0 or nv >= height or nu < 0 or nu >= width:
                            continue
                        if not mask[nv, nu] or labels[nv, nu] != 0:
                            continue
                        labels[nv, nu] = count
                        queue.append((nv, nu))
    return labels, int(count)


def _side_from_center(center_u: float, width: int) -> str:
    ratio = float(center_u) / float(max(width, 1))
    if ratio < 0.38:
        return "left"
    if ratio > 0.62:
        return "right"
    return "center"


def _route_target_u_ratio(route_context: dict[str, Any] | None) -> float:
    local = (route_context or {}).get("local_semantics", {}) if isinstance(route_context, dict) else {}
    turn_type = str((local or {}).get("turn_type") or "").strip().lower()
    turn_hint = str((local or {}).get("instruction_hint") or "").strip().lower()
    if "left" in turn_type or "left" in turn_hint:
        return 0.34
    if "right" in turn_type or "right" in turn_hint:
        return 0.66
    return 0.50


def _regions_from_labels(
    *,
    mask: np.ndarray,
    labels: np.ndarray,
    component_count: int,
    id_to_name: dict[int, str],
    forbidden_ids: set[int],
    min_region_area_ratio: float,
    max_regions: int,
    region_prefix: str,
    region_role: str,
) -> list[dict[str, Any]]:
    height, width = int(mask.shape[0]), int(mask.shape[1])
    total_pixels = float(max(height * width, 1))
    regions: list[dict[str, Any]] = []
    for component_id in range(1, int(component_count) + 1):
        component_pixels = np.argwhere(labels == component_id)
        if component_pixels.size == 0:
            continue
        pixel_count = int(component_pixels.shape[0])
        area_ratio = float(pixel_count) / total_pixels
        if area_ratio < float(min_region_area_ratio):
            continue

        ys = component_pixels[:, 0]
        xs = component_pixels[:, 1]
        v_min = int(ys.min())
        v_max = int(ys.max())
        u_min = int(xs.min())
        u_max = int(xs.max())
        center_u = float(xs.mean())
        center_v = float(ys.mean())

        class_ids, class_counts = np.unique(mask[labels == component_id], return_counts=True)
        dominant_idx = int(np.argmax(class_counts)) if len(class_counts) else 0
        dominant_class_id = int(class_ids[dominant_idx]) if len(class_ids) else -1
        dominant_class_name = str(id_to_name.get(dominant_class_id, f"unknown_{dominant_class_id}"))
        component_classes = {int(class_id) for class_id in class_ids.tolist()}
        forbidden_pixel_count = int(np.isin(mask[labels == component_id], list(forbidden_ids)).sum()) if forbidden_ids else 0
        forbidden_overlap_ratio = float(forbidden_pixel_count) / float(max(pixel_count, 1))
        bottom_touch = bool(v_max >= int(round(height * 0.84)))
        regions.append(
            {
                "region_id": f"{region_prefix}_{len(regions) + 1}",
                "region_role": str(region_role),
                "component_id": int(component_id),
                "center_uv": [round(center_u, 2), round(center_v, 2)],
                "area_ratio": round(area_ratio, 6),
                "bbox_uv": [u_min, v_min, u_max, v_max],
                "side": _side_from_center(center_u, width),
                "bottom_touch": bottom_touch,
                "dominant_class_id": dominant_class_id,
                "dominant_class_name": dominant_class_name,
                "class_ids": sorted(int(class_id) for class_id in component_classes),
                "forbidden_overlap_ratio": round(forbidden_overlap_ratio, 6),
                "mask_polygon_uv": [
                    [u_min, v_min],
                    [u_max, v_min],
                    [u_max, v_max],
                    [u_min, v_max],
                ],
            }
        )

    regions.sort(
        key=lambda item: (
            int(bool(item.get("bottom_touch"))),
            float(item.get("area_ratio") or 0.0),
            -float(item.get("forbidden_overlap_ratio") or 0.0),
        ),
        reverse=True,
    )
    if max_regions > 0:
        regions = regions[: int(max_regions)]
    for index, region in enumerate(regions, start=1):
        region["region_id"] = f"{region_prefix}_{index}"
    return regions


def build_som_context(
    segmentation_context: dict[str, Any] | None,
    *,
    min_region_area_ratio: float = 0.002,
    max_regions: int = 32,
) -> dict[str, Any] | None:
    """Convert semantic-mask class sets into numbered connected regions.

    SoM regions are visual anchors for the LVLM and audit logs. They preserve
    the source-image coordinate space and separate preferred traversable,
    conditional connector, and forbidden components.
    """
    if not isinstance(segmentation_context, dict):
        return None
    mask = segmentation_context.get("mask")
    traversable_ids = set(segmentation_context.get("traversable_ids") or [])
    conditional_traversable_ids = set(segmentation_context.get("conditional_traversable_ids") or [])
    allowed_traversable_ids = set(segmentation_context.get("allowed_traversable_ids") or (traversable_ids | conditional_traversable_ids))
    forbidden_ids = set(segmentation_context.get("forbidden_ids") or [])
    id_to_name = segmentation_context.get("id_to_name") or {}
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        return None
    height, width = int(mask.shape[0]), int(mask.shape[1])

    traversable_labels, traversable_count = _connected_components(np.isin(mask, list(traversable_ids))) if traversable_ids else (np.zeros_like(mask, dtype=np.int32), 0)
    conditional_labels, conditional_count = _connected_components(np.isin(mask, list(conditional_traversable_ids))) if conditional_traversable_ids else (np.zeros_like(mask, dtype=np.int32), 0)
    allowed_labels, allowed_count = _connected_components(np.isin(mask, list(allowed_traversable_ids))) if allowed_traversable_ids else (np.zeros_like(mask, dtype=np.int32), 0)
    forbidden_labels, forbidden_count = _connected_components(np.isin(mask, list(forbidden_ids))) if forbidden_ids else (np.zeros_like(mask, dtype=np.int32), 0)

    regions = _regions_from_labels(
        mask=mask,
        labels=traversable_labels,
        component_count=traversable_count,
        id_to_name=id_to_name,
        forbidden_ids=forbidden_ids,
        min_region_area_ratio=min_region_area_ratio,
        max_regions=max_regions,
        region_prefix="som",
        region_role="traversable",
    )
    forbidden_regions = _regions_from_labels(
        mask=mask,
        labels=forbidden_labels,
        component_count=forbidden_count,
        id_to_name=id_to_name,
        forbidden_ids=forbidden_ids,
        min_region_area_ratio=max(float(min_region_area_ratio), 0.001),
        max_regions=max_regions,
        region_prefix="obs",
        region_role="forbidden",
    )
    conditional_regions = _regions_from_labels(
        mask=mask,
        labels=conditional_labels,
        component_count=conditional_count,
        id_to_name=id_to_name,
        forbidden_ids=forbidden_ids,
        min_region_area_ratio=max(float(min_region_area_ratio), 0.001),
        max_regions=max_regions,
        region_prefix="cond",
        region_role="conditional_traversable",
    )

    return {
        "schema_version": 2,
        "image_shape_hw": [height, width],
        "region_count": len(regions),
        "conditional_region_count": len(conditional_regions),
        "forbidden_region_count": len(forbidden_regions),
        "regions": regions,
        "conditional_regions": conditional_regions,
        "forbidden_regions": forbidden_regions,
        "_component_labels": traversable_labels,
        "_conditional_component_labels": conditional_labels,
        "_allowed_component_labels": allowed_labels,
        "_allowed_component_count": int(allowed_count),
        "_forbidden_component_labels": forbidden_labels,
    }


def rank_som_regions_for_route(
    som_context: dict[str, Any] | None,
    route_context: dict[str, Any] | None,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rank SoM support regions by route intent and image geometry.

    Score = 0.45 * side_score + 0.30 * depth_score + 0.25 * area_score
            + bottom_bonus - 0.35 * forbidden_penalty - conditional_penalty.
    """
    if not isinstance(som_context, dict):
        return []
    regions = list(som_context.get("regions") or []) + list(som_context.get("conditional_regions") or [])
    if not isinstance(regions, list):
        return []

    image_shape = som_context.get("image_shape_hw") or [1, 1]
    image_h = int(image_shape[0]) if len(image_shape) >= 1 else 1
    image_w = int(image_shape[1]) if len(image_shape) >= 2 else 1
    target_u_ratio = _route_target_u_ratio(route_context)

    scored: list[dict[str, Any]] = []
    for region in regions:
        role = str(region.get("region_role") or "traversable")
        if role not in {"traversable", "conditional_traversable"}:
            continue
        center = region.get("center_uv") or [image_w * 0.5, image_h * 0.5]
        try:
            center_u = float(center[0])
            center_v = float(center[1])
        except Exception:
            center_u = float(image_w) * 0.5
            center_v = float(image_h) * 0.5
        u_ratio = center_u / float(max(image_w, 1))
        v_ratio = center_v / float(max(image_h, 1))

        area_ratio = float(region.get("area_ratio") or 0.0)
        area_score = min(area_ratio / 0.08, 1.0)
        side_score = max(0.0, 1.0 - abs(u_ratio - target_u_ratio) / 0.5)
        depth_score = max(0.0, 1.0 - abs(v_ratio - 0.78) / 0.55)
        bottom_bonus = 0.15 if bool(region.get("bottom_touch")) else 0.0
        forbidden_penalty = min(float(region.get("forbidden_overlap_ratio") or 0.0), 1.0)
        conditional_penalty = 0.55 if role == "conditional_traversable" else 0.0

        score = (0.45 * side_score) + (0.30 * depth_score) + (0.25 * area_score) + bottom_bonus - (0.35 * forbidden_penalty) - conditional_penalty
        scored_region = dict(region)
        scored_region["route_intent_score"] = round(float(score), 6)
        scored_region["route_target_u_ratio"] = round(float(target_u_ratio), 4)
        scored.append(scored_region)

    scored.sort(key=lambda item: float(item.get("route_intent_score") or 0.0), reverse=True)
    if top_k > 0:
        scored = scored[: int(top_k)]
    for label, region in enumerate(scored, start=1):
        region["label"] = int(label)
    return scored


def _compact_region(region: dict[str, Any]) -> dict[str, Any]:
    return {
        "region_id": region.get("region_id"),
        "region_role": region.get("region_role"),
        "label": region.get("label"),
        "center_uv": region.get("center_uv"),
        "area_ratio": region.get("area_ratio"),
        "bbox_uv": region.get("bbox_uv"),
        "side": region.get("side"),
        "dominant_class_name": region.get("dominant_class_name"),
        "dominant_class_id": region.get("dominant_class_id"),
        "class_ids": region.get("class_ids"),
        "route_intent_score": region.get("route_intent_score"),
    }


def som_summary_for_llm(
    som_context: dict[str, Any] | None,
    *,
    ranked_regions: list[dict[str, Any]] | None = None,
    max_regions: int = 8,
) -> dict[str, Any] | None:
    if not isinstance(som_context, dict):
        return None
    regions = ranked_regions if isinstance(ranked_regions, list) else (som_context.get("regions") or [])
    compact_regions = [
        _compact_region(region)
        for region in regions[: max(int(max_regions), 1)]
        if str(region.get("region_role") or "traversable") == "traversable"
    ]
    for label, region in enumerate(compact_regions, start=1):
        if region.get("label") is None:
            region["label"] = int(label)
    conditional_regions = [
        _compact_region(region)
        for region in (regions if isinstance(ranked_regions, list) else list(som_context.get("conditional_regions") or []))[: max(int(max_regions), 1)]
        if str(region.get("region_role") or "") == "conditional_traversable"
    ]
    for label, region in enumerate(conditional_regions, start=len(compact_regions) + 1):
        if region.get("label") is None:
            region["label"] = int(label)
    forbidden_regions = list(som_context.get("forbidden_regions") or [])
    forbidden_regions.sort(key=lambda item: float(item.get("area_ratio") or 0.0), reverse=True)
    compact_forbidden = [_compact_region(region) for region in forbidden_regions[: max(int(max_regions), 1)]]
    for label, region in enumerate(compact_forbidden, start=len(compact_regions) + len(conditional_regions) + 1):
        region["label"] = int(label)

    primary_id = compact_regions[0].get("region_id") if compact_regions else None
    return {
        "schema_version": int(som_context.get("schema_version") or 2),
        "region_count": int(som_context.get("region_count") or len(som_context.get("regions") or [])),
        "conditional_region_count": int(som_context.get("conditional_region_count") or len(som_context.get("conditional_regions") or [])),
        "forbidden_region_count": int(som_context.get("forbidden_region_count") or len(som_context.get("forbidden_regions") or [])),
        "primary_region_id": primary_id,
        "regions": compact_regions,
        "conditional_regions": conditional_regions,
        "forbidden_regions": compact_forbidden,
    }
