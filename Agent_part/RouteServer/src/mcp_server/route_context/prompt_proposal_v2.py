from __future__ import annotations

from typing import Any

from .prompt_projection import project_bev_prompt_proposals_to_pov
from .schema import (
    PoVPromptCandidate,
    PoVPromptCandidateBundle,
    PromptProposal,
    RouteContext,
    VisualRegion,
    VisualRegionBundle,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = max((ax1 - ax0) * (ay1 - ay0), 1e-6) + max((bx1 - bx0) * (by1 - by0), 1e-6) - inter
    return inter / max(union, 1e-6)


def _candidate_bbox(center_uv: list[float] | None, span_px: float | None, prompt_type: str) -> list[float]:
    cx = float((center_uv or [0.0, 0.0])[0])
    cy = float((center_uv or [0.0, 0.0])[1])
    span = max(float(span_px or 80.0), 20.0)
    if prompt_type == "arrow":
        return [cx - span * 0.25, cy - span * 0.75, cx + span * 0.25, cy + span * 0.15]
    if prompt_type == "region":
        return [cx - span * 0.55, cy - span * 0.35, cx + span * 0.55, cy + span * 0.35]
    return [cx - span * 0.5, cy - span * 0.28, cx + span * 0.5, cy + span * 0.28]


def _region_center(region: VisualRegion) -> tuple[float, float]:
    x0, y0, x1, y1 = [float(v) for v in region.bbox_uv]
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _fuse_center(candidate: PoVPromptCandidate, region: VisualRegion | None) -> list[float] | None:
    if candidate.center_uv is None:
        return None
    if region is None:
        return candidate.center_uv
    cx, cy = float(candidate.center_uv[0]), float(candidate.center_uv[1])
    rx, ry = _region_center(region)
    visual_weight = 0.42 + 0.25 * float(region.score)
    route_weight = 1.0 - visual_weight
    return [round(route_weight * cx + visual_weight * rx, 2), round(route_weight * cy + visual_weight * ry, 2)]


def _fuse_priority(candidate: PoVPromptCandidate, region: VisualRegion | None, overlap: float) -> tuple[float, float]:
    visual_score = float(region.score) if region is not None else 0.0
    fused = 0.5 * float(candidate.priority) + 0.3 * visual_score + 0.2 * overlap
    return round(_clamp(fused, 0.0, 1.0), 4), round(_clamp(0.6 * visual_score + 0.4 * overlap, 0.0, 1.0), 4)


def _match_region(candidate: PoVPromptCandidate, regions: list[VisualRegion]) -> tuple[VisualRegion | None, float]:
    candidate_box = _candidate_bbox(candidate.center_uv, candidate.span_px, candidate.prompt_type)
    best_region = None
    best_score = -1.0
    best_overlap = 0.0
    for region in regions:
        overlap = _bbox_iou(candidate_box, region.bbox_uv)
        score = 0.7 * overlap + 0.3 * float(region.score)
        if score > best_score:
            best_score = score
            best_overlap = overlap
            best_region = region
    return best_region, best_overlap


def _rank_regions(candidate: PoVPromptCandidate, regions: list[VisualRegion]) -> list[tuple[VisualRegion, float, float]]:
    candidate_box = _candidate_bbox(candidate.center_uv, candidate.span_px, candidate.prompt_type)
    ranked: list[tuple[VisualRegion, float, float]] = []
    for region in regions:
        overlap = _bbox_iou(candidate_box, region.bbox_uv)
        score = 0.7 * overlap + 0.3 * float(region.score)
        ranked.append((region, overlap, score))
    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked


def generate_rgb_aware_pov_prompt_candidates(
    *,
    route_context: RouteContext,
    prompt_proposals: list[PromptProposal],
    visual_regions: VisualRegionBundle,
) -> PoVPromptCandidateBundle:
    base_bundle = project_bev_prompt_proposals_to_pov(
        route_context=route_context,
        prompt_proposals=prompt_proposals,
        image_width=visual_regions.image_width,
        image_height=visual_regions.image_height,
    )

    fused: list[PoVPromptCandidate] = []
    for candidate in base_bundle.candidates:
        ranked_regions = _rank_regions(candidate, visual_regions.candidate_regions)
        selected_regions: list[tuple[VisualRegion | None, float, float]] = []
        if ranked_regions:
            best_region, best_overlap, best_score = ranked_regions[0]
            selected_regions.append((best_region, best_overlap, best_score))

            is_forward_like = str(candidate.semantic_tag or "").strip().lower() in {
                "forward_path",
                "corridor_path",
                "connector_merge",
            }
            if is_forward_like:
                for region, overlap, score in ranked_regions[1:]:
                    if len(selected_regions) >= 2:
                        break
                    if score < max(0.08, best_score * 0.72):
                        continue
                    if region.score < 0.35:
                        continue
                    selected_regions.append((region, overlap, score))
        if not selected_regions:
            selected_regions.append((None, 0.0, 0.0))

        for branch_index, (region, overlap, match_score) in enumerate(selected_regions):
            priority, visual_confidence = _fuse_priority(candidate, region, overlap)
            proposal_id = candidate.proposal_id
            semantic_tag = candidate.semantic_tag
            if region is not None and len(selected_regions) > 1:
                proposal_id = f"{candidate.proposal_id}__{region.region_id}"
                semantic_tag = f"{candidate.semantic_tag or 'prompt'}::{region.region_id}"
            fused.append(
                PoVPromptCandidate(
                    proposal_id=proposal_id,
                    source_bev_proposal_id=candidate.source_bev_proposal_id,
                    prompt_type=candidate.prompt_type,
                    frame=candidate.frame,
                    center_uv=_fuse_center(candidate, region),
                    direction_vec=candidate.direction_vec,
                    span_px=round(max(float(candidate.span_px or 0.0), 24.0), 2) if candidate.span_px is not None else None,
                    mask=(region.polygon_uv if region is not None else None),
                    priority=priority,
                    projection_confidence=candidate.projection_confidence,
                    visual_confidence=visual_confidence,
                    semantic_tag=semantic_tag,
                    rationale=(
                        f"{candidate.rationale or ''} Fused with visual region {region.region_id}."
                        if region is not None
                        else (candidate.rationale or "Route prior only candidate.")
                    ).strip(),
                    debug={
                        **(candidate.debug or {}),
                        "matched_visual_region_id": region.region_id if region is not None else None,
                        "matched_visual_region_score": float(region.score) if region is not None else None,
                        "region_overlap": round(overlap, 4),
                        "region_match_score": round(match_score, 4),
                        "branch_index": branch_index,
                    },
                )
            )

    fused.sort(key=lambda item: item.priority, reverse=True)
    return PoVPromptCandidateBundle(candidates=fused)
