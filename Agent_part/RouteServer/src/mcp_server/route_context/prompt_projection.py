from __future__ import annotations

import math

from .schema import PoVPromptCandidate, PoVPromptCandidateBundle, PromptProposal, RouteContext


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _sorted_bundle(candidates: list[PoVPromptCandidate]) -> PoVPromptCandidateBundle:
    candidates.sort(key=lambda item: item.priority, reverse=True)
    return PoVPromptCandidateBundle(candidates=candidates)


def _proposal_prompt_type(proposal_type: str) -> str:
    if proposal_type == "connector_merge_sector":
        return "arrow"
    if proposal_type in {"entrance_candidate_sector", "crossing_candidate_sector"}:
        return "region"
    return "sector"


def _u_from_sector(center_deg: float, image_width: float) -> float:
    lateral = _clamp(center_deg / 90.0, -1.0, 1.0)
    return _clamp(0.5 + lateral * 0.32, 0.08, 0.92) * image_width


def _v_from_range(range_m: list[float], lookahead_m: float, image_height: float) -> float:
    mid_range = sum(range_m) / 2.0 if range_m else lookahead_m * 0.5
    ratio = _clamp(mid_range / max(lookahead_m, 1e-6), 0.0, 1.0)
    return (0.78 - 0.42 * ratio) * image_height


def _span_px(sector_span_deg: float, image_width: float, range_m: list[float], lookahead_m: float) -> float:
    base = max(40.0, image_width * (float(sector_span_deg) / 180.0) * 0.9)
    mid_range = sum(range_m) / 2.0 if range_m else lookahead_m * 0.5
    distance_scale = 1.15 - 0.45 * _clamp(mid_range / max(lookahead_m, 1e-6), 0.0, 1.0)
    return round(_clamp(base * distance_scale, 36.0, image_width * 0.72), 2)


def _direction_vec(center_deg: float) -> list[float]:
    rad = math.radians(center_deg)
    dx = math.sin(rad)
    dy = -math.cos(rad)
    return [round(dx, 4), round(dy, 4)]


def project_bev_prompt_proposals_to_pov(
    *,
    route_context: RouteContext,
    prompt_proposals: list[PromptProposal],
    image_width: int,
    image_height: int,
) -> PoVPromptCandidateBundle:
    width = max(int(image_width), 64)
    height = max(int(image_height), 64)
    lookahead_m = max(float(route_context.geometry.lookahead_m), 5.0)
    route_confidence = float(route_context.pose_match.route_confidence)
    off_route = bool(route_context.pose_match.off_route)

    candidates: list[PoVPromptCandidate] = []
    for index, proposal in enumerate(prompt_proposals):
        center_u = _u_from_sector(float(proposal.sector_center_deg), width)
        center_v = _v_from_range(proposal.range_m, lookahead_m, height)
        span_px = _span_px(float(proposal.sector_span_deg), width, proposal.range_m, lookahead_m)
        projection_confidence = 0.55 + 0.35 * route_confidence + 0.1 * float(proposal.priority)
        if off_route:
            projection_confidence *= 0.82

        candidates.append(
            PoVPromptCandidate(
                proposal_id=f"pov_{index}_{proposal.proposal_id}",
                source_bev_proposal_id=proposal.proposal_id,
                prompt_type=_proposal_prompt_type(proposal.proposal_type),
                frame="image_plane",
                center_uv=[round(center_u, 2), round(center_v, 2)],
                span_px=span_px,
                direction_vec=_direction_vec(float(proposal.sector_center_deg)),
                priority=round(float(proposal.priority), 4),
                projection_confidence=round(_clamp(projection_confidence, 0.0, 1.0), 4),
                debug={
                    "proposal_type": proposal.proposal_type,
                    "sector_center_deg": proposal.sector_center_deg,
                    "sector_span_deg": proposal.sector_span_deg,
                    "range_m": proposal.range_m,
                    "off_route": off_route,
                    "route_confidence": route_confidence,
                },
            )
        )
    return _sorted_bundle(candidates)
