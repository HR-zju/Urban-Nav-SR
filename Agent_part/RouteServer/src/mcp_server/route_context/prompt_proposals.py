from __future__ import annotations

from typing import Any

from .schema import LocalBEVContext, PromptProposal, PromptProposalBundle, RouteContext


TURN_ACTIONS = {"turn_left", "turn_right", "keep_left", "keep_right", "bear_left", "bear_right", "u_turn"}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _distance_urgency(distance_m: float | None, *, near_m: float, far_m: float) -> float:
    if distance_m is None:
        return 0.35
    if distance_m <= near_m:
        return 1.0
    if distance_m >= far_m:
        return 0.0
    return 1.0 - ((distance_m - near_m) / max(far_m - near_m, 1e-6))


def _sorted_bundle(proposals: list[PromptProposal]) -> PromptProposalBundle:
    proposals.sort(key=lambda item: item.priority, reverse=True)
    return PromptProposalBundle(proposals=proposals)


def _make_proposal(
    *,
    proposal_id: str,
    proposal_type: str,
    sector_center_deg: float,
    sector_span_deg: float,
    range_m: tuple[float, float],
    priority: float,
    geometry_source: str,
    semantic_tag: str,
    rationale: str,
    prompt_text: str,
    debug: dict[str, Any],
) -> PromptProposal:
    return PromptProposal(
        proposal_id=proposal_id,
        proposal_type=proposal_type,
        frame="ego_bev",
        sector_center_deg=round(float(sector_center_deg), 2),
        sector_span_deg=round(float(sector_span_deg), 2),
        range_m=[round(float(range_m[0]), 2), round(float(range_m[1]), 2)],
        priority=round(_clamp(priority, 0.0, 1.0), 4),
        geometry_source=geometry_source,
        semantic_tag=semantic_tag,
        rationale=rationale,
        prompt_text=prompt_text,
        debug=debug,
    )


def generate_bev_prompt_proposals(
    *,
    route_context: RouteContext,
    local_bev_context: LocalBEVContext,
) -> PromptProposalBundle:
    proposals: list[PromptProposal] = []

    distance_to_turn = route_context.progress.distance_to_next_turn_m
    turn_type = route_context.local_semantics.turn_type
    turn_semantics = route_context.local_semantics.turn_semantics
    corridor_type = route_context.local_semantics.corridor_type
    off_route = route_context.pose_match.off_route
    lookahead_m = max(float(local_bev_context.lookahead_m), 5.0)
    route_confidence = float(route_context.pose_match.route_confidence)

    corridor_sector = local_bev_context.corridor_sector_deg or {"center_deg": 0.0, "span_deg": 60.0}
    turn_sector = local_bev_context.turn_sector_deg or {"center_deg": 0.0, "span_deg": 35.0}

    corridor_priority = 0.45 + 0.3 * route_confidence + 0.1 * (1.0 - _distance_urgency(distance_to_turn, near_m=8.0, far_m=30.0))
    if off_route:
        corridor_priority *= 0.65
    proposals.append(
        _make_proposal(
            proposal_id="p_corridor_main",
            proposal_type="corridor_sector",
            sector_center_deg=float(corridor_sector.get("center_deg", 0.0)),
            sector_span_deg=float(corridor_sector.get("span_deg", 60.0)),
            range_m=(0.0, min(lookahead_m, 12.0)),
            priority=corridor_priority,
            geometry_source="corridor_type",
            semantic_tag="forward_path",
            rationale=(
                "The current local route corridor remains the primary forward prior."
                if not off_route
                else "The route corridor remains a weak recovery prior while off route."
            ),
            prompt_text="Prioritize the forward route corridor.",
            debug={
                "corridor_type": corridor_type,
                "distance_to_next_turn_m": distance_to_turn,
                "turn_semantics": turn_semantics,
                "route_confidence": route_confidence,
            },
        )
    )

    if turn_type in TURN_ACTIONS or turn_semantics in {"approach_destination", "arrival_imminent"}:
        urgency = _distance_urgency(distance_to_turn, near_m=6.0, far_m=28.0)
        turn_priority = 0.28 + 0.55 * urgency + 0.1 * route_confidence
        proposals.append(
            _make_proposal(
                proposal_id="p_turn_focus",
                proposal_type="turn_sector",
                sector_center_deg=float(turn_sector.get("center_deg", 0.0)),
                sector_span_deg=float(turn_sector.get("span_deg", 35.0)),
                range_m=(max(0.0, min(distance_to_turn or 0.0, 4.0)), min(lookahead_m, max((distance_to_turn or 10.0) + 4.0, 10.0))),
                priority=turn_priority,
                geometry_source="turn_sector_deg",
                semantic_tag="turn_attention",
                rationale="The route is approaching a key maneuver, so turn-focused attention should increase.",
                prompt_text="Increase attention in the upcoming turn sector.",
                debug={
                    "turn_type": turn_type,
                    "turn_semantics": turn_semantics,
                    "distance_to_next_turn_m": distance_to_turn,
                },
            )
        )

    if local_bev_context.connector_bev or route_context.geometry.connector_active:
        merge_priority = 0.35 + 0.4 * _distance_urgency(distance_to_turn, near_m=0.0, far_m=20.0) + 0.1 * route_confidence
        proposals.append(
            _make_proposal(
                proposal_id="p_connector_merge",
                proposal_type="connector_merge_sector",
                sector_center_deg=0.0,
                sector_span_deg=32.0,
                range_m=(0.0, min(lookahead_m, 10.0)),
                priority=merge_priority,
                geometry_source="connector_bev",
                semantic_tag="merge_region",
                rationale="The current pose is still merging into the planned route from a connector segment.",
                prompt_text="Focus on the route merge corridor directly ahead.",
                debug={
                    "connector_active": route_context.geometry.connector_active,
                    "connector_points": len(local_bev_context.connector_bev),
                    "turn_semantics": turn_semantics,
                },
            )
        )

    entrance_candidates = (local_bev_context.semantic_bev_layers or {}).get("entrance_candidates_bev") or []
    if turn_semantics in {"approach_destination", "arrival_imminent"} or entrance_candidates:
        proposals.append(
            _make_proposal(
                proposal_id="p_entrance_candidate",
                proposal_type="entrance_candidate_sector",
                sector_center_deg=float(turn_sector.get("center_deg", 0.0)),
                sector_span_deg=max(float(turn_sector.get("span_deg", 35.0)), 28.0),
                range_m=(max(0.0, (distance_to_turn or 10.0) - 4.0), min(lookahead_m, max((distance_to_turn or 10.0) + 6.0, 8.0))),
                priority=0.26 + 0.4 * _distance_urgency(distance_to_turn, near_m=3.0, far_m=18.0),
                geometry_source="destination_proximity",
                semantic_tag="entrance_candidate",
                rationale="The destination is close enough that an entrance-side proposal should be preserved.",
                prompt_text="Preserve a candidate entrance sector near the destination side.",
                debug={
                    "turn_semantics": turn_semantics,
                    "entrance_candidates_count": len(entrance_candidates),
                    "distance_to_next_turn_m": distance_to_turn,
                },
            )
        )

    crossing_mask = (local_bev_context.semantic_bev_layers or {}).get("crossing_mask")
    if crossing_mask is not None or (distance_to_turn is not None and distance_to_turn <= 12.0 and turn_type in TURN_ACTIONS):
        proposals.append(
            _make_proposal(
                proposal_id="p_crossing_candidate",
                proposal_type="crossing_candidate_sector",
                sector_center_deg=float(turn_sector.get("center_deg", corridor_sector.get("center_deg", 0.0))),
                sector_span_deg=max(float(turn_sector.get("span_deg", 35.0)), 24.0),
                range_m=(0.0, min(lookahead_m, 8.0)),
                priority=0.18 + 0.35 * _distance_urgency(distance_to_turn, near_m=2.0, far_m=12.0),
                geometry_source="turn_proximity",
                semantic_tag="crossing_candidate",
                rationale="Keep a low-priority crossing-side hypothesis when the next maneuver is close.",
                prompt_text="Keep a nearby crossing candidate in the attention set.",
                debug={
                    "turn_type": turn_type,
                    "distance_to_next_turn_m": distance_to_turn,
                    "has_crossing_mask": crossing_mask is not None,
                },
            )
        )

    return _sorted_bundle(proposals)
