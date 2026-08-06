"""Core metrics for UrbanNav destination and route evaluation.

The evaluator first matches the predicted destination to one acceptable GT
destination, then evaluates the route against that matched destination's own
frozen reference route. This is the key difference from single-point GT
evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable

from mcp_server.eval.urbannav.urbannav_schema import UrbanNavCase, UrbanNavDestination, UrbanNavPoint, UrbanNavPrediction


EARTH_RADIUS_M = 6371000.0


@dataclass(frozen=True)
class UrbanNavEvalConfig:
    """Configuration for coordinate fallback matching.

    Args:
        destination_tolerance_m: Coordinate fallback tolerance for destination
            identity matching when stable POI ids are unavailable.
        endpoint_tolerance_m: Tolerance for checking whether the predicted
            route endpoint reaches the matched destination.
        multi_stop_continuity_tolerance_m: Maximum gap between adjacent
            predicted route legs and their declared transition origin.
    """

    destination_tolerance_m: float = 30.0
    endpoint_tolerance_m: float = 30.0
    multi_stop_continuity_tolerance_m: float = 5.0


@dataclass(frozen=True)
class UrbanNavEvalRow:
    """Per-case UrbanNav evaluation result.

    Args:
        case_id: Case id aligned with the GT and prediction.
        destination_success: Whether top-1 prediction matches any acceptable
            destination.
        hit_at_5: Whether top-5 candidates contain any acceptable destination;
            ``None`` when no candidate list is provided.
        route_success: Whether destination is matched and route endpoint reaches
            that matched destination.
        route_ratio: ``pred_route_distance / matched_reference_distance`` for
            successful routes, otherwise ``None``.
        spl: Success weighted by path length efficiency.
        matched_destination_rank: Offline GT rank of the matched destination.
        matched_destination_id: Stable id of the matched destination.
        pred_route_distance_m: Predicted route distance in meters.
        reference_route_distance_m: Frozen reference distance for the matched
            destination.
        destination_match_method: Match method, such as ``stable_id`` or
            ``name_type_coordinate``.
        endpoint_error_m: Distance from predicted route endpoint to matched
            destination.
        extended_metrics: Optional category-specific metrics for tasks such as
            multi-stop route following and route-update rerouting.
        prediction_exception: Whether the saved prediction row is an exception
            placeholder emitted by a strict external runner rather than a valid
            destination/route prediction.
        prediction_exception_detail: Optional exception detail from the
            prediction trace.
        errors: Non-fatal per-case evaluation errors.
        category: Task category copied from GT for slicing.
        subcategory: Task subcategory copied from GT for slicing.
        city: City copied from GT for slicing.
        scene_id: Scene id copied from GT for slicing.
        difficulty: Difficulty copied from GT for slicing.
    """

    case_id: str
    destination_success: bool
    hit_at_5: bool | None
    route_success: bool
    route_ratio: float | None
    spl: float
    matched_destination_rank: int | None = None
    matched_destination_id: str | None = None
    pred_route_distance_m: float | None = None
    reference_route_distance_m: float | None = None
    destination_match_method: str | None = None
    endpoint_error_m: float | None = None
    extended_metrics: dict[str, Any] = field(default_factory=dict)
    prediction_exception: bool = False
    prediction_exception_detail: str | None = None
    errors: list[str] = field(default_factory=list)
    category: str | None = None
    subcategory: str | None = None
    city: str | None = None
    scene_id: str | None = None
    difficulty: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the per-case result.

        Returns:
            JSON-ready dictionary for per-case JSONL output.
        """

        return {
            "case_id": self.case_id,
            "destination_success": self.destination_success,
            "hit@5": self.hit_at_5,
            "route_success": self.route_success,
            "route_ratio": self.route_ratio,
            "spl": self.spl,
            "matched_destination_rank": self.matched_destination_rank,
            "matched_destination_id": self.matched_destination_id,
            "pred_route_distance_m": self.pred_route_distance_m,
            "reference_route_distance_m": self.reference_route_distance_m,
            "destination_match_method": self.destination_match_method,
            "endpoint_error_m": self.endpoint_error_m,
            "extended_metrics": dict(self.extended_metrics),
            "prediction_exception": self.prediction_exception,
            "prediction_exception_detail": self.prediction_exception_detail,
            "errors": list(self.errors),
            "category": self.category,
            "subcategory": self.subcategory,
            "city": self.city,
            "scene_id": self.scene_id,
            "difficulty": self.difficulty,
        }


@dataclass(frozen=True)
class DestinationMatch:
    """Destination matching result.

    Args:
        destination: The acceptable GT destination that was matched.
        method: Matching method name.
        distance_m: Coordinate fallback distance when applicable.
    """

    destination: UrbanNavDestination
    method: str
    distance_m: float | None = None


def haversine_m(a: UrbanNavPoint, b: UrbanNavPoint) -> float:
    r"""Compute great-circle distance between two coordinates.

    Formula:
        Let ``R`` be Earth radius and ``h`` be the Haversine term:

        \[
        h = \sin^2\left(\frac{\Delta \phi}{2}\right)
          + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)
        \]

        \[
        d = 2R\arcsin(\sqrt{h})
        \]

    Args:
        a: First coordinate.
        b: Second coordinate.

    Returns:
        Distance in meters.
    """

    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    dlat = lat2 - lat1
    dlng = math.radians(b.lng - a.lng)
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def route_distance_m(points: list[UrbanNavPoint]) -> float | None:
    r"""Compute polyline length by summing adjacent Haversine distances.

    Formula:
        \[
        L(P) = \sum_{i=1}^{n-1} d(p_i, p_{i+1})
        \]

    Args:
        points: Route polyline points.

    Returns:
        Route length in meters, or ``None`` when fewer than two points exist.
    """

    if len(points) < 2:
        return None
    return sum(haversine_m(a, b) for a, b in zip(points, points[1:]))


def destination_identity(destination: UrbanNavDestination) -> str | None:
    """Return the preferred stable destination identity.

    Args:
        destination: Destination object.

    Returns:
        ``provider_poi_id`` if present, otherwise ``place_id``; ``None`` when
        no stable id is available.
    """

    return destination.provider_poi_id or destination.place_id


def match_destination(
    predicted: UrbanNavDestination | None,
    acceptable: list[UrbanNavDestination],
    *,
    tolerance_m: float = 30.0,
) -> DestinationMatch | None:
    """Match a predicted destination against acceptable GT destinations.

    Identity match by provider/place id is preferred. Coordinate fallback is
    only accepted when name and type agree when those GT fields are present.

    Args:
        predicted: Top-1 predicted destination.
        acceptable: Acceptable GT destinations for this case.
        tolerance_m: Coordinate fallback radius in meters.

    Returns:
        ``DestinationMatch`` for the first stable-id hit or nearest compatible
        coordinate fallback hit. Returns ``None`` if no acceptable destination
        matches.
    """

    if predicted is None:
        return None
    pred_ids = predicted.stable_ids
    if pred_ids:
        for gt in acceptable:
            if pred_ids & gt.stable_ids:
                return DestinationMatch(destination=gt, method="stable_id", distance_m=0.0)

    pred_point = predicted.point
    if pred_point is None:
        return None

    best: tuple[float, UrbanNavDestination] | None = None
    for gt in acceptable:
        gt_point = gt.point
        if gt_point is None:
            continue
        if not _compatible_name_type(predicted, gt):
            continue
        distance = haversine_m(pred_point, gt_point)
        if distance <= tolerance_m and (best is None or distance < best[0]):
            best = (distance, gt)
    if best is None:
        return None
    return DestinationMatch(destination=best[1], method="name_type_coordinate", distance_m=best[0])


def evaluate_case(
    case: UrbanNavCase,
    prediction: UrbanNavPrediction | None,
    *,
    config: UrbanNavEvalConfig | None = None,
) -> UrbanNavEvalRow:
    r"""Evaluate one GT case and one prediction.

    Core formulas:
        Destination success:

        \[
        DS = \mathbb{1}[\operatorname{match}(\hat d, D_{acc}) \neq \varnothing]
        \]

        Route success:

        \[
        RS = DS \land |\hat P| > 0
             \land endpoint(\hat P) \approx d_j
        \]

        Route ratio for matched destination ``d_j``:

        \[
        route\_ratio = \frac{L(\hat P)}{L(P^*_{d_j})}
        \]

        SPL:

        \[
        SPL = RS \cdot \frac{L(P^*_{d_j})}{\max(L(P^*_{d_j}), L(\hat P))}
        \]

    Args:
        case: Frozen UrbanNav GT case.
        prediction: Saved prediction for the same case, or ``None`` when
            missing.
        config: Evaluation configuration. Defaults to 30m destination and
            endpoint tolerances.

    Returns:
        Per-case ``UrbanNavEvalRow``.
    """

    cfg = config or UrbanNavEvalConfig()
    errors: list[str] = []
    acceptable = case.acceptable_destinations
    if not acceptable and case.gt_destination is not None:
        acceptable = [case.gt_destination]
    if not acceptable:
        errors.append("missing_acceptable_destinations")

    if prediction is None:
        errors.append("missing_prediction")
        return _row(
            case,
            destination_success=False,
            hit_at_5=None,
            route_success=False,
            route_ratio=None,
            spl=0.0,
            errors=errors,
        )

    prediction_exception, prediction_exception_detail = _prediction_exception(prediction)
    if prediction_exception:
        errors.append("prediction_exception")

    match = match_destination(
        prediction.chosen_destination,
        acceptable,
        tolerance_m=cfg.destination_tolerance_m,
    )
    destination_success = match is not None
    hit_at_5 = compute_hit_at_k(prediction.candidate_destinations, acceptable, k=5, tolerance_m=cfg.destination_tolerance_m)

    reference_distance = None
    endpoint_error = None
    route_success = False
    route_ratio = None
    spl = 0.0
    pred_distance = prediction.route_distance_m or route_distance_m(prediction.route_polyline)

    if match is not None:
        reference_distance = _reference_distance(case, match.destination)
        endpoint_error = _endpoint_error(prediction.route_polyline, match.destination)
        route_success = (
            destination_success
            and bool(prediction.route_polyline)
            and endpoint_error is not None
            and endpoint_error <= cfg.endpoint_tolerance_m
            and pred_distance is not None
            and pred_distance > 0
            and reference_distance is not None
            and reference_distance > 0
        )
        if route_success:
            route_ratio = pred_distance / reference_distance
            spl = reference_distance / max(reference_distance, pred_distance)

    if destination_success and reference_distance is None:
        errors.append("missing_reference_distance_for_matched_destination")
    if destination_success and not prediction.route_polyline:
        errors.append("missing_route_polyline")

    extended_metrics = evaluate_extended_case_metrics(case, prediction, config=cfg)

    return _row(
        case,
        destination_success=destination_success,
        hit_at_5=hit_at_5,
        route_success=route_success,
        route_ratio=route_ratio,
        spl=spl,
        matched_destination_rank=match.destination.rank if match else None,
        matched_destination_id=destination_identity(match.destination) if match else None,
        pred_route_distance_m=pred_distance,
        reference_route_distance_m=reference_distance,
        destination_match_method=match.method if match else None,
        endpoint_error_m=endpoint_error,
        extended_metrics=extended_metrics,
        prediction_exception=prediction_exception,
        prediction_exception_detail=prediction_exception_detail,
        errors=errors,
    )


def evaluate_extended_case_metrics(
    case: UrbanNavCase,
    prediction: UrbanNavPrediction | None,
    *,
    config: UrbanNavEvalConfig | None = None,
) -> dict[str, Any]:
    """Evaluate category-specific UrbanNav metrics.

    Args:
        case: Frozen UrbanNav GT case.
        prediction: Saved prediction for the case, or ``None``.
        config: Evaluation tolerance configuration.

    Returns:
        Dictionary of extension metrics. Empty for ordinary single-destination
        cases.
    """

    cfg = config or UrbanNavEvalConfig()
    metrics: dict[str, Any] = {}
    if _is_multi_stop_case(case):
        metrics["multi_stop"] = evaluate_multi_stop_case(case, prediction, config=cfg)
    if _is_route_update_case(case):
        metrics["route_update"] = evaluate_route_update_case(case, prediction, config=cfg)
    return metrics


def evaluate_multi_stop_case(
    case: UrbanNavCase,
    prediction: UrbanNavPrediction | None,
    *,
    config: UrbanNavEvalConfig | None = None,
) -> dict[str, Any]:
    r"""Evaluate an ordered multi-stop prediction.

    Let \(G_i\) be acceptable destinations for stop \(i\), and let
    \(\hat d_i\) be the predicted stop at the same index:

    \[
    stop\_recall = \frac{1}{K}\sum_{i=1}^{K}
      \mathbb{1}[\operatorname{match}(\hat d_i, G_i)]
    \]

    Sequence success requires every stop to match in order. Multi-stop SPL is
    computed on the concatenated route only when the full sequence succeeds.

    Args:
        case: Multi-stop GT case.
        prediction: Multi-stop prediction.
        config: Evaluation tolerance configuration.

    Returns:
        JSON-ready multi-stop metric dictionary.
    """

    cfg = config or UrbanNavEvalConfig()
    gt_legs = _gt_route_legs(case)
    gt_sequence = _gt_destination_sequence(case)
    pred_sequence = list(prediction.destination_sequence) if prediction is not None else []
    pred_legs = list(prediction.route_legs) if prediction is not None else []
    if not gt_legs and gt_sequence:
        gt_legs = [
            {"leg_index": index + 1, "gt_destination": destination.to_dict(), "acceptable_destinations": [destination.to_dict()]}
            for index, destination in enumerate(gt_sequence)
        ]

    stop_results: list[dict[str, Any]] = []
    matched_count = 0
    stop_hit_at_5_count = 0
    stop_candidate_count = 0
    leg_route_success_count = 0
    reference_total = 0.0
    for index, leg in enumerate(gt_legs):
        acceptable = [
            destination
            for destination in (
                UrbanNavDestination.from_dict(item)
                for item in _list_dicts(leg.get("acceptable_destinations") or [leg.get("gt_destination")])
            )
            if destination is not None
        ]
        predicted_destination = pred_sequence[index] if index < len(pred_sequence) else None
        match = match_destination(predicted_destination, acceptable, tolerance_m=cfg.destination_tolerance_m)
        matched = match is not None
        if matched:
            matched_count += 1

        leg_candidates = _leg_candidate_destinations(pred_legs[index]) if index < len(pred_legs) else []
        stop_hit_at_5 = compute_hit_at_k(leg_candidates, acceptable, k=5, tolerance_m=cfg.destination_tolerance_m)
        if stop_hit_at_5 is not None:
            stop_candidate_count += 1
            if stop_hit_at_5:
                stop_hit_at_5_count += 1

        reference_distance = _leg_reference_distance(leg, match.destination if match else None)
        if reference_distance is not None and reference_distance > 0:
            reference_total += reference_distance

        pred_leg_route = _parse_leg_route(pred_legs[index]) if index < len(pred_legs) else []
        endpoint_error = _endpoint_error(pred_leg_route, match.destination) if match else None
        pred_leg_distance = _leg_pred_distance(pred_legs[index]) if index < len(pred_legs) else route_distance_m(pred_leg_route)
        leg_route_success = (
            matched
            and bool(pred_leg_route)
            and endpoint_error is not None
            and endpoint_error <= cfg.endpoint_tolerance_m
            and pred_leg_distance is not None
            and pred_leg_distance > 0
            and reference_distance is not None
            and reference_distance > 0
        )
        if leg_route_success:
            leg_route_success_count += 1
        stop_results.append(
            {
                "leg_index": int(leg.get("leg_index") or index + 1),
                "destination_success": matched,
                "route_success": leg_route_success,
                "hit@5": stop_hit_at_5,
                "matched_destination_rank": match.destination.rank if match else None,
                "matched_destination_id": destination_identity(match.destination) if match else None,
                "destination_match_method": match.method if match else None,
                "endpoint_error_m": endpoint_error,
                "pred_route_distance_m": pred_leg_distance,
                "reference_route_distance_m": reference_distance,
            }
        )

    stop_count = len(gt_legs)
    sequence_success = stop_count > 0 and matched_count == stop_count and len(pred_sequence) == stop_count
    sequence_hit_at_5 = stop_count > 0 and stop_candidate_count == stop_count and stop_hit_at_5_count == stop_count
    continuity = _evaluate_leg_continuity(
        pred_legs,
        tolerance_m=cfg.multi_stop_continuity_tolerance_m,
    )
    # 连续性不仅要求相邻几何连接，还要求没有缺失或多余的 route leg。
    leg_continuity_success = (
        stop_count > 0
        and len(pred_legs) == stop_count
        and continuity["success"]
    )
    for index, result in enumerate(stop_results[:-1]):
        result["continuity_to_next"] = continuity["transitions"][index] if index < len(continuity["transitions"]) else None
    if stop_results:
        stop_results[-1]["continuity_to_next"] = None
    leg_route_success = (
        stop_count > 0
        and leg_route_success_count == stop_count
        and leg_continuity_success
    )
    pred_total = prediction.route_distance_m if prediction is not None else None
    if pred_total is None and prediction is not None:
        pred_total = route_distance_m(prediction.route_polyline)
    multi_stop_spl = 0.0
    if sequence_success and leg_route_success and pred_total is not None and pred_total > 0 and reference_total > 0:
        multi_stop_spl = reference_total / max(reference_total, pred_total)
    return {
        "stop_count": stop_count,
        "pred_stop_count": len(pred_sequence),
        "sequence_success": sequence_success,
        "stop_recall": matched_count / stop_count if stop_count else 0.0,
        "stop_hit@5": stop_hit_at_5_count / stop_count if stop_count else 0.0,
        "sequence_hit@5": sequence_hit_at_5,
        "candidate_coverage": stop_candidate_count / stop_count if stop_count else 0.0,
        "leg_route_success": leg_route_success,
        "leg_route_success_rate": leg_route_success_count / stop_count if stop_count else 0.0,
        "leg_continuity_success": leg_continuity_success,
        "max_leg_gap_m": continuity["max_gap_m"],
        "leg_transitions": continuity["transitions"],
        "multi_stop_spl": multi_stop_spl,
        "reference_total_distance_m": reference_total if reference_total > 0 else None,
        "pred_total_distance_m": pred_total,
        "legs": stop_results,
    }


def _evaluate_leg_continuity(
    pred_legs: list[Any],
    *,
    tolerance_m: float,
) -> dict[str, Any]:
    """Evaluate geometric continuity between adjacent predicted route legs."""

    if len(pred_legs) <= 1:
        return {"success": True, "max_gap_m": 0.0, "transitions": []}
    transitions: list[dict[str, Any]] = []
    observed_gaps: list[float] = []
    for index in range(len(pred_legs) - 1):
        previous_route = _parse_leg_route(pred_legs[index])
        next_route = _parse_leg_route(pred_legs[index + 1])
        route_gap = None
        if previous_route and next_route:
            route_gap = haversine_m(previous_route[-1], next_route[0])
            observed_gaps.append(route_gap)
        next_origin = _leg_origin(pred_legs[index + 1])
        previous_to_origin_gap = None
        origin_to_next_gap = None
        if next_origin is not None and previous_route:
            previous_to_origin_gap = haversine_m(previous_route[-1], next_origin)
            observed_gaps.append(previous_to_origin_gap)
        if next_origin is not None and next_route:
            origin_to_next_gap = haversine_m(next_origin, next_route[0])
            observed_gaps.append(origin_to_next_gap)
        checks = [route_gap]
        if next_origin is not None:
            checks.extend([previous_to_origin_gap, origin_to_next_gap])
        success = all(gap is not None and gap <= tolerance_m for gap in checks)
        transitions.append(
            {
                "from_leg_index": index + 1,
                "to_leg_index": index + 2,
                "success": success,
                "route_gap_m": route_gap,
                "previous_end_to_next_origin_m": previous_to_origin_gap,
                "next_origin_to_next_start_m": origin_to_next_gap,
            }
        )
    return {
        "success": all(item["success"] for item in transitions),
        "max_gap_m": max(observed_gaps) if observed_gaps else None,
        "transitions": transitions,
    }


def _leg_origin(value: Any) -> UrbanNavPoint | None:
    """Parse an optional route-leg origin."""

    if not isinstance(value, dict):
        return None
    return _point_from_dict(value.get("origin"))


def evaluate_route_update_case(
    case: UrbanNavCase,
    prediction: UrbanNavPrediction | None,
    *,
    config: UrbanNavEvalConfig | None = None,
) -> dict[str, Any]:
    r"""Evaluate route-update/reroute behavior.

    Route update success requires the prediction to use the update origin,
    avoid the previous destination for replacement updates, and route to an
    acceptable updated destination:

    \[
    update\_success = RS_{updated} \land current\_origin\_used \land avoid\_previous
    \]

    Args:
        case: Route-update GT case.
        prediction: Saved prediction.
        config: Evaluation tolerance configuration.

    Returns:
        JSON-ready route-update metrics.
    """

    cfg = config or UrbanNavEvalConfig()
    update = _route_update_metadata(case)
    update_type = str(update.get("update_type") or "")
    current_origin = _point_from_dict(update.get("current_origin"))
    previous_destination = UrbanNavDestination.from_dict(update.get("previous_destination")) if isinstance(update.get("previous_destination"), dict) else None
    final_destination = UrbanNavDestination.from_dict(update.get("final_destination")) if isinstance(update.get("final_destination"), dict) else None
    current_origin_error = None
    current_origin_used = True
    if prediction is None or not prediction.route_polyline:
        current_origin_used = False
    elif current_origin is not None:
        current_origin_error = haversine_m(prediction.route_polyline[0], current_origin)
        current_origin_used = current_origin_error <= cfg.endpoint_tolerance_m

    acceptable = case.acceptable_destinations or ([case.gt_destination] if case.gt_destination else [])
    destination_match = match_destination(
        prediction.chosen_destination if prediction else None,
        acceptable,
        tolerance_m=cfg.destination_tolerance_m,
    )
    previous_match = None
    if previous_destination is not None and prediction is not None:
        previous_match = match_destination(
            prediction.chosen_destination,
            [previous_destination],
            tolerance_m=cfg.destination_tolerance_m,
        )
    avoid_previous = previous_match is None
    updated_destination_success = destination_match is not None
    updated_route_success = False
    pred_distance = prediction.route_distance_m if prediction else None
    if pred_distance is None and prediction is not None:
        pred_distance = route_distance_m(prediction.route_polyline)
    reference_distance = _reference_distance(case, destination_match.destination) if destination_match else None
    endpoint_error = _endpoint_error(prediction.route_polyline, destination_match.destination) if prediction and destination_match else None
    via_route_error = None
    final_endpoint_error = None
    final_destination_preserved = True
    if _route_update_requires_final(update_type) and prediction is not None and destination_match is not None:
        via_route_error = _route_min_error(prediction.route_polyline, destination_match.destination)
        final_endpoint_error = _endpoint_error(prediction.route_polyline, final_destination) if final_destination is not None else None
        final_destination_preserved = final_endpoint_error is not None and final_endpoint_error <= cfg.endpoint_tolerance_m
        updated_route_success = (
            updated_destination_success
            and bool(prediction.route_polyline)
            and via_route_error is not None
            and via_route_error <= cfg.endpoint_tolerance_m
            and final_destination_preserved
            and pred_distance is not None
            and pred_distance > 0
        )
    else:
        updated_route_success = (
            updated_destination_success
            and bool(prediction and prediction.route_polyline)
            and endpoint_error is not None
            and endpoint_error <= cfg.endpoint_tolerance_m
            and pred_distance is not None
            and pred_distance > 0
            and reference_distance is not None
            and reference_distance > 0
        )
    route_update_success = bool(updated_route_success and current_origin_used and avoid_previous)
    return {
        "update_type": update_type or update.get("update_type"),
        "route_update_success": route_update_success,
        "updated_destination_success": updated_destination_success,
        "updated_route_success": updated_route_success,
        "current_origin_used": current_origin_used,
        "current_origin_error_m": current_origin_error,
        "final_destination_preserved": final_destination_preserved,
        "final_endpoint_error_m": final_endpoint_error,
        "via_route_error_m": via_route_error,
        "previous_destination_avoided": avoid_previous,
        "previous_destination_selected": previous_match is not None,
        "endpoint_error_m": endpoint_error,
        "pred_route_distance_m": pred_distance,
        "reference_route_distance_m": reference_distance,
    }


def compute_hit_at_k(
    candidates: list[UrbanNavDestination],
    acceptable: list[UrbanNavDestination],
    *,
    k: int,
    tolerance_m: float = 30.0,
) -> bool | None:
    r"""Compute whether top-k candidates contain any acceptable destination.

    Formula:
        \[
        hit@k = \mathbb{1}[\exists \hat d_i,\ i \le k:
        \operatorname{match}(\hat d_i, D_{acc}) \neq \varnothing]
        \]

    Args:
        candidates: Ranked predicted candidate destinations.
        acceptable: Acceptable GT destinations.
        k: Cutoff rank.
        tolerance_m: Coordinate fallback tolerance in meters.

    Returns:
        ``True`` if any top-k candidate matches, ``False`` if candidates are
        present but none match, and ``None`` when no candidate list is provided.
    """

    if not candidates:
        return None
    for candidate in candidates[:k]:
        if match_destination(candidate, acceptable, tolerance_m=tolerance_m) is not None:
            return True
    return False


def evaluate_predictions(
    cases: Iterable[UrbanNavCase],
    predictions: Iterable[UrbanNavPrediction],
    *,
    config: UrbanNavEvalConfig | None = None,
) -> tuple[list[UrbanNavEvalRow], dict[str, Any]]:
    """Evaluate a batch of predictions against cases.

    Args:
        cases: Frozen GT cases.
        predictions: Saved predictions.
        config: Optional evaluation configuration.

    Returns:
        Tuple ``(rows, summary)`` where ``rows`` contains per-case results and
        ``summary`` contains aggregate main metrics.
    """

    predictions_by_id = {prediction.case_id: prediction for prediction in predictions}
    rows = [evaluate_case(case, predictions_by_id.get(case.case_id), config=config) for case in cases]
    return rows, summarize_rows(rows)


def summarize_rows(rows: list[UrbanNavEvalRow]) -> dict[str, Any]:
    r"""Aggregate main UrbanNav metrics over rows.

    Aggregate formulas:
        \[
        destination\_success = \frac{1}{N}\sum_i DS_i
        \]

        \[
        route\_success = \frac{1}{N}\sum_i RS_i
        \]

        \[
        median\_route\_ratio = \operatorname{median}
        \left(\frac{L(\hat P_i)}{L(P^*_{d_j,i})}\right)_{RS_i=1}
        \]

        \[
        spl = \frac{1}{N}\sum_i SPL_i
        \]

    Args:
        rows: Per-case evaluation rows.

    Returns:
        Summary dictionary with total case count and the five main metrics.
    """

    total = len(rows)
    if not rows:
        return {
            "total_cases": 0,
            "destination_success": 0.0,
            "hit@5": 0.0,
            "task_hit@5": 0.0,
            "route_success": 0.0,
            "median_route_ratio": None,
            "spl": 0.0,
            "task_spl": 0.0,
            "missing_prediction_count": 0,
            "prediction_exception_count": 0,
            "prediction_exception_rate": 0.0,
        }

    route_ratios = [row.route_ratio for row in rows if row.route_ratio is not None and row.route_success]
    hit_rows = [row for row in rows if row.hit_at_5 is not None]
    return {
        "total_cases": total,
        "destination_success": _mean_bool(row.destination_success for row in rows),
        "hit@5": _mean_bool(row.hit_at_5 for row in hit_rows) if hit_rows else None,
        "task_hit@5": _mean_float(_task_hit_at_5(row) for row in rows),
        "route_success": _mean_bool(row.route_success for row in rows),
        "median_route_ratio": median(route_ratios) if route_ratios else None,
        "spl": sum(row.spl for row in rows) / total,
        "task_spl": sum(_task_spl(row) for row in rows) / total,
        "missing_prediction_count": sum(1 for row in rows if "missing_prediction" in row.errors),
        "prediction_exception_count": sum(1 for row in rows if row.prediction_exception),
        "prediction_exception_rate": _mean_bool(row.prediction_exception for row in rows),
        **_summarize_extended_rows(rows),
    }


def _task_hit_at_5(row: UrbanNavEvalRow) -> float | None:
    """Return the task-aware hit@5 value for aggregate reporting.

    Args:
        row: Per-case evaluation row.

    Returns:
        Per-stop hit@5 for multi-stop cases, ordinary hit@5 for other cases,
        or ``None`` when no candidate list is available.
    """

    if row.category == "multi_stop_route":
        multi_stop_metrics = row.extended_metrics.get("multi_stop", {})
        value = multi_stop_metrics.get("stop_hit@5")
        if isinstance(value, (int, float)):
            return float(value)
        return None
    if row.hit_at_5 is None:
        return None
    return 1.0 if row.hit_at_5 else 0.0


def _task_spl(row: UrbanNavEvalRow) -> float:
    """Return the task-aware SPL value for aggregate reporting.

    Args:
        row: Per-case evaluation row.

    Returns:
        ``multi_stop_spl`` for ordered multi-stop cases, otherwise the generic
        single-route SPL stored on the row.
    """

    if row.category == "multi_stop_route":
        multi_stop_metrics = row.extended_metrics.get("multi_stop", {})
        value = multi_stop_metrics.get("multi_stop_spl")
        if isinstance(value, (int, float)):
            return float(value)
    return row.spl


def _prediction_exception(prediction: UrbanNavPrediction) -> tuple[bool, str | None]:
    """Return whether a prediction is a strict-run exception placeholder.

    Args:
        prediction: Saved prediction row.

    Returns:
        Pair ``(is_exception, detail)``. ``detail`` is the runner-provided
        reason when available.
    """

    trace = prediction.trace if isinstance(prediction.trace, dict) else {}
    if trace.get("error") != "prediction_exception":
        return False, None
    detail = trace.get("detail")
    return True, str(detail) if detail is not None else None


def summarize_by(rows: list[UrbanNavEvalRow], key: str) -> dict[str, dict[str, Any]]:
    """Aggregate summaries by a row attribute.

    Args:
        rows: Per-case evaluation rows.
        key: Attribute name, such as ``category`` or ``city``.

    Returns:
        Mapping from attribute value to ``summarize_rows`` output.
    """

    groups: dict[str, list[UrbanNavEvalRow]] = {}
    for row in rows:
        value = getattr(row, key, None) or "unknown"
        groups.setdefault(str(value), []).append(row)
    return {name: summarize_rows(group_rows) for name, group_rows in sorted(groups.items())}


def _row(case: UrbanNavCase, **kwargs: Any) -> UrbanNavEvalRow:
    """Build an evaluation row while copying slice metadata from the case.

    Args:
        case: GT case providing id/category/city metadata.
        **kwargs: Metric values and diagnostic fields for ``UrbanNavEvalRow``.

    Returns:
        Fully populated ``UrbanNavEvalRow``.
    """

    return UrbanNavEvalRow(
        case_id=case.case_id,
        category=case.category,
        subcategory=case.subcategory,
        city=case.city,
        scene_id=case.scene_id,
        difficulty=case.difficulty,
        **kwargs,
    )


def _reference_distance(case: UrbanNavCase, destination: UrbanNavDestination) -> float | None:
    """Resolve the frozen reference distance for a matched destination.

    Args:
        case: GT case.
        destination: Matched acceptable destination.

    Returns:
        Reference distance in meters, or ``None`` when unavailable.
    """

    if destination.reference_route_distance_m is not None:
        return destination.reference_route_distance_m
    if destination.reference_route:
        return route_distance_m(destination.reference_route)
    if destination.rank == 1:
        return case.gt_distance_m or route_distance_m(case.gt_route)
    return None


def _endpoint_error(route: list[UrbanNavPoint], destination: UrbanNavDestination) -> float | None:
    """Compute route endpoint error to the matched destination.

    Args:
        route: Predicted route polyline.
        destination: Matched acceptable destination.

    Returns:
        Endpoint error in meters, or ``None`` when route/destination coordinate
        is unavailable.
    """

    if not route:
        return None
    dest_point = destination.point
    if dest_point is None:
        return None
    return haversine_m(route[-1], dest_point)


def _route_min_error(route: list[UrbanNavPoint], destination: UrbanNavDestination) -> float | None:
    """Compute minimum distance from any route point to a destination.

    Args:
        route: Predicted route polyline.
        destination: Destination that should be visited along the route.

    Returns:
        Minimum route-to-destination error in meters, or ``None`` when missing.
    """

    dest_point = destination.point
    if not route or dest_point is None:
        return None
    return min(haversine_m(point, dest_point) for point in route)


def _summarize_extended_rows(rows: list[UrbanNavEvalRow]) -> dict[str, Any]:
    """Aggregate category-specific extension metrics.

    Args:
        rows: Per-case evaluation rows.

    Returns:
        Flat summary dictionary for multi-stop and route-update metrics.
    """

    multi_rows = [row.extended_metrics["multi_stop"] for row in rows if "multi_stop" in row.extended_metrics]
    update_rows = [row.extended_metrics["route_update"] for row in rows if "route_update" in row.extended_metrics]
    summary: dict[str, Any] = {}
    if multi_rows:
        summary.update(
            {
                "multi_stop_case_count": len(multi_rows),
                "multi_stop_sequence_success": _mean_bool(item.get("sequence_success") for item in multi_rows),
                "multi_stop_stop_recall": _mean_float(item.get("stop_recall") for item in multi_rows),
                "multi_stop_stop_hit@5": _mean_float(item.get("stop_hit@5") for item in multi_rows),
                "multi_stop_sequence_hit@5": _mean_bool(item.get("sequence_hit@5") for item in multi_rows),
                "multi_stop_candidate_coverage": _mean_float(item.get("candidate_coverage") for item in multi_rows),
                "multi_stop_leg_route_success": _mean_bool(item.get("leg_route_success") for item in multi_rows),
                "multi_stop_leg_route_success_rate": _mean_float(item.get("leg_route_success_rate") for item in multi_rows),
                "multi_stop_spl": _mean_float(item.get("multi_stop_spl") for item in multi_rows),
            }
        )
    if update_rows:
        summary.update(
            {
                "route_update_case_count": len(update_rows),
                "route_update_success": _mean_bool(item.get("route_update_success") for item in update_rows),
                "route_update_destination_success": _mean_bool(item.get("updated_destination_success") for item in update_rows),
                "route_update_route_success": _mean_bool(item.get("updated_route_success") for item in update_rows),
                "route_update_current_origin_used": _mean_bool(item.get("current_origin_used") for item in update_rows),
                "route_update_previous_destination_avoided": _mean_bool(item.get("previous_destination_avoided") for item in update_rows),
                "route_update_final_destination_preserved": _mean_bool(item.get("final_destination_preserved") for item in update_rows),
            }
        )
    return summary


def _is_multi_stop_case(case: UrbanNavCase) -> bool:
    """Return whether a case carries ordered multi-stop GT.

    Args:
        case: UrbanNav case.

    Returns:
        ``True`` for ``category=multi_stop_route`` or explicit multi-stop raw
        fields.
    """

    return (
        str(case.category or "").strip() == "multi_stop_route"
        or isinstance(case.raw.get("gt_destination_sequence"), list)
        or isinstance(case.raw.get("gt_route_legs"), list)
    )


def _is_route_update_case(case: UrbanNavCase) -> bool:
    """Return whether a case is a route-update/reroute task.

    Args:
        case: UrbanNav case.

    Returns:
        ``True`` when the category or metadata indicates route update.
    """

    return str(case.category or "").strip().startswith("route_update") or bool(_route_update_metadata(case))


def _route_update_metadata(case: UrbanNavCase) -> dict[str, Any]:
    """Return route-update metadata.

    Args:
        case: UrbanNav case.

    Returns:
        Metadata dictionary, or empty dictionary.
    """

    value = case.metadata.get("route_update")
    return dict(value) if isinstance(value, dict) else {}


def _route_update_requires_final(update_type: str) -> bool:
    """Return whether a route-update prediction should end at final destination.

    Args:
        update_type: Route-update action label.

    Returns:
        ``True`` for via/insert style updates, ``False`` for replacement
        updates.
    """

    return str(update_type or "").strip() in {"add_via", "insert_before_final"}


def _gt_destination_sequence(case: UrbanNavCase) -> list[UrbanNavDestination]:
    """Return GT destination sequence from raw multi-stop payload.

    Args:
        case: UrbanNav case.

    Returns:
        Ordered GT destinations.
    """

    sequence = case.raw.get("gt_destination_sequence")
    if not isinstance(sequence, list):
        return []
    return [UrbanNavDestination.from_dict(item) for item in sequence if isinstance(item, dict)]


def _gt_route_legs(case: UrbanNavCase) -> list[dict[str, Any]]:
    """Return raw GT route legs.

    Args:
        case: UrbanNav case.

    Returns:
        Ordered leg dictionaries.
    """

    legs = case.raw.get("gt_route_legs")
    return [dict(item) for item in legs if isinstance(item, dict)] if isinstance(legs, list) else []


def _list_dicts(value: Any) -> list[dict[str, Any]]:
    """Normalize a list-like value to dictionaries.

    Args:
        value: Raw value.

    Returns:
        List of dictionary items.
    """

    if not isinstance(value, list):
        value = [value]
    return [dict(item) for item in value if isinstance(item, dict)]


def _parse_leg_route(leg: dict[str, Any] | Any) -> list[UrbanNavPoint]:
    """Parse a route polyline from a route-leg dictionary.

    Args:
        leg: Raw predicted leg.

    Returns:
        Parsed route points.
    """

    if not isinstance(leg, dict):
        return []
    from mcp_server.eval.urbannav.urbannav_schema import parse_route

    return parse_route(leg.get("route") or leg.get("route_polyline") or [])


def _leg_candidate_destinations(leg: dict[str, Any] | Any) -> list[UrbanNavDestination]:
    """Parse ranked candidate destinations from a route-leg dictionary.

    Args:
        leg: Raw predicted leg.

    Returns:
        Candidate destinations in the model/reranker order.
    """

    if not isinstance(leg, dict):
        return []
    return [
        destination
        for destination in (UrbanNavDestination.from_dict(item) for item in _list_dicts(leg.get("candidate_destinations")))
        if destination is not None
    ]


def _leg_pred_distance(leg: dict[str, Any] | Any) -> float | None:
    """Extract predicted leg distance.

    Args:
        leg: Raw predicted leg.

    Returns:
        Distance in meters, or ``None``.
    """

    if not isinstance(leg, dict):
        return None
    summary = leg.get("route_summary") if isinstance(leg.get("route_summary"), dict) else {}
    return _float_or_none(leg.get("distance_m") or summary.get("distance_m") or summary.get("distance"))


def _leg_reference_distance(leg: dict[str, Any], matched_destination: UrbanNavDestination | None) -> float | None:
    """Resolve reference distance for one GT leg.

    Args:
        leg: Raw GT leg.
        matched_destination: Matched GT destination, if any.

    Returns:
        Reference route distance in meters, or ``None``.
    """

    if matched_destination is not None:
        if matched_destination.reference_route_distance_m is not None:
            return matched_destination.reference_route_distance_m
        if matched_destination.reference_route:
            return route_distance_m(matched_destination.reference_route)
    value = _float_or_none(leg.get("distance_m") or leg.get("reference_route_distance_m"))
    if value is not None:
        return value
    destination = leg.get("gt_destination")
    if isinstance(destination, dict):
        parsed = UrbanNavDestination.from_dict(destination)
        return parsed.reference_route_distance_m or route_distance_m(parsed.reference_route)
    return None


def _point_from_dict(value: Any) -> UrbanNavPoint | None:
    """Parse a point from a dictionary.

    Args:
        value: Raw point value.

    Returns:
        Parsed point, or ``None``.
    """

    if not isinstance(value, dict):
        return None
    try:
        return UrbanNavPoint.from_dict(value)
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    """Convert value to float when possible.

    Args:
        value: Raw value.

    Returns:
        Float, or ``None``.
    """

    try:
        return float(value)
    except Exception:
        return None


def _compatible_name_type(predicted: UrbanNavDestination, gt: UrbanNavDestination) -> bool:
    """Check whether coordinate fallback can compare two destinations.

    Args:
        predicted: Predicted destination.
        gt: Acceptable GT destination.

    Returns:
        ``True`` when GT name/type constraints are satisfied or stable ids are
        available; otherwise ``False``.
    """

    if gt.name and predicted.name and _normalize_text(gt.name) != _normalize_text(predicted.name):
        return False
    if gt.type and predicted.type and _normalize_text(gt.type) != _normalize_text(predicted.type):
        return False
    if gt.name and not predicted.name and not predicted.stable_ids:
        return False
    if gt.type and not predicted.type and not predicted.stable_ids:
        return False
    return True


def _normalize_text(value: str) -> str:
    """Normalize text for exact fallback comparisons.

    Args:
        value: Raw text.

    Returns:
        Lowercased whitespace-normalized text.
    """

    return " ".join(str(value).strip().lower().split())


def _mean_bool(values: Iterable[bool | None]) -> float:
    """Average boolean-like values.

    Args:
        values: Iterable of booleans or ``None`` values.

    Returns:
        Mean after mapping truthy values to 1.0 and falsy/``None`` values to
        0.0. Returns 0.0 for an empty iterable.
    """

    items = [1.0 if bool(value) else 0.0 for value in values]
    return sum(items) / len(items) if items else 0.0


def _mean_float(values: Iterable[Any]) -> float | None:
    """Compute mean over numeric values.

    Args:
        values: Numeric values, possibly containing ``None``.

    Returns:
        Mean value, or ``None`` when no numeric values exist.
    """

    items = [_float_or_none(value) for value in values]
    numeric = [value for value in items if value is not None]
    return sum(numeric) / len(numeric) if numeric else None
