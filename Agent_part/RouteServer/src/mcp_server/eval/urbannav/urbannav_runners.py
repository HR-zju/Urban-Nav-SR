"""Runners that map UrbanNav cases to prediction records."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
import math
from typing import Any, Iterable

from mcp_server.core.live_map_backend import LiveMapBackend, LiveRouteRequest
from mcp_server.core.replay_map_backend import ReplayMapBackend, ReplayPoiResult, ReplayRouteResult
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavCase, UrbanNavDestination, UrbanNavPoint, UrbanNavPrediction
from mcp_server.llm import CandidateReranker, GoalDecomposer, IntentParser


class UrbanNavRunner(ABC):
    """Base class for UrbanNav prediction runners.

    A runner is any deterministic or model-backed component that maps frozen
    ``UrbanNavCase`` objects to saved ``UrbanNavPrediction`` objects.
    """

    @abstractmethod
    def run_case(self, case: UrbanNavCase) -> UrbanNavPrediction:
        """Run one case.

        Args:
            case: Frozen UrbanNav GT case.

        Returns:
            Prediction record for the same case id.
        """


class OracleUrbanNavRunner(UrbanNavRunner):
    """Oracle runner for evaluator and pipeline smoke tests.

    This runner copies the rank-1 GT destination and GT route into prediction
    format. It is not a baseline model; it is a contract test tool.
    """

    def run_case(self, case: UrbanNavCase) -> UrbanNavPrediction:
        """Return an oracle prediction for one case.

        Args:
            case: Frozen UrbanNav GT case.

        Returns:
            ``UrbanNavPrediction`` matching the GT destination and route.
        """

        return UrbanNavPrediction(
            case_id=case.case_id,
            chosen_destination=case.gt_destination,
            candidate_destinations=list(case.ranked_destinations or case.acceptable_destinations),
            route_polyline=list(case.gt_route),
            route_distance_m=case.gt_distance_m,
            route_summary={"distance_m": case.gt_distance_m},
            trace={"runner": "oracle"},
        )


def run_urbannav_cases(runner: UrbanNavRunner, cases: Iterable[UrbanNavCase]) -> list[UrbanNavPrediction]:
    """Run a sequence of UrbanNav cases.

    Args:
        runner: Runner implementation.
        cases: Frozen GT cases.

    Returns:
        List of predictions in the same order as input cases.
    """

    return [runner.run_case(case) for case in cases]


@dataclass(frozen=True)
class RouteServerRunnerConfig:
    """Configuration for RouteServer-style UrbanNav runners.

    Args:
        search_radius_m: Fallback POI search radius when intent parsing does
            not provide one.
        candidate_limit: Maximum POI candidates to route and rank.
        mode: Fallback route mode.
        use_llm_intent: Whether ``IntentParser.parse`` may call an LLM. When
            false, the deterministic heuristic parser is used.
        use_llm_reranker: Whether ``CandidateReranker`` may call an LLM.
        selection_strategy: Candidate selection policy. ``"route_distance"``
            selects the shortest routeable candidate; ``"semantic_then_route"``
            applies semantic reranking first and selects the first routeable
            candidate in that order.
    """

    search_radius_m: int = 1500
    candidate_limit: int = 10
    mode: str = "foot"
    use_llm_intent: bool = False
    use_llm_reranker: bool = False
    selection_strategy: str = "route_distance"


class AsyncUrbanNavRunner(ABC):
    """Base class for async UrbanNav runners."""

    @abstractmethod
    async def run_case_async(self, case: UrbanNavCase) -> UrbanNavPrediction:
        """Run one case asynchronously.

        Args:
            case: Frozen UrbanNav GT case.

        Returns:
            Prediction record for the same case id.
        """


class RouteServerUrbanNavRunner(AsyncUrbanNavRunner):
    r"""RouteServer-style runner using intent parsing, POI search, and routing.

    Args:
        backend: Live map backend or replay adapter.
        config: Runner configuration.
        intent_parser: Optional parser; defaults to ``IntentParser``.
        reranker: Optional semantic reranker; defaults to ``CandidateReranker``.
        goal_decomposer: Optional multi-stop decomposer; defaults to
            ``GoalDecomposer``.

    For each case, the runner computes:

    \[
    \hat{I} = \operatorname{IntentParser}(q), \quad
    \hat{C} = \operatorname{POISearch}(o, \hat{I}), \quad
    \hat{R}_i = \operatorname{Route}(o, \hat{d}_i)
    \]

    The selected prediction is either the shortest routeable candidate or the
    first routeable candidate after optional semantic reranking.
    """

    def __init__(
        self,
        *,
        backend: Any,
        config: RouteServerRunnerConfig | None = None,
        intent_parser: IntentParser | None = None,
        reranker: CandidateReranker | None = None,
        goal_decomposer: GoalDecomposer | None = None,
    ) -> None:
        """Initialize the runner.

        Args:
            backend: Backend implementing UrbanNav POI search and route
                planning methods.
            config: Optional runner configuration.
            intent_parser: Optional intent parser.
            reranker: Optional candidate reranker.
            goal_decomposer: Optional ordered-goal decomposer.

        Returns:
            None.
        """

        self.backend = backend
        self.config = config or RouteServerRunnerConfig()
        self.intent_parser = intent_parser or IntentParser()
        self.reranker = reranker or CandidateReranker()
        self.goal_decomposer = goal_decomposer or GoalDecomposer()

    async def run_case_async(self, case: UrbanNavCase) -> UrbanNavPrediction:
        """Run one UrbanNav case through the RouteServer-style pipeline.

        Args:
            case: Frozen UrbanNav case.

        Returns:
            Prediction record with destination, candidates, route, and trace.
        """

        trace: dict[str, Any] = {"runner": "routeserver", "case_id": case.case_id}
        origin = _effective_origin(case)
        if origin is None:
            return UrbanNavPrediction(case_id=case.case_id, trace={**trace, "error": "missing_origin"})

        if _is_multi_stop_case(case):
            return await self._run_multi_stop_case_async(
                case=case,
                origin=origin,
                instruction=_effective_instruction(case),
                trace=trace,
            )

        instruction = _effective_instruction(case)
        intent = await self._parse_intent(instruction)
        trace["intent"] = intent
        poi_record = await self._search(case=case, origin=origin, intent=intent)
        candidates = list(poi_record.candidates[: self.config.candidate_limit])
        trace["poi_search"] = {
            "cache_key": poi_record.cache_key,
            "provider": poi_record.provider,
            "candidate_count": len(candidates),
        }
        if not candidates:
            return UrbanNavPrediction(case_id=case.case_id, trace={**trace, "error": "no_candidates"})

        candidates = await self._maybe_rerank(instruction=instruction, intent=intent, candidates=candidates, trace=trace)
        route_records = await self._route_candidates(case=case, origin=origin, candidates=candidates)
        selected, selected_route = _select_routeable_destination(
            candidates=candidates,
            route_records=route_records,
            selection_strategy=self.config.selection_strategy,
        )
        trace["routes"] = [
            {
                "cache_key": record.cache_key,
                "provider": record.provider,
                "distance_m": record.distance_m,
                "duration_s": record.duration_s,
                "has_polyline": bool(record.polyline),
                "error": record.response.get("error") if isinstance(record.response, dict) else None,
            }
            for record in route_records
        ]
        trace["route_update"] = _route_update_metadata(case)
        if selected is None or selected_route is None:
            return UrbanNavPrediction(
                case_id=case.case_id,
                candidate_destinations=candidates,
                trace={**trace, "error": "no_routeable_candidate"},
            )
        selected = _destination_with_route_distance(selected, selected_route)
        selected_route = await self._extend_route_update_route(
            case=case,
            selected=selected,
            selected_route=selected_route,
            trace=trace,
        )
        predicted_candidates = _attach_route_distances(candidates, route_records)
        return UrbanNavPrediction(
            case_id=case.case_id,
            chosen_destination=selected,
            candidate_destinations=predicted_candidates,
            route_polyline=list(selected_route.polyline),
            route_distance_m=selected_route.distance_m,
            route_summary={
                "distance_m": selected_route.distance_m,
                "duration_s": selected_route.duration_s,
                "mode": self.config.mode,
            },
            trace=trace,
        )

    async def _extend_route_update_route(
        self,
        *,
        case: UrbanNavCase,
        selected: UrbanNavDestination,
        selected_route: ReplayRouteResult,
        trace: dict[str, Any],
    ) -> ReplayRouteResult:
        """Extend via/insert route-update predictions to the final destination.

        Args:
            case: Current case.
            selected: Selected updated target. For via updates this is the
                inserted stop.
            selected_route: Route from current origin to ``selected``.
            trace: Mutable prediction trace.

        Returns:
            ``selected_route`` for replacement updates, or a concatenated
            current-origin -> via -> final route for add/insert updates.
        """

        update = _route_update_metadata(case)
        if not _route_update_requires_final(update):
            return selected_route
        final_destination = _destination_from_mapping(update.get("final_destination"))
        if final_destination is None or selected.point is None:
            trace["route_update_extension"] = {
                "status": "skipped",
                "reason": "missing_final_destination_or_selected_point",
            }
            return selected_route

        if isinstance(self.backend, ReplayUrbanNavBackendAdapter):
            final_route = self.backend.plan_route(
                case=case,
                destination=final_destination,
                origin=selected.point,
                route_phase="via_to_final",
            )
        else:
            try:
                final_route = await self.backend.plan_route(
                    cache_key=f"runner-route:{case.case_id}:via_to_final",
                    request=LiveRouteRequest(origin=selected.point, destination=final_destination, mode=self.config.mode),
                )
            except Exception as exc:
                trace["route_update_extension"] = {
                    "status": "failed",
                    "error": str(exc),
                }
                return selected_route

        if not final_route.polyline or final_route.distance_m is None or final_route.distance_m <= 0:
            trace["route_update_extension"] = {
                "status": "failed",
                "cache_key": final_route.cache_key,
                "error": final_route.response.get("error") if isinstance(final_route.response, dict) else None,
            }
            return selected_route

        trace["route_update_extension"] = {
            "status": "extended_to_final",
            "cache_key": final_route.cache_key,
            "final_destination": final_destination.to_dict(),
            "first_leg_distance_m": selected_route.distance_m,
            "final_leg_distance_m": final_route.distance_m,
        }
        return ReplayRouteResult(
            cache_key=f"{selected_route.cache_key}+{final_route.cache_key}",
            provider=selected_route.provider or final_route.provider,
            request={
                "origin": selected_route.request.get("origin"),
                "via_destination": selected.to_dict(),
                "final_destination": final_destination.to_dict(),
                "mode": self.config.mode,
            },
            response={"route_update_extension": "via_then_final"},
            polyline=_append_route(list(selected_route.polyline), list(final_route.polyline)),
            distance_m=float(selected_route.distance_m or 0.0) + float(final_route.distance_m or 0.0),
            duration_s=(
                float(selected_route.duration_s or 0.0) + float(final_route.duration_s or 0.0)
                if selected_route.duration_s is not None or final_route.duration_s is not None
                else None
            ),
        )

    async def _run_multi_stop_case_async(
        self,
        *,
        case: UrbanNavCase,
        origin: UrbanNavPoint,
        instruction: str,
        trace: dict[str, Any],
    ) -> UrbanNavPrediction:
        r"""Run an ordered multi-stop case through sequential planning.

        For decomposed goals \(g_1,\ldots,g_K\), each leg starts from the
        previous selected destination:

        \[
        \hat d_i = \operatorname{Select}(\operatorname{Route}(p_{i-1}, C_i))
        \]

        Args:
            case: Frozen UrbanNav case.
            origin: Initial route origin.
            instruction: Effective user instruction.
            trace: Mutable trace dictionary.

        Returns:
            Prediction containing a final destination, ordered destination
            sequence, route legs, and concatenated route polyline.
        """

        decomposition = await self._decompose_goals(instruction)
        goals = _normalized_goals(decomposition)
        trace["goal_decomposition"] = decomposition
        if len(goals) < 2:
            trace["multi_stop_warning"] = "decomposer_returned_single_goal"
        if not goals:
            return UrbanNavPrediction(case_id=case.case_id, trace={**trace, "error": "no_decomposed_goals"})

        current_origin = origin
        selected_sequence: list[UrbanNavDestination] = []
        predicted_candidates: list[UrbanNavDestination] = []
        route_legs: list[dict[str, Any]] = []
        full_route: list[UrbanNavPoint] = [origin]
        total_distance_m = 0.0

        for goal_position, goal in enumerate(goals, start=1):
            intent = _goal_to_intent(goal, fallback_mode=self.config.mode, fallback_radius_m=self.config.search_radius_m)
            poi_record = await self._search(case=case, origin=current_origin, intent=intent, stop_index=goal_position)
            candidates = [
                candidate
                for candidate in poi_record.candidates[: self.config.candidate_limit]
                if not _is_repeated_destination(candidate, selected_sequence)
            ]
            if not candidates:
                return UrbanNavPrediction(
                    case_id=case.case_id,
                    chosen_destination=selected_sequence[-1] if selected_sequence else None,
                    candidate_destinations=predicted_candidates,
                    destination_sequence=selected_sequence,
                    route_polyline=full_route,
                    route_distance_m=total_distance_m if total_distance_m > 0 else None,
                    route_summary={"distance_m": total_distance_m, "mode": self.config.mode},
                    route_legs=route_legs,
                    trace={**trace, "error": f"no_candidates_at_stop_{goal_position}"},
                )

            candidates = await self._maybe_rerank(
                instruction=str(goal.get("request_text") or instruction),
                intent=intent,
                candidates=candidates,
                trace=trace,
            )
            route_records = await self._route_candidates(
                case=case,
                origin=current_origin,
                candidates=candidates,
                stop_index=goal_position,
            )
            selected, selected_route = _select_routeable_destination(
                candidates=candidates,
                route_records=route_records,
                selection_strategy=self.config.selection_strategy,
            )
            ranked_candidates = _attach_route_distances(candidates, route_records)
            predicted_candidates.extend(ranked_candidates)
            route_legs.append(
                {
                    "leg_index": goal_position,
                    "goal": goal,
                    "origin": current_origin.to_dict(),
                    "poi_cache_key": poi_record.cache_key,
                    "candidate_destinations": [candidate.to_dict() for candidate in ranked_candidates],
                    "chosen_destination": selected.to_dict() if selected else None,
                    "route": [point.to_dict() for point in selected_route.polyline] if selected_route else [],
                    "route_summary": {
                        "distance_m": selected_route.distance_m if selected_route else None,
                        "duration_s": selected_route.duration_s if selected_route else None,
                    },
                }
            )
            if selected is None or selected_route is None or selected.point is None:
                return UrbanNavPrediction(
                    case_id=case.case_id,
                    chosen_destination=selected_sequence[-1] if selected_sequence else None,
                    candidate_destinations=predicted_candidates,
                    destination_sequence=selected_sequence,
                    route_polyline=full_route,
                    route_distance_m=total_distance_m if total_distance_m > 0 else None,
                    route_summary={"distance_m": total_distance_m, "mode": self.config.mode},
                    route_legs=route_legs,
                    trace={**trace, "error": f"no_routeable_candidate_at_stop_{goal_position}"},
                )
            selected = _destination_with_route_distance(selected, selected_route)
            route_legs[-1]["chosen_destination"] = selected.to_dict()
            route_legs[-1]["route_summary"] = {
                "distance_m": selected_route.distance_m,
                "duration_s": selected_route.duration_s,
            }
            selected_sequence.append(selected)
            full_route = _append_route(full_route, list(selected_route.polyline))
            total_distance_m += float(selected_route.distance_m or 0.0)
            current_origin = selected.point

        return UrbanNavPrediction(
            case_id=case.case_id,
            chosen_destination=selected_sequence[-1] if selected_sequence else None,
            candidate_destinations=predicted_candidates,
            destination_sequence=selected_sequence,
            route_polyline=full_route,
            route_distance_m=total_distance_m,
            route_summary={
                "distance_m": total_distance_m,
                "mode": self.config.mode,
                "goal_count": len(selected_sequence),
            },
            route_legs=route_legs,
            trace=trace,
        )

    async def _decompose_goals(self, instruction: str) -> dict[str, Any]:
        """Decompose an instruction into ordered retrievable goals.

        Args:
            instruction: User instruction.

        Returns:
            Goal decomposition dictionary.
        """

        if self.config.use_llm_intent:
            return await self.goal_decomposer.parse(instruction)
        parsed = self.goal_decomposer.heuristic_decompose(instruction)
        parsed["decomposition_source"] = "heuristic_runner"
        return parsed

    async def _parse_intent(self, instruction: str) -> dict[str, Any]:
        """Parse an instruction into provider search intent.

        Args:
            instruction: User instruction.

        Returns:
            Intent dictionary.
        """

        if self.config.use_llm_intent:
            return await self.intent_parser.parse(instruction)
        parsed = self.intent_parser.heuristic_parse(instruction)
        parsed["intent_source"] = "heuristic_runner"
        return parsed

    async def _search(
        self,
        *,
        case: UrbanNavCase,
        origin: UrbanNavPoint,
        intent: dict[str, Any],
        stop_index: int | None = None,
    ) -> ReplayPoiResult:
        """Search candidate destinations.

        Args:
            case: Current case.
            origin: Effective origin.
            intent: Parsed intent.
            stop_index: Optional one-based stop index for multi-stop replay.

        Returns:
            Replay-compatible POI result.
        """

        keywords = str(intent.get("search_keywords") or case.subcategory or case.instruction)
        radius_m = int(intent.get("radius_m") or self.config.search_radius_m)
        limit = int(intent.get("max_candidates") or self.config.candidate_limit)
        if isinstance(self.backend, ReplayUrbanNavBackendAdapter):
            return self.backend.search_destinations(case=case, stop_index=stop_index)
        return await self.backend.search_destinations(
            cache_key=f"runner-poi:{case.case_id}" if stop_index is None else f"runner-poi:{case.case_id}:stop{stop_index:02d}",
            origin=origin,
            keywords=keywords,
            radius_m=radius_m,
            limit=limit,
        )

    async def _maybe_rerank(
        self,
        *,
        instruction: str,
        intent: dict[str, Any],
        candidates: list[UrbanNavDestination],
        trace: dict[str, Any],
    ) -> list[UrbanNavDestination]:
        """Optionally rerank candidates by semantic relevance.

        Args:
            instruction: Effective instruction.
            intent: Parsed intent.
            candidates: Candidate destinations.
            trace: Mutable prediction trace.

        Returns:
            Candidate list, possibly reordered.
        """

        if not self.config.use_llm_reranker:
            return candidates
        candidate_payloads = [_candidate_for_reranker(item, index) for index, item in enumerate(candidates)]
        decision = await self.reranker.rerank(
            user_instruction=instruction,
            parsed_intent=intent,
            candidates=candidate_payloads,
        )
        trace["reranker"] = decision
        if not decision:
            return candidates
        by_id = {_candidate_id(item, index): item for index, item in enumerate(candidates)}
        ordered = [by_id[item] for item in decision.get("ranked_candidate_ids", []) if item in by_id]
        ordered.extend(item for item in candidates if item not in ordered)
        return ordered

    async def _route_candidates(
        self,
        *,
        case: UrbanNavCase,
        origin: UrbanNavPoint,
        candidates: list[UrbanNavDestination],
        stop_index: int | None = None,
    ) -> list[ReplayRouteResult]:
        """Plan routes to all candidates.

        Args:
            case: Current case.
            origin: Effective origin.
            candidates: Candidate destinations.
            stop_index: Optional one-based stop index for multi-stop route keys.

        Returns:
            Replay-compatible route records.
        """

        if isinstance(self.backend, ReplayUrbanNavBackendAdapter):
            return [
                self.backend.plan_route(
                    case=case,
                    destination=candidate,
                    origin=origin,
                    stop_index=stop_index,
                )
                for candidate in candidates
            ]
        records: list[ReplayRouteResult] = []
        for index, candidate in enumerate(candidates, start=1):
            cache_key = (
                f"runner-route:{case.case_id}:{index:03d}"
                if stop_index is None
                else f"runner-route:{case.case_id}:stop{stop_index:02d}:{index:03d}"
            )
            try:
                records.append(
                    await self.backend.plan_route(
                        cache_key=cache_key,
                        request=LiveRouteRequest(origin=origin, destination=candidate, mode=self.config.mode),
                    )
                )
            except Exception as exc:
                records.append(
                    ReplayRouteResult(
                        cache_key=cache_key,
                        provider=getattr(self.backend, "provider_name", None),
                        request={"origin": origin.to_dict(), "destination": candidate.to_dict(), "mode": self.config.mode},
                        response={"error": str(exc)},
                    )
                )
        return records


class ReplayUrbanNavBackendAdapter:
    """Replay backend adapter for RouteServer-style runners.

    Args:
        replay_backend: Loaded replay cache backend.

    The adapter intentionally ignores live search arguments and reads the
    frozen cache keys from each case. This gives a deterministic smoke runner
    that exercises intent parsing, candidate ranking, and route selection
    without changing the benchmark state.
    """

    def __init__(self, replay_backend: ReplayMapBackend, *, origin_tolerance_m: float = 5.0) -> None:
        """Initialize the adapter.

        Args:
            replay_backend: Replay cache backend.
            origin_tolerance_m: Maximum stored-request origin error accepted
                for replay route lookup.

        Returns:
            None.
        """

        self.replay_backend = replay_backend
        self.origin_tolerance_m = max(0.0, float(origin_tolerance_m))

    def search_destinations(self, *, case: UrbanNavCase, stop_index: int | None = None) -> ReplayPoiResult:
        """Return frozen POI candidates for a case.

        Args:
            case: UrbanNav case with POI cache keys.
            stop_index: Optional one-based stop index for multi-stop cases.

        Returns:
            Frozen POI result.
        """

        keys = list(case.cache_keys.get("poi_search") or [])
        if not keys:
            return ReplayPoiResult(cache_key="", provider=case.optimality_scope.poi_provider, candidates=[])
        if stop_index is not None:
            if 1 <= stop_index <= len(keys):
                return self.replay_backend.search_destinations(cache_key=str(keys[stop_index - 1]))
            return ReplayPoiResult(cache_key="", provider=case.optimality_scope.poi_provider, candidates=[])
        selected_key = _select_poi_cache_key_for_case(case=case, keys=[str(key) for key in keys])
        return self.replay_backend.search_destinations(cache_key=selected_key)

    def plan_route(
        self,
        *,
        case: UrbanNavCase,
        destination: UrbanNavDestination,
        origin: UrbanNavPoint | None = None,
        stop_index: int | None = None,
        route_phase: str | None = None,
    ) -> ReplayRouteResult:
        """Return the frozen route that corresponds to a candidate.

        Args:
            case: UrbanNav case with route cache keys.
            destination: Candidate destination.
            origin: Effective route origin. Route-update and multi-stop replay
                use this to avoid reusing an old leg for the same candidate.
            stop_index: Optional one-based stop index for multi-stop cases.
            route_phase: Optional explicit route phase such as
                ``"via_to_final"``.

        Returns:
            Matching frozen route, or an error route record if absent.
        """

        destination_ids = destination.stable_ids
        route_keys = _route_cache_keys_for_case(
            case=case,
            keys=[str(key) for key in case.cache_keys.get("routes") or []],
            stop_index=stop_index,
            route_phase=route_phase,
        )
        destination_matches: list[ReplayRouteResult] = []
        for key in route_keys:
            try:
                record = self.replay_backend.plan_route(cache_key=key)
            except KeyError:
                continue
            request_destination = UrbanNavDestination.from_dict(dict(record.request.get("destination") or {}))
            if destination_ids and request_destination.stable_ids & destination_ids:
                destination_matches.append(record)
        if destination_matches:
            matched = _best_origin_matched_route(
                destination_matches,
                origin=origin,
                origin_tolerance_m=self.origin_tolerance_m,
            )
            if matched is not None:
                return matched
            return ReplayRouteResult(
                cache_key=f"missing-origin-route:{case.case_id}:{destination.provider_poi_id or destination.name or 'unknown'}",
                provider=case.optimality_scope.route_provider,
                request={
                    "origin": origin.to_dict() if origin is not None else None,
                    "destination": destination.to_dict(),
                },
                response={"error": "route_cache_origin_not_found"},
            )
        return ReplayRouteResult(
            cache_key=f"missing-route:{case.case_id}:{destination.provider_poi_id or destination.name or 'unknown'}",
            provider=case.optimality_scope.route_provider,
            request={"destination": destination.to_dict()},
            response={"error": "route_cache_key_not_found"},
        )


def run_urbannav_cases_async(runner: AsyncUrbanNavRunner, cases: Iterable[UrbanNavCase]) -> list[UrbanNavPrediction]:
    """Run an async runner over cases from synchronous code.

    Args:
        runner: Async runner implementation.
        cases: Frozen GT cases.

    Returns:
        List of predictions in input order.
    """

    return asyncio.run(_run_urbannav_cases_async(runner, list(cases)))


async def _run_urbannav_cases_async(runner: AsyncUrbanNavRunner, cases: list[UrbanNavCase]) -> list[UrbanNavPrediction]:
    """Run an async runner over cases.

    Args:
        runner: Async runner implementation.
        cases: Frozen GT cases.

    Returns:
        List of predictions in input order.
    """

    return [await runner.run_case_async(case) for case in cases]


def _effective_instruction(case: UrbanNavCase) -> str:
    """Return the instruction used for prediction.

    Args:
        case: UrbanNav case.

    Returns:
        Route-update instruction when present, otherwise the case instruction.
    """

    update = _route_update_metadata(case)
    return str(update.get("updated_instruction") or case.instruction)


def _effective_origin(case: UrbanNavCase) -> UrbanNavPoint | None:
    """Return the origin used for prediction.

    Args:
        case: UrbanNav case.

    Returns:
        Route-update current origin when present, otherwise case origin.
    """

    update = _route_update_metadata(case)
    current = update.get("current_origin")
    if isinstance(current, dict):
        try:
            return UrbanNavPoint.from_dict(current)
        except Exception:
            pass
    return case.origin


def _route_update_metadata(case: UrbanNavCase) -> dict[str, Any]:
    """Return route-update metadata.

    Args:
        case: UrbanNav case.

    Returns:
        Route-update metadata dictionary, or an empty dictionary.
    """

    update = case.metadata.get("route_update")
    return dict(update) if isinstance(update, dict) else {}


def _select_poi_cache_key_for_case(*, case: UrbanNavCase, keys: list[str]) -> str:
    """Select the replay POI cache key for a case phase.

    Args:
        case: UrbanNav case with metadata.
        keys: Ordered POI cache keys from the case.

    Returns:
        Cache key to use for prediction search.

    Route-update cases store two independent POI searches: ``previous`` for
    the pre-update destination and ``update`` for the user's new target. The
    runner prediction must search the updated target from the current origin,
    so it prefers the ``update`` phase. Normal single-target cases preserve
    the historical first-key behavior.
    """

    if not keys:
        return ""
    if _route_update_metadata(case):
        for key in keys:
            if _cache_key_has_phase(key, "update"):
                return key
        for key in reversed(keys):
            if not _cache_key_has_phase(key, "previous"):
                return key
        return keys[-1]
    return keys[0]


def _route_cache_keys_for_case(
    *,
    case: UrbanNavCase,
    keys: list[str],
    stop_index: int | None,
    route_phase: str | None = None,
) -> list[str]:
    """Select candidate route cache keys for the active prediction phase.

    Args:
        case: UrbanNav case.
        keys: Ordered route cache keys from the case.
        stop_index: Optional one-based multi-stop leg index.
        route_phase: Optional explicit route phase.

    Returns:
        Route cache keys narrowed to the active leg/update phase when possible.
    """

    if not keys:
        return []
    if route_phase:
        scoped = [key for key in keys if _cache_key_has_phase(key, route_phase)]
        if scoped:
            return scoped
    if stop_index is not None:
        phase = f"stop{stop_index:02d}"
        scoped = [key for key in keys if _cache_key_has_phase(key, phase)]
        return scoped or keys
    if _route_update_metadata(case):
        scoped = [key for key in keys if _cache_key_has_phase(key, "update")]
        if scoped:
            return scoped
    return keys


def _cache_key_has_phase(cache_key: str, phase: str) -> bool:
    """Return whether a colon-delimited cache key contains a phase token.

    Args:
        cache_key: Replay cache key.
        phase: Phase token such as ``"update"`` or ``"stop02"``.

    Returns:
        ``True`` when the key contains the exact phase token.
    """

    return phase in [item.strip() for item in str(cache_key).split(":")]


def _route_update_requires_final(update: dict[str, Any]) -> bool:
    """Return whether a route update must continue to the final destination.

    Args:
        update: Route-update metadata dictionary.

    Returns:
        ``True`` for add/insert updates where the selected updated target is a
        via point rather than the final endpoint.
    """

    update_type = str(update.get("update_type") or "").strip()
    return update_type in {"add_via", "insert_before_final"}


def _destination_from_mapping(value: Any) -> UrbanNavDestination | None:
    """Parse a destination from a mapping value.

    Args:
        value: Candidate destination payload.

    Returns:
        Parsed destination or ``None``.
    """

    if not isinstance(value, dict):
        return None
    try:
        return UrbanNavDestination.from_dict(value)
    except Exception:
        return None


def _best_origin_matched_route(
    records: list[ReplayRouteResult],
    *,
    origin: UrbanNavPoint | None,
    origin_tolerance_m: float = 5.0,
) -> ReplayRouteResult | None:
    """Choose the route whose stored request origin best matches ``origin``.

    Args:
        records: Route cache records already matched by destination.
        origin: Effective route origin.
        origin_tolerance_m: Maximum accepted request-origin error.

    Returns:
        Best route record, or ``None`` when no stored request origin is within
        tolerance. If no origin is provided, the first destination match is
        returned.
    """

    if not records:
        return None
    if origin is None:
        return records[0]

    scored: list[tuple[float, ReplayRouteResult]] = []
    for record in records:
        request_origin = _point_from_mapping(record.request.get("origin"))
        if request_origin is None:
            continue
        scored.append((_distance_m(origin, request_origin), record))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    if scored[0][0] > max(0.0, origin_tolerance_m):
        return None
    return scored[0][1]


def _point_from_mapping(value: Any) -> UrbanNavPoint | None:
    """Parse a point from a mapping-like value.

    Args:
        value: Candidate point payload.

    Returns:
        Parsed point or ``None``.
    """

    if not isinstance(value, dict):
        return None
    try:
        return UrbanNavPoint.from_dict(value)
    except Exception:
        return None


def _distance_m(a: UrbanNavPoint, b: UrbanNavPoint) -> float:
    """Compute great-circle distance in meters between two points.

    Args:
        a: First point.
        b: Second point.

    Returns:
        Approximate WGS84 distance in meters.
    """

    radius_m = 6_371_000.0
    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    dlat = math.radians(b.lat - a.lat)
    dlng = math.radians(b.lng - a.lng)
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * radius_m * math.asin(math.sqrt(hav))


def _is_multi_stop_case(case: UrbanNavCase) -> bool:
    """Return whether a case should be run as ordered multi-stop navigation.

    Args:
        case: UrbanNav case.

    Returns:
        ``True`` for ``category=multi_stop_route`` or rows carrying a
        destination sequence / route legs.
    """

    return (
        str(case.category or "").strip() == "multi_stop_route"
        or isinstance(case.raw.get("gt_destination_sequence"), list)
        or isinstance(case.raw.get("gt_route_legs"), list)
    )


def _normalized_goals(decomposition: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize decomposed goals into a small ordered list.

    Args:
        decomposition: Output from ``GoalDecomposer``.

    Returns:
        Ordered goal dictionaries.
    """

    goals = decomposition.get("goals")
    if not isinstance(goals, list):
        return []
    output = [dict(goal) for goal in goals if isinstance(goal, dict)]
    output.sort(key=lambda item: int(item.get("goal_index", len(output)) or 0))
    return output[:4]


def _goal_to_intent(goal: dict[str, Any], *, fallback_mode: str, fallback_radius_m: int) -> dict[str, Any]:
    """Convert one decomposed goal into RouteServer search intent.

    Args:
        goal: Decomposed goal dictionary.
        fallback_mode: Route mode fallback.
        fallback_radius_m: Search radius fallback.

    Returns:
        Intent dictionary usable by existing search/routing methods.
    """

    return {
        "destination_type": goal.get("destination_type") or goal.get("goal_type") or "poi",
        "destination_kind": goal.get("destination_kind") or "generic_poi",
        "search_keywords": goal.get("search_keywords") or goal.get("keywords") or goal.get("request_text") or "poi",
        "mode": goal.get("mode") or fallback_mode,
        "radius_m": int(goal.get("radius_m") or fallback_radius_m),
        "optimize_for": goal.get("optimize_for") or "distance",
        "max_candidates": goal.get("max_candidates"),
        "hard_constraints": goal.get("hard_constraints") or {},
    }


def _is_repeated_destination(candidate: UrbanNavDestination, selected: list[UrbanNavDestination]) -> bool:
    """Check whether a candidate repeats an earlier selected stop.

    Args:
        candidate: Candidate destination for the current leg.
        selected: Previously selected destinations.

    Returns:
        ``True`` when stable ids match or coordinates are nearly identical.
    """

    candidate_ids = candidate.stable_ids
    candidate_point = candidate.point
    for existing in selected:
        if candidate_ids and candidate_ids.intersection(existing.stable_ids):
            return True
        existing_point = existing.point
        if candidate_point is None or existing_point is None:
            continue
        if abs(candidate_point.lat - existing_point.lat) < 2e-5 and abs(candidate_point.lng - existing_point.lng) < 2e-5:
            return True
    return False


def _append_route(base: list[UrbanNavPoint], leg: list[UrbanNavPoint]) -> list[UrbanNavPoint]:
    """Append a route leg while removing one adjacent duplicate endpoint.

    Args:
        base: Existing full route.
        leg: Next route leg.

    Returns:
        Concatenated route.
    """

    if not leg:
        return list(base)
    if not base:
        return list(leg)
    if abs(base[-1].lat - leg[0].lat) < 1e-9 and abs(base[-1].lng - leg[0].lng) < 1e-9:
        return [*base, *leg[1:]]
    return [*base, *leg]


def _select_routeable_destination(
    *,
    candidates: list[UrbanNavDestination],
    route_records: list[ReplayRouteResult],
    selection_strategy: str,
) -> tuple[UrbanNavDestination | None, ReplayRouteResult | None]:
    """Select one routeable destination.

    Args:
        candidates: Candidate destinations in semantic order.
        route_records: Route records aligned with candidates.
        selection_strategy: Selection strategy.

    Returns:
        Tuple of selected destination and selected route.
    """

    pairs = [
        (candidate, route)
        for candidate, route in zip(candidates, route_records)
        if route.distance_m is not None and route.distance_m > 0 and route.polyline
    ]
    if not pairs:
        return None, None
    if selection_strategy == "semantic_then_route":
        return pairs[0]
    return min(pairs, key=lambda item: float(item[1].distance_m or float("inf")))


def _attach_route_distances(
    candidates: list[UrbanNavDestination],
    route_records: list[ReplayRouteResult],
) -> list[UrbanNavDestination]:
    """Attach predicted route distances to candidate destinations.

    Args:
        candidates: Candidate destinations.
        route_records: Route records aligned with candidates.

    Returns:
        Candidate destinations with ``route_distance_m`` populated when known.
    """

    output: list[UrbanNavDestination] = []
    for index, candidate in enumerate(candidates):
        payload = candidate.to_dict()
        if index < len(route_records) and route_records[index].distance_m is not None:
            payload["route_distance_m"] = route_records[index].distance_m
        payload["rank"] = index + 1
        output.append(UrbanNavDestination.from_dict(payload))
    return output


def _destination_with_route_distance(
    destination: UrbanNavDestination,
    route_record: ReplayRouteResult,
) -> UrbanNavDestination:
    """Return a destination copy with selected route distance attached.

    Args:
        destination: Selected destination.
        route_record: Route record selected for the destination.

    Returns:
        Destination with ``route_distance_m`` populated from the route record.
    """

    payload = destination.to_dict()
    if route_record.distance_m is not None:
        payload["route_distance_m"] = route_record.distance_m
    return UrbanNavDestination.from_dict(payload)


def _candidate_for_reranker(candidate: UrbanNavDestination, index: int) -> dict[str, Any]:
    """Build a reranker payload for one candidate.

    Args:
        candidate: Candidate destination.
        index: Zero-based candidate index.

    Returns:
        JSON-ready candidate payload.
    """

    return {
        "id": _candidate_id(candidate, index),
        "name": candidate.name,
        "type": candidate.type,
        "distance_m": candidate.route_distance_m,
        "candidate_kind": candidate.type,
        "is_concrete_destination": bool(candidate.point),
    }


def _candidate_id(candidate: UrbanNavDestination, index: int) -> str:
    """Return a stable candidate id for reranking.

    Args:
        candidate: Candidate destination.
        index: Zero-based candidate index.

    Returns:
        Candidate id string.
    """

    return str(candidate.provider_poi_id or candidate.place_id or f"candidate-{index}")
