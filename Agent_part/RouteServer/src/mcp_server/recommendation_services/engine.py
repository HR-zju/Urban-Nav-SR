"""Engine responsibilities for destination recommendation."""
from __future__ import annotations

from datetime import datetime
from typing import Any
import os
from time import perf_counter
from mcp_server.config import default_intent_rules, split_search_keywords
from mcp_server.llm import IntentParser, rerank_candidates_with_llm, translate_instruction_for_cn_map, translate_result_bundle_to_english
from mcp_server.map_providers import MapProviderClient, parse_location_text
from mcp_server.recommendation_services.infrastructure import _LOCALITY_HINT_CACHE, _NAMED_GEOCODE_CACHE, _PARSE_CACHE, _SEMANTIC_SEARCH_CACHE, _cache_get, _cache_put, _ctx_info, _ctx_report_progress, _ctx_warning, _log_timing, _run_limited, _stable_cache_key
from mcp_server.recommendation_services.policy import _annotate_candidate, _candidate_priority, _candidate_structured_text, _candidate_text_blob, _candidate_unavailable_reason, _collect_query_terms, _estimated_route_distance_m, _estimated_route_duration_s, _is_explicit_destination_request, _looks_english_instruction, _named_place_queries, _parsed_confidence, _parsed_named_entity_confidence, _query_affinity_score, _resolve_destination_kind, _scope_limited_radius_m, _semantic_candidate_bias, _session_local_via_bias, _should_enable_llm_rerank, _should_use_named_place_geocode
from mcp_server.recommendation_services.presentation import _apply_english_translations, _recommendation_debug_snapshot
from mcp_server.recommendation_services.regional import _expand_regional_scenic_area_candidates
from mcp_server.recommendation_services.candidates import _extract_locality_hints, _normalize_named_place_candidates, _normalize_semantic_poi_candidates
from mcp_server.recommendation_services.ranking import _operationalize_hard_constraints, apply_hard_constraints, clamp_score, keyword_match_score, soft_rank
from mcp_server.recommendation_services.geometry import extract_key_waypoints, extract_waypoint_sequence, haversine_m

class DestinationRecommendationEngine:
    def __init__(
        self,
        map_client: MapProviderClient,
        provider: str,
        intent_parser: IntentParser | None = None,
    ) -> None:
        self.map_client = map_client
        self.provider = provider
        self.intent_parser = intent_parser or IntentParser()

    async def recommend(
        self,
        user_instruction: str,
        current_latitude: float,
        current_longitude: float,
        max_candidates: int = 5,
        search_radius: float = 3000,
        mode: str | None = None,
        resolution_scope: str = "global_destination",
        anchor_location: str | None = None,
        allow_llm_rerank: bool | None = None,
        compute_chosen_route: bool | None = None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        overall_started_at = perf_counter()
        eval_limit = int(os.getenv("ROUTE_EVAL_LIMIT", "25"))
        if self.provider == "amap":
            eval_limit = min(eval_limit, int(os.getenv("ROUTE_EVAL_LIMIT_AMAP", "8")))
        normalized_scope = str(resolution_scope or "global_destination").strip().lower()
        if normalized_scope not in {"global_destination", "session_local_via"}:
            normalized_scope = "global_destination"
        if normalized_scope == "session_local_via":
            eval_limit = min(eval_limit, int(os.getenv("ROUTE_EVAL_LIMIT_LOCAL_VIA", "8") or 8))
        default_opt_target = os.getenv("DEFAULT_OPTIMIZE_FOR", "distance").strip() or "distance"

        original_instruction = user_instruction
        working_instruction = user_instruction
        translated_instruction = None
        english_localization_active = False
        if self.provider in {"amap", "baidu"} and _looks_english_instruction(user_instruction):
            localization_started_at = perf_counter()
            translated_instruction = await translate_instruction_for_cn_map(
                user_instruction=user_instruction,
                provider=self.provider,
            )
            localization_elapsed = perf_counter() - localization_started_at
            if translated_instruction:
                working_instruction = translated_instruction
                english_localization_active = True
                await _ctx_info(
                    ctx,
                    f"Localized English instruction for {self.provider} in {localization_elapsed:.2f}s: {translated_instruction}",
                )
                _log_timing(
                    "llm_translate_instruction",
                    localization_elapsed,
                    provider=self.provider,
                    status="ok",
                )
            else:
                await _ctx_warning(
                    ctx,
                    f"English instruction localization for {self.provider} failed after {localization_elapsed:.2f}s; continuing with original instruction.",
                )
                _log_timing(
                    "llm_translate_instruction",
                    localization_elapsed,
                    provider=self.provider,
                    status="failed",
                )

        parse_started_at = perf_counter()
        parse_cache_key = _stable_cache_key(["intent_parse", self.provider, working_instruction])
        cached_parse = _cache_get(_PARSE_CACHE, parse_cache_key)
        if cached_parse is not None:
            parsed = cached_parse
            _log_timing("cache_hit", 0.0, provider=self.provider, cache_kind="intent_parse")
        else:
            parsed = await self.intent_parser.parse(working_instruction)
            _cache_put(_PARSE_CACHE, parse_cache_key, parsed)
        parse_elapsed = perf_counter() - parse_started_at
        await _ctx_info(ctx, f"Intent parse completed in {parse_elapsed:.2f}s.")
        _log_timing("llm_intent_parse", parse_elapsed, provider=self.provider)
        intent_source = str(parsed.get("intent_source") or "")
        if intent_source.startswith("heuristic_llm_"):
            error_detail = str(parsed.get("llm_error") or "").strip()
            warning = f"LLM parse fallback activated: {intent_source}"
            if error_detail:
                warning = f"{warning}; detail={error_detail}"
            await _ctx_warning(ctx, warning)

        search_keywords = parsed.get("search_keywords", "")
        keywords = parsed.get("keywords") or split_search_keywords(search_keywords)
        hard_constraints = _operationalize_hard_constraints(
            working_instruction,
            parsed=parsed,
            hard_constraints=parsed.get("hard_constraints") or default_intent_rules()["default_hard_constraints"],
        )

        parsed_mode = str(parsed.get("mode") or "car").strip().lower()
        user_mode = str(mode or "").strip().lower()
        chosen_mode = user_mode if user_mode in {"car", "foot", "bike"} else parsed_mode
        if chosen_mode not in {"car", "foot", "bike"}:
            chosen_mode = "car"

        parsed_radius = parsed.get("radius_m")
        if not isinstance(parsed_radius, (int, float)) or parsed_radius <= 0:
            parsed_radius = 3000
        user_radius = search_radius if isinstance(search_radius, (int, float)) and search_radius > 0 else None
        radius = float(user_radius if user_radius is not None else parsed_radius)

        optimize_for = parsed.get("optimize_for", default_opt_target)

        parsed_top_k = parsed.get("max_candidates")
        if not isinstance(parsed_top_k, (int, float)) or parsed_top_k < 1:
            parsed_top_k = 5
        user_top_k = max_candidates if isinstance(max_candidates, (int, float)) and max_candidates >= 1 else None
        top_k = int(user_top_k if user_top_k is not None else parsed_top_k)
        top_k = max(1, top_k)

        semantic_keywords = search_keywords or "|".join(keywords)
        search_longitude = float(current_longitude)
        search_latitude = float(current_latitude)
        search_anchor_location = None
        if anchor_location:
            try:
                search_longitude, search_latitude = parse_location_text(anchor_location)
                search_anchor_location = f"{float(search_longitude)},{float(search_latitude)}"
            except Exception:
                search_longitude = float(current_longitude)
                search_latitude = float(current_latitude)
                search_anchor_location = None
        llm_rerank_requested = allow_llm_rerank
        if compute_chosen_route is None:
            compute_chosen_route = normalized_scope == "global_destination"
        radius = _scope_limited_radius_m(radius, mode=chosen_mode, resolution_scope=normalized_scope)
        origin_location = f"{search_longitude},{search_latitude}"
        poi_limit = max(top_k, min(eval_limit, 25))
        destination_kind = _resolve_destination_kind(working_instruction, parsed)
        query_terms = _collect_query_terms(working_instruction, parsed)
        parsed["destination_kind"] = destination_kind
        explicit_destination = _is_explicit_destination_request(working_instruction, parsed)
        parsed_confidence = _parsed_confidence(parsed)
        named_entity_confidence = _parsed_named_entity_confidence(
            user_instruction=working_instruction,
            parsed=parsed,
        )
        parsed["confidence"] = parsed_confidence
        parsed["named_entity_confidence"] = named_entity_confidence
        if translated_instruction:
            parsed["provider_instruction_zh"] = translated_instruction
            parsed["original_user_instruction"] = original_instruction
        rerank_meta: dict[str, Any] | None = None

        await _ctx_info(
            ctx,
            f"Provider={self.provider}, mode={chosen_mode}, radius={radius}m, search_keywords={semantic_keywords}, destination_kind={destination_kind}, resolution_scope={normalized_scope}",
        )

        normalized_candidates: list[dict[str, Any]] = []
        retrieval_errors: list[dict[str, str]] = []
        retrieval_started_at = perf_counter()
        use_named_place_geocode = _should_use_named_place_geocode(
            resolution_scope=normalized_scope,
            explicit_destination=explicit_destination,
            destination_kind=destination_kind,
            named_entity_confidence=named_entity_confidence,
        )
        _log_timing(
            "recommendation_execution_plan",
            0.0,
            provider=self.provider,
            resolution_scope=normalized_scope,
            explicit_destination=explicit_destination,
            parsed_confidence=round(parsed_confidence, 4),
            named_entity_confidence=round(named_entity_confidence, 4),
            named_geocode_enabled=use_named_place_geocode,
            chosen_route_enabled=bool(compute_chosen_route),
        )
        if use_named_place_geocode:
            locality_hints: list[str] = []
            locality_started_at = perf_counter()
            locality_cache_key = _stable_cache_key(
                ["locality_hint", self.provider, round(search_latitude, 6), round(search_longitude, 6)]
            )
            cached_locality_hints = _cache_get(_LOCALITY_HINT_CACHE, locality_cache_key)
            if cached_locality_hints is not None:
                locality_hints = cached_locality_hints
                _log_timing("cache_hit", 0.0, provider=self.provider, cache_kind="locality_hint")
            else:
                try:
                    locality_hints = _extract_locality_hints(
                        await self.map_client.reverse_geocode(search_latitude, search_longitude)
                    )
                except Exception:
                    locality_hints = []
                _cache_put(_LOCALITY_HINT_CACHE, locality_cache_key, locality_hints)
            _log_timing(
                "map_reverse_geocode_locality_hint",
                perf_counter() - locality_started_at,
                provider=self.provider,
                locality_hint_count=len(locality_hints),
            )

            named_query_started_at = perf_counter()
            queries = _named_place_queries(
                semantic_keywords,
                user_instruction=working_instruction,
                locality_hints=locality_hints,
            )
            _log_timing(
                "named_place_query_build",
                perf_counter() - named_query_started_at,
                provider=self.provider,
                query_count=len(queries),
            )
            await _ctx_info(ctx, f"Named-place retrieval via geocode, queries={queries}")
            geocode_concurrency = int(os.getenv("GEOCODE_CONCURRENCY", "4") or 4)
            geocode_started_at = perf_counter()
            geocode_cache_key = _stable_cache_key(["named_geocode", self.provider, queries])
            cached_geocode_hits = _cache_get(_NAMED_GEOCODE_CACHE, geocode_cache_key)
            if cached_geocode_hits is not None:
                geocode_hits = cached_geocode_hits
                _log_timing("cache_hit", 0.0, provider=self.provider, cache_kind="named_geocode")
            else:
                geocode_results = await _run_limited(
                    [self.map_client.geocode(query) for query in queries],
                    concurrency=geocode_concurrency,
                )
                geocode_hits = []
                for result in geocode_results:
                    if isinstance(result, Exception):
                        continue
                    geocode_hits.extend(result)
                _cache_put(_NAMED_GEOCODE_CACHE, geocode_cache_key, geocode_hits)
            _log_timing(
                "map_named_place_geocode",
                perf_counter() - geocode_started_at,
                provider=self.provider,
                query_count=len(queries),
                geocode_hit_count=len(geocode_hits),
            )
            normalize_named_started_at = perf_counter()
            normalized_candidates = _normalize_named_place_candidates(
                geocode_hits,
                current_latitude=search_latitude,
                current_longitude=search_longitude,
                max_direct_distance_m=max(float(radius) * 1.5, 5000.0),
            )
            normalized_candidates = [
                _annotate_candidate(candidate, destination_kind=destination_kind)
                for candidate in normalized_candidates
            ]
            for candidate in normalized_candidates:
                candidate["query_affinity"] = _query_affinity_score(candidate, query_terms=query_terms)
            _log_timing(
                "normalize_named_place_candidates",
                perf_counter() - normalize_named_started_at,
                provider=self.provider,
                candidate_count=len(normalized_candidates),
            )
            if destination_kind == "regional_scenic_area":
                regional_grounding_started_at = perf_counter()
                grounded_candidates = await _expand_regional_scenic_area_candidates(
                    base_candidates=normalized_candidates,
                    map_client=self.map_client,
                    provider=self.provider,
                    current_latitude=search_latitude,
                    current_longitude=search_longitude,
                    radius=radius,
                    query_terms=query_terms,
                    ctx=ctx,
                )
                _log_timing(
                    "regional_scenic_grounding",
                    perf_counter() - regional_grounding_started_at,
                    provider=self.provider,
                    input_candidate_count=len(normalized_candidates),
                    grounded_candidate_count=len(grounded_candidates),
                )
                if grounded_candidates:
                    normalized_candidates = grounded_candidates
                elif normalized_candidates and not any(
                    candidate.get("is_concrete_destination")
                    and candidate.get("candidate_kind") in {"scenic_destination", "named_landmark", "specific_poi"}
                    for candidate in normalized_candidates
                ):
                    await _ctx_info(
                        ctx,
                        "Named-place grounding produced only non-concrete regional candidates; fallback to semantic POI search.",
                    )
                    normalized_candidates = []
            if not normalized_candidates:
                await _ctx_info(ctx, "Named-place geocode returned no candidates; fallback to semantic POI search.")

        if not normalized_candidates:
            semantic_search_started_at = perf_counter()
            semantic_cache_key = _stable_cache_key(
                [
                    "semantic_search",
                    self.provider,
                    origin_location,
                    semantic_keywords,
                    int(radius),
                    poi_limit,
                    destination_kind,
                ]
            )
            cached_semantic_candidates = _cache_get(_SEMANTIC_SEARCH_CACHE, semantic_cache_key)
            if cached_semantic_candidates is not None:
                normalized_candidates = cached_semantic_candidates
                candidates_raw = []
                _log_timing("cache_hit", 0.0, provider=self.provider, cache_kind="semantic_search")
            else:
                try:
                    candidates_raw = await self.map_client.search_semantic_pois(
                        location=origin_location,
                        keywords=semantic_keywords,
                        radius=int(radius),
                        sortrule="distance",
                        limit=poi_limit,
                    )
                except Exception as exc:
                    candidates_raw = []
                    retrieval_errors.append(
                        {
                            "stage": "map_semantic_poi_search",
                            "provider": self.provider,
                            "detail": str(exc),
                        }
                    )
                    await _ctx_warning(ctx, f"Semantic POI search failed for provider={self.provider}: {exc}")
                    _log_timing(
                        "map_semantic_poi_search",
                        perf_counter() - semantic_search_started_at,
                        provider=self.provider,
                        candidate_count=0,
                        poi_limit=poi_limit,
                        status="error",
                    )
                normalize_semantic_started_at = perf_counter()
                normalized_candidates = _normalize_semantic_poi_candidates(
                    candidates_raw,
                    destination_kind=destination_kind,
                )
                for candidate in normalized_candidates:
                    candidate["query_affinity"] = _query_affinity_score(candidate, query_terms=query_terms)
                _cache_put(_SEMANTIC_SEARCH_CACHE, semantic_cache_key, normalized_candidates)
                _log_timing(
                    "normalize_semantic_poi_candidates",
                    perf_counter() - normalize_semantic_started_at,
                    provider=self.provider,
                    candidate_count=len(normalized_candidates),
                )
            _log_timing(
                "map_semantic_poi_search",
                perf_counter() - semantic_search_started_at,
                provider=self.provider,
                candidate_count=len(candidates_raw),
                poi_limit=poi_limit,
            )
        await _ctx_info(
            ctx,
            f"Candidate retrieval completed in {perf_counter() - retrieval_started_at:.2f}s with {len(normalized_candidates)} raw candidates.",
        )
        _log_timing(
            "map_candidate_retrieval",
            perf_counter() - retrieval_started_at,
            provider=self.provider,
            candidate_count=len(normalized_candidates),
            explicit_destination=explicit_destination,
        )

        business_filter_started_at = perf_counter()
        availability_rejections = []
        business_filtered = []
        for candidate in normalized_candidates:
            unavailable_reason = _candidate_unavailable_reason(candidate)
            if unavailable_reason:
                availability_rejections.append(
                    {
                        "id": str(candidate.get("id") or ""),
                        "name": str(candidate.get("name") or ""),
                        "reason": unavailable_reason,
                    }
                )
                continue
            business_filtered.append(candidate)
        _log_timing(
            "candidate_business_filter",
            perf_counter() - business_filter_started_at,
            provider=self.provider,
            input_candidate_count=len(normalized_candidates),
            filtered_candidate_count=len(business_filtered),
            rejected_candidate_count=len(availability_rejections),
        )

        dedupe_started_at = perf_counter()
        deduped = []
        seen: set[str] = set()
        for candidate in business_filtered:
            if candidate["id"] in seen:
                continue
            seen.add(candidate["id"])
            candidate["query_affinity"] = max(
                float(candidate.get("query_affinity", 0.0) or 0.0),
                _query_affinity_score(candidate, query_terms=query_terms),
            )
            deduped.append(candidate)
        _log_timing(
            "candidate_deduplication",
            perf_counter() - dedupe_started_at,
            provider=self.provider,
            input_candidate_count=len(business_filtered),
            deduped_candidate_count=len(deduped),
        )

        if destination_kind == "regional_scenic_area":
            regional_filter_started_at = perf_counter()
            concrete_candidates = [
                candidate
                for candidate in deduped
                if candidate.get("is_concrete_destination")
                and candidate.get("candidate_kind") in {"scenic_destination", "named_landmark", "specific_poi"}
                and (
                    float(candidate.get("query_affinity", 0.0) or 0.0) >= 0.2
                    or float(candidate.get("anchor_distance_m", 999999.0) or 999999.0) <= 900
                )
            ]
            if concrete_candidates:
                deduped = concrete_candidates
            else:
                deduped = [
                    candidate
                    for candidate in deduped
                    if candidate.get("candidate_kind") not in {"administrative_area", "road_label", "institution", "commercial_place"}
                ]
            _log_timing(
                "regional_candidate_filter",
                perf_counter() - regional_filter_started_at,
                provider=self.provider,
                candidate_count=len(deduped),
            )

        ranking_started_at = perf_counter()
        rerank_reason = "disabled_by_request" if llm_rerank_requested is False else None
        auto_enable_llm_rerank, auto_rerank_reason = _should_enable_llm_rerank(
            explicit_destination=explicit_destination,
            resolution_scope=normalized_scope,
            parsed_confidence=parsed_confidence,
            intent_source=intent_source,
            candidate_count=len(deduped),
        )
        if llm_rerank_requested is True:
            rerank_enabled = True
        elif llm_rerank_requested is False:
            rerank_enabled = False
        else:
            rerank_enabled = auto_enable_llm_rerank
        if rerank_enabled and llm_rerank_requested is None:
            rerank_reason = auto_rerank_reason
        elif llm_rerank_requested is True:
            rerank_reason = "enabled_by_request"
        elif rerank_reason is None:
            rerank_reason = auto_rerank_reason
        _log_timing(
            "recommendation_rerank_gate",
            0.0,
            provider=self.provider,
            resolution_scope=normalized_scope,
            candidate_count=len(deduped),
            parsed_confidence=round(parsed_confidence, 4),
            rerank_enabled=rerank_enabled,
            rerank_reason=rerank_reason,
        )
        if deduped and explicit_destination and rerank_enabled:
            rerank_started_at = perf_counter()
            rerank_input = [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "type": (item.get("tags", {}) or {}).get("type", ""),
                    "address": (item.get("tags", {}) or {}).get("address", ""),
                    "distance_m": item.get("direct_distance_m"),
                    "candidate_kind": item.get("candidate_kind", ""),
                    "is_concrete_destination": item.get("is_concrete_destination", False),
                    "query_affinity": round(float(item.get("query_affinity", 0.0) or 0.0), 4),
                    "anchor_name": str(item.get("anchor_name", "")),
                    "anchor_distance_m": item.get("anchor_distance_m"),
                }
                for item in deduped[: min(len(deduped), 10)]
            ]
            rerank_meta = await rerank_candidates_with_llm(
                user_instruction=working_instruction,
                parsed_intent=parsed,
                candidates=rerank_input,
            )
            await _ctx_info(
                ctx,
                f"Candidate rerank completed in {perf_counter() - rerank_started_at:.2f}s for {len(rerank_input)} candidates.",
            )
            _log_timing(
                "llm_candidate_rerank",
                perf_counter() - rerank_started_at,
                provider=self.provider,
                candidate_count=len(rerank_input),
            )
            if rerank_meta and rerank_meta.get("ranked_candidate_ids"):
                order_map = {
                    candidate_id: index
                    for index, candidate_id in enumerate(rerank_meta["ranked_candidate_ids"])
                }
                deduped.sort(key=lambda item: order_map.get(item["id"], len(order_map) + item["direct_distance_m"]))
            else:
                deduped.sort(
                    key=lambda item: (
                        _candidate_priority(str(item.get("candidate_kind", ""))),
                        float(item.get("direct_distance_m", 0) or 0),
                    )
                )
        else:
            deduped.sort(
                key=lambda item: (
                    _candidate_priority(str(item.get("candidate_kind", ""))),
                    float(item.get("direct_distance_m", 0) or 0),
                )
            )
        _log_timing(
            "candidate_ordering",
            perf_counter() - ranking_started_at,
            provider=self.provider,
            candidate_count=len(deduped),
            used_llm_rerank=bool(deduped and explicit_destination and rerank_enabled),
        )

        stage1_filtered = []
        stage1_rejected = []
        route_failures = []
        lightweight_scoring_started_at = perf_counter()
        progress_total = max(1, len(deduped))

        for index, candidate in enumerate(deduped):
            await _ctx_report_progress(ctx, index, progress_total)
            direct_distance_m = round(
                    candidate.get("direct_distance_m")
                    or haversine_m(
                    search_latitude,
                    search_longitude,
                    candidate["latitude"],
                    candidate["longitude"],
                ),
                1,
            )
            estimated_distance_m = round(_estimated_route_distance_m(direct_distance_m, chosen_mode), 1)
            estimated_duration_s = round(_estimated_route_duration_s(estimated_distance_m, chosen_mode), 1)

            if destination_kind == "regional_scenic_area":
                text_blob = _candidate_structured_text(candidate["raw_candidate"])
            else:
                text_blob = _candidate_text_blob(candidate["raw_candidate"])

            staged = {
                "candidate": candidate,
                "route": None,
                "text_blob": text_blob,
                "summary": {
                    "distance_m": estimated_distance_m,
                    "duration_s": estimated_duration_s,
                    "duration_min": round(estimated_duration_s / 60, 1),
                },
                "summary_source": "estimated",
                "direct_distance_m": direct_distance_m,
                "keyword_score": clamp_score(
                    keyword_match_score(text_blob, keywords)
                    + _semantic_candidate_bias(
                        candidate,
                        parsed=parsed,
                        explicit_destination=explicit_destination,
                    )
                    + (
                        _session_local_via_bias(
                            candidate,
                            direct_distance_m=direct_distance_m,
                            search_radius_m=radius,
                        )
                        if normalized_scope == "session_local_via"
                        else 0.0
                    )
                ),
                "semantic_affinity": clamp_score(
                    max(
                        float(candidate.get("query_affinity", 0.0) or 0.0),
                        keyword_match_score(text_blob, keywords),
                    )
                ),
                "semantic_weight": (
                    0.45
                    if destination_kind == "regional_scenic_area"
                    else (0.18 if normalized_scope == "session_local_via" else 0.0)
                ),
            }

            passed, reason = apply_hard_constraints(staged, hard_constraints)
            if passed:
                stage1_filtered.append(staged)
            else:
                stage1_rejected.append(
                    {
                        "id": candidate["id"],
                        "name": candidate["name"],
                        "reason": reason,
                    }
                )

        await _ctx_info(
            ctx,
            f"Lightweight candidate scoring completed in {perf_counter() - lightweight_scoring_started_at:.2f}s for {len(deduped)} candidates.",
        )
        _log_timing(
            "candidate_lightweight_scoring",
            perf_counter() - lightweight_scoring_started_at,
            provider=self.provider,
            candidate_count=len(deduped),
        )

        soft_rank_started_at = perf_counter()
        scored = soft_rank(stage1_filtered, optimize_for)
        picked = scored[:top_k]
        _log_timing(
            "candidate_soft_rank",
            perf_counter() - soft_rank_started_at,
            provider=self.provider,
            candidate_count=len(stage1_filtered),
            picked_count=len(picked),
            optimize_for=optimize_for,
        )
        await _ctx_report_progress(ctx, 1, 1)

        translation_started_at = perf_counter()
        recommendations = []
        for item in picked:
            candidate = item["candidate"]
            recommendations.append(
                {
                    "id": candidate["id"],
                    "name": candidate["name"],
                    "coordinates": {
                        "latitude": candidate["latitude"],
                        "longitude": candidate["longitude"],
                    },
                    "score": item["score"],
                    "score_breakdown": item.get("score_breakdown", {}),
                    "route_summary": item["summary"],
                    "tags": candidate["tags"],
                    "candidate_kind": candidate.get("candidate_kind", "unknown"),
                    "is_concrete_destination": candidate.get("is_concrete_destination", False),
                    "query_affinity": round(float(candidate.get("query_affinity", 0.0) or 0.0), 4),
                    "route_summary_source": item.get("summary_source", "estimated"),
                    "location": candidate["location"],
                }
            )
        _log_timing(
            "recommendation_result_materialization",
            perf_counter() - translation_started_at,
            provider=self.provider,
            recommendation_count=len(recommendations),
        )

        if not picked:
            no_candidate_meta = {
                "candidate_count": len(deduped),
                "evaluated_count": len(stage1_filtered) + len(stage1_rejected),
                "hard_filter_pass_count": len(stage1_filtered),
                "hard_filter_reject_count": len(stage1_rejected),
                "availability_reject_count": len(availability_rejections),
                "selection_mode": chosen_mode,
                "selection_radius_m": radius,
                "selection_top_k": top_k,
                "optimization_target": optimize_for,
                "destination_kind": destination_kind,
                "resolution_scope": normalized_scope,
                "llm_rerank_enabled": bool(rerank_enabled),
                "llm_rerank_reason": rerank_reason,
                "parse_confidence": round(parsed_confidence, 4),
                "named_entity_confidence": round(named_entity_confidence, 4),
                "named_geocode_enabled": bool(use_named_place_geocode),
            }
            return {
                "provider": self.provider,
                "input": {
                    "user_instruction": original_instruction,
                    "provider_instruction": translated_instruction or original_instruction,
                    "current_latitude": current_latitude,
                    "current_longitude": current_longitude,
                    "search_anchor_location": search_anchor_location or f"{current_longitude},{current_latitude}",
                },
                "parsed_intent": parsed,
                "recommendations": [],
                "hard_constraint_rejections": stage1_rejected,
                "route_failures": route_failures,
                "retrieval_errors": retrieval_errors,
                "semantic_rerank": rerank_meta,
                "meta": no_candidate_meta,
                "debug": _recommendation_debug_snapshot(
                    parsed=parsed,
                    meta=no_candidate_meta,
                    recommendations=[],
                    retrieval_errors=retrieval_errors,
                ),
                "message": "No candidates matched constraints.",
            }

        chosen_route_data = None
        chosen_route_summary = recommendations[0]["route_summary"].copy()
        chosen_route_started_at = perf_counter()
        if compute_chosen_route:
            try:
                chosen_route_data = await self.map_client.get_coarse_waypoints(
                    origin=origin_location,
                    destination=recommendations[0]["location"],
                    mode=chosen_mode,
                )
                actual_summary = (chosen_route_data.get("route_summary", {}) or {})
                actual_distance_m = float(actual_summary.get("distance_m", 0) or 0)
                actual_duration_s = float(actual_summary.get("duration_s", 0) or 0)
                if actual_distance_m > 0 or actual_duration_s > 0:
                    chosen_route_summary = {
                        "distance_m": round(actual_distance_m, 1),
                        "duration_s": round(actual_duration_s, 1),
                        "duration_min": round(actual_duration_s / 60, 1),
                    }
                    recommendations[0]["route_summary"] = chosen_route_summary
                    recommendations[0]["route_summary_source"] = "actual"
                else:
                    route_failures.append(
                        {
                            "id": recommendations[0]["id"],
                            "name": recommendations[0]["name"],
                            "reason": "empty_route_summary",
                        }
                    )
                    chosen_route_data = None
            except Exception as exc:
                route_failures.append(
                    {
                        "id": recommendations[0]["id"],
                        "name": recommendations[0]["name"],
                        "reason": "route_exception",
                        "detail": str(exc),
                    }
                )
                chosen_route_data = None

        await _ctx_info(
            ctx,
            f"Chosen-destination route computation completed in {perf_counter() - chosen_route_started_at:.2f}s.",
        )
        _log_timing(
            "map_route_for_chosen_destination",
            perf_counter() - chosen_route_started_at,
            provider=self.provider,
            status="ok" if chosen_route_data else "fallback",
        )

        result = {
            # Rich semantic waypoints for explainability and debugging.
            "provider": self.provider,
            "input": {
                "user_instruction": original_instruction,
                "provider_instruction": translated_instruction or original_instruction,
                "current_latitude": current_latitude,
                "current_longitude": current_longitude,
                "search_anchor_location": search_anchor_location or f"{current_longitude},{current_latitude}",
            },
            "parsed_intent": parsed,
            "recommendations": recommendations,
            "chosen_destination": recommendations[0],
            "key_waypoints": extract_key_waypoints(chosen_route_data or {}),
            "waypoint_sequence": extract_waypoint_sequence(chosen_route_data or {}),
            "route_polyline": [
                {
                    "longitude": float(point[0]),
                    "latitude": float(point[1]),
                }
                for point in ((chosen_route_data or {}).get("route_polyline") or [])
                if isinstance(point, (list, tuple)) and len(point) == 2
            ],
            "route_summary": chosen_route_summary,
            "meta": {
                "candidate_count": len(deduped),
                "evaluated_count": len(stage1_filtered) + len(stage1_rejected),
                "hard_filter_pass_count": len(stage1_filtered),
                "hard_filter_reject_count": len(stage1_rejected),
                "availability_reject_count": len(availability_rejections),
                "selection_mode": chosen_mode,
                "selection_radius_m": radius,
                "selection_top_k": top_k,
                "optimization_target": optimize_for,
                "destination_kind": destination_kind,
                "candidate_routing_mode": "estimated",
                "chosen_destination_routing_mode": (
                    "actual" if chosen_route_data else ("estimated_only" if not compute_chosen_route else "estimated_fallback")
                ),
                "resolution_scope": normalized_scope,
                "llm_rerank_enabled": bool(rerank_enabled),
                "llm_rerank_reason": rerank_reason,
                "parse_confidence": round(parsed_confidence, 4),
                "named_entity_confidence": round(named_entity_confidence, 4),
                "named_geocode_enabled": bool(use_named_place_geocode),
            },
            "availability_rejections": availability_rejections,
            "hard_constraint_rejections": stage1_rejected,
            "route_failures": route_failures,
            "retrieval_errors": retrieval_errors,
            "semantic_rerank": rerank_meta,
            "timestamp": datetime.now().isoformat(),
        }
        result["debug"] = _recommendation_debug_snapshot(
            parsed=parsed,
            meta=result["meta"],
            recommendations=recommendations,
            retrieval_errors=retrieval_errors,
        )

        if english_localization_active:
            result_localization_started_at = perf_counter()
            translated_bundle = await translate_result_bundle_to_english(
                result_bundle={
                    "recommendations": [
                        {
                            "id": item["id"],
                            "name": item["name"],
                            "tags": {
                                "type": (item.get("tags") or {}).get("type", ""),
                                "address": (item.get("tags") or {}).get("address", ""),
                            },
                        }
                        for item in result["recommendations"]
                    ],
                    "chosen_destination": {
                        "id": result["chosen_destination"]["id"],
                        "name": result["chosen_destination"]["name"],
                        "tags": {
                            "type": (result["chosen_destination"].get("tags") or {}).get("type", ""),
                            "address": (result["chosen_destination"].get("tags") or {}).get("address", ""),
                        },
                    },
                    "semantic_rerank": result.get("semantic_rerank") or {},
                }
            )
            if translated_bundle:
                result = _apply_english_translations(result, translated_bundle)
                await _ctx_info(
                    ctx,
                    f"English result localization completed in {perf_counter() - result_localization_started_at:.2f}s.",
                )
                _log_timing(
                    "llm_translate_result_bundle",
                    perf_counter() - result_localization_started_at,
                    provider=self.provider,
                    status="ok",
                )
            else:
                await _ctx_warning(
                    ctx,
                    f"English result localization failed after {perf_counter() - result_localization_started_at:.2f}s; returning provider-language text.",
                )
                _log_timing(
                    "llm_translate_result_bundle",
                    perf_counter() - result_localization_started_at,
                    provider=self.provider,
                    status="failed",
                )

        await _ctx_info(ctx, f"Recommendation pipeline completed in {perf_counter() - overall_started_at:.2f}s.")
        _log_timing(
            "recommendation_pipeline_total",
            perf_counter() - overall_started_at,
            provider=self.provider,
            recommendation_count=len(recommendations),
        )

        return result
