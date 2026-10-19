"""Public destination-recommendation API; responsibilities live in recommendation_services."""
from mcp_server.recommendation_services.candidates import _extract_locality_hints, _normalize_named_place_candidates, _normalize_semantic_poi_candidates, _normalize_nearby_poi_candidates, _bbox_from_center, _normalize_feature_candidates
from mcp_server.recommendation_services.engine import DestinationRecommendationEngine
from mcp_server.recommendation_services.geometry import haversine_m, extract_key_waypoints, extract_waypoint_sequence
from mcp_server.recommendation_services.infrastructure import _ctx_info, _ctx_warning, _ctx_report_progress, _log_timing, _cache_max_entries, _stable_cache_key, _cache_get, _cache_put, _run_limited, TIMING_LOGGER, _PARSE_CACHE, _LOCALITY_HINT_CACHE, _NAMED_GEOCODE_CACHE, _SEMANTIC_SEARCH_CACHE
from mcp_server.recommendation_services.policy import _compact_candidate_tags, _candidate_text_blob, _candidate_unavailable_reason, _candidate_structured_text, _estimated_route_distance_m, _estimated_route_duration_s, _contains_cjk, _looks_english_instruction, _normalize_text_key, _scope_limited_radius_m, _confidence_value, _parsed_confidence, _parsed_named_entity_confidence, _should_use_named_place_geocode, _should_enable_llm_rerank, _is_identifier_like_name, _humanize_type_label, _display_name_from_tags, _split_camel_case, _tokenize_semantic_terms, _strip_place_descriptors, _collect_query_terms, _candidate_affinity_terms, _is_generic_scenic_container, _query_affinity_score, _regional_anchor_terms, _regional_expansion_acceptance, _resolve_destination_kind, _is_explicit_destination_request, _candidate_priority, _classify_candidate_kind, _is_concrete_destination, _regional_diversify_candidates, _annotate_candidate, _named_place_queries, _candidate_type_text, _semantic_candidate_bias, _session_local_via_bias
from mcp_server.recommendation_services.presentation import _apply_english_translations, _recommendation_debug_snapshot
from mcp_server.recommendation_services.ranking import keyword_match_score, clamp_score, apply_hard_constraints, _instruction_has_explicit_time_limit, _instruction_has_explicit_distance_limit, _instruction_has_explicit_include, _instruction_has_explicit_exclude, _operationalize_hard_constraints, soft_rank
from mcp_server.recommendation_services.regional import _expand_regional_scenic_area_candidates

__all__ = [
    "DestinationRecommendationEngine",
    "apply_hard_constraints",
    "clamp_score",
    "extract_key_waypoints",
    "extract_waypoint_sequence",
    "haversine_m",
    "keyword_match_score",
    "soft_rank",
]
