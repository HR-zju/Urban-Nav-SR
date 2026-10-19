"""Presentation responsibilities for destination recommendation."""
from __future__ import annotations

from typing import Any

def _apply_english_translations(
    result: dict[str, Any],
    translated_bundle: dict[str, Any],
) -> dict[str, Any]:
    translated_recommendations = translated_bundle.get("recommendations")
    if isinstance(translated_recommendations, list):
        translated_map = {
            str(item.get("id", "")): item
            for item in translated_recommendations
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }
        for recommendation in result.get("recommendations", []) or []:
            translated = translated_map.get(str(recommendation.get("id", "")))
            if not translated:
                continue
            if isinstance(translated.get("name"), str) and translated.get("name"):
                recommendation["name"] = translated["name"]
            translated_tags = translated.get("tags")
            if isinstance(translated_tags, dict) and isinstance(recommendation.get("tags"), dict):
                for field in ("type", "address"):
                    if isinstance(translated_tags.get(field), str):
                        recommendation["tags"][field] = translated_tags[field]

    translated_chosen = translated_bundle.get("chosen_destination")
    if isinstance(translated_chosen, dict) and isinstance(result.get("chosen_destination"), dict):
        if isinstance(translated_chosen.get("name"), str) and translated_chosen.get("name"):
            result["chosen_destination"]["name"] = translated_chosen["name"]
        translated_tags = translated_chosen.get("tags")
        if isinstance(translated_tags, dict) and isinstance(result["chosen_destination"].get("tags"), dict):
            for field in ("type", "address"):
                if isinstance(translated_tags.get(field), str):
                    result["chosen_destination"]["tags"][field] = translated_tags[field]

    translated_waypoints = translated_bundle.get("key_waypoints")
    if isinstance(translated_waypoints, list):
        for original, translated in zip(result.get("key_waypoints", []) or [], translated_waypoints):
            if not isinstance(translated, dict):
                continue
            for field in ("instruction", "road"):
                if isinstance(translated.get(field), str):
                    original[field] = translated[field]

    translated_rerank = translated_bundle.get("semantic_rerank")
    if isinstance(translated_rerank, dict) and isinstance(result.get("semantic_rerank"), dict):
        if isinstance(translated_rerank.get("summary"), str):
            result["semantic_rerank"]["summary"] = translated_rerank["summary"]

    return result


def _recommendation_debug_snapshot(
    *,
    parsed: dict[str, Any],
    meta: dict[str, Any],
    recommendations: list[dict[str, Any]],
    retrieval_errors: list[dict[str, str]],
    top_k: int = 3,
) -> dict[str, Any]:
    chosen = recommendations[0] if recommendations else {}
    top_candidates: list[dict[str, Any]] = []
    for item in recommendations[: max(0, int(top_k))]:
        tags = item.get("tags") if isinstance(item.get("tags"), dict) else {}
        top_candidates.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "location": item.get("location"),
                "candidate_kind": item.get("candidate_kind"),
                "is_concrete_destination": item.get("is_concrete_destination"),
                "query_affinity": item.get("query_affinity"),
                "score": item.get("score"),
                "score_breakdown": item.get("score_breakdown"),
                "tags": {
                    "type": tags.get("type"),
                    "address": tags.get("address"),
                },
            }
        )
    return {
        "parsed_intent": {
            "intent_source": parsed.get("intent_source"),
            "search_keywords": parsed.get("search_keywords"),
            "destination_type": parsed.get("destination_type"),
            "destination_kind": parsed.get("destination_kind"),
            "confidence": parsed.get("confidence"),
            "named_entity_confidence": parsed.get("named_entity_confidence"),
            "llm_error": parsed.get("llm_error"),
        },
        "meta": {
            "resolution_scope": meta.get("resolution_scope"),
            "selection_radius_m": meta.get("selection_radius_m"),
            "selection_top_k": meta.get("selection_top_k"),
            "llm_rerank_enabled": meta.get("llm_rerank_enabled"),
            "llm_rerank_reason": meta.get("llm_rerank_reason"),
            "parse_confidence": meta.get("parse_confidence"),
            "named_entity_confidence": meta.get("named_entity_confidence"),
            "named_geocode_enabled": meta.get("named_geocode_enabled"),
        },
        "chosen": {
            "id": chosen.get("id"),
            "name": chosen.get("name"),
            "location": chosen.get("location"),
            "score": chosen.get("score"),
            "score_breakdown": chosen.get("score_breakdown"),
            "query_affinity": chosen.get("query_affinity"),
        },
        "top_candidates": top_candidates,
        "retrieval_errors": retrieval_errors,
    }
