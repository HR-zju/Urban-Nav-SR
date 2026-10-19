"""Regional responsibilities for destination recommendation."""
from __future__ import annotations

from typing import Any
import os
from mcp_server.map_providers import MapProviderClient
from mcp_server.recommendation_services.candidates import _bbox_from_center, _normalize_feature_candidates, _normalize_nearby_poi_candidates
from mcp_server.recommendation_services.policy import _candidate_priority, _is_generic_scenic_container, _query_affinity_score, _regional_anchor_terms, _regional_diversify_candidates, _regional_expansion_acceptance
from mcp_server.recommendation_services.infrastructure import _ctx_warning, _run_limited
from mcp_server.recommendation_services.ranking import clamp_score
from mcp_server.recommendation_services.geometry import haversine_m

async def _expand_regional_scenic_area_candidates(
    *,
    base_candidates: list[dict[str, Any]],
    map_client: MapProviderClient,
    provider: str,
    current_latitude: float,
    current_longitude: float,
    radius: float,
    query_terms: list[str],
    ctx: Any = None,
) -> list[dict[str, Any]]:
    sorted_base = sorted(
        base_candidates,
        key=lambda item: (
            _candidate_priority(str(item.get("candidate_kind", ""))),
            -float(item.get("query_affinity", 0.0) or 0.0),
            float(item.get("direct_distance_m", 0.0) or 0.0),
        ),
    )
    anchor_terms = _regional_anchor_terms(sorted_base, query_terms)
    anchors = [
        candidate
        for candidate in sorted_base
        if candidate.get("candidate_kind") in {"regional_area", "scenic_destination", "named_landmark"}
        and float(candidate.get("query_affinity", 0.0) or 0.0) >= 0.15
    ][:3]
    if not anchors:
        anchors = [
            candidate
            for candidate in sorted_base
            if candidate.get("candidate_kind") in {"regional_area", "scenic_destination", "named_landmark", "specific_poi"}
        ][:2]

    expanded: list[dict[str, Any]] = []
    for candidate in sorted_base:
        affinity = _query_affinity_score(candidate, query_terms=query_terms, anchor_terms=anchor_terms)
        candidate["query_affinity"] = max(float(candidate.get("query_affinity", 0.0) or 0.0), affinity)
        if candidate.get("candidate_kind") in {"scenic_destination", "named_landmark", "specific_poi"} and candidate.get(
            "is_concrete_destination"
        ):
            if candidate["query_affinity"] >= 0.15 or not _is_generic_scenic_container(candidate):
                expanded.append(candidate)
        elif candidate.get("candidate_kind") == "regional_area" and candidate["query_affinity"] >= 0.25:
            expanded.append(candidate)

    search_radius = max(900, min(int(radius * 0.2), 2200))

    expand_concurrency = int(os.getenv("REGIONAL_EXPANSION_CONCURRENCY", "3") or 3)

    async def _fetch_anchor_nearby(anchor: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        nearby = await map_client.get_nearby_pois(
            lat=float(anchor["latitude"]),
            lon=float(anchor["longitude"]),
            radius=search_radius,
            categories=["tourism", "leisure"],
        )
        return anchor, nearby

    nearby_results = await _run_limited(
        [_fetch_anchor_nearby(anchor) for anchor in anchors],
        concurrency=expand_concurrency,
    )

    for result in nearby_results:
        if isinstance(result, Exception):
            continue

        anchor, nearby = result
        normalized_nearby = _normalize_nearby_poi_candidates(
            nearby,
            current_latitude=current_latitude,
            current_longitude=current_longitude,
            destination_kind="regional_scenic_area",
            source_prefix="regional_expansion",
        )
        for candidate in normalized_nearby:
            anchor_distance_m = haversine_m(
                float(anchor["latitude"]),
                float(anchor["longitude"]),
                float(candidate["latitude"]),
                float(candidate["longitude"]),
            )
            direct_affinity = _query_affinity_score(candidate, query_terms=query_terms, anchor_terms=anchor_terms)
            inherited_affinity = clamp_score(max(direct_affinity, float(anchor.get("query_affinity", 0.0) or 0.0) * 0.55))
            if anchor_distance_m <= 900:
                inherited_affinity = clamp_score(max(inherited_affinity, float(anchor.get("query_affinity", 0.0) or 0.0) * 0.75))
            candidate["query_affinity"] = inherited_affinity
            candidate["anchor_name"] = anchor.get("name", "")
            candidate["anchor_distance_m"] = round(anchor_distance_m, 1)
            if _regional_expansion_acceptance(
                candidate,
                query_affinity=float(candidate["query_affinity"]),
                anchor_distance_m=anchor_distance_m,
            ):
                expanded.append(candidate)

    concrete_count = sum(1 for candidate in expanded if candidate.get("is_concrete_destination"))
    if provider == "osm" and concrete_count < 3 and anchors:
        feature_queries = []
        for anchor in anchors[:2]:
            bbox = _bbox_from_center(
                float(anchor["latitude"]),
                float(anchor["longitude"]),
                max(search_radius * 1.25, 1400),
            )
            feature_queries.extend(
                [
                    (anchor, bbox, "tourism", ["attraction", "viewpoint", "museum", "gallery", "theme_park"]),
                    (anchor, bbox, "leisure", ["park", "garden", "nature_reserve"]),
                ]
            )

        async def _fetch_anchor_features(
            anchor: dict[str, Any],
            bbox: tuple[float, float, float, float],
            category: str,
            subcategories: list[str],
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            features = await map_client.search_features_by_category(
                bbox=bbox,
                category=category,
                subcategories=subcategories,
            )
            return anchor, features

        feature_results = await _run_limited(
            [
                _fetch_anchor_features(anchor, bbox, category, subcategories)
                for anchor, bbox, category, subcategories in feature_queries
            ],
            concurrency=expand_concurrency,
        )

        for result in feature_results:
            if isinstance(result, Exception):
                await _ctx_warning(ctx, f"OSM regional feature expansion failed: {result}")
                continue

            anchor, features = result
            normalized_features = _normalize_feature_candidates(
                features,
                current_latitude=current_latitude,
                current_longitude=current_longitude,
                destination_kind="regional_scenic_area",
                source_prefix="osm_feature_expansion",
            )
            for candidate in normalized_features:
                anchor_distance_m = haversine_m(
                    float(anchor["latitude"]),
                    float(anchor["longitude"]),
                    float(candidate["latitude"]),
                    float(candidate["longitude"]),
                )
                direct_affinity = _query_affinity_score(candidate, query_terms=query_terms, anchor_terms=anchor_terms)
                inherited_affinity = clamp_score(
                    max(
                        direct_affinity,
                        float(anchor.get("query_affinity", 0.0) or 0.0) * 0.65,
                    )
                )
                candidate["query_affinity"] = max(float(candidate.get("query_affinity", 0.0) or 0.0), inherited_affinity)
                candidate["anchor_name"] = anchor.get("name", "")
                candidate["anchor_distance_m"] = round(anchor_distance_m, 1)
                if _regional_expansion_acceptance(
                    candidate,
                    query_affinity=float(candidate["query_affinity"]),
                    anchor_distance_m=anchor_distance_m,
                ):
                    expanded.append(candidate)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in expanded:
        if candidate["id"] in seen:
            continue
        seen.add(candidate["id"])
        deduped.append(candidate)

    deduped.sort(
        key=lambda item: (
            _candidate_priority(str(item.get("candidate_kind", ""))),
            -float(item.get("query_affinity", 0.0) or 0.0),
            float(item.get("direct_distance_m", 0) or 0),
        )
    )
    return _regional_diversify_candidates(
        deduped,
        desired_count=max(12, min(len(deduped), 18)),
        min_spacing_m=float(os.getenv("REGIONAL_CANDIDATE_MIN_SPACING_M", "650") or 650),
    )
