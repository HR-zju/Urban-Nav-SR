"""Destination recommendation orchestration."""

from __future__ import annotations

from datetime import datetime
from typing import Any
import json
import math
import os

from mcp_server.config import (
    default_intent_rules,
    default_scoring_weights,
    split_search_keywords,
    load_json_env,
)
from mcp_server.llm import IntentParser
from mcp_server.map_providers import MapProviderClient


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_m * math.asin(math.sqrt(a))


def extract_key_waypoints(route_data: dict[str, Any], max_points: int = 12) -> list[dict[str, Any]]:
    coarse_waypoints = route_data.get("coarse_waypoints", [])
    if coarse_waypoints:
        extracted = []
        for item in coarse_waypoints:
            coordinate = item.get("coordinate", {}) or {}
            longitude = coordinate.get("longitude")
            latitude = coordinate.get("latitude")
            if longitude is None or latitude is None:
                location_text = item.get("location", "")
                try:
                    lon_str, lat_str = str(location_text).split(",", 1)
                    longitude = float(lon_str)
                    latitude = float(lat_str)
                except Exception:
                    continue
            extracted.append(
                {
                    "longitude": longitude,
                    "latitude": latitude,
                    "instruction": item.get("instruction", ""),
                    "road": item.get("road", ""),
                    "action": item.get("action", ""),
                }
            )
        if len(extracted) <= max_points:
            return extracted
        stride = max(1, len(extracted) // max_points)
        return extracted[::stride][:max_points]

    waypoints: list[dict[str, Any]] = []
    routes = route_data.get("routes", [])
    if not routes:
        return waypoints

    route = routes[0]
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            maneuver = step.get("maneuver", {})
            location = maneuver.get("location")
            if location and isinstance(location, list) and len(location) == 2:
                waypoints.append(
                    {
                        "longitude": location[0],
                        "latitude": location[1],
                        "instruction": maneuver.get("instruction", ""),
                        "road": step.get("name", ""),
                    }
                )

    if len(waypoints) <= max_points:
        return waypoints

    stride = max(1, len(waypoints) // max_points)
    return waypoints[::stride][:max_points]


def extract_waypoint_sequence(route_data: dict[str, Any], max_points: int = 12) -> list[dict[str, float]]:
    semantic_points = extract_key_waypoints(route_data, max_points=max_points)
    return [
        {
            "longitude": float(item["longitude"]),
            "latitude": float(item["latitude"]),
        }
        for item in semantic_points
        if item.get("longitude") is not None and item.get("latitude") is not None
    ]


def keyword_match_score(text_blob: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    matches = sum(1 for keyword in keywords if keyword.lower() in text_blob)
    return matches / max(len(keywords), 1)


def apply_hard_constraints(candidate: dict[str, Any], hard_constraints: dict[str, Any]) -> tuple[bool, str]:
    text_blob = candidate["text_blob"]
    duration_min = candidate["summary"]["duration_min"]
    distance_m = candidate["summary"]["distance_m"]

    max_duration = hard_constraints.get("max_route_duration_min")
    if isinstance(max_duration, (int, float)) and duration_min > float(max_duration):
        return False, "duration_exceeded"

    max_distance = hard_constraints.get("max_route_distance_m")
    if isinstance(max_distance, (int, float)) and distance_m > float(max_distance):
        return False, "distance_exceeded"

    for keyword in hard_constraints.get("must_include_keywords", []) or []:
        if keyword.lower() not in text_blob:
            return False, "missing_required_keyword"

    for keyword in hard_constraints.get("exclude_keywords", []) or []:
        if keyword.lower() in text_blob:
            return False, "excluded_keyword_hit"

    return True, "ok"


def soft_rank(candidates: list[dict[str, Any]], optimize_for: str) -> list[dict[str, Any]]:
    if not candidates:
        return []

    weights = load_json_env("RANKING_WEIGHTS_JSON", default_scoring_weights())
    time_weight = float(weights.get("time", 0.5))
    distance_weight = float(weights.get("distance", 0.3))
    proximity_weight = float(weights.get("proximity", 0.1))
    keyword_weight = float(weights.get("keyword", 0.1))

    durations = [item["summary"]["duration_s"] for item in candidates]
    distances = [item["summary"]["distance_m"] for item in candidates]
    direct_distances = [item["direct_distance_m"] for item in candidates]
    min_duration, max_duration = min(durations), max(durations)
    min_distance, max_distance = min(distances), max(distances)
    min_direct, max_direct = min(direct_distances), max(direct_distances)

    def inv_norm(value: float, v_min: float, v_max: float) -> float:
        if abs(v_max - v_min) < 1e-9:
            return 1.0
        return 1.0 - ((value - v_min) / (v_max - v_min))

    for item in candidates:
        duration_score = inv_norm(item["summary"]["duration_s"], min_duration, max_duration)
        distance_score = inv_norm(item["summary"]["distance_m"], min_distance, max_distance)
        proximity_score = inv_norm(item["direct_distance_m"], min_direct, max_direct)
        keyword_score = item["keyword_score"]

        if optimize_for == "time":
            time_component = 0.55 * duration_score
            distance_component = 0.15 * distance_score
        elif optimize_for == "distance":
            time_component = 0.15 * duration_score
            distance_component = 0.55 * distance_score
        else:
            time_component = time_weight * duration_score
            distance_component = distance_weight * distance_score

        item["score"] = round(
            time_component
            + distance_component
            + proximity_weight * proximity_score
            + keyword_weight * keyword_score,
            4,
        )
        item["score_breakdown"] = {
            "duration": round(duration_score, 4),
            "distance": round(distance_score, 4),
            "proximity": round(proximity_score, 4),
            "keyword": round(keyword_score, 4),
        }

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


async def _ctx_info(ctx: Any, message: str) -> None:
    if ctx and hasattr(ctx, "info"):
        await ctx.info(message)


async def _ctx_report_progress(ctx: Any, current: int, total: int) -> None:
    if ctx and hasattr(ctx, "report_progress"):
        await ctx.report_progress(current, total)


def _compact_candidate_tags(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": candidate.get("type", ""),
        "address": candidate.get("address", ""),
        "provider": candidate.get("provider", ""),
        "provider_poi_id": candidate.get("provider_poi_id", ""),
    }


def _candidate_text_blob(candidate: dict[str, Any]) -> str:
    parts = [
        str(candidate.get("name", "")),
        str(candidate.get("type", "")),
        str(candidate.get("address", "")),
    ]
    raw = candidate.get("raw")
    if isinstance(raw, dict):
        parts.append(json.dumps(raw, ensure_ascii=False))
    return " ".join(part for part in parts if part).lower()


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
        ctx: Any = None,
    ) -> dict[str, Any]:
        eval_limit = int(os.getenv("ROUTE_EVAL_LIMIT", "25"))
        default_opt_target = os.getenv("DEFAULT_OPTIMIZE_FOR", "distance").strip() or "distance"

        parsed = await self.intent_parser.parse(user_instruction)
        search_keywords = parsed.get("search_keywords", "")
        keywords = parsed.get("keywords") or split_search_keywords(search_keywords)
        hard_constraints = parsed.get("hard_constraints") or default_intent_rules()["default_hard_constraints"]
        chosen_mode = mode or parsed.get("mode") or "car"
        radius = parsed.get("radius_m", search_radius) or search_radius
        optimize_for = parsed.get("optimize_for", default_opt_target)
        top_k = int(parsed.get("max_candidates", max_candidates) or max_candidates)
        top_k = max(1, min(top_k, max_candidates))
        semantic_keywords = search_keywords or "|".join(keywords)
        origin_location = f"{current_longitude},{current_latitude}"
        poi_limit = max(top_k, min(eval_limit, 25))

        await _ctx_info(
            ctx,
            f"Provider={self.provider}, mode={chosen_mode}, radius={radius}m, search_keywords={semantic_keywords}",
        )
        candidates_raw = await self.map_client.search_semantic_pois(
            location=origin_location,
            keywords=semantic_keywords,
            radius=int(radius),
            sortrule="distance",
            limit=poi_limit,
        )

        normalized_candidates = []
        for item in candidates_raw:
            coordinates = item.get("coordinates", {}) or {}
            lat = coordinates.get("latitude")
            lon = coordinates.get("longitude")
            if lat is None or lon is None:
                continue

            name = item.get("name") or "Unnamed"
            text_blob = _candidate_text_blob(item)
            if keywords and not any(keyword.lower() in text_blob for keyword in keywords):
                continue

            normalized_candidates.append(
                {
                    "id": item.get("provider_poi_id") or item.get("location") or f"{lat}-{lon}-{name}",
                    "name": name,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "location": item.get("location") or f"{lon},{lat}",
                    "tags": _compact_candidate_tags(item),
                    "raw_candidate": item,
                    "direct_distance_m": float(item.get("distance_m", 0) or 0),
                }
            )

        deduped = []
        seen: set[str] = set()
        for candidate in normalized_candidates:
            if candidate["id"] in seen:
                continue
            seen.add(candidate["id"])
            deduped.append(candidate)

        stage1_filtered = []
        stage1_rejected = []
        progress_total = max(1, min(eval_limit, len(deduped)))
        for index, candidate in enumerate(deduped[:eval_limit]):
            await _ctx_report_progress(ctx, index, progress_total)

            try:
                route_data = await self.map_client.get_coarse_waypoints(
                    origin=origin_location,
                    destination=candidate["location"],
                    mode=chosen_mode,
                )
            except Exception:
                continue

            route_summary = route_data.get("route_summary", {}) or {}
            distance_m = float(route_summary.get("distance_m", 0) or 0)
            duration_s = float(route_summary.get("duration_s", 0) or 0)
            if distance_m <= 0 and duration_s <= 0:
                continue

            text_blob = _candidate_text_blob(candidate["raw_candidate"])
            staged = {
                "candidate": candidate,
                "route": route_data,
                "text_blob": text_blob,
                "summary": {
                    "distance_m": round(distance_m, 1),
                    "duration_s": round(duration_s, 1),
                    "duration_min": round(duration_s / 60, 1),
                },
                "direct_distance_m": round(
                    candidate.get("direct_distance_m")
                    or haversine_m(
                        current_latitude,
                        current_longitude,
                        candidate["latitude"],
                        candidate["longitude"],
                    ),
                    1,
                ),
                "keyword_score": keyword_match_score(text_blob, keywords),
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

        scored = soft_rank(stage1_filtered, optimize_for)
        picked = scored[:top_k]
        await _ctx_report_progress(ctx, 1, 1)

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
                    "location": candidate["location"],
                }
            )

        if not picked:
            return {
                "provider": self.provider,
                "input": {
                    "user_instruction": user_instruction,
                    "current_latitude": current_latitude,
                    "current_longitude": current_longitude,
                },
                "parsed_intent": parsed,
                "recommendations": [],
                "hard_constraint_rejections": stage1_rejected,
                "message": "No candidates matched constraints.",
            }

        return {
            # Rich semantic waypoints for explainability and debugging.
            "provider": self.provider,
            "input": {
                "user_instruction": user_instruction,
                "current_latitude": current_latitude,
                "current_longitude": current_longitude,
            },
            "parsed_intent": parsed,
            "recommendations": recommendations,
            "chosen_destination": recommendations[0],
            "key_waypoints": extract_key_waypoints(picked[0]["route"]),
            "waypoint_sequence": extract_waypoint_sequence(picked[0]["route"]),
            "route_summary": recommendations[0]["route_summary"],
            "meta": {
                "candidate_count": len(deduped),
                "evaluated_count": len(stage1_filtered) + len(stage1_rejected),
                "hard_filter_pass_count": len(stage1_filtered),
                "hard_filter_reject_count": len(stage1_rejected),
                "selection_mode": chosen_mode,
                "optimization_target": optimize_for,
            },
            "hard_constraint_rejections": stage1_rejected,
            "timestamp": datetime.now().isoformat(),
        }


__all__ = [
    "DestinationRecommendationEngine",
    "apply_hard_constraints",
    "extract_key_waypoints",
    "extract_waypoint_sequence",
    "haversine_m",
    "keyword_match_score",
    "soft_rank",
]
