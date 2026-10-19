"""Goals responsibilities for language and vision services."""
from __future__ import annotations

import asyncio
from typing import Any
import json
import os
import re
from mcp_server.config import default_intent_rules, default_search_keywords
from mcp_server.llm_services.base import BaseLLMService
from mcp_server.llm_services.intent import IntentParser
from mcp_server.llm_services.transport import _apply_chat_template_kwargs, _close_async_client, _create_async_client_with_timeout, _extract_text_from_chat_response, _extract_text_from_responses_response
from mcp_server.llm_services.json_codec import _json_only_contract, _parse_json_object_from_text

def _split_itinerary_segments(user_instruction: str) -> list[str]:
    text = str(user_instruction or "").strip()
    if not text:
        return []

    # Treat sequence markers as boundaries, but do not split on weak fillers such as "想".
    normalized = re.sub(r"[；;。]", " | ", text)
    normalized = re.sub(r"[，,]\s*(?:再|然后|之后|接着|最后)\s*", " | ", normalized)
    normalized = re.sub(r"(?:^|[，,\s])先\s*", " | ", normalized)
    normalized = re.sub(
        r"\s+(?:and then|then|after that|next|finally)\s+",
        " | ",
        normalized,
        flags=re.IGNORECASE,
    )
    segments = [
        re.sub(r"\s+", " ", segment).strip(" ,|，")
        for segment in normalized.split("|")
    ]
    stopwords = {"想", "我想", "want", "i want", "would like", "想要"}
    cleaned: list[str] = []
    for segment in segments:
        if not segment or segment.lower() in stopwords:
            continue
        if _is_origin_only_segment(segment):
            continue
        if _is_context_only_segment(segment):
            continue
        if len(segment) <= 1:
            continue
        cleaned.append(segment)
    return cleaned


def _is_origin_only_segment(segment: str) -> bool:
    """Return whether a decomposed segment only describes the start point.

    Args:
        segment: Candidate itinerary segment.

    Returns:
        ``True`` if the segment is an origin phrase rather than a destination
        goal.
    """

    text = str(segment or "").strip()
    return bool(re.search(r"(?:从|由).{0,30}(?:出发|开始)$", text)) or text in {"出发", "开始"}


def _is_context_only_segment(segment: str) -> bool:
    """Return whether a segment only states user context, not a stop.

    Args:
        segment: Candidate itinerary segment.

    Returns:
        ``True`` when the segment is a condition such as hunger, illness, or
        fatigue without an action verb that asks for a place.
    """

    text = str(segment or "").strip()
    if re.search(r"(?:去|找|买|吃|喝|取|看|坐|休息|办公|看书)", text):
        return False
    return bool(re.search(r"(?:不舒服|有点饿|饿了|累了|渴了|手机.*没电)", text))


def _normalize_goal_label(raw_label: Any, fallback: str = "generic_stop") -> str:
    label = re.sub(r"[^0-9A-Za-z_\-\u3400-\u9fff]+", "_", str(raw_label or "").strip().lower())
    label = re.sub(r"_+", "_", label).strip("_")
    return label or fallback


class GoalDecomposer(BaseLLMService):
    """Decompose a multi-activity user instruction into an ordered goal list."""

    def _prompt_profile(self) -> str:
        """Return the configured itinerary prompt profile."""

        profile = str(os.getenv("LLM_ITINERARY_PROFILE", "full") or "full").strip().lower()
        return profile if profile in {"minimal", "compact", "full"} else "full"

    def _build_minimal_messages(self, user_instruction: str) -> list[dict[str, str]]:
        """Build the smallest ordered-goal contract accepted by the runtime."""

        schema = {
            "goals": [
                {
                    "goal_index": 0,
                    "request_text": "buy coffee",
                    "destination_type": "cafe",
                    "destination_kind": "generic_poi",
                    "search_keywords": "coffee|cafe",
                }
            ],
            "global_mode": "foot",
        }
        example = {
            "goals": [
                {
                    "goal_index": 0,
                    "request_text": "eat at a restaurant",
                    "destination_type": "restaurant",
                    "destination_kind": "generic_poi",
                    "search_keywords": "restaurant|food",
                },
                {
                    "goal_index": 1,
                    "request_text": "buy a drink",
                    "destination_type": "beverage_shop",
                    "destination_kind": "generic_poi",
                    "search_keywords": "beverage shop|cafe|drink store",
                },
            ],
            "global_mode": "foot",
        }
        system_prompt = (
            "Decompose the request into ordered map-search goals. Return JSON only. "
            "Top-level keys must be exactly goals and global_mode. Each goal must contain exactly "
            "goal_index, request_text, destination_type, destination_kind, and search_keywords. Preserve every requested "
            "stop and its order. Use global_mode=foot unless another mode is explicitly requested. "
            "Do not return explanations, POI ids, coordinates, routes, constraints, rationale, or extra fields."
            + " "
            + _json_only_contract(schema)
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "request: First eat at a restaurant, then buy a drink."},
            {"role": "assistant", "content": json.dumps(example, ensure_ascii=True)},
            {"role": "user", "content": f"request: {user_instruction}\nReturn the JSON object now."},
        ]

    def _build_compact_messages(self, user_instruction: str) -> list[dict[str, str]]:
        """Build a short ordered-goal contract for format-sensitive models."""

        schema = {
            "goals": [
                {
                    "goal_index": 0,
                    "goal_type": "coffee_stop",
                    "request_text": "buy coffee",
                    "destination_type": "cafe",
                    "destination_kind": "generic_poi",
                    "search_keywords": "coffee|cafe",
                    "mode": "foot",
                    "radius_m": 2000,
                    "optimize_for": "distance",
                }
            ],
            "global_mode": "foot",
            "notes": "ordered_goal_list",
        }
        example = {
            "goals": [
                {
                    "goal_index": 0,
                    "goal_type": "meal_stop",
                    "request_text": "eat at a restaurant",
                    "destination_type": "restaurant",
                    "destination_kind": "generic_poi",
                    "search_keywords": "restaurant|food",
                    "mode": "foot",
                    "radius_m": 2000,
                    "optimize_for": "distance",
                },
                {
                    "goal_index": 1,
                    "goal_type": "drink_stop",
                    "request_text": "buy a drink",
                    "destination_type": "beverage_shop",
                    "destination_kind": "generic_poi",
                    "search_keywords": "beverage shop|cafe|drink store",
                    "mode": "foot",
                    "radius_m": 2000,
                    "optimize_for": "distance",
                },
            ],
            "global_mode": "foot",
            "notes": "ordered_multi_goal",
        }
        system_prompt = (
            "Decompose the request into ordered map-search goals. Return JSON only. "
            "Top-level keys must be exactly goals, global_mode, notes. Each goal must contain exactly "
            "goal_index, goal_type, request_text, destination_type, destination_kind, search_keywords, "
            "mode, radius_m, optimize_for. Preserve every requested stop and its order. Use mode=foot "
            "unless the user explicitly requests another mode. Do not return POI ids, coordinates, routes, "
            "hard_constraints, or rationale."
            + " "
            + _json_only_contract(schema)
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "request: First eat at a restaurant, then buy a drink."},
            {"role": "assistant", "content": json.dumps(example, ensure_ascii=True)},
            {"role": "user", "content": f"request: {user_instruction}\nReturn the JSON object now."},
        ]

    def _itinerary_timeout_s(self) -> float:
        raw = os.getenv("LLM_ITINERARY_TIMEOUT_S", "").strip()
        if raw:
            try:
                return max(float(raw), 10.0)
            except Exception:
                pass
        return max(float(self.settings.timeout_s), 45.0)

    def heuristic_decompose(self, user_instruction: str) -> dict[str, Any]:
        parser = IntentParser(settings=self.settings)
        global_intent = parser.heuristic_parse(user_instruction)
        segments = _split_itinerary_segments(user_instruction)
        if not segments:
            segments = [str(user_instruction or "").strip()]

        goals: list[dict[str, Any]] = []
        for index, segment in enumerate(segments[:4]):
            parsed = parser.heuristic_parse(segment)
            goal_type = _normalize_goal_label(parsed.get("destination_type"), fallback="generic_stop")
            goals.append(
                {
                    "goal_index": index,
                    "goal_type": goal_type,
                    "request_text": segment,
                    "destination_type": parsed.get("destination_type", "poi"),
                    "destination_kind": parsed.get("destination_kind", "generic_poi"),
                    "search_keywords": parsed.get("search_keywords", default_search_keywords()),
                    "mode": parsed.get("mode", global_intent.get("mode", "car")),
                    "radius_m": parsed.get("radius_m", global_intent.get("radius_m", 3000)),
                    "optimize_for": parsed.get("optimize_for", global_intent.get("optimize_for", "distance")),
                    "hard_constraints": parsed.get("hard_constraints", default_intent_rules()["default_hard_constraints"]),
                    "rationale": "heuristic fallback only preserves structure; semantic generalization is provided by the LLM path",
                }
            )

        return {
            "goals": goals,
            "global_mode": global_intent.get("mode", "car"),
            "global_radius_m": global_intent.get("radius_m", 3000),
            "is_multi_goal": len(goals) > 1,
            "decomposition_source": "heuristic",
        }

    def build_messages(self, user_instruction: str) -> list[dict[str, str]]:
        prompt_profile = self._prompt_profile()
        if prompt_profile == "minimal":
            return self._build_minimal_messages(user_instruction)
        if prompt_profile == "compact":
            return self._build_compact_messages(user_instruction)

        schema = {
            "goals": [
                {
                    "goal_index": 0,
                    "goal_type": "coffee_stop",
                    "request_text": "buy coffee",
                    "destination_type": "cafe",
                    "destination_kind": "generic_poi",
                    "search_keywords": "coffee|cafe",
                    "mode": "foot",
                    "radius_m": 2000,
                    "optimize_for": "distance",
                    "hard_constraints": {
                        "max_route_duration_min": None,
                        "max_route_distance_m": None,
                        "must_include_keywords": [],
                        "exclude_keywords": [],
                    },
                    "rationale": "coffee can be satisfied by a nearby cafe",
                }
            ],
            "global_mode": "foot",
            "notes": "ordered_goal_list",
        }
        examples = [
            {
                "request": "I am tired, so first find somewhere to sit and rest, then dinner, then a movie.",
                "output": {
                    "goals": [
                        {
                            "goal_index": 0,
                            "goal_type": "rest_stop",
                            "request_text": "find somewhere to sit and rest",
                            "destination_type": "rest_stop",
                            "destination_kind": "generic_poi",
                            "search_keywords": "cafe|tea house|bookstore|park",
                            "mode": "foot",
                            "radius_m": 3000,
                            "optimize_for": "distance",
                            "hard_constraints": {
                                "max_route_duration_min": None,
                                "max_route_distance_m": None,
                                "must_include_keywords": [],
                                "exclude_keywords": [],
                            },
                            "rationale": "The user is tired, so the first stop should be a nearby place where they can sit and rest.",
                        },
                        {
                            "goal_index": 1,
                            "goal_type": "dinner",
                            "request_text": "eat dinner",
                            "destination_type": "dinner_restaurant",
                            "destination_kind": "generic_poi",
                            "search_keywords": "restaurant|dinner|food",
                            "mode": "foot",
                            "radius_m": 3000,
                            "optimize_for": "balanced",
                            "hard_constraints": {
                                "max_route_duration_min": None,
                                "max_route_distance_m": None,
                                "must_include_keywords": [],
                                "exclude_keywords": [],
                            },
                            "rationale": "Dinner should retrieve food-service POIs, not generic rest stops.",
                        },
                        {
                            "goal_index": 2,
                            "goal_type": "movie",
                            "request_text": "watch a movie",
                            "destination_type": "cinema",
                            "destination_kind": "generic_poi",
                            "search_keywords": "cinema|movie theater",
                            "mode": "foot",
                            "radius_m": 5000,
                            "optimize_for": "distance",
                            "hard_constraints": {
                                "max_route_duration_min": None,
                                "max_route_distance_m": None,
                                "must_include_keywords": [],
                                "exclude_keywords": [],
                            },
                            "rationale": "The final activity is watching a movie, so the retrievable supply is cinemas.",
                        },
                    ],
                    "global_mode": "foot",
                    "notes": "ordered_multi_goal",
                },
            },
            {
                "request": "I'm thirsty and want a quick coffee nearby",
                "output": {
                    "goals": [
                        {
                            "goal_index": 0,
                            "goal_type": "quick_drink_or_coffee",
                            "request_text": "find a quick coffee or drink stop nearby",
                            "destination_type": "drink_stop",
                            "destination_kind": "generic_poi",
                            "search_keywords": "coffee|cafe|tea shop|juice bar|convenience store",
                            "mode": "foot",
                            "radius_m": 2000,
                            "optimize_for": "distance",
                            "hard_constraints": {
                                "max_route_duration_min": None,
                                "max_route_distance_m": None,
                                "must_include_keywords": [],
                                "exclude_keywords": [],
                            },
                            "rationale": "The need is thirst plus coffee, so nearby beverage-service POIs are appropriate.",
                        },
                    ],
                    "global_mode": "foot",
                    "notes": "single_goal_need",
                },
            },
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trip-goal decomposer for map recommendation. "
                    "Return JSON only. The top-level JSON keys must be exactly: goals, global_mode, notes. "
                    "goals must be an ordered list. Each goal object must contain exactly: "
                    "goal_index, goal_type, request_text, destination_type, destination_kind, "
                    "search_keywords, mode, radius_m, optimize_for, hard_constraints, rationale. "
                    "For English instructions, write request_text, goal_type, destination_type, search_keywords, and rationale in English. "
                    "Use open-vocabulary reasoning: goal_type and destination_type are semantic labels, not a closed taxonomy. "
                    "search_keywords must be a concise provider-facing query string, preferably pipe-delimited, that a domestic map API can retrieve. "
                    "For implicit human needs, infer the real-world POI/service supply that would satisfy the need, then write searchable keywords. "
                    "Use hard_constraints only for constraints that are explicitly stated by the user. "
                    "Allowed destination_kind values: generic_poi, named_landmark, regional_scenic_area, administrative_area, road_or_route_label, specific_poi. "
                    "Allowed mode values: car, foot, bike. Allowed optimize_for values: time, distance, balanced. "
                    "This UrbanNav benchmark is pedestrian navigation: use mode=foot and global_mode=foot unless the user explicitly asks to drive or use a car. "
                    "If the request is single-goal, still return a one-item goals list. "
                    "Do not solve routing. Do not return destinations. Only decompose the user's instruction into ordered retrievable sub-goals."
                    + " "
                    + _json_only_contract(schema)
                ),
            }
        ]
        for example in examples:
            messages.append({"role": "user", "content": f"request: {example['request']}"})
            messages.append({"role": "assistant", "content": json.dumps(example["output"], ensure_ascii=True)})
        messages.append({"role": "user", "content": f"request: {user_instruction}\nReturn the JSON object now."})
        return messages

    def build_prompt(self, user_instruction: str) -> str:
        sections = [self.build_messages(user_instruction)[0]["content"], "", "Examples:"]
        for message in self.build_messages(user_instruction)[1:-1]:
            if message["role"] == "user":
                sections.append(f"Request: {message['content'].replace('request: ', '')}")
            else:
                sections.append(f"JSON: {message['content']}")
        sections.extend(["", "Now decompose this request and return JSON only.", f"Request: {user_instruction}"])
        return "\n".join(sections)

    async def _call_chat_completions(self, client: Any, user_instruction: str) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": self.build_messages(user_instruction),
        }
        max_tokens_env = os.getenv("LLM_ITINERARY_MAX_TOKENS", "").strip()
        payload["max_tokens"] = max(128, int(max_tokens_env)) if max_tokens_env else 1536
        if self.settings.response_format and self.settings.provider != "ark":
            payload["response_format"] = {"type": "json_object"}
        payload = _apply_chat_template_kwargs(payload, self.settings)
        response = await client.chat.completions.create(**payload)
        return _extract_text_from_chat_response(response)

    async def _call_responses_api(self, client: Any, user_instruction: str) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": self.build_prompt(user_instruction)}],
                }
            ],
        }
        max_tokens_env = os.getenv("LLM_ITINERARY_MAX_TOKENS", "").strip()
        payload["max_output_tokens"] = max(128, int(max_tokens_env)) if max_tokens_env else 1536
        response = await client.responses.create(**payload)
        return _extract_text_from_responses_response(response)

    async def _call_llm_with_fallback(self, client: Any, user_instruction: str) -> tuple[str, str]:
        errors: list[str] = []
        retry_count = max(1, int(os.getenv("LLM_ITINERARY_CALL_ATTEMPTS", "2") or 2))
        retry_sleep_s = max(0.0, float(os.getenv("LLM_ITINERARY_RETRY_SLEEP_S", "1.0") or 1.0))
        for attempt in range(retry_count):
            for style in self._api_style_sequence():
                try:
                    content = (
                        await self._call_responses_api(client, user_instruction)
                        if style == "responses"
                        else await self._call_chat_completions(client, user_instruction)
                    )
                    if str(content or "").strip():
                        return content, style
                    errors.append(f"{style}: empty_content")
                except Exception as exc:
                    errors.append(f"{style}: {exc}")
            if attempt + 1 < retry_count and retry_sleep_s:
                await asyncio.sleep(retry_sleep_s * (attempt + 1))
        raise RuntimeError(" | ".join(errors) if errors else "unknown_llm_call_error")

    def _normalize_output(self, user_instruction: str, parsed: dict[str, Any]) -> dict[str, Any]:
        heuristic = self.heuristic_decompose(user_instruction)
        parser = IntentParser(settings=self.settings)
        raw_goals = (
            parsed.get("goals")
            or parsed.get("ordered_goals")
            or parsed.get("stops")
            or parsed.get("waypoints")
            or parsed.get("destinations")
        )
        if not isinstance(raw_goals, list) or not raw_goals:
            return heuristic

        goals: list[dict[str, Any]] = []
        for index, raw_goal in enumerate(raw_goals[:4]):
            if not isinstance(raw_goal, dict):
                continue
            request_text = str(
                raw_goal.get("request_text")
                or raw_goal.get("instruction")
                or raw_goal.get("query")
                or raw_goal.get("description")
                or raw_goal.get("goal")
                or raw_goal.get("name")
                or ""
            ).strip()
            if not request_text:
                continue
            normalization_goal = dict(raw_goal)
            normalization_goal.setdefault(
                "mode",
                parsed.get("global_mode") or heuristic.get("global_mode", "car"),
            )
            normalized_intent = parser._normalize_llm_output(  # type: ignore[attr-defined]
                request_text,
                normalization_goal,
            )
            raw_goal_type = str(raw_goal.get("goal_type") or raw_goal.get("destination_type") or "").strip()
            goal_type = _normalize_goal_label(
                raw_goal_type,
                fallback=_normalize_goal_label(normalized_intent.get("destination_type"), fallback="generic_stop"),
            )
            goals.append(
                {
                    "goal_index": index,
                    "goal_type": goal_type,
                    "request_text": request_text,
                    "destination_type": normalized_intent.get("destination_type", "poi"),
                    "destination_kind": normalized_intent.get("destination_kind", "generic_poi"),
                    "search_keywords": normalized_intent.get("search_keywords", default_search_keywords()),
                    "mode": normalized_intent.get("mode", heuristic.get("global_mode", "car")),
                    "radius_m": normalized_intent.get("radius_m", heuristic.get("global_radius_m", 3000)),
                    "optimize_for": normalized_intent.get("optimize_for", "distance"),
                    "hard_constraints": normalized_intent.get(
                        "hard_constraints",
                        default_intent_rules()["default_hard_constraints"],
                    ),
                    "rationale": str(raw_goal.get("rationale") or "").strip(),
                }
            )

        if not goals:
            return heuristic

        global_mode = str(parsed.get("global_mode") or heuristic.get("global_mode") or "car").strip().lower()
        if global_mode not in {"car", "foot", "bike"}:
            global_mode = heuristic.get("global_mode", "car")

        radii = [
            int(goal.get("radius_m", heuristic.get("global_radius_m", 3000)) or 3000)
            for goal in goals
            if isinstance(goal.get("radius_m", 3000), (int, float))
        ]

        return {
            "goals": goals,
            "global_mode": global_mode,
            "global_radius_m": max(radii or [int(heuristic.get("global_radius_m", 3000) or 3000)]),
            "is_multi_goal": len(goals) > 1,
            "decomposition_source": "llm",
            "notes": str(parsed.get("notes") or "").strip(),
        }

    async def parse(self, user_instruction: str) -> dict[str, Any]:
        if not self.is_available():
            parsed = self.heuristic_decompose(user_instruction)
            parsed["decomposition_source"] = "heuristic_no_llm"
            return parsed

        client = _create_async_client_with_timeout(self.settings, self._itinerary_timeout_s())
        try:
            try:
                content, api_style_used = await self._call_llm_with_fallback(client, user_instruction)
            except Exception as exc:
                parsed = self.heuristic_decompose(user_instruction)
                parsed["decomposition_source"] = "heuristic_llm_api_error"
                parsed["llm_error"] = str(exc)
                return parsed

            try:
                parsed = _parse_json_object_from_text(content)
            except Exception as exc:
                fallback = self.heuristic_decompose(user_instruction)
                fallback["decomposition_source"] = "heuristic_llm_parse_error"
                fallback["llm_error"] = str(exc)
                return fallback

            normalized = self._normalize_output(user_instruction, parsed)
            normalized["llm_api_style_used"] = api_style_used
            return normalized
        except Exception:
            parsed = self.heuristic_decompose(user_instruction)
            parsed["decomposition_source"] = "heuristic_llm_error"
            return parsed
        finally:
            await _close_async_client(client)
