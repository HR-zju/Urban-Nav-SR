"""Intent responsibilities for language and vision services."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any
import json
import os
from mcp_server.config import default_intent_rules, default_search_keywords, default_search_categories, get_few_shot_examples, join_search_keywords, load_json_env, split_search_keywords
from mcp_server.llm_services.base import BaseLLMService
from mcp_server.llm_services.intent_rules import DESTINATION_KIND_VALUES, _cjk_heuristic_intent, _clamp_probability, _heuristic_named_entity_confidence, _heuristic_parse_confidence, extract_explicit_destination_query, infer_destination_kind
from mcp_server.llm_services.settings import LLMSettings
from mcp_server.llm_services.transport import _apply_chat_template_kwargs, _close_async_client, _create_async_client, _extract_text_from_chat_response, _extract_text_from_responses_response
from mcp_server.llm_services.json_codec import _json_only_contract, _parse_json_object_from_text

class IntentParser(BaseLLMService):
    """Parse recommendation instructions with an LLM and a heuristic fallback."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        super().__init__(settings=settings)
        intent_model = os.getenv("LLM_INTENT_MODEL", "").strip()
        intent_timeout = os.getenv("LLM_INTENT_TIMEOUT_S", "").strip()
        intent_retries = os.getenv("LLM_INTENT_MAX_RETRIES", "").strip()
        updates: dict[str, Any] = {}
        if intent_model:
            updates["model"] = intent_model
        if intent_timeout:
            updates["timeout_s"] = float(intent_timeout)
        if intent_retries:
            updates["max_retries"] = max(0, int(intent_retries))
        if updates:
            self.settings = replace(self.settings, **updates)

    def _prompt_profile(self) -> str:
        profile = os.getenv("LLM_INTENT_PROFILE", os.getenv("INTENT_PARSE_PROFILE", "compact"))
        profile = str(profile or "compact").strip().lower()
        return profile if profile in {"compact", "full"} else "compact"

    async def _call_llm_with_fallback(self, client: Any, user_instruction: str) -> tuple[str, str]:
        errors: list[str] = []
        retry_count = max(1, int(os.getenv("LLM_INTENT_CALL_ATTEMPTS", "2") or 2))
        retry_sleep_s = max(0.0, float(os.getenv("LLM_INTENT_RETRY_SLEEP_S", "1.0") or 1.0))
        for attempt in range(retry_count):
            for style in self._api_style_sequence():
                try:
                    if style == "responses":
                        content = await self._call_responses_api(client, user_instruction)
                    else:
                        content = await self._call_chat_completions(client, user_instruction)

                    if str(content or "").strip():
                        return content, style
                    errors.append(f"{style}: empty_content")
                except Exception as exc:
                    errors.append(f"{style}: {exc}")
            if attempt + 1 < retry_count and retry_sleep_s:
                await asyncio.sleep(retry_sleep_s * (attempt + 1))

        raise RuntimeError(" | ".join(errors) if errors else "unknown_llm_call_error")

    def _extract_explicit_destination_query(self, user_instruction: str) -> str:
        return extract_explicit_destination_query(user_instruction)

    def heuristic_parse(self, user_instruction: str) -> dict[str, Any]:
        rules = load_json_env("INTENT_RULES_JSON", default_intent_rules())
        text = user_instruction.lower()

        mode = "car"
        for candidate_mode, tokens in rules.get("mode_tokens", {}).items():
            if any(token.lower() in text for token in tokens):
                mode = candidate_mode
                break

        destination_type = "poi"
        search_keywords = rules.get("destination_search_keywords", {}).get("poi", default_search_keywords())

        explicit_query = self._extract_explicit_destination_query(user_instruction)
        cjk_intent = _cjk_heuristic_intent(user_instruction)
        if cjk_intent:
            destination_type = cjk_intent["destination_type"]
            search_keywords = cjk_intent["search_keywords"]
        elif explicit_query:
            destination_type = "named_place"
            search_keywords = explicit_query
        else:
            for destination_rule in rules.get("destination_rules", []):
                tokens = destination_rule.get("tokens", [])
                if any(token.lower() in text for token in tokens):
                    destination_type = destination_rule.get("type", "poi")
                    search_keywords = rules.get("destination_search_keywords", {}).get(destination_type, search_keywords)
                    break

        optimize_for = "distance"
        if any(token.lower() in text for token in rules.get("time_optimization_tokens", [])):
            optimize_for = "time"

        radius = rules.get("default_radius_m", 3000)
        if any(token.lower() in text for token in rules.get("nearby_tokens", [])):
            radius = rules.get("nearby_radius_m", 2000)

        destination_kind = infer_destination_kind(
            user_instruction=user_instruction,
            destination_type=destination_type,
            search_keywords=search_keywords,
        )

        return {
            "destination_type": destination_type,
            "destination_kind": destination_kind,
            "search_keywords": search_keywords,
            "mode": mode,
            "radius_m": radius,
            "optimize_for": optimize_for,
            "max_candidates": rules.get("default_max_candidates", 5),
            "hard_constraints": rules.get("default_hard_constraints", {}),
            "confidence": _heuristic_parse_confidence(
                user_instruction=user_instruction,
                destination_type=destination_type,
                destination_kind=destination_kind,
            ),
            "named_entity_confidence": _heuristic_named_entity_confidence(
                user_instruction=user_instruction,
                destination_type=destination_type,
                destination_kind=destination_kind,
                search_keywords=search_keywords,
            ),
            "intent_source": "heuristic",
        }

    def build_messages(self, user_instruction: str) -> list[dict[str, str]]:
        if self._prompt_profile() == "compact":
            schema = {
                "destination_type": "poi",
                "destination_kind": "generic_poi",
                "search_keywords": "restaurant|cafe",
                "mode": "foot",
                "radius_m": 2000,
                "optimize_for": "distance",
                "max_candidates": 10,
                "hard_constraints": {
                    "max_route_duration_min": None,
                    "max_route_distance_m": None,
                    "must_include_keywords": [],
                    "exclude_keywords": [],
                },
                "confidence": 0.8,
                "named_entity_confidence": 0.0,
            }
            return [
                {
                    "role": "system",
                    "content": (
                        "Parse a navigation/recommendation request. Return one JSON object only with keys: "
                        "destination_type, destination_kind, search_keywords, mode, radius_m, optimize_for, "
                        "max_candidates, hard_constraints, confidence, named_entity_confidence. "
                        "destination_kind must be one of generic_poi,named_landmark,regional_scenic_area,"
                        "administrative_area,road_or_route_label,specific_poi. mode is car/foot/bike. "
                        "This UrbanNav benchmark is pedestrian navigation: use mode=foot unless the user "
                        "explicitly asks to drive or use a car. "
                        "search_keywords must be concise provider-facing POI/place search text, pipe-delimited "
                        "when multiple categories are useful. hard_constraints has max_route_duration_min, "
                        "max_route_distance_m, must_include_keywords, exclude_keywords. JSON only."
                        + " "
                        + _json_only_contract(schema)
                    ),
                },
                {
                    "role": "user",
                    "content": f"Request: {user_instruction}\nReturn the JSON object now.",
                },
            ]

        messages = [
            {
                "role": "system",
                "content": (
                "You are an intent parser for map recommendation and routing. "
                "Return JSON only. The JSON schema must contain exactly these top-level keys: "
                "destination_type, destination_kind, search_keywords, mode, radius_m, optimize_for, "
                "max_candidates, hard_constraints, confidence, named_entity_confidence. "
                "Allowed destination_kind values: generic_poi, named_landmark, regional_scenic_area, "
                "administrative_area, road_or_route_label, specific_poi. "
                "Allowed mode values: car, foot, bike. "
                "Allowed optimize_for values: time, distance, balanced. "
                "radius_m and max_candidates must be integers. "
                "confidence and named_entity_confidence must be numbers between 0 and 1. "
                "search_keywords must be a provider-facing keyword string, preferably pipe-delimited, "
                "for example 'bookstore|cafe' or 'convenience store|tea shop'. "
                "If the user gives an explicit destination name (for example 'West Lake' or 'Times Square'), "
                "set destination_type appropriately, use destination_kind=named_landmark or "
                "destination_kind=regional_scenic_area when the destination is a broad scenic region, and put that destination name directly into "
                "search_keywords instead of generic categories. "
                "For broad need-based requests such as thirsty, hungry, tired, need coffee, need groceries, or need a restroom, "
                "output broad retrievable provider-facing search_keywords that cover several suitable POI types. "
                "Use hard_constraints only for constraints that are explicitly stated by the user. "
                "Do not invent max route duration, max route distance, must_include_keywords, or exclude_keywords from vague preferences alone. "
                "hard_constraints must include keys: max_route_duration_min, max_route_distance_m, "
                "must_include_keywords, exclude_keywords. "
                "Do not add explanation or markdown."
                ),
            }
        ]

        for example in get_few_shot_examples():
            request = example.get("request", "")
            output = example.get("output", {})
            if not request or not output:
                continue
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Parse this request into domestic-map recommendation constraints. "
                        "Use provider-facing search_keywords instead of category taxonomy. "
                        f"request: {request}"
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(output, ensure_ascii=True),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": (
                    "Parse this request into domestic-map recommendation constraints. "
                    "Prefer a concise provider-facing search_keywords string that can be sent directly to "
                    "AMap/Baidu nearby search APIs. Avoid category taxonomy. "
                    f"request: {user_instruction}"
                ),
            }
        )
        return messages

    def build_prompt(self, user_instruction: str) -> str:
        if self._prompt_profile() == "compact":
            return "\n".join(
                [
                    self.build_messages(user_instruction)[0]["content"],
                    f"Request: {user_instruction}",
                ]
            )

        sections = [self.build_messages(user_instruction)[0]["content"], "", "Few-shot examples:"]

        for example in get_few_shot_examples():
            request = example.get("request", "")
            output = example.get("output", {})
            if not request or not output:
                continue
            sections.append(f"Request: {request}")
            sections.append(f"JSON: {json.dumps(output, ensure_ascii=True)}")

        sections.extend(
            [
                "",
                "Now parse the following request and return JSON only.",
                f"Request: {user_instruction}",
            ]
        )
        return "\n".join(sections)

    async def _call_chat_completions(self, client: Any, user_instruction: str) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": self.build_messages(user_instruction),
        }
        max_tokens_env = os.getenv("LLM_INTENT_MAX_TOKENS", "").strip()
        if max_tokens_env:
            payload["max_tokens"] = max(32, int(max_tokens_env))
        elif self._prompt_profile() == "compact":
            payload["max_tokens"] = 768
        # ARK chat completions can reject JSON response_format on some endpoints/models.
        # Keep the prompt JSON-only and fall back to regex extraction instead.
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
                    "content": [
                        {
                            "type": "input_text",
                            "text": self.build_prompt(user_instruction),
                        }
                    ],
                }
            ],
        }
        max_tokens_env = os.getenv("LLM_INTENT_MAX_TOKENS", "").strip()
        if max_tokens_env:
            payload["max_output_tokens"] = max(32, int(max_tokens_env))
        elif self._prompt_profile() == "compact":
            payload["max_output_tokens"] = 768
        response = await client.responses.create(**payload)
        return _extract_text_from_responses_response(response)

    def _normalize_llm_output(self, user_instruction: str, parsed: dict[str, Any]) -> dict[str, Any]:
        heuristic_defaults = self.heuristic_parse(user_instruction)
        normalized = dict(heuristic_defaults)
        normalized.update(parsed)

        mode = str(normalized.get("mode", "car")).strip().lower()
        if mode not in {"car", "foot", "bike"}:
            normalized["mode"] = "car"
        else:
            normalized["mode"] = mode

        optimize_for = str(normalized.get("optimize_for", "distance")).strip().lower()
        if optimize_for not in {"time", "distance", "balanced"}:
            normalized["optimize_for"] = "distance"
        else:
            normalized["optimize_for"] = optimize_for

        destination_type = str(normalized.get("destination_type", "poi")).strip() or "poi"
        destination_kind = str(normalized.get("destination_kind", "")).strip().lower()
        if destination_kind not in DESTINATION_KIND_VALUES:
            destination_kind = infer_destination_kind(
                user_instruction=user_instruction,
                destination_type=destination_type,
                search_keywords=str(normalized.get("search_keywords") or normalized.get("keywords") or ""),
            )
        normalized["destination_kind"] = destination_kind
        normalized["search_keywords"] = join_search_keywords(
            normalized.get("search_keywords") or normalized.get("keywords"),
            default=default_intent_rules().get("destination_search_keywords", {}).get(
                destination_type,
                default_search_keywords(),
            ),
        )
        normalized["confidence"] = _clamp_probability(
            normalized.get("confidence"),
            default=_heuristic_parse_confidence(
                user_instruction=user_instruction,
                destination_type=destination_type,
                destination_kind=destination_kind,
            ),
        )
        normalized["named_entity_confidence"] = _clamp_probability(
            normalized.get("named_entity_confidence"),
            default=_heuristic_named_entity_confidence(
                user_instruction=user_instruction,
                destination_type=destination_type,
                destination_kind=destination_kind,
                search_keywords=str(normalized.get("search_keywords") or normalized.get("keywords") or ""),
            ),
        )
        normalized["categories"] = default_search_categories()
        normalized["keywords"] = split_search_keywords(normalized["search_keywords"])

        radius = normalized.get("radius_m", 3000)
        if not isinstance(radius, (int, float)) or radius <= 0:
            normalized["radius_m"] = heuristic_defaults["radius_m"]
        else:
            normalized["radius_m"] = int(radius)

        max_candidates = normalized.get("max_candidates", 5)
        if not isinstance(max_candidates, (int, float)) or max_candidates < 1:
            normalized["max_candidates"] = heuristic_defaults["max_candidates"]
        else:
            normalized["max_candidates"] = int(max_candidates)

        hard_constraints = normalized.get("hard_constraints")
        if not isinstance(hard_constraints, dict):
            normalized["hard_constraints"] = default_intent_rules()["default_hard_constraints"]
        else:
            normalized["hard_constraints"] = {
                "max_route_duration_min": hard_constraints.get("max_route_duration_min")
                if isinstance(hard_constraints.get("max_route_duration_min"), (int, float))
                else None,
                "max_route_distance_m": hard_constraints.get("max_route_distance_m")
                if isinstance(hard_constraints.get("max_route_distance_m"), (int, float))
                else None,
                "must_include_keywords": [
                    str(item).strip()
                    for item in hard_constraints.get("must_include_keywords", [])
                    if str(item).strip()
                ],
                "exclude_keywords": [
                    str(item).strip()
                    for item in hard_constraints.get("exclude_keywords", [])
                    if str(item).strip()
                ],
            }

        return normalized

    async def parse(self, user_instruction: str) -> dict[str, Any]:
        if not self.is_available():
            parsed = self.heuristic_parse(user_instruction)
            parsed["intent_source"] = "heuristic_no_llm"
            return parsed

        client = _create_async_client(self.settings)
        try:
            try:
                content, api_style_used = await self._call_llm_with_fallback(client, user_instruction)
            except Exception as exc:
                parsed = self.heuristic_parse(user_instruction)
                parsed["intent_source"] = "heuristic_llm_api_error"
                parsed["llm_error"] = str(exc)
                return parsed

            try:
                parsed = _parse_json_object_from_text(content)
            except Exception as exc:
                fallback = self.heuristic_parse(user_instruction)
                fallback["intent_source"] = "heuristic_llm_parse_error"
                fallback["llm_error"] = str(exc)
                return fallback

            normalized = self._normalize_llm_output(user_instruction, parsed)
            normalized["intent_source"] = "llm"
            normalized["llm_api_style_used"] = api_style_used
            return normalized
        except Exception:
            parsed = self.heuristic_parse(user_instruction)
            parsed["intent_source"] = "heuristic_llm_error"
            return parsed
        finally:
            await _close_async_client(client)
