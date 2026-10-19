"""Ranking responsibilities for language and vision services."""
from __future__ import annotations

import asyncio
from typing import Any
import json
import os
import re
from mcp_server.llm_services.base import BaseLLMService
from mcp_server.llm_services.transport import _apply_chat_template_kwargs, _close_async_client, _create_async_client, _create_async_client_with_timeout, _extract_text_from_chat_response, _extract_text_from_responses_response
from mcp_server.llm_services.json_codec import _extract_json_array_field, _json_only_contract, _parse_json_object_from_text

class ItineraryCandidateJudge(BaseLLMService):
    """Select the best concrete POI for one itinerary goal from Top-K map candidates."""

    def _itinerary_timeout_s(self) -> float:
        raw = os.getenv("LLM_ITINERARY_TIMEOUT_S", "").strip()
        if raw:
            try:
                return max(float(raw), 10.0)
            except Exception:
                pass
        return max(float(self.settings.timeout_s), 45.0)

    def _simplify_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        tags = candidate.get("tags") or {}
        route_summary = candidate.get("route_summary") or {}
        return {
            "id": str(candidate.get("id") or ""),
            "name": str(candidate.get("name") or ""),
            "type": str(tags.get("type") or ""),
            "address": str(tags.get("address") or ""),
            "candidate_kind": str(candidate.get("candidate_kind") or ""),
            "is_concrete_destination": bool(candidate.get("is_concrete_destination", False)),
            "location": str(candidate.get("location") or ""),
            "score": candidate.get("score"),
            "distance_m": route_summary.get("distance_m"),
            "duration_min": route_summary.get("duration_min"),
        }

    def build_messages(
        self,
        *,
        goal: dict[str, Any],
        candidates: list[dict[str, Any]],
        previous_stops: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        payload = {
            "goal": {
                "goal_index": goal.get("goal_index"),
                "goal_type": goal.get("goal_type"),
                "request_text": goal.get("request_text"),
                "destination_type": goal.get("destination_type"),
                "destination_kind": goal.get("destination_kind"),
                "search_keywords": goal.get("search_keywords"),
                "rationale": goal.get("rationale"),
            },
            "previous_stops": [
                {
                    "name": str(item.get("name") or ""),
                    "location": str(item.get("location") or ""),
                    "goal_type": str(item.get("goal_type") or ""),
                }
                for item in previous_stops
            ],
            "candidates": [self._simplify_candidate(item) for item in candidates],
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are an itinerary candidate judge for map navigation. "
                    "Return JSON only with exactly: selected_candidate_id, confidence, reason, reject_all. "
                    "Use open-vocabulary reasoning over the goal, its rationale/search query, and each candidate's name/type/address. "
                    "Do not rely on a closed taxonomy. Select a candidate only if it plausibly satisfies the goal as a real-world service/place. "
                    "Reject candidates that appear closed, unavailable, irrelevant to the activity, or duplicate a previous stop unless the goal explicitly asks to stay there. "
                    "If no candidate satisfies the goal, set selected_candidate_id=null and reject_all=true. "
                    "Prefer concrete POIs over vague administrative areas. Consider route distance and duration only after semantic suitability."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]

    def build_prompt(
        self,
        *,
        goal: dict[str, Any],
        candidates: list[dict[str, Any]],
        previous_stops: list[dict[str, Any]],
    ) -> str:
        messages = self.build_messages(goal=goal, candidates=candidates, previous_stops=previous_stops)
        return f"{messages[0]['content']}\n\nInput JSON:\n{messages[1]['content']}"

    async def _call_chat_completions(
        self,
        client: Any,
        *,
        goal: dict[str, Any],
        candidates: list[dict[str, Any]],
        previous_stops: list[dict[str, Any]],
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": self.build_messages(goal=goal, candidates=candidates, previous_stops=previous_stops),
        }
        if self.settings.response_format and self.settings.provider != "ark":
            payload["response_format"] = {"type": "json_object"}
        payload = _apply_chat_template_kwargs(payload, self.settings)
        response = await client.chat.completions.create(**payload)
        return _extract_text_from_chat_response(response)

    async def _call_responses_api(
        self,
        client: Any,
        *,
        goal: dict[str, Any],
        candidates: list[dict[str, Any]],
        previous_stops: list[dict[str, Any]],
    ) -> str:
        response = await client.responses.create(
            model=self.settings.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self.build_prompt(
                                goal=goal,
                                candidates=candidates,
                                previous_stops=previous_stops,
                            ),
                        }
                    ],
                }
            ],
        )
        return _extract_text_from_responses_response(response)

    def _normalize_decision(self, parsed: dict[str, Any], candidate_ids: set[str]) -> dict[str, Any]:
        selected = parsed.get("selected_candidate_id")
        if selected is not None:
            selected = str(selected).strip()
        if selected and selected not in candidate_ids:
            selected = None

        confidence = parsed.get("confidence", 0.0)
        try:
            confidence_value = max(0.0, min(float(confidence), 1.0))
        except Exception:
            confidence_value = 0.0

        reject_all = bool(parsed.get("reject_all", False)) or selected is None
        return {
            "selected_candidate_id": selected,
            "confidence": confidence_value,
            "reason": str(parsed.get("reason") or "").strip(),
            "reject_all": reject_all,
            "judge_source": "llm",
        }

    async def select_candidate(
        self,
        *,
        goal: dict[str, Any],
        candidates: list[dict[str, Any]],
        previous_stops: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        previous_stops = previous_stops or []
        candidate_ids = {str(item.get("id") or "").strip() for item in candidates if str(item.get("id") or "").strip()}
        if not candidates or not candidate_ids:
            return {
                "selected_candidate_id": None,
                "confidence": 0.0,
                "reason": "No candidates were provided.",
                "reject_all": True,
                "judge_source": "no_candidates",
            }
        if not self.is_available():
            return {
                "selected_candidate_id": None,
                "confidence": 0.0,
                "reason": "LLM candidate judge is not configured.",
                "reject_all": True,
                "judge_source": "unavailable",
            }

        client = _create_async_client_with_timeout(self.settings, self._itinerary_timeout_s())
        try:
            try:
                if self.settings.api_style == "responses":
                    content = await self._call_responses_api(
                        client,
                        goal=goal,
                        candidates=candidates,
                        previous_stops=previous_stops,
                    )
                    api_style_used = "responses"
                else:
                    content = await self._call_chat_completions(
                        client,
                        goal=goal,
                        candidates=candidates,
                        previous_stops=previous_stops,
                    )
                    api_style_used = "chat_completions"
            except Exception as exc:
                return {
                    "selected_candidate_id": None,
                    "confidence": 0.0,
                    "reason": f"LLM candidate judge failed: {exc}",
                    "reject_all": True,
                    "judge_source": "llm_api_error",
                }

            try:
                parsed = _parse_json_object_from_text(content)
            except Exception as exc:
                return {
                    "selected_candidate_id": None,
                    "confidence": 0.0,
                    "reason": f"LLM candidate judge returned non-JSON output: {exc}",
                    "reject_all": True,
                    "judge_source": "llm_parse_error",
                    "raw_text": content,
                }

            decision = self._normalize_decision(parsed, candidate_ids)
            decision["llm_api_style_used"] = api_style_used
            decision["raw"] = parsed
            return decision
        finally:
            await _close_async_client(client)


class CandidateReranker(BaseLLMService):
    """Rerank map candidates by semantic relevance to the user request."""

    def build_messages(
        self,
        *,
        user_instruction: str,
        parsed_intent: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        candidate_ids = [str(item.get("id", "")).strip() for item in candidates if str(item.get("id", "")).strip()]
        simplified_candidates = [
            {
                "id": str(item.get("id", "")),
                "name": str(item.get("name", "")),
                "type": str(item.get("type", "")),
                "address": str(item.get("address", "")),
                "distance_m": item.get("distance_m"),
                "candidate_kind": str(item.get("candidate_kind", "")),
                "is_concrete_destination": bool(item.get("is_concrete_destination", False)),
                "query_affinity": item.get("query_affinity"),
                "anchor_name": str(item.get("anchor_name", "")),
                "anchor_distance_m": item.get("anchor_distance_m"),
            }
            for item in candidates
        ]
        schema = {
            "ranked_candidate_ids": candidate_ids[: min(len(candidate_ids), 5)] or ["candidate_id"],
            "summary": "short",
        }
        return [
            {
                "role": "system",
                "content": (
                    "You rank map destination candidates for an embodied navigation system. "
                    "Return JSON only with exactly two keys: ranked_candidate_ids and summary. "
                    "ranked_candidate_ids must be a non-empty array of candidate ids ordered best to worst. "
                    "The summary must be 12 words or fewer. "
                    "Every id must be copied exactly from the candidate_ids list. Do not invent ids. "
                    "Do not use candidate names, numeric ranks, coordinates, or explanations inside ranked_candidate_ids. "
                    "Prefer destinations that truly satisfy the user's semantic intent. "
                    "If the user asks for a landmark, scenic area, lake, park, tourist attraction, or named place, "
                    "rank concrete navigable scenic destinations ahead of administrative regions, roads, "
                    "institutions, hotels, stores, residences, or branded places that only contain the same "
                    "words in their names or addresses. For regional scenic areas, prioritize the destination "
                    "itself or canonical sub-features directly attached to that area. Do not rank merely nearby "
                    "scenic places highly unless the candidate is clearly part of the requested area."
                    + " "
                    + _json_only_contract(schema)
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_instruction": user_instruction,
                        "parsed_intent": parsed_intent,
                        "candidate_ids": candidate_ids,
                        "candidates": simplified_candidates,
                        "instruction": "Rank all candidates. Return the JSON object now.",
                    },
                    ensure_ascii=True,
                ),
            },
        ]

    async def _call_chat_completions(
        self,
        client: Any,
        *,
        user_instruction: str,
        parsed_intent: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "messages": self.build_messages(
                user_instruction=user_instruction,
                parsed_intent=parsed_intent,
                candidates=candidates,
            ),
            "max_tokens": max(64, int(os.getenv("LLM_RERANK_MAX_TOKENS", "768") or 768)),
        }
        if self.settings.response_format and self.settings.provider != "ark":
            payload["response_format"] = {"type": "json_object"}
        payload = _apply_chat_template_kwargs(payload, self.settings)
        response = await client.chat.completions.create(**payload)
        return _extract_text_from_chat_response(response)

    async def _call_responses_api(
        self,
        client: Any,
        *,
        user_instruction: str,
        parsed_intent: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> str:
        response = await client.responses.create(
            model=self.settings.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(
                                {
                                    "instruction": (
                                        "Rerank these destination candidates by semantic relevance and "
                                        "return JSON only with ranked_candidate_ids and summary."
                                    ),
                                    "user_instruction": user_instruction,
                                    "parsed_intent": parsed_intent,
                                    "candidates": candidates,
                                },
                                ensure_ascii=True,
                            ),
                        }
                    ],
                }
            ],
        )
        return _extract_text_from_responses_response(response)

    async def _call_llm_with_fallback(
        self,
        client: Any,
        *,
        user_instruction: str,
        parsed_intent: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> tuple[str, str]:
        errors: list[str] = []
        retry_count = max(1, int(os.getenv("LLM_RERANK_CALL_ATTEMPTS", "2") or 2))
        retry_sleep_s = max(0.0, float(os.getenv("LLM_RERANK_RETRY_SLEEP_S", "1.0") or 1.0))
        for attempt in range(retry_count):
            for style in self._api_style_sequence():
                try:
                    if style == "responses":
                        content = await self._call_responses_api(
                            client,
                            user_instruction=user_instruction,
                            parsed_intent=parsed_intent,
                            candidates=candidates,
                        )
                    else:
                        content = await self._call_chat_completions(
                            client,
                            user_instruction=user_instruction,
                            parsed_intent=parsed_intent,
                            candidates=candidates,
                        )
                    if str(content or "").strip():
                        return content, style
                    errors.append(f"{style}: empty_content")
                except Exception as exc:
                    errors.append(f"{style}: {exc}")
            if attempt + 1 < retry_count and retry_sleep_s:
                await asyncio.sleep(retry_sleep_s * (attempt + 1))

        raise RuntimeError(" | ".join(errors) if errors else "unknown_llm_call_error")

    async def rerank(
        self,
        *,
        user_instruction: str,
        parsed_intent: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not candidates or not self.is_available():
            return None

        client = _create_async_client(self.settings)
        content = ""
        api_style_used = ""
        try:
            content, api_style_used = await self._call_llm_with_fallback(
                client,
                user_instruction=user_instruction,
                parsed_intent=parsed_intent,
                candidates=candidates,
            )
            parsed = _parse_json_object_from_text(content)
        except Exception as exc:
            recovered_ids = _extract_json_array_field(content, "ranked_candidate_ids")
            if recovered_ids:
                parsed = {
                    "ranked_candidate_ids": recovered_ids,
                    "summary": "",
                    "recovered_from_partial_json": True,
                    "llm_parse_warning": str(exc),
                }
            else:
                preview = str(content or "").strip().replace("\n", " ")
                if len(preview) > 500:
                    preview = preview[:500] + "..."
                return {
                    "ranked_candidate_ids": [],
                    "summary": "",
                    "error": "llm_parse_or_api_error",
                    "llm_error": str(exc),
                    "raw_response_preview": preview,
                    "api_style": api_style_used,
                }
        finally:
            await _close_async_client(client)

        raw_ranked = (
            parsed.get("ranked_candidate_ids")
            or parsed.get("ranked_ids")
            or parsed.get("candidate_ids")
            or parsed.get("ranking")
            or parsed.get("ranked_candidates")
            or []
        )
        if isinstance(raw_ranked, str):
            raw_ranked = [item.strip() for item in re.split(r"[,|\n]", raw_ranked) if item.strip()]
        ranked_ids: list[str] = []
        if isinstance(raw_ranked, list):
            for item in raw_ranked:
                if isinstance(item, dict):
                    value = item.get("id") or item.get("candidate_id") or item.get("provider_poi_id") or item.get("poi_id")
                else:
                    value = item
                value_text = str(value or "").strip()
                if value_text:
                    ranked_ids.append(value_text)
        candidate_ids = [str(item.get("id", "")).strip() for item in candidates if str(item.get("id", "")).strip()]
        if candidate_ids:
            candidate_id_set = set(candidate_ids)
            ranked_ids = [candidate_id for candidate_id in ranked_ids if candidate_id in candidate_id_set]
            seen = set(ranked_ids)
            ranked_ids.extend(candidate_id for candidate_id in candidate_ids if candidate_id not in seen)
        if not ranked_ids:
            return None
        return {
            "ranked_candidate_ids": ranked_ids,
            "summary": str(parsed.get("summary", "")).strip(),
            "api_style": api_style_used,
        }
