"""Vision responsibilities for language and vision services."""
from __future__ import annotations

from typing import Any
import json
import os
import traceback
from time import perf_counter
from mcp_server.llm_services.base import BaseLLMService
from mcp_server.llm_services.transport import _apply_chat_template_kwargs, _apply_reasoning_effort, _chat_response_metadata, _chat_stream_finish_reason, _chat_stream_metadata, _close_async_client, _create_async_client_with_timeout, _extract_chat_stream_delta_text, _extract_text_from_chat_response, _extract_text_from_responses_response, _response_usage_dict, _responses_metadata, _truthy_env, _vision_call_debug_enabled, _vision_request_debug_summary, _vision_url_payload_stats

class VisionAnalyzer(BaseLLMService):
    """Analyze an image through the configured multimodal LLM endpoint."""

    def _vision_timeout_s(self) -> float:
        raw = os.getenv("LLM_VISION_TIMEOUT_S", "").strip()
        if raw:
            try:
                return max(5.0, float(raw))
            except ValueError:
                pass
        return max(90.0, self.settings.timeout_s)

    def _vision_max_tokens(self) -> int | None:
        """Return the optional response-token cap for multimodal requests."""

        raw = os.getenv("LLM_VISION_MAX_TOKENS", "").strip()
        if not raw:
            return None
        try:
            return max(64, int(raw))
        except ValueError as exc:
            raise ValueError(
                "LLM_VISION_MAX_TOKENS must be an integer"
            ) from exc

    def _vision_json_response_enabled(self) -> bool:
        """Return whether multimodal chat calls should request JSON mode."""

        return _truthy_env("LLM_VISION_RESPONSE_FORMAT_JSON")

    def _stream_timing_enabled(self) -> bool:
        return _truthy_env("LLM_VISION_STREAM_TIMING") or _truthy_env("LLM_STREAM_TIMING")

    def build_messages(
        self,
        image_url: str,
        question: str,
        detail: str = "auto",
        system_prompt: str | None = None,
        extra_image_urls: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        prompt = system_prompt or (
            "You are a visual assistant. Answer the user's question about the image directly and concisely. "
            "If the image is unclear or inaccessible, say so plainly."
        )
        image_urls = [str(image_url).strip()]
        for extra in list(extra_image_urls or []):
            value = str(extra or "").strip()
            if value:
                image_urls.append(value)
        content: list[dict[str, Any]] = []
        for url in image_urls:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": url,
                        "detail": detail,
                    },
                }
            )
        content.append(
            {
                "type": "text",
                "text": question,
            }
        )
        return [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": content,
            },
        ]

    async def _stream_chat_completions(self, client: Any, payload: dict[str, Any]) -> dict[str, Any]:
        """Run chat-completions streaming and expose first-token timing.

        Without provider-side Server-Timing headers, the best client-side split is:

            time_to_first_token ~= upload + provider queue + image encode + prefill
            stream_decode_ms    ~= output decoding + streamed network return
        """

        started_at = perf_counter()
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        if _truthy_env("LLM_STREAM_INCLUDE_USAGE", default=True):
            stream_payload["stream_options"] = {"include_usage": True}

        try:
            stream = await client.chat.completions.create(**stream_payload)
        except Exception:
            if "stream_options" not in stream_payload:
                raise
            # Some OpenAI-compatible providers support streaming but reject
            # stream_options. Keep first-token timing and lose usage details.
            stream_payload.pop("stream_options", None)
            stream = await client.chat.completions.create(**stream_payload)
        chunks = 0
        content_chunks = 0
        first_chunk_ms: int | None = None
        first_content_ms: int | None = None
        last_content_ms: int | None = None
        response_id: str | None = None
        created: Any = None
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        parts: list[str] = []

        async for chunk in stream:
            now_ms = int(round((perf_counter() - started_at) * 1000.0))
            chunks += 1
            if first_chunk_ms is None:
                first_chunk_ms = now_ms
            response_id = response_id or getattr(chunk, "id", None)
            created = created or getattr(chunk, "created", None)
            finish_reason = _chat_stream_finish_reason(chunk) or finish_reason
            chunk_usage = _response_usage_dict(chunk)
            if chunk_usage:
                usage = chunk_usage
            text = _extract_chat_stream_delta_text(chunk)
            if text:
                parts.append(text)
                content_chunks += 1
                if first_content_ms is None:
                    first_content_ms = now_ms
                last_content_ms = now_ms

        roundtrip_ms = int(round((perf_counter() - started_at) * 1000.0))
        if first_content_ms is None:
            first_content_ms = first_chunk_ms
        stream_decode_ms = max(0, roundtrip_ms - int(first_content_ms)) if first_content_ms is not None else None
        return {
            "answer": "".join(parts),
            "roundtrip_ms": roundtrip_ms,
            "stream_timing": {
                "enabled": True,
                "time_to_first_chunk_ms": first_chunk_ms,
                "time_to_first_token_ms": first_content_ms,
                "stream_decode_ms": stream_decode_ms,
                "last_content_chunk_ms": last_content_ms,
                "chunks": chunks,
                "content_chunks": content_chunks,
            },
            "response": _chat_stream_metadata(
                chunks=chunks,
                response_id=response_id,
                created=created,
                finish_reason=finish_reason,
                usage=usage,
            ),
        }

    async def _call_chat_completions(
        self,
        client: Any,
        image_url: str,
        question: str,
        detail: str,
        system_prompt: str | None,
        extra_image_urls: list[str] | None = None,
    ) -> dict[str, Any]:
        messages = self.build_messages(
            image_url,
            question,
            detail,
            system_prompt,
            extra_image_urls=extra_image_urls,
        )
        image_urls = [str(image_url).strip(), *[str(item).strip() for item in list(extra_image_urls or []) if str(item or "").strip()]]
        started_at = perf_counter()
        payload = _apply_reasoning_effort(
            {
                "model": self.settings.model,
                "temperature": self.settings.temperature,
                "messages": messages,
            },
            self.settings,
        )
        payload = _apply_chat_template_kwargs(payload, self.settings)
        max_tokens = self._vision_max_tokens()
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self._vision_json_response_enabled():
            payload["response_format"] = {"type": "json_object"}
        try:
            if self._stream_timing_enabled():
                streamed = await self._stream_chat_completions(client, payload)
                roundtrip_ms = int(streamed.get("roundtrip_ms") or 0)
                answer = str(streamed.get("answer") or "")
                response_metadata = streamed.get("response") if isinstance(streamed.get("response"), dict) else {}
                stream_timing = streamed.get("stream_timing") if isinstance(streamed.get("stream_timing"), dict) else {}
            else:
                response = await client.chat.completions.create(**payload)
                roundtrip_ms = int(round((perf_counter() - started_at) * 1000.0))
                answer = _extract_text_from_chat_response(response)
                response_metadata = _chat_response_metadata(response)
                stream_timing = {"enabled": False}
        except Exception as exc:
            if _vision_call_debug_enabled():
                debug_summary = _vision_request_debug_summary(
                    api_style="chat_completions",
                    settings=self.settings,
                    image_urls=image_urls,
                    question=question,
                    system_prompt=system_prompt,
                    detail=detail,
                )
                debug_summary["timeout_s"] = self._vision_timeout_s()
                setattr(exc, "vision_debug", debug_summary)
                setattr(exc, "traceback", traceback.format_exc())
            raise
        request_json = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        return {
            "answer": answer,
            "telemetry": {
                "api_style": "chat_completions",
                "provider": self.settings.provider,
                "model": self.settings.model,
                "roundtrip_ms": roundtrip_ms,
                "request_text_chars": len(str(system_prompt or "")) + len(str(question or "")),
                "request_payload_chars": len(request_json),
                "response_text_chars": len(answer),
                "temperature": self.settings.temperature,
                "reasoning_effort": self.settings.reasoning_effort,
                "max_tokens": max_tokens,
                "json_response_format": (
                    self._vision_json_response_enabled()
                ),
                "timeout_s": self._vision_timeout_s(),
                "max_retries": self.settings.max_retries,
                "image_payload": _vision_url_payload_stats(image_urls),
                "response": response_metadata,
                "prompt_cached_tokens": (response_metadata or {}).get("cached_tokens"),
                "stream_timing": stream_timing,
                "server_timing_available": False,
            },
        }

    async def _call_responses_api(
        self,
        client: Any,
        image_url: str,
        question: str,
        detail: str,
        system_prompt: str | None,
        extra_image_urls: list[str] | None = None,
    ) -> dict[str, Any]:
        prompt = system_prompt or (
            "You are a visual assistant. Answer the user's question about the image directly and concisely."
        )
        images = [str(image_url).strip()]
        for extra in list(extra_image_urls or []):
            value = str(extra or "").strip()
            if value:
                images.append(value)
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for url in images:
            content.append({"type": "input_image", "image_url": url, "detail": detail})
        content.append({"type": "input_text", "text": question})
        request_input = [
            {
                "role": "user",
                "content": content,
            }
        ]
        started_at = perf_counter()
        payload = _apply_reasoning_effort(
            {
                "model": self.settings.model,
                "input": request_input,
            },
            self.settings,
        )
        max_tokens = self._vision_max_tokens()
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        try:
            response = await client.responses.create(**payload)
            roundtrip_ms = int(round((perf_counter() - started_at) * 1000.0))
            answer = _extract_text_from_responses_response(response)
            response_metadata = _responses_metadata(response)
        except Exception as exc:
            if _vision_call_debug_enabled():
                debug_summary = _vision_request_debug_summary(
                    api_style="responses",
                    settings=self.settings,
                    image_urls=images,
                    question=question,
                    system_prompt=prompt,
                    detail=detail,
                )
                debug_summary["timeout_s"] = self._vision_timeout_s()
                setattr(exc, "vision_debug", debug_summary)
                setattr(exc, "traceback", traceback.format_exc())
            raise
        request_json = json.dumps(request_input, ensure_ascii=False, separators=(",", ":"))
        return {
            "answer": answer,
            "telemetry": {
                "api_style": "responses",
                "provider": self.settings.provider,
                "model": self.settings.model,
                "roundtrip_ms": roundtrip_ms,
                "request_text_chars": len(str(prompt or "")) + len(str(question or "")),
                "request_payload_chars": len(request_json),
                "response_text_chars": len(answer),
                "temperature": self.settings.temperature,
                "reasoning_effort": self.settings.reasoning_effort,
                "max_tokens": max_tokens,
                "json_response_format": False,
                "timeout_s": self._vision_timeout_s(),
                "max_retries": self.settings.max_retries,
                "image_payload": _vision_url_payload_stats(images),
                "response": response_metadata,
                "prompt_cached_tokens": response_metadata.get("cached_tokens"),
                "server_timing_available": False,
            },
        }

    async def analyze(
        self,
        image_url: str,
        question: str = "What is in this image?",
        detail: str = "auto",
        system_prompt: str | None = None,
        extra_image_urls: list[str] | None = None,
    ) -> dict[str, Any]:
        if not image_url.strip():
            raise ValueError("image_url is required")
        if detail not in {"auto", "low", "high"}:
            detail = "auto"
        resolved_extra_image_urls = [
            str(item).strip() for item in list(extra_image_urls or []) if str(item or "").strip()
        ]

        if not self.is_available():
            raise RuntimeError("LLM image analysis is not configured. Set ARK_API_KEY or LLM_API_KEY.")

        client = _create_async_client_with_timeout(self.settings, self._vision_timeout_s())
        try:
            try:
                if self.settings.api_style == "responses":
                    call = await self._call_responses_api(
                        client,
                        image_url,
                        question,
                        detail,
                        system_prompt,
                        extra_image_urls=resolved_extra_image_urls,
                    )
                else:
                    call = await self._call_chat_completions(
                        client,
                        image_url,
                        question,
                        detail,
                        system_prompt,
                        extra_image_urls=resolved_extra_image_urls,
                    )
            except Exception as exc:
                if _vision_call_debug_enabled() and not isinstance(getattr(exc, "vision_debug", None), dict):
                    image_urls = [str(image_url).strip(), *resolved_extra_image_urls]
                    debug_summary = _vision_request_debug_summary(
                        api_style=self.settings.api_style,
                        settings=self.settings,
                        image_urls=image_urls,
                        question=question,
                        system_prompt=system_prompt,
                        detail=detail,
                    )
                    debug_summary["timeout_s"] = self._vision_timeout_s()
                    setattr(exc, "vision_debug", debug_summary)
                    setattr(exc, "traceback", traceback.format_exc())
                raise
            answer = str(call.get("answer") or "")

            return {
                "provider": self.settings.provider,
                "api_style": self.settings.api_style,
                "model": self.settings.model,
                "image_url": image_url,
                "image_urls": [str(image_url).strip(), *resolved_extra_image_urls],
                "question": question,
                "detail": detail,
                "answer": answer.strip(),
                "llm_telemetry": call.get("telemetry") if isinstance(call, dict) else None,
            }
        finally:
            await _close_async_client(client)
