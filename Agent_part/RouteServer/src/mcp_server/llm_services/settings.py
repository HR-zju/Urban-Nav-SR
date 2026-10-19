"""Settings responsibilities for language and vision services."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os

@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_style: str
    api_key: str
    model: str
    base_url: str
    temperature: float = 0.0
    timeout_s: float = 45.0
    max_retries: int = 5
    allow_api_style_fallback: bool = False
    response_format: bool = True
    reasoning_effort: str | None = None
    chat_template_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_env(cls) -> "LLMSettings":
        explicit_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        ark_api_key = os.getenv("ARK_API_KEY", "").strip()
        provider = explicit_provider or ("ark" if ark_api_key else "openai")
        default_model = "doubao-seed-2-0-pro-260215" if provider == "ark" else "gpt-4o-mini"
        default_base_url = (
            "https://ark.cn-beijing.volces.com/api/v3"
            if provider == "ark"
            else "https://api.openai.com/v1"
        )
        api_key = os.getenv("LLM_API_KEY", "").strip() or ark_api_key
        api_style = os.getenv("LLM_API_STYLE", "").strip().lower() or (
            "chat_completions" if provider == "ark" else "chat_completions"
        )
        if api_style not in {"responses", "chat_completions"}:
            api_style = "chat_completions" if provider == "ark" else "chat_completions"
        model = (
            os.getenv("LLM_MODEL", "").strip()
            or os.getenv("VLM_MODEL", "").strip()
            or os.getenv("ARK_MODEL", "").strip()
            or default_model
        )
        base_url = (
            os.getenv("LLM_API_BASE_URL", "").strip()
            or os.getenv("ARK_BASE_URL", "").strip()
            or os.getenv("ARK_API_BASE", "").strip()
            or os.getenv("OPENAI_BASE_URL", "").strip()
            or default_base_url
        )
        default_timeout_s = 45.0 if provider == "ark" else 20.0
        default_max_retries = 5 if provider == "ark" else 2
        reasoning_effort = (
            os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
            or os.getenv("ARK_REASONING_EFFORT", "").strip().lower()
        )
        if not reasoning_effort and provider == "ark":
            reasoning_effort = "minimal"
        if reasoning_effort not in {"minimal", "low", "medium", "high"}:
            reasoning_effort = None
        chat_template_kwargs: dict[str, Any] | None = None
        enable_thinking_env = os.getenv("LLM_CHAT_TEMPLATE_ENABLE_THINKING", "").strip().lower()
        if enable_thinking_env:
            chat_template_kwargs = {
                "enable_thinking": enable_thinking_env in {"1", "true", "yes", "on"},
            }
        fallback_env = os.getenv("LLM_API_STYLE_FALLBACK", "").strip().lower()
        if fallback_env:
            allow_api_style_fallback = fallback_env in {"1", "true", "yes"}
        else:
            allow_api_style_fallback = provider != "ark"
        return cls(
            provider=provider,
            api_style=api_style,
            api_key=api_key,
            model=model,
            base_url=base_url.rstrip("/"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0") or 0),
            timeout_s=float(os.getenv("LLM_TIMEOUT_S", str(default_timeout_s)) or default_timeout_s),
            max_retries=max(0, int(os.getenv("LLM_MAX_RETRIES", str(default_max_retries)) or default_max_retries)),
            allow_api_style_fallback=allow_api_style_fallback,
            response_format=os.getenv("LLM_RESPONSE_FORMAT_JSON", "true").strip().lower() not in {"0", "false", "no"},
            reasoning_effort=reasoning_effort,
            chat_template_kwargs=chat_template_kwargs,
        )
