"""Base responsibilities for language and vision services."""
from __future__ import annotations

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from mcp_server.llm_services.settings import LLMSettings

class BaseLLMService:
    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings.from_env()

    def is_available(self) -> bool:
        return bool(self.settings.api_key and AsyncOpenAI is not None)

    def _api_style_sequence(self) -> list[str]:
        preferred_style = self.settings.api_style
        if not self.settings.allow_api_style_fallback:
            return [preferred_style]
        fallback_style = "responses" if preferred_style == "chat_completions" else "chat_completions"
        ordered: list[str] = []
        for style in (preferred_style, fallback_style):
            if style not in ordered:
                ordered.append(style)
        return ordered
