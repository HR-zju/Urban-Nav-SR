"""Api responsibilities for language and vision services."""
from __future__ import annotations

from mcp_server.visualization.images import local_image_path_to_data_url
from typing import Any
from pathlib import Path
from mcp_server.llm_services.ranking import CandidateReranker
from mcp_server.llm_services.intent import IntentParser
from mcp_server.llm_services.localization import MapLocalizationService
from mcp_server.llm_services.vision import VisionAnalyzer

async def parse_instruction_with_llm(user_instruction: str) -> dict[str, Any]:
    return await IntentParser().parse(user_instruction)


async def rerank_candidates_with_llm(
    *,
    user_instruction: str,
    parsed_intent: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    return await CandidateReranker().rerank(
        user_instruction=user_instruction,
        parsed_intent=parsed_intent,
        candidates=candidates,
    )


async def analyze_image_with_llm(
    image_url: str,
    question: str = "What is in this image?",
    detail: str = "auto",
    system_prompt: str | None = None,
) -> dict[str, Any]:
    return await VisionAnalyzer().analyze(
        image_url=image_url,
        question=question,
        detail=detail,
        system_prompt=system_prompt,
    )


async def analyze_local_image_with_llm(
    image_path: str,
    question: str = "What is in this image?",
    detail: str = "auto",
    system_prompt: str | None = None,
) -> dict[str, Any]:
    data_url = local_image_path_to_data_url(image_path)
    result = await VisionAnalyzer().analyze(
        image_url=data_url,
        question=question,
        detail=detail,
        system_prompt=system_prompt,
    )
    result["image_path"] = str(Path(image_path).expanduser())
    result["image_source"] = "local_path"
    return result


async def translate_instruction_for_cn_map(
    *,
    user_instruction: str,
    provider: str,
) -> str | None:
    return await MapLocalizationService().translate_instruction_to_chinese(
        user_instruction=user_instruction,
        provider=provider,
    )


async def translate_result_bundle_to_english(
    *,
    result_bundle: dict[str, Any],
) -> dict[str, Any] | None:
    return await MapLocalizationService().translate_result_bundle_to_english(result_bundle=result_bundle)
