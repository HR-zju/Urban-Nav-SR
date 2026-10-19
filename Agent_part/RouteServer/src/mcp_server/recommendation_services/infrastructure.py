"""Infrastructure responsibilities for destination recommendation."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
import copy
from typing import Any
import hashlib
import json
import logging
import os

async def _ctx_info(ctx: Any, message: str) -> None:
    if ctx and hasattr(ctx, "info"):
        await ctx.info(message)


async def _ctx_warning(ctx: Any, message: str) -> None:
    if ctx and hasattr(ctx, "warning"):
        await ctx.warning(message)


async def _ctx_report_progress(ctx: Any, current: int, total: int) -> None:
    if ctx and hasattr(ctx, "report_progress"):
        await ctx.report_progress(current, total)


def _log_timing(stage: str, elapsed_s: float, **fields: Any) -> None:
    payload = {
        "component": "recommendation",
        "stage": stage,
        "elapsed_ms": round(float(elapsed_s) * 1000.0, 2),
    }
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    TIMING_LOGGER.info("TIMING %s", json.dumps(payload, ensure_ascii=True, sort_keys=True))


def _cache_max_entries() -> int:
    return max(8, int(os.getenv("RECOMMENDATION_CACHE_MAX_ENTRIES", "256") or 256))


def _stable_cache_key(parts: list[Any]) -> str:
    return hashlib.sha256(
        json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _cache_get(cache: "OrderedDict[str, Any]", key: str) -> Any | None:
    if key not in cache:
        return None
    value = cache.pop(key)
    cache[key] = value
    return copy.deepcopy(value)


def _cache_put(cache: "OrderedDict[str, Any]", key: str, value: Any) -> None:
    if key in cache:
        cache.pop(key)
    cache[key] = copy.deepcopy(value)
    while len(cache) > _cache_max_entries():
        cache.popitem(last=False)


async def _run_limited(coroutines: list[Any], concurrency: int) -> list[Any]:
    if not coroutines:
        return []

    limit = max(1, int(concurrency))
    semaphore = asyncio.Semaphore(limit)

    async def _runner(coroutine: Any) -> Any:
        async with semaphore:
            return await coroutine

    return await asyncio.gather(*[_runner(coroutine) for coroutine in coroutines], return_exceptions=True)


TIMING_LOGGER = logging.getLogger("mcp_server.timing")


_PARSE_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


_LOCALITY_HINT_CACHE: "OrderedDict[str, list[str]]" = OrderedDict()


_NAMED_GEOCODE_CACHE: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()


_SEMANTIC_SEARCH_CACHE: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
