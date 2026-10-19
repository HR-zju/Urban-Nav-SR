"""Timing utilities for current batch pipelines and service telemetry."""

from __future__ import annotations

from time import perf_counter
from typing import Any


def timer_start() -> float:
    """Return an opaque monotonic timestamp for elapsed-time measurements."""

    return perf_counter()


def elapsed_ms(started_at: float) -> int:
    """Return elapsed milliseconds since a timestamp produced by timer_start()."""

    return int(round((perf_counter() - started_at) * 1000.0))


def telemetry_stage_elapsed_ms(telemetry: dict[str, Any] | None, stage: str) -> int | None:
    """Read one named stage duration from a service telemetry payload."""

    if not isinstance(telemetry, dict):
        return None
    stages = telemetry.get("stages")
    if not isinstance(stages, dict):
        return None
    payload = stages.get(stage)
    if not isinstance(payload, dict):
        return None
    try:
        return int(payload.get("elapsed_ms"))
    except Exception:
        return None


def telemetry_llm_call(telemetry: dict[str, Any] | None, stage: str) -> dict[str, Any] | None:
    """Read detailed LLM call telemetry for one stage when the provider exposes it."""

    if not isinstance(telemetry, dict):
        return None
    calls = telemetry.get("llm_calls")
    if isinstance(calls, dict) and isinstance(calls.get(stage), dict):
        return calls[stage]
    stages = telemetry.get("stages")
    if isinstance(stages, dict):
        stage_payload = stages.get(stage)
        if isinstance(stage_payload, dict) and isinstance(stage_payload.get("llm_call"), dict):
            return stage_payload["llm_call"]
    return None




def timing_percentile(values: list[int], percentile: float) -> int | None:
    """Return a nearest-rank percentile from integer millisecond values."""

    if not values:
        return None
    ordered = sorted(int(v) for v in values)
    index = int(round((len(ordered) - 1) * float(percentile)))
    return ordered[max(0, min(index, len(ordered) - 1))]
