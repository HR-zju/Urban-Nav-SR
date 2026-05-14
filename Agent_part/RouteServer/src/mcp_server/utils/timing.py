"""Small timing utilities for batch pipelines and service telemetry.

The visual prompt stack already emits per-stage service telemetry. Batch tools
still need a consistent way to convert that nested telemetry into frame-level
logs and manifest summaries. Keeping that logic here prevents orchestration
scripts from growing their own incompatible profiling code.
"""

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


def compact_visual_prompt_frame_timing(
    *,
    payload: dict[str, Any] | None,
    runner_elapsed_ms: int,
    artifact_elapsed_ms: int,
) -> dict[str, Any]:
    """Flatten visual-prompt service telemetry into a manifest-friendly record."""

    telemetry = payload.get("telemetry") if isinstance(payload, dict) else None
    if not isinstance(telemetry, dict):
        telemetry = {}
    timing = {
        "runner_elapsed_ms": int(runner_elapsed_ms),
        "service_elapsed_ms": int(telemetry.get("elapsed_ms") or 0),
        "artifact_write_ms": int(artifact_elapsed_ms),
        "route_context_ms": telemetry_stage_elapsed_ms(telemetry, "route_context"),
        "bev_map_provider_ms": telemetry_stage_elapsed_ms(telemetry, "bev_map_provider"),
        "image_preprocess_ms": telemetry_stage_elapsed_ms(telemetry, "image_preprocess"),
        "segmentation_ms": telemetry_stage_elapsed_ms(telemetry, "segmentation"),
        "som_and_extra_images_ms": telemetry_stage_elapsed_ms(telemetry, "multi_image_inputs"),
        "bev_map_overlay_ms": telemetry_stage_elapsed_ms(telemetry, "bev_map_overlay"),
        "generator_input_build_ms": telemetry_stage_elapsed_ms(telemetry, "A_build_inputs"),
        "generator_llm_ms": telemetry_stage_elapsed_ms(telemetry, "generator_llm"),
        "generator_constraint_ms": telemetry_stage_elapsed_ms(telemetry, "V_constraint_generator"),
        "verifier_llm_ms": telemetry_stage_elapsed_ms(telemetry, "verifier_llm"),
        "qa_bundle_ms": telemetry_stage_elapsed_ms(telemetry, "qa_bundle"),
        "render_ms": telemetry_stage_elapsed_ms(telemetry, "render"),
    }
    generator_call = telemetry_llm_call(telemetry, "generator_llm")
    if isinstance(generator_call, dict):
        response = generator_call.get("response") if isinstance(generator_call.get("response"), dict) else {}
        usage = response.get("usage") if isinstance(response, dict) and isinstance(response.get("usage"), dict) else {}
        image_payload = (
            generator_call.get("image_payload")
            if isinstance(generator_call.get("image_payload"), dict)
            else {}
        )
        timing["generator_llm_roundtrip_ms"] = generator_call.get("roundtrip_ms")
        timing["generator_llm_request_payload_chars"] = generator_call.get("request_payload_chars")
        timing["generator_llm_request_text_chars"] = generator_call.get("request_text_chars")
        timing["generator_llm_response_text_chars"] = generator_call.get("response_text_chars")
        timing["generator_llm_image_count"] = image_payload.get("image_count") if isinstance(image_payload, dict) else None
        timing["generator_llm_approx_image_bytes"] = (
            image_payload.get("approx_image_bytes") if isinstance(image_payload, dict) else None
        )
        timing["generator_llm_prompt_tokens"] = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("prompt_token_count")
        )
        timing["generator_llm_completion_tokens"] = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("completion_token_count")
        )
        timing["generator_llm_total_tokens"] = (
            usage.get("total_tokens")
            or usage.get("total_token_count")
        )
        stream_timing = (
            generator_call.get("stream_timing")
            if isinstance(generator_call.get("stream_timing"), dict)
            else {}
        )
        if stream_timing:
            timing["generator_llm_stream_enabled"] = bool(stream_timing.get("enabled"))
            timing["generator_llm_time_to_first_chunk_ms"] = stream_timing.get("time_to_first_chunk_ms")
            timing["generator_llm_time_to_first_token_ms"] = stream_timing.get("time_to_first_token_ms")
            timing["generator_llm_stream_decode_ms"] = stream_timing.get("stream_decode_ms")
            timing["generator_llm_stream_chunks"] = stream_timing.get("chunks")
            timing["generator_llm_stream_content_chunks"] = stream_timing.get("content_chunks")
    timing["known_stage_ms"] = sum(
        int(timing.get(key) or 0)
        for key in (
            "route_context_ms",
            "bev_map_provider_ms",
            "image_preprocess_ms",
            "segmentation_ms",
            "som_and_extra_images_ms",
            "generator_input_build_ms",
            "generator_llm_ms",
            "generator_constraint_ms",
            "verifier_llm_ms",
            "qa_bundle_ms",
            "render_ms",
        )
        if isinstance(timing.get(key), int)
    )
    return timing


def timing_percentile(values: list[int], percentile: float) -> int | None:
    """Return a nearest-rank percentile from integer millisecond values."""

    if not values:
        return None
    ordered = sorted(int(v) for v in values)
    index = int(round((len(ordered) - 1) * float(percentile)))
    return ordered[max(0, min(index, len(ordered) - 1))]


def summarize_visual_prompt_manifest_timings(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-frame timing records into a compact manifest summary."""

    ok_frames = [frame for frame in frames if frame.get("status") == "ok" and isinstance(frame.get("timing_ms"), dict)]
    runner_values = [int(frame["timing_ms"].get("runner_elapsed_ms") or 0) for frame in ok_frames]
    service_values = [int(frame["timing_ms"].get("service_elapsed_ms") or 0) for frame in ok_frames]
    stage_keys = [
        "bev_map_provider_ms",
        "image_preprocess_ms",
        "segmentation_ms",
        "som_and_extra_images_ms",
        "bev_map_overlay_ms",
        "generator_llm_ms",
        "generator_constraint_ms",
        "verifier_llm_ms",
        "qa_bundle_ms",
        "render_ms",
        "artifact_write_ms",
        "generator_llm_roundtrip_ms",
        "generator_llm_request_payload_chars",
        "generator_llm_approx_image_bytes",
        "generator_llm_prompt_tokens",
        "generator_llm_completion_tokens",
        "generator_llm_total_tokens",
        "generator_llm_time_to_first_token_ms",
        "generator_llm_time_to_first_chunk_ms",
        "generator_llm_stream_decode_ms",
        "generator_llm_stream_chunks",
        "generator_llm_stream_content_chunks",
    ]
    stage_summary: dict[str, Any] = {}
    for key in stage_keys:
        values = [
            int(frame["timing_ms"].get(key))
            for frame in ok_frames
            if isinstance(frame["timing_ms"].get(key), int)
        ]
        if values:
            stage_summary[key] = {
                "avg_ms": int(round(sum(values) / max(len(values), 1))),
                "p95_ms": timing_percentile(values, 0.95),
                "max_ms": max(values),
            }
    slowest = sorted(
        (
            {
                "frame_name": str(frame.get("frame_name")),
                "runner_elapsed_ms": int(frame["timing_ms"].get("runner_elapsed_ms") or 0),
                "generator_llm_ms": frame["timing_ms"].get("generator_llm_ms"),
                "segmentation_ms": frame["timing_ms"].get("segmentation_ms"),
                "bev_map_overlay_ms": frame["timing_ms"].get("bev_map_overlay_ms"),
            }
            for frame in ok_frames
        ),
        key=lambda item: int(item.get("runner_elapsed_ms") or 0),
        reverse=True,
    )[:8]
    return {
        "frames_profiled": len(ok_frames),
        "runner_elapsed_ms": {
            "avg_ms": int(round(sum(runner_values) / max(len(runner_values), 1))) if runner_values else None,
            "p50_ms": timing_percentile(runner_values, 0.50),
            "p95_ms": timing_percentile(runner_values, 0.95),
            "max_ms": max(runner_values) if runner_values else None,
        },
        "service_elapsed_ms": {
            "avg_ms": int(round(sum(service_values) / max(len(service_values), 1))) if service_values else None,
            "p50_ms": timing_percentile(service_values, 0.50),
            "p95_ms": timing_percentile(service_values, 0.95),
            "max_ms": max(service_values) if service_values else None,
        },
        "stages": stage_summary,
        "slowest_frames": slowest,
    }


def format_visual_prompt_frame_timing(frame_name: str, timing: dict[str, Any], *, status: str) -> str:
    """Format one frame timing record as a stable single-line log message."""

    return (
        "[TIME] "
        f"frame={frame_name} status={status} "
        f"runner={timing.get('runner_elapsed_ms')}ms "
        f"service={timing.get('service_elapsed_ms')}ms "
        f"llm={timing.get('generator_llm_ms')}ms "
        f"seg={timing.get('segmentation_ms')}ms "
        f"bev_overlay={timing.get('bev_map_overlay_ms')}ms "
        f"artifacts={timing.get('artifact_write_ms')}ms"
    )
