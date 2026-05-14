from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FrameLlmLatency:
    frame_name: str
    runner_ms: int
    roundtrip_ms: int
    request_payload_bytes: int
    request_text_chars: int
    response_text_bytes: int
    approx_image_bytes: int
    image_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_network_ms: int
    estimated_upload_ms: int
    estimated_download_ms: int
    estimated_service_ms: int
    time_to_first_chunk_ms: int
    time_to_first_token_ms: int
    stream_decode_ms: int


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(round(float(value)))
    except Exception:
        return default


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = int(math.ceil(float(percentile) * len(ordered))) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def _summary(values: list[int]) -> dict[str, int]:
    if not values:
        return {"count": 0, "avg": 0, "p50": 0, "p95": 0, "max": 0}
    return {
        "count": len(values),
        "avg": int(round(statistics.mean(values))),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _bytes_transfer_ms(byte_count: int, mbps: float) -> int:
    if byte_count <= 0 or mbps <= 0:
        return 0
    bits = float(byte_count) * 8.0
    return int(math.ceil((bits / (float(mbps) * 1_000_000.0)) * 1000.0))


def _resolve_manifest_path(args: argparse.Namespace) -> Path:
    if args.manifest:
        return Path(args.manifest)
    if args.run_dir:
        return Path(args.run_dir) / "manifest.json"
    raise ValueError("Either --run-dir or --manifest is required.")


def _iter_ok_frame_timings(manifest: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for item in manifest.get("frames") or []:
        if not isinstance(item, dict) or item.get("status") != "ok":
            continue
        timing = item.get("timing_ms")
        if not isinstance(timing, dict):
            continue
        rows.append((str(item.get("frame_name") or ""), timing))
    return rows


def _estimate_frames(
    manifest: dict[str, Any],
    *,
    upload_mbps: float,
    download_mbps: float,
    rtt_ms: int,
    protocol_overhead_ms: int,
) -> list[FrameLlmLatency]:
    frames: list[FrameLlmLatency] = []
    for frame_name, timing in _iter_ok_frame_timings(manifest):
        roundtrip_ms = _as_int(timing.get("generator_llm_roundtrip_ms") or timing.get("generator_llm_ms"))
        payload_bytes = _as_int(timing.get("generator_llm_request_payload_bytes"))
        if payload_bytes <= 0:
            # Payload chars are emitted from JSON text. For this UTF-8-heavy payload,
            # chars are a close lower-bound for wire bytes before HTTP/TLS framing.
            payload_bytes = _as_int(timing.get("generator_llm_request_payload_chars"))
        response_bytes = _as_int(timing.get("generator_llm_response_text_chars"))
        upload_ms = _bytes_transfer_ms(payload_bytes, upload_mbps)
        download_ms = _bytes_transfer_ms(response_bytes, download_mbps)
        network_ms = max(0, int(rtt_ms) + int(protocol_overhead_ms) + upload_ms + download_ms)
        frames.append(
            FrameLlmLatency(
                frame_name=frame_name,
                runner_ms=_as_int(timing.get("runner_elapsed_ms")),
                roundtrip_ms=roundtrip_ms,
                request_payload_bytes=payload_bytes,
                request_text_chars=_as_int(timing.get("generator_llm_request_text_chars")),
                response_text_bytes=response_bytes,
                approx_image_bytes=_as_int(timing.get("generator_llm_approx_image_bytes")),
                image_count=_as_int(timing.get("generator_llm_image_count")),
                prompt_tokens=_as_int(timing.get("generator_llm_prompt_tokens")),
                completion_tokens=_as_int(timing.get("generator_llm_completion_tokens")),
                total_tokens=_as_int(timing.get("generator_llm_total_tokens")),
                estimated_network_ms=network_ms,
                estimated_upload_ms=upload_ms,
                estimated_download_ms=download_ms,
                estimated_service_ms=max(0, roundtrip_ms - network_ms),
                time_to_first_chunk_ms=_as_int(timing.get("generator_llm_time_to_first_chunk_ms"), default=-1),
                time_to_first_token_ms=_as_int(timing.get("generator_llm_time_to_first_token_ms"), default=-1),
                stream_decode_ms=_as_int(timing.get("generator_llm_stream_decode_ms"), default=-1),
            )
        )
    return frames


def _format_bytes(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000.0:.2f}MB"
    if value >= 1_000:
        return f"{value / 1_000.0:.1f}KB"
    return f"{value}B"


def _print_table(frames: list[FrameLlmLatency], *, top_n: int) -> None:
    if not frames:
        print("No successful frame LLM timing records found.")
        return
    rows = sorted(frames, key=lambda item: item.roundtrip_ms, reverse=True)[: max(1, int(top_n))]
    header = (
        f"{'frame':<8} {'rt_ms':>7} {'net_est':>8} {'svc_est':>8} "
        f"{'ttft':>7} {'decode':>7} {'upload':>7} {'payload':>9} {'img':>8} {'prompt':>7} {'comp':>6}"
    )
    print(header)
    print("-" * len(header))
    for item in rows:
        print(
            f"{item.frame_name:<8} "
            f"{item.roundtrip_ms:>7} "
            f"{item.estimated_network_ms:>8} "
            f"{item.estimated_service_ms:>8} "
            f"{item.time_to_first_token_ms if item.time_to_first_token_ms >= 0 else '-':>7} "
            f"{item.stream_decode_ms if item.stream_decode_ms >= 0 else '-':>7} "
            f"{item.estimated_upload_ms:>7} "
            f"{_format_bytes(item.request_payload_bytes):>9} "
            f"{_format_bytes(item.approx_image_bytes):>8} "
            f"{item.prompt_tokens:>7} "
            f"{item.completion_tokens:>6}"
        )


def _print_summary(frames: list[FrameLlmLatency], *, upload_mbps: float, download_mbps: float, rtt_ms: int) -> None:
    if not frames:
        return
    roundtrip = [item.roundtrip_ms for item in frames]
    network = [item.estimated_network_ms for item in frames]
    service = [item.estimated_service_ms for item in frames]
    ttft = [item.time_to_first_token_ms for item in frames if item.time_to_first_token_ms >= 0]
    decode = [item.stream_decode_ms for item in frames if item.stream_decode_ms >= 0]
    payload = [item.request_payload_bytes for item in frames if item.request_payload_bytes > 0]
    prompt_tokens = [item.prompt_tokens for item in frames if item.prompt_tokens > 0]
    completion_tokens = [item.completion_tokens for item in frames if item.completion_tokens > 0]
    missing_payload = sum(1 for item in frames if item.request_payload_bytes <= 0)

    print()
    print("Assumptions")
    print(f"  upload_mbps={upload_mbps:g} download_mbps={download_mbps:g} rtt_ms={rtt_ms}")
    print("  network_est = RTT + protocol_overhead + payload/upload_bw + response/download_bw")
    print("  service_est = observed_llm_roundtrip - network_est")
    print("  service_est still includes Ark gateway queueing, image parsing, model prefill, and decoding.")
    if missing_payload:
        print(f"  warning: {missing_payload} frame(s) have no payload/token telemetry; network_est is a lower bound.")

    print()
    print("Summary")
    for name, values in (
        ("roundtrip_ms", roundtrip),
        ("network_est_ms", network),
        ("service_est_ms", service),
        ("time_to_first_token_ms", ttft),
        ("stream_decode_ms", decode),
        ("payload_bytes", payload),
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
    ):
        if not values and name in {"time_to_first_token_ms", "stream_decode_ms"}:
            print(f"  {name:<18} unavailable; enable LLM_VISION_STREAM_TIMING=1")
            continue
        row = _summary(values)
        if name == "payload_bytes":
            print(
                f"  {name:<18} avg={_format_bytes(row['avg'])} "
                f"p50={_format_bytes(row['p50'])} p95={_format_bytes(row['p95'])} max={_format_bytes(row['max'])}"
            )
        else:
            print(
                f"  {name:<18} avg={row['avg']} p50={row['p50']} p95={row['p95']} max={row['max']}"
            )

    avg_roundtrip = max(1, int(round(statistics.mean(roundtrip))))
    avg_network = int(round(statistics.mean(network)))
    print()
    print(
        "Attribution"
        f"  network_est_share={avg_network / avg_roundtrip:.1%} "
        f"service_est_share={(avg_roundtrip - avg_network) / avg_roundtrip:.1%}"
    )


def _probe_url(url: str, *, count: int, timeout_s: float) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for _ in range(max(1, int(count))):
        started = time.perf_counter()
        status = None
        body_bytes = 0
        error = None
        try:
            request = urllib.request.Request(url, method="GET", headers={"User-Agent": "UrbanNav-latency-probe/1.0"})
            with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
                status = int(getattr(response, "status", 0) or 0)
                body = response.read(4096)
                body_bytes = len(body)
        except Exception as exc:  # pragma: no cover - depends on local network/provider state.
            error = str(exc)
        elapsed_ms = int(round((time.perf_counter() - started) * 1000.0))
        samples.append({"elapsed_ms": elapsed_ms, "status": status, "body_bytes": body_bytes, "error": error})
    values = [int(item["elapsed_ms"]) for item in samples]
    return {
        "url": url,
        "count": len(samples),
        "avg_ms": int(round(statistics.mean(values))) if values else 0,
        "p50_ms": _percentile(values, 0.50),
        "min_ms": min(values) if values else 0,
        "max_ms": max(values) if values else 0,
        "samples": samples,
    }


def _to_json_summary(frames: list[FrameLlmLatency], probe: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "frames": [item.__dict__ for item in frames],
        "summary": {
            "roundtrip_ms": _summary([item.roundtrip_ms for item in frames]),
            "network_est_ms": _summary([item.estimated_network_ms for item in frames]),
            "service_est_ms": _summary([item.estimated_service_ms for item in frames]),
            "time_to_first_token_ms": _summary([item.time_to_first_token_ms for item in frames if item.time_to_first_token_ms >= 0]),
            "stream_decode_ms": _summary([item.stream_decode_ms for item in frames if item.stream_decode_ms >= 0]),
            "request_payload_bytes": _summary([item.request_payload_bytes for item in frames]),
            "prompt_tokens": _summary([item.prompt_tokens for item in frames]),
            "completion_tokens": _summary([item.completion_tokens for item in frames]),
        },
        "probe": probe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate visual-prompt LVLM latency attribution from pipeline manifest telemetry."
    )
    parser.add_argument("--run-dir", help="Pipeline output directory containing manifest.json.")
    parser.add_argument("--manifest", help="Path to manifest.json. Overrides --run-dir.")
    parser.add_argument("--upload-mbps", type=float, default=20.0, help="Assumed client upload bandwidth.")
    parser.add_argument("--download-mbps", type=float, default=100.0, help="Assumed client download bandwidth.")
    parser.add_argument("--rtt-ms", type=int, default=80, help="Assumed RTT to the API region.")
    parser.add_argument("--protocol-overhead-ms", type=int, default=40, help="TLS/HTTP client overhead estimate.")
    parser.add_argument("--top-n", type=int, default=8, help="Number of slowest frames to print.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table.")
    parser.add_argument("--probe-url", help="Optional URL to probe for coarse HTTP latency, e.g. Ark base URL.")
    parser.add_argument("--probe-count", type=int, default=3, help="Number of GET probes when --probe-url is set.")
    parser.add_argument("--probe-timeout-s", type=float, default=10.0)
    args = parser.parse_args()

    manifest_path = _resolve_manifest_path(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = _estimate_frames(
        manifest,
        upload_mbps=float(args.upload_mbps),
        download_mbps=float(args.download_mbps),
        rtt_ms=int(args.rtt_ms),
        protocol_overhead_ms=int(args.protocol_overhead_ms),
    )
    probe = _probe_url(args.probe_url, count=args.probe_count, timeout_s=args.probe_timeout_s) if args.probe_url else None

    if args.json:
        print(json.dumps(_to_json_summary(frames, probe), ensure_ascii=False, indent=2))
        return

    print(f"Manifest: {manifest_path}")
    _print_table(frames, top_n=int(args.top_n))
    _print_summary(frames, upload_mbps=float(args.upload_mbps), download_mbps=float(args.download_mbps), rtt_ms=int(args.rtt_ms))
    if probe is not None:
        print()
        print("HTTP Probe")
        print(
            f"  url={probe['url']} count={probe['count']} "
            f"avg={probe['avg_ms']}ms p50={probe['p50_ms']}ms min={probe['min_ms']}ms max={probe['max_ms']}ms"
        )


if __name__ == "__main__":
    main()
