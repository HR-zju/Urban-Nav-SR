#!/usr/bin/env python3
"""Generate UrbanNav predictions with explicit tool-use agent traces."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from time import monotonic


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_server.core.replay_map_backend import ReplayMapBackend  # noqa: E402
from mcp_server.eval.urbannav.urbannav_io import load_urbannav_cases
from mcp_server.eval.urbannav.urbannav_runners import ReplayUrbanNavBackendAdapter
from mcp_server.eval.urbannav.urbannav_schema import UrbanNavCase, UrbanNavPrediction
from mcp_server.eval.agent_tools.urbannav_tool_agent_runner import UrbanNavToolAgentConfig, UrbanNavToolAgentRunner
from mcp_server.llm import LLMSettings  # noqa: E402
from mcp_server.utils.async_batch import run_ordered_bounded  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        None.

    Returns:
        Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="UrbanNav task_gt.jsonl.")
    parser.add_argument("--poi-cache", required=True, type=Path, help="Frozen poi_cache.jsonl.")
    parser.add_argument("--route-cache", required=True, type=Path, help="Frozen route_cache.jsonl.")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Output UrbanNavPrediction JSONL.")
    parser.add_argument("--manifest-json", type=Path, default=None, help="Optional manifest JSON output.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional smoke-test case cap.")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum concurrent cases.")
    parser.add_argument("--candidate-limit", type=int, default=10, help="Candidates exposed by search_destination.")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Maximum assistant turns per case. Set 0 for category-adaptive defaults.",
    )
    parser.add_argument("--max-tokens", type=int, default=768, help="Maximum output tokens per assistant turn.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM sampling temperature.")
    parser.add_argument("--mode", default="foot", choices=["foot", "bike", "car"], help="Route mode label.")
    parser.add_argument("--api-keys-file", type=Path, default=None, help="Newline-separated LLM API keys.")
    parser.add_argument(
        "--api-keys-env",
        default="ARK_API_KEYS,LLM_API_KEYS,ARK_API_KEY,LLM_API_KEY,OPENAI_API_KEY",
        help="Comma-separated environment variables for API keys.",
    )
    parser.add_argument("--llm-provider", default=None, help="Provider override.")
    parser.add_argument("--llm-model", default=None, help="Model override.")
    parser.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--llm-timeout-s", type=float, default=None, help="LLM timeout override.")
    parser.add_argument("--llm-max-retries", type=int, default=None, help="LLM max retries override.")
    parser.add_argument("--chunk-size", type=int, default=50, help="Durable write chunk size. Set 0 to write at end.")
    parser.add_argument("--resume", action="store_true", help="Skip case ids already in --output-jsonl.")
    parser.add_argument("--stop-after-pending", type=int, default=None, help="Process only N pending cases then exit.")
    parser.add_argument("--progress-interval", type=int, default=100, help="Progress log interval. Set 0 to disable.")
    parser.add_argument("--heartbeat-interval-s", type=float, default=15.0, help="Heartbeat interval. Set 0 to disable.")
    return parser.parse_args()


def main() -> None:
    """Run prediction generation.

    Args:
        None.

    Returns:
        None.
    """

    args = parse_args()
    _disable_env_proxy_for_local_llm(args)
    cases = load_urbannav_cases(args.cases)
    if args.max_cases is not None:
        cases = cases[: max(0, int(args.max_cases))]
    api_keys = _load_api_keys(path=args.api_keys_file, env_names=args.api_keys_env)
    key_count = len(api_keys)
    if not api_keys:
        api_keys = ["EMPTY"]
    backend = ReplayUrbanNavBackendAdapter(
        ReplayMapBackend(poi_cache_path=args.poi_cache, route_cache_path=args.route_cache)
    )
    print(
        json.dumps(
            {
                "event": "tool_agent_prediction_start",
                "case_count": len(cases),
                "concurrency": max(1, int(args.concurrency)),
                "key_count": key_count,
                "model": args.llm_model or os.getenv("LLM_MODEL", ""),
                "base_url": args.llm_base_url or os.getenv("LLM_API_BASE_URL", ""),
                "max_turns": args.max_turns,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    prediction_count = asyncio.run(
        _generate_chunked(args=args, cases=cases, backend=backend, api_keys=api_keys)
    )
    if args.manifest_json:
        _write_json(
            {
                "schema_version": "urbannav_tool_agent_prediction_generation_v1",
                "cases": str(args.cases),
                "poi_cache": str(args.poi_cache),
                "route_cache": str(args.route_cache),
                "output_jsonl": str(args.output_jsonl),
                "case_count": len(cases),
                "prediction_count": prediction_count,
                "concurrency": max(1, int(args.concurrency)),
                "candidate_limit": int(args.candidate_limit),
                "max_turns": int(args.max_turns),
                "mode": args.mode,
                "llm_provider": args.llm_provider or os.getenv("LLM_PROVIDER", ""),
                "llm_model": args.llm_model or os.getenv("LLM_MODEL", ""),
                "llm_base_url": args.llm_base_url or os.getenv("LLM_API_BASE_URL", ""),
                "api_key_count": key_count,
                "chunk_size": int(args.chunk_size or 0),
                "resume": bool(args.resume),
            },
            args.manifest_json,
        )
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "prediction_count": prediction_count,
                "output_jsonl": str(args.output_jsonl),
                "manifest_json": str(args.manifest_json) if args.manifest_json else None,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


async def _generate_chunked(
    *,
    args: argparse.Namespace,
    cases: list[UrbanNavCase],
    backend: ReplayUrbanNavBackendAdapter,
    api_keys: list[str],
) -> int:
    existing_ids = _existing_prediction_ids(args.output_jsonl) if args.resume else set()
    pending_cases = [case for case in cases if case.case_id not in existing_ids]
    if args.stop_after_pending is not None:
        pending_cases = pending_cases[: max(0, int(args.stop_after_pending))]
    if not args.resume:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.output_jsonl.write_text("", encoding="utf-8")
    chunk_size = int(args.chunk_size or 0)
    if chunk_size <= 0:
        predictions = await _generate_predictions(args=args, cases=pending_cases, backend=backend, api_keys=api_keys)
        _append_predictions(predictions, args.output_jsonl)
        return len(existing_ids) + len(predictions)

    written = len(existing_ids)
    for start in range(0, len(pending_cases), max(1, chunk_size)):
        chunk = pending_cases[start : start + max(1, chunk_size)]
        predictions = await _generate_predictions(args=args, cases=chunk, backend=backend, api_keys=api_keys)
        _append_predictions(predictions, args.output_jsonl)
        written += len(predictions)
        print(
            json.dumps(
                {
                    "event": "tool_agent_prediction_chunk_written",
                    "chunk_size": len(predictions),
                    "written": written,
                    "target_total": len(cases),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return written


async def _generate_predictions(
    *,
    args: argparse.Namespace,
    cases: list[UrbanNavCase],
    backend: ReplayUrbanNavBackendAdapter,
    api_keys: list[str],
) -> list[UrbanNavPrediction]:
    settings_pool = [_llm_settings_for_key(args, key) for key in api_keys]
    runners = [
        UrbanNavToolAgentRunner(
            backend=backend,
            settings=settings,
            config=UrbanNavToolAgentConfig(
                max_turns=args.max_turns,
                candidate_limit=args.candidate_limit,
                mode=args.mode,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            ),
        )
        for settings in settings_pool
    ]
    completed = 0
    started_at = monotonic()
    progress_interval = max(0, int(args.progress_interval or 0))
    heartbeat_interval_s = max(0.0, float(args.heartbeat_interval_s or 0.0))
    progress_lock = asyncio.Lock()

    async def _run_worker(index: int, case: UrbanNavCase) -> UrbanNavPrediction:
        nonlocal completed
        try:
            runner = runners[index % len(runners)]
            return await runner.run_case_async(case)
        finally:
            if progress_interval:
                async with progress_lock:
                    completed += 1
                    if completed == len(cases) or completed % progress_interval == 0:
                        elapsed_s = max(monotonic() - started_at, 1e-6)
                        print(
                            json.dumps(
                                {
                                    "event": "tool_agent_prediction_progress",
                                    "completed": completed,
                                    "total": len(cases),
                                    "rate_cases_per_s": round(completed / elapsed_s, 4),
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            flush=True,
                        )

    async def _heartbeat() -> None:
        while True:
            await asyncio.sleep(heartbeat_interval_s)
            async with progress_lock:
                elapsed_s = max(monotonic() - started_at, 1e-6)
                print(
                    json.dumps(
                        {
                            "event": "tool_agent_prediction_heartbeat",
                            "completed": completed,
                            "total": len(cases),
                            "elapsed_s": round(elapsed_s, 1),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )

    heartbeat_task: asyncio.Task[None] | None = None
    if heartbeat_interval_s > 0:
        heartbeat_task = asyncio.create_task(_heartbeat())
    try:
        results = await run_ordered_bounded(cases, _run_worker, concurrency=max(1, int(args.concurrency)))
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        for runner in runners:
            await runner.close()

    predictions: list[UrbanNavPrediction] = []
    for result in results:
        if result.ok and result.value is not None:
            predictions.append(result.value)
            continue
        case = result.item if isinstance(result.item, UrbanNavCase) else None
        predictions.append(
            UrbanNavPrediction(
                case_id=case.case_id if case is not None else f"tool-agent-error-{result.index:06d}",
                trace={
                    "runner": "urbannav_tool_agent",
                    "case_id": case.case_id if case is not None else None,
                    "instruction": case.instruction if case is not None else None,
                    "category": case.category if case is not None else None,
                    "subcategory": case.subcategory if case is not None else None,
                    "city": case.city if case is not None else None,
                    "error": "prediction_exception",
                    "detail": result.error,
                },
            )
        )
    return predictions


def _llm_settings_for_key(args: argparse.Namespace, api_key: str) -> LLMSettings:
    updates: dict[str, object] = {"api_key": api_key}
    if args.llm_provider:
        updates["provider"] = args.llm_provider
    if args.llm_model:
        updates["model"] = args.llm_model
    if args.llm_base_url:
        updates["base_url"] = str(args.llm_base_url).rstrip("/")
    if args.llm_timeout_s is not None:
        updates["timeout_s"] = float(args.llm_timeout_s)
    if args.llm_max_retries is not None:
        updates["max_retries"] = max(0, int(args.llm_max_retries))
    return replace(LLMSettings.from_env(), response_format=False, **updates)


def _load_api_keys(*, path: Path | None, env_names: str) -> list[str]:
    raw_values: list[str] = []
    if path:
        raw_values.append(path.read_text(encoding="utf-8"))
    for name in [item.strip() for item in str(env_names or "").split(",") if item.strip()]:
        raw_values.append(os.getenv(name, ""))
    keys: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        for item in raw.replace(",", "\n").splitlines():
            key = item.strip()
            if not key or key.startswith("#") or key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return keys


def _append_predictions(predictions: list[UrbanNavPrediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()


def _write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _existing_prediction_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    output: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = str(payload.get("id") or payload.get("case_id") or "").strip()
            if case_id:
                output.add(case_id)
    return output


def _disable_env_proxy_for_local_llm(args: argparse.Namespace) -> None:
    base_url = str(args.llm_base_url or os.getenv("LLM_API_BASE_URL", "") or "").strip().lower()
    if not base_url.startswith(("http://127.", "https://127.", "http://localhost", "https://localhost")):
        return
    no_proxy_items = ["127.0.0.1", "localhost", "::1"]
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    for item in existing.split(","):
        item = item.strip()
        if item and item not in no_proxy_items:
            no_proxy_items.append(item)
    os.environ["NO_PROXY"] = ",".join(no_proxy_items)
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)


if __name__ == "__main__":
    main()
