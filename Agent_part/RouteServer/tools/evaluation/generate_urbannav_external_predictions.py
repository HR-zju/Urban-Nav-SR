#!/usr/bin/env python3
"""Generate UrbanNavPrediction JSONL for external model evaluation.

This script is intentionally separate from the evaluator.  It may call LLMs or
live map providers to produce predictions, while
``run_urbannav_module1_benchmark.py --runner external`` remains an offline
deterministic scorer over the saved prediction JSONL.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from time import monotonic
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_server.core.live_map_backend import LiveMapBackend  # noqa: E402
from mcp_server.core.replay_map_backend import ReplayMapBackend  # noqa: E402
from mcp_server.eval.urbannav_io import load_urbannav_cases, write_urbannav_predictions  # noqa: E402
from mcp_server.eval.urbannav_runners import (  # noqa: E402
    ReplayUrbanNavBackendAdapter,
    RouteServerRunnerConfig,
    RouteServerUrbanNavRunner,
)
from mcp_server.eval.urbannav_schema import UrbanNavCase, UrbanNavPrediction  # noqa: E402
from mcp_server.llm import CandidateReranker, GoalDecomposer, IntentParser, LLMSettings  # noqa: E402
from mcp_server.utils.async_batch import run_ordered_bounded  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="UrbanNav task_gt.jsonl.")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Output UrbanNavPrediction JSONL.")
    parser.add_argument(
        "--backend",
        choices=["replay", "live"],
        default="replay",
        help="Map backend used by the RouteServer prediction runner.",
    )
    parser.add_argument("--poi-cache", type=Path, default=None, help="POI replay cache for --backend replay.")
    parser.add_argument("--route-cache", type=Path, default=None, help="Route replay cache for --backend replay.")
    parser.add_argument("--map-provider", default=None, help="Map provider override for --backend live.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional smoke-test case cap.")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum concurrent prediction workers.")
    parser.add_argument("--candidate-limit", type=int, default=10, help="Maximum POI candidates to rank and route.")
    parser.add_argument("--search-radius-m", type=int, default=1500, help="Fallback POI search radius.")
    parser.add_argument("--mode", default="foot", choices=["foot", "bike", "car"], help="Route mode.")
    parser.add_argument(
        "--selection-strategy",
        choices=["route_distance", "semantic_then_route"],
        default="semantic_then_route",
        help="Destination selection policy.",
    )
    parser.add_argument("--use-llm-intent", action="store_true", help="Use LLM intent parsing.")
    parser.add_argument("--use-llm-reranker", action="store_true", help="Use LLM candidate reranking.")
    parser.add_argument(
        "--strict-llm",
        action="store_true",
        help="Fail a case when a requested LLM step falls back or returns no decision.",
    )
    parser.add_argument("--api-keys-file", type=Path, default=None, help="Newline-separated LLM API keys.")
    parser.add_argument(
        "--api-keys-env",
        default="ARK_API_KEYS,LLM_API_KEYS,ARK_API_KEY,LLM_API_KEY",
        help="Comma-separated environment variables to read API keys from.",
    )
    parser.add_argument("--llm-provider", default=None, help="Provider override, for example ark.")
    parser.add_argument("--llm-model", default=None, help="Model override.")
    parser.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL override.")
    parser.add_argument("--llm-timeout-s", type=float, default=None, help="LLM request timeout override.")
    parser.add_argument("--llm-max-retries", type=int, default=None, help="LLM max retries override.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print prediction progress every N completed cases. Set 0 to disable.",
    )
    parser.add_argument(
        "--heartbeat-interval-s",
        type=float,
        default=15.0,
        help="Print a heartbeat every N seconds while prediction is running. Set 0 to disable.",
    )
    parser.add_argument("--manifest-json", type=Path, default=None, help="Optional manifest JSON output.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Write predictions after each chunk so long external runs keep partial results. Set 0 to write only at end.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip case ids already present in --output-jsonl.")
    return parser.parse_args()


def main() -> None:
    """Generate predictions from the configured runner.

    Returns:
        None. Predictions and an optional manifest are written to disk.
    """

    args = parse_args()
    cases = load_urbannav_cases(args.cases)
    if args.max_cases is not None:
        if args.max_cases <= 0:
            raise ValueError("--max-cases must be positive when provided")
        cases = cases[: args.max_cases]

    keys = _load_api_keys(path=args.api_keys_file, env_names=args.api_keys_env)
    if (args.use_llm_intent or args.use_llm_reranker) and not keys:
        raise ValueError("LLM mode requires at least one key from --api-keys-file or configured environment variables")
    print(
        json.dumps(
            {
                "event": "prediction_start",
                "backend": args.backend,
                "case_count": len(cases),
                "concurrency": max(1, int(args.concurrency)),
                "key_count": len(keys),
                "strict_llm": bool(args.strict_llm),
                "use_llm_intent": bool(args.use_llm_intent),
                "use_llm_reranker": bool(args.use_llm_reranker),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    if int(args.chunk_size or 0) > 0:
        prediction_count = asyncio.run(
            _generate_predictions_chunked(
                args=args,
                cases=cases,
                api_keys=keys,
                output_path=args.output_jsonl,
            )
        )
    else:
        predictions = asyncio.run(_generate_predictions(args=args, cases=cases, api_keys=keys))
        write_urbannav_predictions(predictions, args.output_jsonl)
        prediction_count = len(predictions)
    if args.manifest_json:
        _write_json(
            {
                "schema_version": "urbannav_external_prediction_generation_v1",
                "cases": str(args.cases),
                "output_jsonl": str(args.output_jsonl),
                "case_count": len(cases),
                "prediction_count": prediction_count,
                "backend": args.backend,
                "map_provider": args.map_provider,
                "concurrency": max(1, int(args.concurrency)),
                "selection_strategy": args.selection_strategy,
                "use_llm_intent": bool(args.use_llm_intent),
                "use_llm_reranker": bool(args.use_llm_reranker),
                "strict_llm": bool(args.strict_llm),
                "llm_provider": args.llm_provider or os.getenv("LLM_PROVIDER", ""),
                "llm_model": args.llm_model or os.getenv("LLM_MODEL", ""),
                "api_key_count": len(keys),
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
        )
    )


async def _generate_predictions_chunked(
    *,
    args: argparse.Namespace,
    cases: list[UrbanNavCase],
    api_keys: list[str],
    output_path: Path,
) -> int:
    r"""Generate predictions in chunks and append them to disk.

    For expensive external model calls, this changes the persistence model from
    one all-or-nothing write:

    \[
    \{\hat y_i\}_{i=1}^{N} \rightarrow file
    \]

    to periodic durable chunks:

    \[
    \{\hat y_i\}_{i=b}^{b+B-1} \rightarrow file
    \]

    Args:
        args: Parsed CLI arguments.
        cases: Frozen UrbanNav cases.
        api_keys: API keys used for LLM-backed workers.
        output_path: JSONL file to append chunk predictions to.

    Returns:
        Number of prediction rows present in the output file after completion.
    """

    existing_ids = _existing_prediction_ids(output_path) if args.resume else set()
    pending_cases = [case for case in cases if case.case_id not in existing_ids]
    if not args.resume:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    elif existing_ids:
        print(
            json.dumps(
                {
                    "event": "prediction_resume",
                    "existing_predictions": len(existing_ids),
                    "pending_cases": len(pending_cases),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    chunk_size = max(1, int(args.chunk_size or 1))
    written = len(existing_ids)
    for start in range(0, len(pending_cases), chunk_size):
        chunk = pending_cases[start : start + chunk_size]
        chunk_predictions = await _generate_predictions(args=args, cases=chunk, api_keys=api_keys)
        _append_predictions(chunk_predictions, output_path)
        written += len(chunk_predictions)
        print(
            json.dumps(
                {
                    "event": "prediction_chunk_written",
                    "chunk_size": len(chunk_predictions),
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
    api_keys: list[str],
) -> list[UrbanNavPrediction]:
    """Generate predictions with bounded concurrency.

    Args:
        args: Parsed CLI arguments.
        cases: Frozen UrbanNav cases.
        api_keys: API keys used for LLM-backed workers. Empty means heuristic
            runner only.

    Returns:
        Predictions in the same order as input cases.
    """

    backend = _build_backend(args)
    runner_pool = _build_runner_pool(args=args, backend=backend, api_keys=api_keys)
    progress_interval = max(0, int(args.progress_interval or 0))
    heartbeat_interval_s = max(0.0, float(args.heartbeat_interval_s or 0.0))
    completed = 0
    started_at = monotonic()
    progress_lock = asyncio.Lock()

    async def _run_worker(index: int, case: UrbanNavCase) -> UrbanNavPrediction:
        nonlocal completed
        try:
            runner = runner_pool[index % len(runner_pool)]
            prediction = await runner.run_case_async(case)
            if args.strict_llm:
                _validate_strict_llm_prediction(
                    prediction,
                    require_intent=bool(args.use_llm_intent),
                    require_reranker=bool(args.use_llm_reranker),
                )
            return prediction
        finally:
            if progress_interval:
                async with progress_lock:
                    completed += 1
                    if completed == len(cases) or completed % progress_interval == 0:
                        elapsed_s = max(monotonic() - started_at, 1e-6)
                        rate = completed / elapsed_s
                        print(
                            json.dumps(
                                {
                                    "event": "prediction_progress",
                                    "completed": completed,
                                    "total": len(cases),
                                    "rate_cases_per_s": round(rate, 4),
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            flush=True,
                        )

    async def _heartbeat() -> None:
        """Emit periodic progress logs while remote calls are pending.

        Args:
            None.

        Returns:
            None.
        """

        while True:
            await asyncio.sleep(heartbeat_interval_s)
            async with progress_lock:
                elapsed_s = max(monotonic() - started_at, 1e-6)
                print(
                    json.dumps(
                        {
                            "event": "prediction_heartbeat",
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

    if args.backend == "live":
        await backend.connect()
    try:
        results = await run_ordered_bounded(cases, _run_worker, concurrency=max(1, int(args.concurrency)))
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        if args.backend == "live":
            await backend.disconnect()

    predictions: list[UrbanNavPrediction] = []
    for result in results:
        if result.ok and result.value is not None:
            predictions.append(result.value)
            continue
        case = result.item if isinstance(result.item, UrbanNavCase) else None
        predictions.append(
            UrbanNavPrediction(
                case_id=case.case_id if case is not None else f"prediction-error-{result.index:06d}",
                trace={
                    "runner": "external_routeserver_llm",
                    "error": "prediction_exception",
                    "detail": result.error,
                },
            )
        )
    return predictions


def _build_backend(args: argparse.Namespace) -> Any:
    """Build the configured map backend.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Replay adapter or live map backend.

    Raises:
        ValueError: If replay cache paths are missing.
    """

    if args.backend == "replay":
        if not args.poi_cache or not args.route_cache:
            raise ValueError("--poi-cache and --route-cache are required for --backend replay")
        return ReplayUrbanNavBackendAdapter(
            ReplayMapBackend(
                poi_cache_path=args.poi_cache,
                route_cache_path=args.route_cache,
            )
        )
    return LiveMapBackend.from_env(map_provider=args.map_provider)


def _validate_strict_llm_prediction(
    prediction: UrbanNavPrediction,
    *,
    require_intent: bool,
    require_reranker: bool,
) -> None:
    """Reject predictions that silently fell back from requested LLM steps.

    Args:
        prediction: Prediction produced by the RouteServer runner.
        require_intent: Whether the case requires LLM intent parsing.
        require_reranker: Whether the case requires an LLM reranking decision.

    Returns:
        None.

    Raises:
        RuntimeError: If a requested LLM step is missing or fell back.
    """

    trace = prediction.trace if isinstance(prediction.trace, dict) else {}
    if require_intent:
        intent = trace.get("intent") if isinstance(trace.get("intent"), dict) else {}
        source = str(intent.get("intent_source") or "").strip()
        if source != "llm":
            raise RuntimeError(f"strict_llm_intent_failed:{source or 'missing'}")
    if require_reranker:
        decision = trace.get("reranker")
        if not isinstance(decision, dict) or not decision.get("ranked_candidate_ids"):
            raise RuntimeError("strict_llm_reranker_failed")


def _build_runner_pool(
    *,
    args: argparse.Namespace,
    backend: Any,
    api_keys: list[str],
) -> list[RouteServerUrbanNavRunner]:
    """Build model-backed runner instances.

    Args:
        args: Parsed CLI arguments.
        backend: Map backend shared by runners.
        api_keys: API keys. One runner is created per key when LLM is enabled.

    Returns:
        Non-empty runner pool.
    """

    config = RouteServerRunnerConfig(
        search_radius_m=args.search_radius_m,
        candidate_limit=args.candidate_limit,
        mode=args.mode,
        use_llm_intent=bool(args.use_llm_intent),
        use_llm_reranker=bool(args.use_llm_reranker),
        selection_strategy=str(args.selection_strategy),
    )
    if not (args.use_llm_intent or args.use_llm_reranker):
        return [RouteServerUrbanNavRunner(backend=backend, config=config)]

    return [
        RouteServerUrbanNavRunner(
            backend=backend,
            config=config,
            intent_parser=IntentParser(settings=_llm_settings_for_key(args, key)),
            reranker=CandidateReranker(settings=_llm_settings_for_key(args, key)),
            goal_decomposer=GoalDecomposer(settings=_llm_settings_for_key(args, key)),
        )
        for key in api_keys
    ]


def _llm_settings_for_key(args: argparse.Namespace, api_key: str) -> LLMSettings:
    """Build LLM settings for one API key.

    Args:
        args: Parsed CLI arguments.
        api_key: Secret API key. The value is not persisted.

    Returns:
        LLM settings with CLI overrides applied.
    """

    updates: dict[str, Any] = {"api_key": api_key}
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
    return replace(LLMSettings.from_env(), **updates)


def _load_api_keys(*, path: Path | None, env_names: str) -> list[str]:
    """Load API keys without logging them.

    Args:
        path: Optional newline-separated key file.
        env_names: Comma-separated environment variables. Each value may
            contain comma- or newline-separated keys.

    Returns:
        Deduplicated keys in stable order.
    """

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


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Write a JSON object.

    Args:
        payload: JSON-serializable object.
        path: Output path.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_predictions(predictions: list[UrbanNavPrediction], path: Path) -> None:
    """Append prediction rows to JSONL.

    Args:
        predictions: Predictions to append.
        path: Target JSONL path.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()


def _existing_prediction_ids(path: Path) -> set[str]:
    """Read existing prediction ids from a partial output JSONL.

    Args:
        path: Prediction JSONL path.

    Returns:
        Set of already written prediction ids.
    """

    if not path.exists() or path.stat().st_size == 0:
        return set()
    existing: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = str(payload.get("id") or payload.get("case_id") or "").strip()
            if case_id:
                existing.add(case_id)
    return existing


if __name__ == "__main__":
    main()
