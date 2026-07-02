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
from urllib import error as urlerror
from urllib import request as urlrequest


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
    parser.add_argument(
        "--skip-llm-preflight",
        action="store_true",
        help="Skip the strict LLM health check before generation.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run the configured LLM health check and exit without writing predictions.",
    )
    parser.add_argument(
        "--max-prediction-exception-rate",
        type=float,
        default=None,
        help=(
            "Abort when prediction_exception rows exceed this rate after enough samples. "
            "Defaults to 0.1 for --strict-llm and disabled otherwise."
        ),
    )
    parser.add_argument(
        "--exception-rate-min-samples",
        type=int,
        default=20,
        help="Minimum written prediction rows before enforcing --max-prediction-exception-rate.",
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
    parser.add_argument(
        "--stop-after-pending",
        type=int,
        default=None,
        help=(
            "Process at most this many pending cases, then exit cleanly. "
            "Use with --resume from a wrapper script to periodically restart "
            "the Python process and clear provider/client state."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Generate predictions from the configured runner.

    Returns:
        None. Predictions and an optional manifest are written to disk.
    """

    args = parse_args()
    _disable_env_proxy_for_local_llm(args)
    keys = _load_api_keys(path=args.api_keys_file, env_names=args.api_keys_env)
    if (args.use_llm_intent or args.use_llm_reranker) and not keys:
        raise ValueError("LLM mode requires at least one key from --api-keys-file or configured environment variables")
    if _strict_llm_health_required(args):
        asyncio.run(_run_strict_llm_preflight(args=args, api_key=keys[0]))
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "event": "prediction_preflight_ok",
                    "llm_model": args.llm_model or os.getenv("LLM_MODEL", ""),
                    "llm_provider": args.llm_provider or os.getenv("LLM_PROVIDER", ""),
                    "llm_base_url": args.llm_base_url or os.getenv("LLM_API_BASE_URL", ""),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return

    cases = load_urbannav_cases(args.cases)
    if args.max_cases is not None:
        if args.max_cases <= 0:
            raise ValueError("--max-cases must be positive when provided")
        cases = cases[: args.max_cases]

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
        _enforce_prediction_exception_rate_counts(
            args=args,
            exception_count=sum(1 for prediction in predictions if _is_prediction_exception(prediction)),
            written_count=len(predictions),
        )
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
    if args.stop_after_pending is not None:
        if args.stop_after_pending <= 0:
            raise ValueError("--stop-after-pending must be positive when provided")
        pending_cases = pending_cases[: int(args.stop_after_pending)]
    if not args.resume:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    if args.resume:
        print(
            json.dumps(
                {
                    "event": "prediction_resume",
                    "existing_predictions": len(existing_ids),
                    "pending_cases": len(pending_cases),
                    "stop_after_pending": args.stop_after_pending,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    chunk_size = max(1, int(args.chunk_size or 1))
    written = len(existing_ids)
    written_exceptions = _existing_prediction_exception_count(output_path) if args.resume else 0
    for start in range(0, len(pending_cases), chunk_size):
        chunk = pending_cases[start : start + chunk_size]
        chunk_predictions = await _generate_predictions(args=args, cases=chunk, api_keys=api_keys)
        chunk_exceptions = sum(1 for prediction in chunk_predictions if _is_prediction_exception(prediction))
        _enforce_prediction_exception_rate_counts(
            args=args,
            exception_count=written_exceptions + chunk_exceptions,
            written_count=written + len(chunk_predictions),
        )
        _append_predictions(chunk_predictions, output_path)
        written += len(chunk_predictions)
        written_exceptions += chunk_exceptions
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
                try:
                    _validate_strict_llm_prediction(
                        prediction,
                        require_intent=bool(args.use_llm_intent),
                        require_reranker=bool(args.use_llm_reranker),
                    )
                except RuntimeError as exc:
                    trace = prediction.trace if isinstance(prediction.trace, dict) else {}
                    return UrbanNavPrediction(
                        case_id=case.case_id,
                        trace={
                            **trace,
                            "runner": trace.get("runner") or "external_routeserver_llm",
                            "case_id": case.case_id,
                            "instruction": case.instruction,
                            "category": case.category,
                            "subcategory": case.subcategory,
                            "city": case.city,
                            "error": "prediction_exception",
                            "detail": str(exc),
                        },
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
        source = _strict_intent_source(trace)
        if source != "llm":
            raise RuntimeError(f"strict_llm_intent_failed:{source or 'missing'}")
    if require_reranker:
        decision = trace.get("reranker")
        if not isinstance(decision, dict) or not decision.get("ranked_candidate_ids"):
            raise RuntimeError("strict_llm_reranker_failed")


def _strict_intent_source(trace: dict[str, Any]) -> str:
    """Return the LLM source marker for single- or multi-goal parsing.

    Args:
        trace: Prediction trace emitted by the runner.

    Returns:
        ``"llm"`` when the requested intent step was handled by an LLM,
        otherwise the observed fallback source or an empty string.

    Multi-stop cases use ``GoalDecomposer`` rather than ``IntentParser`` and
    therefore record ``goal_decomposition.decomposition_source`` instead of
    ``intent.intent_source``. Treating that field as missing incorrectly marks
    every strict multi-stop prediction as an exception.
    """

    intent = trace.get("intent") if isinstance(trace.get("intent"), dict) else {}
    source = str(intent.get("intent_source") or "").strip()
    if source:
        return source
    decomposition = trace.get("goal_decomposition")
    if isinstance(decomposition, dict):
        return str(decomposition.get("decomposition_source") or "").strip()
    return ""


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


def _strict_llm_health_required(args: argparse.Namespace) -> bool:
    """Return whether strict LLM preflight should run.

    Args:
        args: Parsed command-line arguments.

    Returns:
        True when strict mode requires an LLM-backed step and the preflight was
        not explicitly disabled.
    """

    return bool(args.strict_llm and (args.use_llm_intent or args.use_llm_reranker) and not args.skip_llm_preflight)


async def _run_strict_llm_preflight(*, args: argparse.Namespace, api_key: str) -> None:
    """Check that the configured LLM endpoint works before a long run.

    Args:
        args: Parsed command-line arguments.
        api_key: One loaded API key for the configured provider.

    Returns:
        None.

    Raises:
        RuntimeError: If the LLM endpoint fails and would trigger heuristic
        fallback under strict mode.
    """

    settings = _llm_settings_for_key(args, api_key)
    if settings.base_url:
        await asyncio.to_thread(_run_chat_completions_http_preflight, settings)
        return
    parsed = await IntentParser(settings=settings).parse("Please find a nearby restaurant.")
    source = str(parsed.get("intent_source") or "").strip()
    if source != "llm":
        detail = str(parsed.get("llm_error") or source or "missing_llm_source").strip()
        raise RuntimeError(f"strict_llm_preflight_failed:{detail}")


def _run_chat_completions_http_preflight(settings: LLMSettings) -> None:
    """Run a short OpenAI-compatible HTTP health check.

    Args:
        settings: LLM settings for one API key.

    Returns:
        None.

    Raises:
        RuntimeError: If the endpoint cannot produce a non-empty chat
        completion response.
    """

    base_url = str(settings.base_url or "").rstrip("/")
    url = f"{base_url}/chat/completions"
    payload = {
        "model": settings.model,
        "messages": [{"role": "user", "content": "Only output OK."}],
        "temperature": 0,
        "max_tokens": 8,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    request = urlrequest.Request(url, data=body, headers=headers, method="POST")
    timeout_s = max(1.0, min(float(settings.timeout_s or 15.0), 15.0))
    opener = urlrequest.build_opener(urlrequest.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            status = int(response.status)
            raw = response.read(4096).decode("utf-8", errors="replace")
    except urlerror.HTTPError as exc:
        detail = exc.read(1024).decode("utf-8", errors="replace")
        raise RuntimeError(f"strict_llm_preflight_failed:http_{exc.code}:{detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"strict_llm_preflight_failed:{type(exc).__name__}:{exc}") from exc
    if not 200 <= status < 300:
        raise RuntimeError(f"strict_llm_preflight_failed:http_{status}:{raw[:200]}")
    if not raw.strip():
        raise RuntimeError("strict_llm_preflight_failed:empty_response")


def _effective_max_exception_rate(args: argparse.Namespace) -> float | None:
    """Return the active prediction-exception threshold.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Maximum allowed exception rate, or None when disabled.
    """

    if args.max_prediction_exception_rate is not None:
        rate = float(args.max_prediction_exception_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError("--max-prediction-exception-rate must be between 0 and 1")
        return rate
    if args.strict_llm:
        return 0.1
    return None


def _enforce_prediction_exception_rate_counts(
    *,
    args: argparse.Namespace,
    exception_count: int,
    written_count: int,
) -> None:
    """Abort invalid runs before they are mistaken for benchmark results.

    Args:
        args: Parsed command-line arguments.
        exception_count: Cumulative prediction_exception rows.
        written_count: Total prediction rows that would be present after the
            current write.

    Returns:
        None.

    Raises:
        RuntimeError: If prediction_exception rows exceed the configured
        threshold after the minimum sample count.
    """

    max_rate = _effective_max_exception_rate(args)
    if max_rate is None:
        return
    min_samples = max(1, int(args.exception_rate_min_samples or 1))
    if written_count < min_samples:
        return
    exception_rate = exception_count / max(written_count, 1)
    print(
        json.dumps(
            {
                "event": "prediction_exception_rate_check",
                "exception_count": exception_count,
                "exception_rate": round(exception_rate, 6),
                "max_prediction_exception_rate": max_rate,
                "written_count": written_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    if exception_rate > max_rate:
        raise RuntimeError(
            "prediction_exception_rate_exceeded:"
            f"{exception_rate:.6f}>{max_rate:.6f}; "
            "run is invalid and should not be used as a benchmark result"
        )


def _is_prediction_exception(prediction: UrbanNavPrediction) -> bool:
    """Return whether a prediction row is an exception placeholder.

    Args:
        prediction: Prediction object.

    Returns:
        True when the row contains a prediction_exception trace.
    """

    trace = prediction.trace if isinstance(prediction.trace, dict) else {}
    return str(trace.get("error") or "") == "prediction_exception"


def _disable_env_proxy_for_local_llm(args: argparse.Namespace) -> None:
    """Bypass environment proxies for local OpenAI-compatible relays.

    Args:
        args: Parsed command-line arguments.

    Returns:
        None.
    """

    base_url = str(args.llm_base_url or os.getenv("LLM_API_BASE_URL", "") or "").strip().lower()
    if not _is_local_base_url(base_url):
        return
    no_proxy_items = ["127.0.0.1", "localhost", "::1"]
    existing_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    for item in existing_no_proxy.split(","):
        item = item.strip()
        if item and item not in no_proxy_items:
            no_proxy_items.append(item)
    os.environ["NO_PROXY"] = ",".join(no_proxy_items)
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(name, None)


def _is_local_base_url(base_url: str) -> bool:
    """Return whether an LLM base URL points to loopback.

    Args:
        base_url: Lower-cased URL string.

    Returns:
        True for localhost or loopback base URLs.
    """

    return base_url.startswith(("http://127.", "https://127.", "http://localhost", "https://localhost", "http://[::1]", "https://[::1]"))


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


def _existing_prediction_exception_count(path: Path) -> int:
    """Count prediction_exception rows in a partial output JSONL.

    Args:
        path: Prediction JSONL path.

    Returns:
        Number of existing exception placeholder rows.
    """

    if not path.exists() or path.stat().st_size == 0:
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                count += 1
                continue
            trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
            if trace.get("error") == "prediction_exception":
                count += 1
    return count


if __name__ == "__main__":
    main()
