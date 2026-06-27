"""UrbanNav Module 1 benchmark orchestration.

Module 1 is the instruction-to-destination-and-route benchmark.  It evaluates
whether a system can map a user instruction and origin to a reasonable
destination and an executable walking route.  This module intentionally
orchestrates existing schema, runner, and metric components instead of
redefining them.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from mcp_server.core.live_map_backend import LiveMapBackend
from mcp_server.core.replay_map_backend import ReplayMapBackend
from mcp_server.eval.urbannav_io import (
    load_urbannav_cases,
    load_urbannav_predictions,
    write_urbannav_predictions,
)
from mcp_server.eval.urbannav_metrics import UrbanNavEvalConfig, evaluate_predictions, summarize_by
from mcp_server.eval.urbannav_runners import (
    OracleUrbanNavRunner,
    ReplayUrbanNavBackendAdapter,
    RouteServerRunnerConfig,
    RouteServerUrbanNavRunner,
    run_urbannav_cases,
    run_urbannav_cases_async,
)
from mcp_server.eval.urbannav_schema import UrbanNavCase, UrbanNavPrediction


URBANNAV_MODULE1_BENCHMARK_VERSION = "urbannav_module1_benchmark_v1"
DEFAULT_MODULE1_DATASET = Path("data/benchmark/urbannav_global_osm_local_full_en_v1/task_gt.jsonl")

RunnerName = Literal["oracle", "routeserver-replay", "routeserver-live", "external"]


@dataclass(frozen=True)
class UrbanNavModule1BenchmarkConfig:
    """Configuration for one Module 1 benchmark run.

    Args:
        dataset_path: Frozen UrbanNav GT JSONL.  The default points to the
            10K English global OSM-local split.
        output_dir: Directory where predictions, per-case rows, summaries,
            and manifest files will be written.
        runner: Prediction source.  ``external`` loads ``predictions_path``;
            ``oracle`` copies GT for contract tests; ``routeserver-replay``
            runs the RouteServer baseline against frozen replay cache;
            ``routeserver-live`` calls the configured live map provider.
        predictions_path: Existing prediction JSONL for external model
            evaluation.  Required when ``runner="external"``.
        poi_cache_path: POI replay cache for ``routeserver-replay``.
        route_cache_path: Route replay cache for ``routeserver-replay``.
        map_provider: Optional map provider override for ``routeserver-live``.
        max_cases: Optional cap for smoke tests.  ``None`` evaluates all cases.
        candidate_limit: Maximum candidates requested by RouteServer runners.
        search_radius_m: Fallback POI search radius for RouteServer runners.
        mode: Route mode, usually ``foot``.
        use_llm_intent: Whether the RouteServer runner may call LLM intent
            parsing.
        use_llm_reranker: Whether the RouteServer runner may call LLM
            reranking.
        selection_strategy: RouteServer candidate selection policy.
        destination_radius_m: Coordinate fallback radius for destination
            identity matching when stable POI ids are unavailable.
        endpoint_radius_m: Route endpoint tolerance for route success.
        run_id: Optional stable run id.  Defaults to the output directory name.
        metadata: Extra user-supplied metadata persisted in the manifest.
    """

    dataset_path: str | Path = DEFAULT_MODULE1_DATASET
    output_dir: str | Path = "data/benchmark/urbannav_module1_runs/default"
    runner: RunnerName = "external"
    predictions_path: str | Path | None = None
    poi_cache_path: str | Path | None = None
    route_cache_path: str | Path | None = None
    map_provider: str | None = None
    max_cases: int | None = None
    candidate_limit: int = 10
    search_radius_m: int = 1500
    mode: str = "foot"
    use_llm_intent: bool = False
    use_llm_reranker: bool = False
    selection_strategy: str = "route_distance"
    destination_radius_m: float = 30.0
    endpoint_radius_m: float = 30.0
    run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the configuration.

        Returns:
            JSON-serializable configuration dictionary.
        """

        return {
            "dataset_path": str(self.dataset_path),
            "output_dir": str(self.output_dir),
            "runner": self.runner,
            "predictions_path": str(self.predictions_path) if self.predictions_path else None,
            "poi_cache_path": str(self.poi_cache_path) if self.poi_cache_path else None,
            "route_cache_path": str(self.route_cache_path) if self.route_cache_path else None,
            "map_provider": self.map_provider,
            "max_cases": self.max_cases,
            "candidate_limit": self.candidate_limit,
            "search_radius_m": self.search_radius_m,
            "mode": self.mode,
            "use_llm_intent": self.use_llm_intent,
            "use_llm_reranker": self.use_llm_reranker,
            "selection_strategy": self.selection_strategy,
            "destination_radius_m": self.destination_radius_m,
            "endpoint_radius_m": self.endpoint_radius_m,
            "run_id": self.run_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class UrbanNavModule1BenchmarkResult:
    """Result paths and summary for one Module 1 benchmark run.

    Args:
        run_id: Stable run id.
        output_dir: Directory containing all written artifacts.
        dataset_path: GT dataset path.
        predictions_path: Normalized prediction JSONL written by this run.
        rows_path: Per-case evaluation rows JSONL.
        summary_path: Aggregate JSON report.
        manifest_path: Run manifest.
        case_count: Number of evaluated GT cases.
        prediction_count: Number of loaded/generated predictions.
        summary: Aggregate metric report.
    """

    run_id: str
    output_dir: str
    dataset_path: str
    predictions_path: str
    rows_path: str
    summary_path: str
    manifest_path: str
    case_count: int
    prediction_count: int
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize benchmark result.

        Returns:
            JSON-ready benchmark result dictionary.
        """

        return {
            "run_id": self.run_id,
            "output_dir": self.output_dir,
            "dataset_path": self.dataset_path,
            "predictions_path": self.predictions_path,
            "rows_path": self.rows_path,
            "summary_path": self.summary_path,
            "manifest_path": self.manifest_path,
            "case_count": self.case_count,
            "prediction_count": self.prediction_count,
            "summary": self.summary,
        }


def run_module1_benchmark(config: UrbanNavModule1BenchmarkConfig) -> UrbanNavModule1BenchmarkResult:
    r"""Run the complete UrbanNav Module 1 benchmark.

    The benchmark computes predictions, evaluates them, and writes a reusable
    report package.  Its core evaluation target is:

    \[
    f_\theta(q,o) \rightarrow (\hat d,\hat C,\hat P)
    \]

    where ``q`` is the user instruction, ``o`` is the origin, ``\hat d`` is the
    selected destination, ``\hat C`` is an optional ranked candidate list, and
    ``\hat P`` is the predicted route polyline.

    Args:
        config: Benchmark configuration.

    Returns:
        Paths and summary metrics for the completed run.

    Raises:
        ValueError: If runner-specific required inputs are missing.
    """

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = config.run_id or output_dir.name
    cases = _load_cases(config.dataset_path, max_cases=config.max_cases)
    predictions = _load_or_generate_predictions(config, cases)

    predictions_path = output_dir / "predictions.jsonl"
    rows_path = output_dir / "eval_rows.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    write_urbannav_predictions(predictions, predictions_path)
    rows, summary = evaluate_predictions(
        cases,
        predictions,
        config=UrbanNavEvalConfig(
            destination_tolerance_m=config.destination_radius_m,
            endpoint_tolerance_m=config.endpoint_radius_m,
        ),
    )
    report = {
        "schema_version": URBANNAV_MODULE1_BENCHMARK_VERSION,
        "run_id": run_id,
        "summary": summary,
        "by_category": summarize_by(rows, "category"),
        "by_subcategory": summarize_by(rows, "subcategory"),
        "by_city": summarize_by(rows, "city"),
        "by_scene_id": summarize_by(rows, "scene_id"),
        "by_difficulty": summarize_by(rows, "difficulty"),
    }
    _write_json(report, summary_path)
    _write_jsonl((row.to_dict() for row in rows), rows_path)
    manifest = _build_manifest(
        config=config,
        run_id=run_id,
        case_count=len(cases),
        prediction_count=len(predictions),
        predictions_path=predictions_path,
        rows_path=rows_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
    )
    _write_json(manifest, manifest_path)

    return UrbanNavModule1BenchmarkResult(
        run_id=run_id,
        output_dir=str(output_dir),
        dataset_path=str(config.dataset_path),
        predictions_path=str(predictions_path),
        rows_path=str(rows_path),
        summary_path=str(summary_path),
        manifest_path=str(manifest_path),
        case_count=len(cases),
        prediction_count=len(predictions),
        summary=report,
    )


def module1_prediction_contract() -> dict[str, Any]:
    """Return the prediction contract for external models.

    Returns:
        JSON-ready schema summary that external model runners should satisfy.
    """

    return {
        "schema_version": "urbannav_prediction_contract_v1",
        "format": "JSONL, one object per case",
        "required": {
            "id": "Case id matching task_gt.jsonl id.",
            "chosen_destination": "Top-1 destination object with stable provider id or coordinates.",
        },
        "recommended": {
            "candidate_destinations": "Ranked candidate list used for hit@5.",
            "route_polyline": "Predicted route polyline as [{lat,lng}, ...] or [[lng,lat], ...].",
            "route_summary.distance_m": "Predicted route length in meters.",
            "trace": "Optional diagnostic model/tool trace. It is not part of main scoring.",
        },
        "destination_fields": [
            "provider_poi_id",
            "place_id",
            "name",
            "type",
            "lat",
            "lng",
        ],
        "main_metrics": [
            "destination_success",
            "hit@5",
            "route_success",
            "median_route_ratio",
            "spl",
        ],
    }


def _load_cases(path: str | Path, *, max_cases: int | None) -> list[UrbanNavCase]:
    """Load and optionally truncate benchmark cases.

    Args:
        path: GT JSONL path.
        max_cases: Optional maximum number of cases.

    Returns:
        Loaded cases.
    """

    cases = load_urbannav_cases(path)
    if max_cases is not None:
        if max_cases <= 0:
            raise ValueError("max_cases must be positive when provided")
        cases = cases[:max_cases]
    return cases


def _load_or_generate_predictions(
    config: UrbanNavModule1BenchmarkConfig,
    cases: list[UrbanNavCase],
) -> list[UrbanNavPrediction]:
    """Load external predictions or run a configured baseline.

    Args:
        config: Benchmark configuration.
        cases: Loaded GT cases.

    Returns:
        Prediction records.
    """

    if config.runner == "external":
        if not config.predictions_path:
            raise ValueError("predictions_path is required when runner='external'")
        predictions = load_urbannav_predictions(config.predictions_path)
        if config.max_cases is not None:
            allowed = {case.case_id for case in cases}
            predictions = [prediction for prediction in predictions if prediction.case_id in allowed]
        return predictions
    if config.runner == "oracle":
        return run_urbannav_cases(OracleUrbanNavRunner(), cases)
    if config.runner == "routeserver-replay":
        if not config.poi_cache_path or not config.route_cache_path:
            raise ValueError("poi_cache_path and route_cache_path are required for routeserver-replay")
        backend = ReplayUrbanNavBackendAdapter(
            ReplayMapBackend(
                poi_cache_path=config.poi_cache_path,
                route_cache_path=config.route_cache_path,
            )
        )
        runner = RouteServerUrbanNavRunner(backend=backend, config=_runner_config(config))
        return run_urbannav_cases_async(runner, cases)
    if config.runner == "routeserver-live":
        return _run_live_baseline(config, cases)
    raise ValueError(f"Unsupported runner: {config.runner}")


def _run_live_baseline(
    config: UrbanNavModule1BenchmarkConfig,
    cases: list[UrbanNavCase],
) -> list[UrbanNavPrediction]:
    """Run RouteServer live baseline with provider lifecycle management.

    Args:
        config: Benchmark configuration.
        cases: GT cases.

    Returns:
        Prediction records.
    """

    async def _run() -> list[UrbanNavPrediction]:
        backend = LiveMapBackend.from_env(map_provider=config.map_provider)
        runner = RouteServerUrbanNavRunner(backend=backend, config=_runner_config(config))
        await backend.connect()
        try:
            return [await runner.run_case_async(case) for case in cases]
        finally:
            await backend.disconnect()

    return asyncio.run(_run())


def _runner_config(config: UrbanNavModule1BenchmarkConfig) -> RouteServerRunnerConfig:
    """Build RouteServer runner configuration.

    Args:
        config: Module 1 benchmark configuration.

    Returns:
        RouteServer runner configuration.
    """

    return RouteServerRunnerConfig(
        search_radius_m=config.search_radius_m,
        candidate_limit=config.candidate_limit,
        mode=config.mode,
        use_llm_intent=config.use_llm_intent,
        use_llm_reranker=config.use_llm_reranker,
        selection_strategy=config.selection_strategy,
    )


def _build_manifest(
    *,
    config: UrbanNavModule1BenchmarkConfig,
    run_id: str,
    case_count: int,
    prediction_count: int,
    predictions_path: Path,
    rows_path: Path,
    summary_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Build a reproducibility manifest.

    Args:
        config: Benchmark configuration.
        run_id: Stable run id.
        case_count: Number of evaluated cases.
        prediction_count: Number of predictions.
        predictions_path: Written prediction file.
        rows_path: Written per-case rows file.
        summary_path: Written summary file.
        manifest_path: Manifest file path.

    Returns:
        JSON-ready manifest.
    """

    return {
        "schema_version": URBANNAV_MODULE1_BENCHMARK_VERSION,
        "run_id": run_id,
        "module": "urbannav_module1_instruction_to_destination_route",
        "config": config.to_dict(),
        "case_count": case_count,
        "prediction_count": prediction_count,
        "prediction_contract": module1_prediction_contract(),
        "artifacts": {
            "dataset": str(config.dataset_path),
            "predictions": str(predictions_path),
            "eval_rows": str(rows_path),
            "summary": str(summary_path),
            "manifest": str(manifest_path),
        },
    }


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    """Write a formatted JSON object.

    Args:
        payload: JSON-serializable payload.
        path: Output path.

    Returns:
        None.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write dictionaries as JSONL.

    Args:
        records: JSON-serializable dictionaries.
        path: Output path.

    Returns:
        None.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
