#!/usr/bin/env python3
"""Run the UrbanNav Module 1 instruction-to-destination/route benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_server.eval.urbannav_module1_benchmark import (  # noqa: E402
    DEFAULT_MODULE1_DATASET,
    UrbanNavModule1BenchmarkConfig,
    module1_prediction_contract,
    run_module1_benchmark,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_MODULE1_DATASET, help="Frozen UrbanNav task_gt.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for benchmark artifacts")
    parser.add_argument(
        "--runner",
        choices=["external", "oracle", "routeserver-replay", "routeserver-live"],
        default="external",
        help="Prediction source",
    )
    parser.add_argument("--predictions", type=Path, default=None, help="External UrbanNavPrediction JSONL")
    parser.add_argument("--poi-cache", type=Path, default=None, help="POI replay cache for routeserver-replay")
    parser.add_argument("--route-cache", type=Path, default=None, help="Route replay cache for routeserver-replay")
    parser.add_argument("--map-provider", default=None, help="Map provider override for routeserver-live")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional smoke-test case cap")
    parser.add_argument("--candidate-limit", type=int, default=10)
    parser.add_argument("--search-radius-m", type=int, default=1500)
    parser.add_argument("--mode", default="foot", choices=["foot", "bike", "car"])
    parser.add_argument("--use-llm-intent", action="store_true")
    parser.add_argument("--use-llm-reranker", action="store_true")
    parser.add_argument(
        "--selection-strategy",
        choices=["route_distance", "semantic_then_route"],
        default="route_distance",
    )
    parser.add_argument("--destination-radius-m", type=float, default=30.0)
    parser.add_argument("--endpoint-radius-m", type=float, default=30.0)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum concurrent case runners for routeserver-replay/live.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--max-prediction-exception-rate",
        type=float,
        default=0.1,
        help="Reject external prediction JSONL when prediction_exception rows exceed this rate.",
    )
    parser.add_argument(
        "--prediction-exception-min-samples",
        type=int,
        default=20,
        help="Minimum external prediction rows before enforcing --max-prediction-exception-rate.",
    )
    parser.add_argument(
        "--allow-invalid-predictions",
        action="store_true",
        help="Disable external prediction_exception-rate guard for debugging only.",
    )
    parser.add_argument(
        "--print-contract",
        action="store_true",
        help="Print the UrbanNavPrediction contract and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the configured benchmark.

    Returns:
        None.  A compact JSON result is printed to stdout.
    """

    args = parse_args()
    if args.print_contract:
        print(json.dumps(module1_prediction_contract(), ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.runner == "external" and args.predictions and not args.allow_invalid_predictions:
        _validate_external_prediction_health(
            predictions_path=args.predictions,
            max_exception_rate=float(args.max_prediction_exception_rate),
            min_samples=int(args.prediction_exception_min_samples),
        )
    result = run_module1_benchmark(
        UrbanNavModule1BenchmarkConfig(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            runner=args.runner,
            predictions_path=args.predictions,
            poi_cache_path=args.poi_cache,
            route_cache_path=args.route_cache,
            map_provider=args.map_provider,
            max_cases=args.max_cases,
            candidate_limit=args.candidate_limit,
            search_radius_m=args.search_radius_m,
            mode=args.mode,
            use_llm_intent=args.use_llm_intent,
            use_llm_reranker=args.use_llm_reranker,
            selection_strategy=args.selection_strategy,
            destination_radius_m=args.destination_radius_m,
            endpoint_radius_m=args.endpoint_radius_m,
            runner_concurrency=args.concurrency,
            run_id=args.run_id,
        )
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


def _validate_external_prediction_health(
    *,
    predictions_path: Path,
    max_exception_rate: float,
    min_samples: int,
) -> None:
    """Reject external prediction files that are mostly placeholders.

    Args:
        predictions_path: External UrbanNavPrediction JSONL path.
        max_exception_rate: Maximum allowed prediction_exception fraction.
        min_samples: Minimum row count before the rate check is enforced.

    Returns:
        None.

    Raises:
        SystemExit: If the file is invalid for benchmark reporting.
    """

    if not predictions_path.exists():
        raise SystemExit(f"external predictions not found: {predictions_path}")
    if not 0.0 <= max_exception_rate <= 1.0:
        raise SystemExit("--max-prediction-exception-rate must be between 0 and 1")
    rows = 0
    exceptions = 0
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                exceptions += 1
                continue
            trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
            if trace.get("error") == "prediction_exception":
                exceptions += 1
    if rows <= 0:
        raise SystemExit(f"external predictions are empty: {predictions_path}")
    exception_rate = exceptions / rows
    print(
        json.dumps(
            {
                "event": "external_prediction_health_check",
                "prediction_rows": rows,
                "prediction_exception_count": exceptions,
                "prediction_exception_rate": round(exception_rate, 6),
                "max_prediction_exception_rate": max_exception_rate,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if rows >= max(1, min_samples) and exception_rate > max_exception_rate:
        raise SystemExit(
            "external prediction_exception_rate exceeded threshold; "
            "run is invalid and summary.json will not be written"
        )


if __name__ == "__main__":
    main()
