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
    parser.add_argument("--run-id", default=None)
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
            run_id=args.run_id,
        )
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
