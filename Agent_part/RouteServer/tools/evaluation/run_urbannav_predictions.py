#!/usr/bin/env python3
"""Run UrbanNav prediction runners over frozen benchmark cases."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcp_server.core.live_map_backend import LiveMapBackend
from mcp_server.core.replay_map_backend import ReplayMapBackend
from mcp_server.eval.urbannav_io import load_urbannav_cases, write_urbannav_predictions
from mcp_server.eval.urbannav_runners import (
    OracleUrbanNavRunner,
    ReplayUrbanNavBackendAdapter,
    RouteServerRunnerConfig,
    RouteServerUrbanNavRunner,
    run_urbannav_cases,
    run_urbannav_cases_async,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with case path, runner name, and output path.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path, help="Path to UrbanNav task_gt.jsonl")
    parser.add_argument(
        "--runner",
        choices=["oracle", "routeserver-replay", "routeserver-live"],
        default="oracle",
        help="Runner implementation",
    )
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Output predictions JSONL")
    parser.add_argument("--poi-cache", type=Path, default=None, help="POI replay cache for routeserver-replay")
    parser.add_argument("--route-cache", type=Path, default=None, help="Route replay cache for routeserver-replay")
    parser.add_argument("--map-provider", default=None, help="Map provider override for routeserver-live, e.g. amap")
    parser.add_argument("--candidate-limit", type=int, default=10, help="Maximum candidates for RouteServer runners")
    parser.add_argument("--search-radius-m", type=int, default=1500, help="Fallback search radius for RouteServer runners")
    parser.add_argument("--mode", default="foot", choices=["foot", "bike", "car"], help="Route mode")
    parser.add_argument("--use-llm-intent", action="store_true", help="Allow LLM intent parsing")
    parser.add_argument("--use-llm-reranker", action="store_true", help="Allow LLM candidate reranking")
    parser.add_argument(
        "--selection-strategy",
        choices=["route_distance", "semantic_then_route"],
        default="route_distance",
        help="Destination selection policy for RouteServer runners",
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected UrbanNav runner and write predictions.

    Returns:
        None.
    """

    args = parse_args()
    cases = load_urbannav_cases(args.cases)
    if args.runner == "oracle":
        runner = OracleUrbanNavRunner()
        predictions = run_urbannav_cases(runner, cases)
    elif args.runner == "routeserver-replay":
        if not args.poi_cache or not args.route_cache:
            raise ValueError("--poi-cache and --route-cache are required for routeserver-replay")
        config = _runner_config(args)
        backend = ReplayUrbanNavBackendAdapter(
            ReplayMapBackend(poi_cache_path=args.poi_cache, route_cache_path=args.route_cache)
        )
        runner = RouteServerUrbanNavRunner(backend=backend, config=config)
        predictions = run_urbannav_cases_async(runner, cases)
    elif args.runner == "routeserver-live":
        config = _runner_config(args)
        backend = LiveMapBackend.from_env(map_provider=args.map_provider)
        runner = RouteServerUrbanNavRunner(backend=backend, config=config)
        predictions = _run_live_runner(runner, backend, cases)
    else:
        raise ValueError(f"Unsupported runner: {args.runner}")
    write_urbannav_predictions(predictions, args.output_jsonl)


def _runner_config(args: argparse.Namespace) -> RouteServerRunnerConfig:
    """Build runner configuration from CLI arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        RouteServer runner configuration.
    """

    return RouteServerRunnerConfig(
        search_radius_m=args.search_radius_m,
        candidate_limit=args.candidate_limit,
        mode=args.mode,
        use_llm_intent=bool(args.use_llm_intent),
        use_llm_reranker=bool(args.use_llm_reranker),
        selection_strategy=str(args.selection_strategy),
    )


def _run_live_runner(
    runner: RouteServerUrbanNavRunner,
    backend: LiveMapBackend,
    cases: list,
) -> list:
    """Run a live runner with provider connection lifecycle.

    Args:
        runner: RouteServer runner.
        backend: Live map backend.
        cases: UrbanNav cases.

    Returns:
        Prediction records.
    """

    import asyncio

    async def _run() -> list:
        """Run all cases while the backend is connected.

        Returns:
            Prediction records.
        """

        await backend.connect()
        try:
            return [await runner.run_case_async(case) for case in cases]
        finally:
            await backend.disconnect()

    return asyncio.run(_run())


if __name__ == "__main__":
    main()
