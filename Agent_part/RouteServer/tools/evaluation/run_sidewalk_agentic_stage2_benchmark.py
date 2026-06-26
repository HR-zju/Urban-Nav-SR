#!/usr/bin/env python3
"""Run or collect SideWalk Agentic Stage2 benchmark artifacts.

This is a thin RouteServer-side orchestration CLI.  It either calls the
existing ms-swift Stage2 pipeline, collects an already completed pipeline
output directory, or does both.  The produced review package is offline and
replayable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_server.eval.agentic_stage2_benchmark import (  # noqa: E402
    AgenticStage2BenchmarkConfig,
    collect_agentic_stage2_benchmark,
    run_ms_swift_stage2_pipeline,
)


DEFAULT_ACTIONS = ("prepare", "benchmark_base", "benchmark_stage2a", "benchmark_stage2b", "compare")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=["collect", "run", "run-and-collect"], default="collect")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--heldout-manifest", required=True)
    parser.add_argument("--ms-swift-output-root", required=True)
    parser.add_argument("--report-output-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--stage2a-checkpoint", default=None)
    parser.add_argument("--stage2b-checkpoint", default=None)
    parser.add_argument("--ms-swift-root", default=None)
    parser.add_argument("--python-bin", default="python")
    parser.add_argument("--docker-container", default=None)
    parser.add_argument("--container-ms-swift-root", default=None)
    parser.add_argument(
        "--pipeline-action",
        action="append",
        choices=list(DEFAULT_ACTIONS),
        help="Pipeline action to run. May be repeated. Defaults to the full Step8 sequence.",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment variable passed to the ms-swift pipeline. May be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only write command logs for run actions.")
    return parser.parse_args()


def main() -> None:
    """Execute the selected benchmark action.

    Returns:
        ``None``.  A compact JSON summary is printed to stdout.

    Raises:
        ValueError: If ``--env`` values are malformed.
    """

    args = parse_args()
    config = AgenticStage2BenchmarkConfig(
        run_id=args.run_id,
        heldout_manifest=args.heldout_manifest,
        ms_swift_output_root=args.ms_swift_output_root,
        report_output_dir=args.report_output_dir,
        base_model=args.base_model,
        stage2a_checkpoint=args.stage2a_checkpoint,
        stage2b_checkpoint=args.stage2b_checkpoint,
        ms_swift_root=args.ms_swift_root,
        python_bin=args.python_bin,
        docker_container=args.docker_container,
        container_ms_swift_root=args.container_ms_swift_root,
        env=_parse_env_pairs(args.env),
    )
    result: dict[str, object] = {"run_id": args.run_id, "action": args.action}
    if args.action in {"run", "run-and-collect"}:
        pipeline_results = run_ms_swift_stage2_pipeline(
            config,
            actions=tuple(args.pipeline_action or DEFAULT_ACTIONS),
            dry_run=args.dry_run,
        )
        result["pipeline_commands"] = [item.to_dict() for item in pipeline_results]
    if args.action in {"collect", "run-and-collect"}:
        result["collection"] = collect_agentic_stage2_benchmark(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _parse_env_pairs(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--env must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--env has empty key: {item}")
        env[key] = value
    return env


if __name__ == "__main__":
    main()
