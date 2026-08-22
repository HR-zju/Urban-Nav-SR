#!/usr/bin/env python3
"""Build the production cached trajectory-query temporal pair manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from mcp_server.data.stage_ii.cached_pairs import (
    build_stage_ii_cached_traj_query_pairs_v1,
    build_stage_ii_cached_traj_query_pairs_v2,
)


def _dataset_manifest(value: str) -> tuple[str, Path]:
    """Parse one repeatable ``dataset=/path.jsonl`` argument."""

    dataset, separator, raw_path = value.partition("=")
    if not separator or not dataset.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected dataset=/absolute/path.jsonl")
    return dataset.strip(), Path(raw_path).expanduser()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse temporal pair materialization arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--stage-ii-views-jsonl", type=Path)
    input_group.add_argument(
        "--source-views-jsonl",
        type=Path,
        help="Pure-latent V2 source rows from Stage-I visible input plus sidecar.",
    )
    parser.add_argument(
        "--condition-mode",
        choices=("cached_traj_query_v1", "cached_traj_query_v2"),
        default="cached_traj_query_v1",
    )
    parser.add_argument(
        "--dataset-manifest",
        action="append",
        required=True,
        type=_dataset_manifest,
        help="Repeat dataset=/absolute/path.jsonl; sidewalk may repeat for main/extra.",
    )
    parser.add_argument("--trajectory-source-registry-json", required=True, type=Path)
    parser.add_argument("--output-pairs-jsonl", required=True, type=Path)
    parser.add_argument("--output-reject-jsonl", required=True, type=Path)
    parser.add_argument("--output-report-json", required=True, type=Path)
    parser.add_argument("--query-period-sec", required=True, type=float)
    parser.add_argument("--max-query-age-sec", required=True, type=float)
    parser.add_argument("--progress-interval", type=int, default=10000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the canonical materializer and print its report."""

    args = parse_args(argv)
    manifests: dict[str, list[Path]] = defaultdict(list)
    for dataset, path in args.dataset_manifest:
        manifests[dataset].append(path)
    progress_callback = lambda payload: print(
        json.dumps({"cached_pair_progress": payload}, ensure_ascii=False),
        file=sys.stderr,
        flush=True,
    )
    common = dict(
        dataset_manifests=manifests,
        trajectory_source_registry_json=args.trajectory_source_registry_json,
        output_pairs_jsonl=args.output_pairs_jsonl,
        output_reject_jsonl=args.output_reject_jsonl,
        output_report_json=args.output_report_json,
        query_period_sec=args.query_period_sec,
        max_query_age_sec=args.max_query_age_sec,
        progress_interval=args.progress_interval,
        progress_callback=progress_callback,
    )
    if args.condition_mode == "cached_traj_query_v1":
        if args.stage_ii_views_jsonl is None:
            raise ValueError("V1 materialization requires --stage-ii-views-jsonl")
        report = build_stage_ii_cached_traj_query_pairs_v1(
            stage_ii_views_jsonl=args.stage_ii_views_jsonl,
            **common,
        )
    else:
        if args.source_views_jsonl is None:
            raise ValueError("V2 materialization requires --source-views-jsonl")
        report = build_stage_ii_cached_traj_query_pairs_v2(
            source_views_jsonl=args.source_views_jsonl,
            **common,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
