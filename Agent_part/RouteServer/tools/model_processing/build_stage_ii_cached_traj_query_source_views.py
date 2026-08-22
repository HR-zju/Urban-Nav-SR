#!/usr/bin/env python3
"""Bind Stage-I V2 visible rows to accepted 4-second trajectory sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from mcp_server.data.stage_ii.cached_pairs import (
    build_stage_ii_cached_traj_query_source_views_v2,
)


def _dataset_manifest(value: str) -> tuple[str, Path]:
    """Parse one repeatable ``dataset=/path.jsonl`` argument."""

    dataset, separator, raw_path = value.partition("=")
    if not separator or not dataset.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected dataset=/absolute/path.jsonl")
    return dataset.strip(), Path(raw_path).expanduser()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse source-view materialization arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-i-views-jsonl", required=True, type=Path)
    parser.add_argument(
        "--trajectory-sidecar-jsonl",
        action="append",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--dataset-manifest",
        action="append",
        required=True,
        type=_dataset_manifest,
        help="Repeat dataset=/absolute/path.jsonl; sidewalk may repeat.",
    )
    parser.add_argument("--trajectory-source-registry-json", required=True, type=Path)
    parser.add_argument("--output-source-views-jsonl", required=True, type=Path)
    parser.add_argument("--output-reject-jsonl", required=True, type=Path)
    parser.add_argument("--output-report-json", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-interval", type=int, default=10000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run source-view materialization and print its immutable report."""

    args = parse_args(argv)
    manifests: dict[str, list[Path]] = defaultdict(list)
    for dataset, path in args.dataset_manifest:
        manifests[dataset].append(path)
    progress_callback = lambda payload: print(
        json.dumps({"cached_source_view_progress": payload}, ensure_ascii=False),
        file=sys.stderr,
        flush=True,
    )
    report = build_stage_ii_cached_traj_query_source_views_v2(
        stage_i_views_jsonl=args.stage_i_views_jsonl,
        trajectory_sidecars_jsonl=args.trajectory_sidecar_jsonl,
        dataset_manifests=manifests,
        trajectory_source_registry_json=args.trajectory_source_registry_json,
        output_source_views_jsonl=args.output_source_views_jsonl,
        output_reject_jsonl=args.output_reject_jsonl,
        output_report_json=args.output_report_json,
        limit=args.limit,
        progress_interval=args.progress_interval,
        progress_callback=progress_callback,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
