#!/usr/bin/env python3
"""Generate the source-pool UrbanNav 4s/32-point trajectory data contract."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from mcp_server.data.stage_ii_trajectory_v2_corpus import (
    build_stage_ii_trajectory_corpus_v2,
)


def _key_value(value: str) -> tuple[str, Path]:
    """Parse one repeatable ``dataset=/path`` argument."""

    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected dataset=/absolute/path")
    return name.strip(), Path(raw_path).expanduser()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse V2 source-pool generation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-manifest",
        action="append",
        required=True,
        type=_key_value,
        help="Repeat dataset=/path.jsonl; SideWalk may have main and extra.",
    )
    parser.add_argument(
        "--raw-root",
        action="append",
        required=True,
        type=_key_value,
        help="Repeat once per dataset, for example sidewalk=/raw/sidewalk.",
    )
    parser.add_argument("--identity-registry-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--limit-per-manifest", type=int, default=None)
    parser.add_argument("--raw-cache-size", type=int, default=8)
    parser.add_argument("--progress-interval", type=int, default=1000)
    parser.add_argument(
        "--allow-partial-horizon",
        action="store_true",
        help=(
            "Accept source-covered trajectory prefixes and emit the explicit "
            "zero-padded timestep mask. Full-horizon admission remains the default."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Build V2 sidecars and fail if any scanned row is unaccounted."""

    args = parse_args(argv)
    manifests: dict[str, list[Path]] = defaultdict(list)
    for dataset, path in args.dataset_manifest:
        manifests[dataset].append(path)
    raw_roots = dict(args.raw_root)
    if len(raw_roots) != len(args.raw_root):
        raise ValueError("duplicate --raw-root dataset")
    report = build_stage_ii_trajectory_corpus_v2(
        dataset_manifests=manifests,
        raw_roots=raw_roots,
        identity_registry_json=args.identity_registry_json,
        output_dir=args.output_dir,
        limit_per_dataset=args.limit_per_dataset,
        limit_per_manifest=args.limit_per_manifest,
        raw_cache_size=args.raw_cache_size,
        progress_interval=args.progress_interval,
        full_horizon_required=not args.allow_partial_horizon,
        progress_callback=lambda payload: print(
            json.dumps({"stage_ii_v2_progress": payload}, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
