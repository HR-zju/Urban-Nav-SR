#!/usr/bin/env python3
"""Generate the current 4-second history training view for all datasets.

Datasets are processed sequentially so that the receipt identifies the first
failed dataset and the output disk is not saturated by concurrent decoders.
The per-dataset implementation remains in
``generate_history_training_corpus.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generate_history_training_corpus import generate


DATASETS = ("egowalk", "sidewalk", "foxglove")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse one explicit plan and destination for every dataset."""

    parser = argparse.ArgumentParser(description=__doc__)
    for dataset in DATASETS:
        parser.add_argument(f"--{dataset}-plan", required=True, type=Path,
                            help=f"Completed {dataset} plan-batch directory")
    parser.add_argument("--output-root", required=True, type=Path,
                        help="New root directory containing one dataset output per subdirectory")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--history-max-lag-ns", type=int,
                        help="Override all plan lag values; omitted means use each plan value")
    parser.add_argument("--query-count", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent clip workers within one dataset")
    parser.add_argument("--target-profile-id",
                        default="ego_degree10_12way_terminal_centroid_v2")
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _write_receipt(path: Path, value: Any) -> None:
    """Write the aggregate receipt once all datasets have completed."""

    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv: list[str] | None = None) -> None:
    """Run EgoWalk, SideWalk, and Foxglove in the fixed order."""

    args = _parse_args(argv)
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"output already exists; choose a new directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    root_receipt = output_root / "generation_receipt.json"
    if args.resume and root_receipt.is_file():
        print(root_receipt.read_text(encoding="utf-8"), end="")
        return

    results = {}
    for dataset in DATASETS:
        dataset_args = argparse.Namespace(
            plan=getattr(args, f"{dataset}_plan").resolve(),
            output=output_root / dataset,
            dataset_source=dataset,
            jpeg_quality=args.jpeg_quality,
            history_max_lag_ns=args.history_max_lag_ns,
            query_count=args.query_count,
            workers=args.workers,
            target_profile_id=args.target_profile_id,
            archive=args.archive,
            resume=args.resume,
        )
        # 单个数据集失败直接抛出；已完成的数据集保留 receipt，后续数据集不伪装成完成。
        results[dataset] = generate(dataset_args)

    receipt = {
        "status": "generated",
        "dataset_order": list(DATASETS),
        "output_root": str(output_root),
        "datasets": results,
        "semantic_label_status": "preserved_source_fields; FullCoT/VQA regeneration is separate",
    }
    _write_receipt(output_root / "generation_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
