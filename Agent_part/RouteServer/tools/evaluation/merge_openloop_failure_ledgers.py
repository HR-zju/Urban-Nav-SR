#!/usr/bin/env python3
"""Merge pass-specific failure ledgers with duplicate-id rejection."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcp_server.contracts.openloop_benchmark import OpenLoopFailureV1
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic


def merge_failure_ledgers(
    input_paths: list[Path],
    output_path: Path,
) -> int:
    """Merge canonical failure rows in input order.

    Args:
        input_paths: Pass-R/D/T failure artifacts.
        output_path: Unique merged failure destination.

    Returns:
        Number of merged failures.

    Raises:
        ValueError: If a sample appears in more than one failure artifact.
    """

    rows: list[dict] = []
    seen: set[str] = set()
    for path in input_paths:
        for payload in iter_jsonl(path):
            failure = OpenLoopFailureV1.model_validate(payload)
            if failure.sample_id in seen:
                raise ValueError(
                    f"duplicate failure sample_id: {failure.sample_id}"
                )
            seen.add(failure.sample_id)
            rows.append(failure.model_dump(mode="json", exclude_none=True))
    write_jsonl_atomic(output_path, rows)
    return len(rows)


def main() -> None:
    """Parse arguments and write one merged failure ledger."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    count = merge_failure_ledgers(args.input, args.output)
    print(f"merged_failure_count={count}")


if __name__ == "__main__":
    main()
