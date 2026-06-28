#!/usr/bin/env python3
"""Compare UrbanNav Module 1 runs by internal relative scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mcp_server.eval.urbannav_module1_compare import (  # noqa: E402
    Module1ComparisonConfig,
    Module1RunSpec,
    compare_module1_runs,
    comparison_to_markdown,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argparse namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Run directory or summary.json path. Repeat for each model.",
    )
    parser.add_argument("--baseline", default=None, help="Baseline label. Defaults to the first --run label.")
    parser.add_argument("--output-json", type=Path, default=None, help="Write full JSON comparison report.")
    parser.add_argument("--output-md", type=Path, default=None, help="Write compact Markdown comparison report.")
    parser.add_argument("--win-epsilon", type=float, default=1e-9, help="Per-case score margin counted as win/loss.")
    return parser.parse_args()


def main() -> None:
    """Run relative comparison and write optional reports.

    Returns:
        None. The JSON report is printed to stdout unless ``--output-json`` is
        supplied.
    """

    args = parse_args()
    report = compare_module1_runs(
        Module1ComparisonConfig(
            runs=[_parse_run_spec(item) for item in args.run],
            baseline_label=args.baseline,
            win_epsilon=args.win_epsilon,
        )
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(comparison_to_markdown(report), encoding="utf-8")


def _parse_run_spec(text: str) -> Module1RunSpec:
    if "=" not in text:
        raise ValueError(f"--run must be LABEL=PATH, got: {text}")
    label, path = text.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"--run has empty label: {text}")
    return Module1RunSpec(label=label, path=Path(path.strip()))


if __name__ == "__main__":
    main()
