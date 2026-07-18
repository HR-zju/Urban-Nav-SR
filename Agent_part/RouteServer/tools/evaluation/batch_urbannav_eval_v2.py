#!/usr/bin/env python3
"""Run UrbanNav v2 evaluation for a JSON manifest of saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SINGLE_RUNNER = ROOT / "tools" / "evaluation" / "run_urbannav_eval_v2.py"


def parse_args() -> argparse.Namespace:
    """Parse batch evaluator command-line arguments.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="Recompute runs whose eval_v2 summary already exists.")
    return parser.parse_args()


def main() -> None:
    """Evaluate every manifest entry and write aggregate JSON/Markdown tables."""

    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset = Path(payload["dataset"])
    results: list[dict[str, Any]] = []
    for item in payload.get("runs") or []:
        model = str(item["model"])
        predictions = Path(item["predictions"])
        output_dir = Path(item.get("output_dir") or predictions.parent / "eval_v2")
        if not predictions.exists():
            results.append({"model": model, "status": "missing_predictions", "predictions": str(predictions)})
            continue
        summary_path = output_dir / "summary.json"
        if args.force or not summary_path.exists():
            command = [
                sys.executable,
                str(SINGLE_RUNNER),
                "--dataset",
                str(dataset),
                "--predictions",
                str(predictions),
                "--output-dir",
                str(output_dir),
                "--run-id",
                model,
            ]
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
            if completed.returncode != 0:
                results.append(
                    {
                        "model": model,
                        "status": "evaluation_failed",
                        "predictions": str(predictions),
                        "error": completed.stderr.strip() or completed.stdout.strip(),
                    }
                )
                continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))["summary"]
        results.append(
            {
                "model": model,
                "status": "complete",
                "predictions": str(predictions),
                "output_dir": str(output_dir),
                **summary,
            }
        )

    report = {
        "schema_version": "urbannav_eval_v2_batch",
        "dataset": str(dataset),
        "runs": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    args.output_md.write_text(_markdown_table(results), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _markdown_table(results: list[dict[str, Any]]) -> str:
    """Render the v2 main metrics as a Markdown table.

    Args:
        results: Batch result dictionaries.

    Returns:
        Markdown table text.
    """

    lines = [
        "| Method | DS | MRR@5 | Hit@3 | Task Success | Task SPL | Multi-stop Seq | Stop Recall | Update |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        if row.get("status") != "complete":
            lines.append(f"| {row['model']} | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |")
            continue
        lines.append(
            "| {model} | {ds:.4f} | {mrr:.4f} | {hit:.4f} | {success:.4f} | {spl:.4f} | {seq:.4f} | {recall:.4f} | {update:.4f} |".format(
                model=row["model"],
                ds=float(row["destination_success"]),
                mrr=float(row["task_mrr@5"]),
                hit=float(row["task_hit@3"]),
                success=float(row["task_success"]),
                spl=float(row["task_spl"]),
                seq=float(row.get("multi_stop_sequence_success") or 0.0),
                recall=float(row.get("multi_stop_stop_recall") or 0.0),
                update=float(row.get("route_update_success") or 0.0),
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
