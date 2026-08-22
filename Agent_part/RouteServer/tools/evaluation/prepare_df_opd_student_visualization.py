#!/usr/bin/env python3
"""Create the planned visualization manifest before a DF-OPD evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def prepare_visualization_manifest(
    *,
    research_workflow_root: Path,
    output: Path,
    run_id: str,
    run_root: Path,
    planned_outputs: Sequence[str],
    tensorboard_logdir: Path | None = None,
    dashboard_url: str | None = None,
) -> None:
    """Delegate planned-manifest creation to the ResearchWorkflow owner.

    Args:
        research_workflow_root: Checkout containing the scripts directory.
        output: Planned manifest path for this run.
        run_id: Stable identifier shared by the run lifecycle artifacts.
        run_root: Run-specific visualization root.
        planned_outputs: artifact_id=path entries expected after evaluation.
        tensorboard_logdir: Optional TensorBoard directory reserved for this run.
        dashboard_url: Optional loopback dashboard URL.

    Raises:
        FileNotFoundError: If the authoritative ResearchWorkflow script is absent.
        subprocess.CalledProcessError: If the owner rejects the manifest request.
    """

    generator = research_workflow_root / "scripts" / "experiment_visualization_manifest.py"
    if not generator.is_file():
        raise FileNotFoundError(f"ResearchWorkflow manifest generator not found: {generator}")
    run_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(generator),
        "--out",
        str(output),
        "--run-id",
        run_id,
        "--run-kind",
        "evaluation",
        "--run-root",
        str(run_root),
        "--phase",
        "planned",
    ]
    if tensorboard_logdir is not None:
        command.extend(("--tensorboard-logdir", str(tensorboard_logdir)))
    if dashboard_url is not None:
        command.extend(("--dashboard-url", dashboard_url))
    for planned_output in planned_outputs:
        command.extend(("--planned-output", planned_output))
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the planned-manifest CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-workflow-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--planned-output", action="append", default=[])
    parser.add_argument("--tensorboard-logdir", type=Path)
    parser.add_argument("--dashboard-url")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Create one planned evaluation manifest and report its path."""

    args = build_parser().parse_args(argv)
    prepare_visualization_manifest(
        research_workflow_root=args.research_workflow_root,
        output=args.out,
        run_id=args.run_id,
        run_root=args.run_root,
        planned_outputs=args.planned_output,
        tensorboard_logdir=args.tensorboard_logdir,
        dashboard_url=args.dashboard_url,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
