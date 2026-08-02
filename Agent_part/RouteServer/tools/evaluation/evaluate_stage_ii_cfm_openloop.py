#!/usr/bin/env python3
"""Evaluate a frozen UrbanNav Stage II CFM open-loop run package."""

from __future__ import annotations

import argparse
from pathlib import Path

from mcp_server.contracts.openloop_benchmark import (
    CheckpointExposureRegistryV1,
    OpenLoopFailureV1,
    OpenLoopPredictionV1,
)
from mcp_server.eval.openloop_common import (
    audit_checkpoint_exposure_v1,
    audit_output_input_digests_v1,
    join_openloop_artifacts_v1,
    load_unique_jsonl_models_v1,
    validate_openloop_package_v1,
)
from mcp_server.eval.openloop_reporting import write_openloop_scorecard_v1
from mcp_server.eval.stage_ii_cfm_openloop import (
    evaluate_stage_ii_cfm_openloop_v1,
)


def parse_args() -> argparse.Namespace:
    """Parse formal Stage II evaluator arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-manifest", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--failures", required=True, type=Path)
    parser.add_argument("--exposure-registry", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    """Validate package lineage, evaluate, and export three scorecard forms."""

    args = parse_args()
    artifacts = validate_openloop_package_v1(
        args.package_manifest,
        args.run_manifest,
    )
    registry = CheckpointExposureRegistryV1.model_validate_json(
        args.exposure_registry.read_text(encoding="utf-8")
    )
    audit_checkpoint_exposure_v1(
        artifacts.inputs.values(),
        registry,
        checkpoint_digest=artifacts.run.checkpoint.checkpoint_digest,
    )
    predictions = load_unique_jsonl_models_v1(
        args.predictions,
        OpenLoopPredictionV1,
    )
    failures = load_unique_jsonl_models_v1(
        args.failures,
        OpenLoopFailureV1,
    )
    audit_output_input_digests_v1(
        artifacts.inputs,
        predictions,
        failures,
    )
    rows = join_openloop_artifacts_v1(
        artifacts.references,
        predictions,
        failures,
    )
    report = evaluate_stage_ii_cfm_openloop_v1(
        rows,
        run_manifest=artifacts.run,
        metric_policy=artifacts.package.metric_policy,
    )
    report["benchmark_id"] = artifacts.package.benchmark_id
    report["split"] = artifacts.package.split
    write_openloop_scorecard_v1(
        report,
        json_path=args.output_json,
        csv_path=args.output_csv,
        html_path=args.output_html,
        title="UrbanNav Stage II CFM Open-loop Scorecard",
    )


if __name__ == "__main__":
    main()
