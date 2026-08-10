#!/usr/bin/env python3
"""Evaluate a frozen plugin package against prediction/failure ledgers offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mcp_server.contracts.benchmark_plugin import BenchmarkFailureV1, BenchmarkPredictionV1
from mcp_server.eval.openloop.benchmark_plugin_package import (
    join_benchmark_artifacts_v1,
    load_benchmark_package_v1,
    load_unique_models_v1,
)
from mcp_server.eval.openloop.benchmark_plugin_review import write_benchmark_plugin_review_html_v1
from mcp_server.eval.openloop.benchmark_plugin_review_base import (
    load_benchmark_native_adapter_traces_v1,
    summarize_benchmark_native_adapter_funnel_v1,
)
from mcp_server.eval.openloop.benchmark_plugin_runner import (
    benchmark_evaluation_admission_v1,
    evaluate_benchmark_plugin_v1,
)
from mcp_server.eval.openloop.openloop_reporting import write_openloop_scorecard_v1


def main() -> int:
    """Load immutable artifacts, run metrics, and persist scorecard/review outputs.

    Returns:
        Process exit code zero after deterministic evaluation completes.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-json", required=True)
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--failure-jsonl")
    parser.add_argument(
        "--adapter-trace-jsonl",
        help=("optional public-only native-adapter trace sidecar; it augments "
              "review/funnel only and never alters the scorecard"),
    )
    parser.add_argument(
        "--require-complete-adapter-traces",
        action="store_true",
        help=
        "fail unless the optional trace sidecar has one bound row per input",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--minimum-successful-predictions",
        type=int,
        default=1,
        help=("absolute prediction admission floor; the default prevents an "
              "all-failure ledger from being reported as passed"),
    )
    parser.add_argument("--title",
                        default="UrbanNav pluggable benchmark scorecard")
    args = parser.parse_args()
    if args.require_complete_adapter_traces and args.adapter_trace_jsonl is None:
        parser.error(
            "--require-complete-adapter-traces requires --adapter-trace-jsonl")
    package, inputs, references = load_benchmark_package_v1(args.package_json)
    predictions = load_unique_models_v1(args.prediction_jsonl,
                                        BenchmarkPredictionV1)
    failures = load_unique_models_v1(
        args.failure_jsonl, BenchmarkFailureV1) if args.failure_jsonl else {}
    report = evaluate_benchmark_plugin_v1(
        profile=package.benchmark_profile,
        inputs=inputs,
        references=references,
        predictions=predictions,
        failures=failures,
    )
    output = Path(args.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    admission = benchmark_evaluation_admission_v1(
        profile=package.benchmark_profile,
        reference_count=len(references),
        prediction_count=len(predictions),
        minimum_successful_predictions=args.minimum_successful_predictions,
    )
    report['evaluation_admission'] = admission
    write_openloop_scorecard_v1(
        report,
        json_path=output / "scorecard.json",
        csv_path=output / "scorecard.csv",
        html_path=output / "scorecard.html",
        title=args.title,
    )
    joined = join_benchmark_artifacts_v1(references, predictions, failures)
    adapter_traces = None
    adapter_trace_funnel = None
    adapter_trace_path = None
    if args.adapter_trace_jsonl is not None:
        adapter_trace_path = str(
            Path(args.adapter_trace_jsonl).expanduser().resolve())
        adapter_traces = load_benchmark_native_adapter_traces_v1(
            args.adapter_trace_jsonl,
            inputs=inputs,
            require_complete=args.require_complete_adapter_traces,
        )
        adapter_trace_funnel = summarize_benchmark_native_adapter_funnel_v1(
            rows=joined,
            inputs=inputs,
            traces=adapter_traces,
            horizons_s=package.benchmark_profile.time_policy.horizons_s,
        )
        (output / "adapter_trace_funnel.json").write_text(
            json.dumps(adapter_trace_funnel,
                       ensure_ascii=False,
                       indent=2,
                       sort_keys=True) + "\n",
            encoding="utf-8",
        )
    write_benchmark_plugin_review_html_v1(
        path=output / "review.html",
        rows=joined,
        inputs=inputs,
        horizon_s=package.benchmark_profile.time_policy.horizons_s[0],
        title=args.title + " review",
        adapter_traces=adapter_traces,
        adapter_trace_funnel=adapter_trace_funnel,
        coverage_horizons_s=package.benchmark_profile.time_policy.horizons_s,
    )
    (output / 'evaluation_admission.json').write_text(
        json.dumps(admission, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding='utf-8',
    )
    print(
        json.dumps(
            {
                "output_root":
                str(output),
                "report":
                report,
                "adapter_trace_path":
                adapter_trace_path,
                "adapter_trace_funnel_path":
                (None if adapter_trace_funnel is None else str(
                    output / "adapter_trace_funnel.json")),
            },
            ensure_ascii=False,
            indent=2,
        ))
    return 0 if admission['status'] == 'passed' else 2


if __name__ == "__main__":
    raise SystemExit(main())
