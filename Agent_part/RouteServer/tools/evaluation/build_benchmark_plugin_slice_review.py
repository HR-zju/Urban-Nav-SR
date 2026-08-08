#!/usr/bin/env python3
"""Build a portable deterministic review subset from a completed benchmark ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from mcp_server.contracts.benchmark_plugin import BenchmarkFailureV1, BenchmarkPredictionV1
from mcp_server.data.io.jsonl import write_jsonl_atomic
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
from mcp_server.eval.openloop.benchmark_plugin_slice_review import (
    build_benchmark_slice_review_diagnosis_template_v1,
    build_benchmark_slice_review_selection_v1,
    evaluate_benchmark_coordinate_diagnostic_v1,
    evaluate_benchmark_reference_motion_diagnostic_v1,
    select_benchmark_slice_review_rows_v1,
)


def main() -> int:
    """Write a fixed, evaluator-only review bundle and coordinate diagnostic.

    Returns:
        Process exit code zero when immutable ledgers, media, and generated
        local review artifacts have all been verified.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-json", required=True)
    parser.add_argument("--prediction-jsonl", required=True)
    parser.add_argument("--failure-jsonl")
    parser.add_argument(
        "--adapter-trace-jsonl",
        help=("optional public-only native-adapter trace sidecar; it augments "
              "review/funnel only and never alters diagnostics or scorecards"),
    )
    parser.add_argument(
        "--require-complete-adapter-traces",
        action="store_true",
        help=
        "fail unless the optional trace sidecar has one bound row per input",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--slice-label", action="append", required=True)
    parser.add_argument("--horizon-s", type=float, default=2.0)
    parser.add_argument("--per-slice", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--title", default="UrbanNav benchmark deterministic slice review")
    parser.add_argument(
        "--copy-media",
        action="store_true",
        help=
        "copy selected source media into output-root/media and use relative review links",
    )
    parser.add_argument(
        "--require-readable-media",
        action="store_true",
        help="fail if a selected model-visible image cannot be read",
    )
    args = parser.parse_args()
    if args.require_complete_adapter_traces and args.adapter_trace_jsonl is None:
        parser.error(
            "--require-complete-adapter-traces requires --adapter-trace-jsonl")

    package, inputs, references = load_benchmark_package_v1(args.package_json)
    predictions = load_unique_models_v1(args.prediction_jsonl,
                                        BenchmarkPredictionV1)
    failures = load_unique_models_v1(
        args.failure_jsonl, BenchmarkFailureV1) if args.failure_jsonl else {}
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
    selection = build_benchmark_slice_review_selection_v1(
        joined,
        slice_labels=args.slice_label,
        horizon_s=args.horizon_s,
        per_slice=args.per_slice,
        seed=args.seed,
    )
    selected_rows = select_benchmark_slice_review_rows_v1(joined,
                                                          selection=selection)
    output = Path(args.output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    media_report, media_href_resolver = _prepare_review_media_v1(
        rows=selected_rows,
        inputs=inputs,
        output_root=output,
        copy_media=args.copy_media,
        require_readable=args.require_readable_media,
    )
    diagnostics = {
        "full_denominator":
        evaluate_benchmark_coordinate_diagnostic_v1(joined,
                                                    horizon_s=args.horizon_s),
        "selected_review_subset":
        evaluate_benchmark_coordinate_diagnostic_v1(
            selected_rows,
            horizon_s=args.horizon_s,
        ),
        "requested_slices": {
            label:
            evaluate_benchmark_coordinate_diagnostic_v1(
                [
                    row
                    for row in joined if label in row.reference.scenario_labels
                ],
                horizon_s=args.horizon_s,
            )
            for label in args.slice_label
        },
    }
    reference_motion = {
        "full_denominator":
        evaluate_benchmark_reference_motion_diagnostic_v1(
            joined, horizon_s=args.horizon_s),
        "selected_review_subset":
        evaluate_benchmark_reference_motion_diagnostic_v1(
            selected_rows,
            horizon_s=args.horizon_s,
        ),
        "requested_slices": {
            label:
            evaluate_benchmark_reference_motion_diagnostic_v1(
                [
                    row
                    for row in joined if label in row.reference.scenario_labels
                ],
                horizon_s=args.horizon_s,
            )
            for label in args.slice_label
        },
    }
    _write_json_v1(output / "selection.json", selection)
    _write_json_v1(output / "coordinate_diagnostic.json", diagnostics)
    _write_json_v1(output / "reference_motion_diagnostic.json",
                   reference_motion)
    if adapter_trace_funnel is not None:
        _write_json_v1(output / "adapter_trace_funnel.json",
                       adapter_trace_funnel)
    diagnosis_rows = _populate_pre_review_evidence_v1(
        build_benchmark_slice_review_diagnosis_template_v1(selection),
        selection=selection,
        inputs=inputs,
    )
    written_template_count = write_jsonl_atomic(
        output / "diagnosis_template.jsonl", diagnosis_rows)
    write_benchmark_plugin_review_html_v1(
        path=output / "review.html",
        rows=selected_rows,
        inputs=inputs,
        horizon_s=args.horizon_s,
        title=args.title,
        media_href_resolver=media_href_resolver,
        review_annotations={
            entry["sample_id"]: _selection_annotation_v1(entry)
            for entry in selection["entries"]
        },
        embed_media=True,
        adapter_traces=adapter_traces,
        adapter_trace_funnel=adapter_trace_funnel,
        coverage_horizons_s=package.benchmark_profile.time_policy.horizons_s,
    )
    summary = {
        "schema_version":
        "urbannav_benchmark_slice_review_bundle_v1",
        "evaluator_only":
        True,
        "not_model_visible":
        True,
        "package_id":
        package.package_id,
        "package_input_digest":
        package.input_digest,
        "package_reference_digest":
        package.reference_digest,
        "prediction_path":
        str(Path(args.prediction_jsonl).expanduser().resolve()),
        "failure_path":
        None if args.failure_jsonl is None else str(
            Path(args.failure_jsonl).expanduser().resolve()),
        "selection_path":
        "selection.json",
        "coordinate_diagnostic_path":
        "coordinate_diagnostic.json",
        "reference_motion_diagnostic_path":
        "reference_motion_diagnostic.json",
        "adapter_trace_path":
        adapter_trace_path,
        "adapter_trace_funnel_path":
        None if adapter_trace_funnel is None else "adapter_trace_funnel.json",
        "diagnosis_template_path":
        "diagnosis_template.jsonl",
        "review_html_path":
        "review.html",
        "selected_unique_count":
        len(selected_rows),
        "diagnosis_template_count":
        written_template_count,
        "media":
        media_report,
    }
    _write_json_v1(output / "summary.json", summary)
    print(
        json.dumps({
            "output_root": str(output),
            **summary
        },
                   ensure_ascii=False,
                   indent=2))
    return 0


def _prepare_review_media_v1(
    *,
    rows: list[Any],
    inputs: dict[str, Any],
    output_root: Path,
    copy_media: bool,
    require_readable: bool,
) -> tuple[dict[str, Any], Any]:
    """Validate selected media and optionally materialize deterministic local copies."""

    paths: list[Path] = []
    for row in rows:
        input_row = inputs[row.reference.sample_id]
        paths.extend(Path(value).expanduser() for value in input_row.images)
    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    missing = [str(path) for path in unique_paths if not path.is_file()]
    if require_readable and missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"selected review media is unavailable ({len(missing)}): {preview}"
        )
    hrefs: dict[str, str] = {}
    copied_count = 0
    if copy_media:
        media_root = output_root / "media"
        media_root.mkdir(parents=True, exist_ok=True)
        for source in unique_paths:
            if not source.is_file():
                continue
            destination = media_root / _review_media_name_v1(source)
            if not destination.exists():
                shutil.copy2(source, destination)
            hrefs[str(source)] = destination.relative_to(
                output_root).as_posix()
            copied_count += 1

    def resolve(image: str) -> str:
        """Resolve an input image to copied relative media or legacy source URI."""

        source = str(Path(image).expanduser().resolve())
        return hrefs.get(source, f"file://{source}")

    return (
        {
            "referenced_unique_count": len(unique_paths),
            "missing_count": len(missing),
            "missing_paths": missing,
            "copied_count": copied_count,
            "copy_media": copy_media,
        },
        resolve,
    )


def _review_media_name_v1(path: Path) -> str:
    """Create a collision-resistant, stable review asset filename."""

    suffix = path.suffix.lower() or ".bin"
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    return f"{digest}_{path.stem}{suffix}"


def _selection_annotation_v1(entry: dict[str, Any]) -> str:
    """Render one concise deterministic slice/stratum rationale for HTML review."""

    return "; ".join(
        f"{item['slice_label']} / {item['stratum']} / L2={float(item['l2_m']):.3f}m"
        for item in entry["selection_reasons"])


def _populate_pre_review_evidence_v1(
    diagnosis_rows: list[dict[str, Any]],
    *,
    selection: dict[str, Any],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add deterministic evidence while preserving all subjective fields for humans.

    Args:
        diagnosis_rows: Pending human rows emitted by the typed selection helper.
        selection: Fixed evaluator-only selection ledger.
        inputs: Exact model-visible cases keyed by stable sample id.

    Returns:
        The same pending-human rows enriched with media and coordinate evidence.

    Raises:
        ValueError: If the selection/template stable identities drift.
    """

    entries = {entry["sample_id"]: entry for entry in selection["entries"]}
    for row in diagnosis_rows:
        sample_id = row["sample_id"]
        entry = entries.get(sample_id)
        input_row = inputs.get(sample_id)
        if entry is None or input_row is None:
            raise ValueError(
                f"slice-review template identity drift: {sample_id}")
        media_paths = [
            Path(value).expanduser().resolve() for value in input_row.images
        ]
        row["input_media_readable"] = all(path.is_file()
                                          for path in media_paths)
        row["pre_review_evidence"] = {
            "status":
            "deterministic_evidence_only",
            "reviewer_action_required": [
                "verify point-goal semantics against visible scene",
                "verify visible context and GT/pred overlay plausibility",
                "assign final diagnosis without using this pre-review as a substitute",
            ],
            "model_visible_image_count":
            len(media_paths),
            "reference_xy_at_horizon_m":
            entry["reference_xy_at_horizon_m"],
            "prediction_xy_at_horizon_m":
            entry["prediction_xy_at_horizon_m"],
            "reference_displacement_m":
            entry["reference_displacement_m"],
            "prediction_displacement_m":
            entry["prediction_displacement_m"],
            "coordinate_stamps": {
                "reference_coordinate_frame":
                entry["reference_coordinate_frame"],
                "prediction_geometry_id": entry["prediction_geometry_id"],
                "prediction_time_basis": entry["prediction_time_basis"],
                "coordinate_projection_id": entry["coordinate_projection_id"],
            },
        }
    return diagnosis_rows


def _write_json_v1(path: Path, payload: dict[str, Any]) -> None:
    """Write one human-readable deterministic JSON artifact."""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) +
        "\n",
        encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
