#!/usr/bin/env python3
"""Run the canonical four-dataset Reasoning V3.4 minimal smoke pipeline.

The default path first uses ``prepare_reasoning_v3_smoke_inputs.py`` to select
three deterministic samples from each real dataset, then freezes V3.4 sources,
runs the two bounded LVLM passes, exports both safe view profiles, audits both
profiles, and builds one self-contained review page.  The resulting ``passed``
flag means structural/provenance/availability checks passed; human semantic
review is always a separate pending step.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Keep direct ``python tools/...py`` execution equivalent to ``python -m``.
_ROUTE_SERVER_ROOT = Path(__file__).resolve().parents[2]
if str(_ROUTE_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROUTE_SERVER_ROOT))

from mcp_server.contracts.reasoning_dataset_v3 import (
    ReasoningAnnotationV3,
    ReasoningSourceV3,
)
from mcp_server.data.jsonl_io import iter_jsonl, write_jsonl_atomic
from mcp_server.data.reasoning_source import ReasoningSourceBuildConfig
from mcp_server.data.reasoning_v3_audit import audit_reasoning_v3_artifacts
from mcp_server.data.reasoning_v3_quality import (
    ReasoningV3QualityThresholds,
    evaluate_reasoning_v3_quality,
)
from mcp_server.data.reasoning_v3_source import build_reasoning_source_manifest_v3
from mcp_server.data.reasoning_v3_views import (
    ReasoningViewBuildConfigV3,
    export_reasoning_training_views_v3,
)
from mcp_server.teacher.reasoning_v3_batch import (
    ReasoningV3BatchConfig,
    label_reasoning_source_manifest_v3,
)
from tools.model_processing.build_reasoning_v3_review_html import (
    build_reasoning_v3_review_html,
)
from tools.model_processing.prepare_reasoning_v3_smoke_inputs import (
    SELECTION_RULE_VERSION,
    main as prepare_reasoning_v3_smoke_inputs,
)


_CANONICAL_FAMILIES = ("sidewalk", "foxglove", "frodobots", "egowalk")
_CLAIMS_BOUNDARY = (
    "structural/provenance/availability smoke only; no semantic accuracy, "
    "calibration, projection, tracking, mask, bbox, TTC, or geometry claim"
)
_DEFAULT_MS_SWIFT_ROOT = Path(__file__).resolve().parents[5] / "ms-swift"
_PROVIDER_DEFAULT_BASES = {
    "ark": "https://ark.cn-beijing.volces.com/api/v3",
    "openai": "https://api.openai.com/v1",
}
_PROVIDER_SECRET_ENV_NAMES = {
    "ARK_API_KEY",
    "ARK_API_KEYS",
    "LLM_API_KEY",
    "LLM_API_KEYS",
    "OPENAI_API_KEY",
    "OPENAI_API_KEYS",
}


def _dataset_arg(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError(
            "dataset must be sidewalk|foxglove|frodobots|egowalk=JSONL"
        )
    try:
        family = _dataset_family(name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return family, Path(path).expanduser()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the V3.4 smoke CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=_dataset_arg,
        required=True,
        help=(
            "One real source manifest per canonical family, formatted as "
            "FAMILY=JSONL. Repeat exactly four times."
        ),
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--ms-swift-root",
        type=Path,
        default=_DEFAULT_MS_SWIFT_ROOT,
        help="ms-swift checkout used by the direction adapter and final converter.",
    )
    parser.add_argument(
        "--prepared-inputs",
        action="store_true",
        help=(
            "Treat --dataset paths as already curated manifests. By default the "
            "canonical deterministic three-sample selector runs first."
        ),
    )
    parser.add_argument("--limit-per-dataset", type=int, default=3)
    parser.add_argument("--required-fpv-frames", type=int, default=5)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--skip-source-errors", action="store_true")
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Stop after deterministic selection and frozen-source validation.",
    )
    parser.add_argument("--provider", choices=("ark", "openai"))
    parser.add_argument("--api-base")
    parser.add_argument("--model")
    parser.add_argument(
        "--api-keys-file",
        type=Path,
        help="Optional newline/assignment key file; values stay in memory only.",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--image-detail", choices=("auto", "low", "high"), default="low")
    parser.add_argument("--min-valid-annotation-fraction", type=float, default=1.0)
    parser.add_argument("--max-failed-annotation-fraction", type=float, default=0.0)
    parser.add_argument("--review-per-stratum", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run deterministic selection, annotation, dual export, audit, and review."""

    args = parse_args(argv)
    if args.limit_per_dataset < 1:
        raise SystemExit("--limit-per-dataset must be positive")
    if args.required_fpv_frames < 1:
        raise SystemExit("--required-fpv-frames must be positive")
    if not args.prepared_inputs and args.limit_per_dataset != 3:
        raise SystemExit(
            f"the canonical {SELECTION_RULE_VERSION} selector emits exactly three "
            "samples per dataset; use --limit-per-dataset 3"
        )
    dataset_paths = _canonical_dataset_paths(args.dataset)
    ms_swift_root = args.ms_swift_root.expanduser().resolve()
    if not ms_swift_root.is_dir():
        raise SystemExit(f"ms-swift checkout is missing: {ms_swift_root}")

    run_root = args.run_root.expanduser().resolve()
    for name in (
        "annotations",
        "ms_swift",
        "reports",
        "review",
        "sources",
        "views",
    ):
        (run_root / name).mkdir(parents=True, exist_ok=True)

    selected_inputs = _select_or_reuse_inputs(
        dataset_paths,
        run_root=run_root,
        ms_swift_root=ms_swift_root,
        prepared_inputs=args.prepared_inputs,
    )
    source_paths: dict[str, Path] = {}
    source_reports: dict[str, dict[str, Any]] = {}
    for family in _CANONICAL_FAMILIES:
        output = run_root / "sources" / f"{family}.jsonl"
        report = build_reasoning_source_manifest_v3(
            selected_inputs[family],
            output,
            config=ReasoningSourceBuildConfig(
                dataset_source=family,
                max_fpv_frames=args.required_fpv_frames,
                media_root=args.media_root,
            ),
            limit=args.limit_per_dataset,
            skip_errors=args.skip_source_errors,
        )
        _write_json_report(run_root / "reports" / f"source_{family}.json", report)
        _assert_fpv_count(output, expected=args.required_fpv_frames)
        source_paths[family] = output
        source_reports[family] = report

    merged_source = run_root / "sources" / "reasoning_v3_4_smoke_all.jsonl"
    write_jsonl_atomic(
        merged_source,
        (
            row
            for family in _CANONICAL_FAMILIES
            for row in iter_jsonl(source_paths[family])
        ),
    )
    expected_rows = args.limit_per_dataset * len(_CANONICAL_FAMILIES)
    source_summary = _source_summary(
        merged_source,
        expected_per_family=args.limit_per_dataset,
    )
    source_summary.update(
        {
            "schema_version": "urbannav_reasoning_v3_4_source_smoke_report_v1",
            "selection_mode": (
                "caller_prepared_inputs"
                if args.prepared_inputs
                else SELECTION_RULE_VERSION
            ),
            "original_datasets": {
                family: str(dataset_paths[family])
                for family in _CANONICAL_FAMILIES
            },
            "selected_inputs": {
                family: str(selected_inputs[family])
                for family in _CANONICAL_FAMILIES
            },
            "source_reports": source_reports,
        }
    )
    _write_json_report(run_root / "reports/source_smoke_summary.json", source_summary)
    print(json.dumps(source_summary, ensure_ascii=False, indent=2), flush=True)

    if args.source_only:
        final = {
            "schema_version": "urbannav_reasoning_v3_4_smoke_source_only_report_v1",
            "run_root": str(run_root),
            "source_summary": source_summary,
            "expected_rows": expected_rows,
            "stage": "source_only",
            "claims_boundary": _CLAIMS_BOUNDARY,
            "human_review_status": "not_started",
            "passed": source_summary["rows"] == expected_rows,
        }
        _write_json_report(run_root / "reports/final.json", final)
        print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
        if not final["passed"]:
            raise SystemExit(2)
        return

    provider, model, runtime_keys, secret_values = _configure_teacher(args)
    annotations = run_root / "annotations/reasoning_v3_4_smoke.jsonl"
    annotation_report = asyncio.run(
        label_reasoning_source_manifest_v3(
            merged_source,
            annotations,
            rejected_jsonl=run_root
            / "annotations/reasoning_v3_4_smoke.rejected.jsonl",
            report_json=run_root / "reports/annotation.json",
            config=ReasoningV3BatchConfig(
                max_retries=max(0, args.max_retries),
                image_detail=args.image_detail,
                concurrency=max(1, args.concurrency),
                retry_failed_on_resume=True,
                progress_interval=1,
            ),
            api_keys=runtime_keys,
        )
    )
    latest_annotations = (
        run_root / "annotations/reasoning_v3_4_smoke.latest.jsonl"
    )
    _write_latest_annotations(
        merged_source,
        annotations,
        latest_annotations,
    )

    required_families = set(_CANONICAL_FAMILIES)
    quality_report = evaluate_reasoning_v3_quality(
        merged_source,
        latest_annotations,
        thresholds=ReasoningV3QualityThresholds(
            min_acceptance_fraction=args.min_valid_annotation_fraction,
            max_reject_fraction=args.max_failed_annotation_fraction,
        ),
        required_dataset_families=required_families,
    )
    _write_json_report(run_root / "reports/quality.json", quality_report)

    view_paths: dict[str, Path] = {}
    view_reports: dict[str, dict[str, Any]] = {}
    audit_reports: dict[str, dict[str, Any]] = {}
    for profile in ("audit_only", "strong_action_only"):
        view_path = run_root / "views" / f"reasoning_v3_4_{profile}.jsonl"
        view_report = export_reasoning_training_views_v3(
            merged_source,
            latest_annotations,
            view_path,
            config=ReasoningViewBuildConfigV3(training_profile=profile),
            skip_errors=False,
        )
        audit_report = audit_reasoning_v3_artifacts(
            merged_source,
            latest_annotations,
            view_path,
            required_dataset_families=required_families,
        )
        _write_json_report(
            run_root / "reports" / f"views_{profile}.json",
            view_report,
        )
        _write_json_report(
            run_root / "reports" / f"acceptance_audit_{profile}.json",
            audit_report,
        )
        view_paths[profile] = view_path
        view_reports[profile] = view_report
        audit_reports[profile] = audit_report

    review_report = build_reasoning_v3_review_html(
        merged_source,
        latest_annotations,
        run_root
        / "review"
        / "reasoning_v3_4_smoke_review_all.complete.html",
        per_stratum=max(1, args.review_per_stratum),
        media_root=args.media_root,
    )
    _write_json_report(run_root / "reports/review_all.json", review_report)

    ms_swift_output = (
        run_root / "ms_swift/reasoning_v3_4_strong_action_only.jsonl"
    )
    ms_swift_report_path = run_root / "reports/ms_swift_manifest.json"
    _convert_ms_swift(
        ms_swift_root=ms_swift_root,
        input_jsonl=view_paths["strong_action_only"],
        output_jsonl=ms_swift_output,
        report_json=ms_swift_report_path,
    )
    ms_swift_report = _read_json_object(ms_swift_report_path)

    latest_rows = [
        ReasoningAnnotationV3.model_validate(row)
        for row in iter_jsonl(latest_annotations)
    ]
    status_counts = Counter(item.status for item in latest_rows)
    dataset_counts = _annotation_dataset_counts(merged_source, latest_annotations)
    audit_view_rows = _count_jsonl_rows(view_paths["audit_only"])
    strong_view_rows = _count_jsonl_rows(view_paths["strong_action_only"])
    ms_swift_rows = _count_jsonl_rows(ms_swift_output)
    valid_rows = status_counts["accepted"] + status_counts["observation_only"]
    checks = {
        "source_rows_complete": source_summary["rows"] == expected_rows,
        "all_latest_annotations_valid": valid_rows == expected_rows,
        "quality_passed": bool(quality_report["passed"]),
        "audit_only_passed": bool(audit_reports["audit_only"]["passed"]),
        "strong_action_only_passed": bool(
            audit_reports["strong_action_only"]["passed"]
        ),
        "review_contains_every_sample": (
            review_report["selected_rows"] == expected_rows
        ),
        "review_has_no_missing_images": review_report["missing_images"] == 0,
        "ms_swift_matches_strong_view": ms_swift_rows == strong_view_rows,
    }
    _assert_no_key_material(run_root, secret_values)
    checks["api_key_material_not_serialized"] = True
    structural_provenance_passed = all(checks.values())
    final = {
        "schema_version": "urbannav_reasoning_v3_4_smoke_pipeline_report_v1",
        "run_root": str(run_root),
        "provider": provider,
        "model": model,
        "source_summary": source_summary,
        "annotation_report": annotation_report,
        "status_counts": dict(sorted(status_counts.items())),
        "dataset_counts": dataset_counts,
        "provider_telemetry": _provider_telemetry(latest_rows),
        "weak_label_availability": _weak_label_availability(latest_rows),
        "view_reports": view_reports,
        "view_rows": {
            "audit_only": audit_view_rows,
            "strong_action_only": strong_view_rows,
        },
        "quality_report": quality_report,
        "audit_reports": audit_reports,
        "review_report": review_report,
        "ms_swift_report": ms_swift_report,
        "ms_swift_conversion_rows": ms_swift_rows,
        "checks": checks,
        "review_html": review_report["output_html"],
        "claims_boundary": _CLAIMS_BOUNDARY,
        "human_review_status": "pending",
        "map_provider_used": False,
        "api_key_material_serialized": False,
        "structural_provenance_passed": structural_provenance_passed,
        "passed": structural_provenance_passed,
    }
    _write_json_report(run_root / "reports/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    if not structural_provenance_passed:
        raise SystemExit(2)


def _canonical_dataset_paths(
    values: Iterable[tuple[str, Path]],
) -> dict[str, Path]:
    """Require exactly one manifest for each canonical dataset family."""

    result: dict[str, Path] = {}
    duplicates: list[str] = []
    for family, path in values:
        if family in result:
            duplicates.append(family)
        result[family] = path.expanduser().resolve()
    missing = sorted(set(_CANONICAL_FAMILIES) - set(result))
    extra = sorted(set(result) - set(_CANONICAL_FAMILIES))
    if missing or extra or duplicates:
        raise SystemExit(
            "Reasoning V3.4 smoke requires exactly one manifest per family; "
            f"missing={missing}, extra={extra}, duplicates={sorted(set(duplicates))}"
        )
    for family, path in result.items():
        if not path.is_file():
            raise SystemExit(f"{family} manifest is missing: {path}")
    return result


def _select_or_reuse_inputs(
    dataset_paths: Mapping[str, Path],
    *,
    run_root: Path,
    ms_swift_root: Path,
    prepared_inputs: bool,
) -> dict[str, Path]:
    """Run the canonical selector unless the caller supplied curated inputs."""

    if prepared_inputs:
        return dict(dataset_paths)
    argv: list[str] = []
    for family in _CANONICAL_FAMILIES:
        argv.extend(("--dataset", f"{family}={dataset_paths[family]}"))
    argv.extend(("--output-root", str(run_root), "--ms-swift-root", str(ms_swift_root)))
    prepare_reasoning_v3_smoke_inputs(argv)
    selected = {
        family: run_root / "curated_inputs" / f"{family}.jsonl"
        for family in _CANONICAL_FAMILIES
    }
    for family, path in selected.items():
        if not path.is_file():
            raise RuntimeError(f"canonical selector did not write {family}: {path}")
    return selected


def _assert_fpv_count(path: Path, *, expected: int) -> None:
    """Require an exact, non-padded FPV frame count for every frozen source."""

    invalid: list[tuple[str, int]] = []
    for row in iter_jsonl(path):
        source = ReasoningSourceV3.model_validate(row)
        roles = source.model_input.image_roles
        count = sum(str(role).startswith("fpv.") for role in roles)
        if count != expected:
            invalid.append((source.model_input.source_group_id, count))
            if len(invalid) >= 20:
                break
    if invalid:
        raise RuntimeError(
            f"V3.4 smoke requires exactly {expected} FPV frames: {invalid}"
        )


def _source_summary(
    path: Path,
    *,
    expected_per_family: int,
) -> dict[str, Any]:
    """Summarize source availability without making geometry claims."""

    counts: Counter[str] = Counter()
    for row in iter_jsonl(path):
        source = ReasoningSourceV3.model_validate(row)
        family = _dataset_family(source.model_input.dataset_source)
        counts["rows"] += 1
        counts[f"dataset:{family}"] += 1
        counts[f"dataset:{family}:direction_available"] += int(
            _direction_available(source)
        )
        counts[f"dataset:{family}:system1_available"] += int(
            _system1_available(source)
        )
        flags = source.teacher_supervision_context.get("p0_supervision_flags")
        counts[f"dataset:{family}:p0_flagged"] += int(bool(flags))
    for family in _CANONICAL_FAMILIES:
        actual = counts[f"dataset:{family}"]
        if actual != expected_per_family:
            raise RuntimeError(
                f"{family} smoke source count mismatch: "
                f"expected={expected_per_family}, actual={actual}"
            )
    return {
        "rows": counts["rows"],
        "expected_rows": expected_per_family * len(_CANONICAL_FAMILIES),
        "counts": dict(sorted(counts.items())),
        "report_semantics": (
            "availability_only:no_calibration_projection_fusion_tracking_or_geometry_claim"
        ),
    }


def _direction_available(source: ReasoningSourceV3) -> bool:
    fixed = source.teacher_supervision_context.get("fixed_labels")
    direction = fixed.get("canonical_direction") if isinstance(fixed, Mapping) else None
    provenance = source.label_provenance.get("decision.direction")
    return bool(
        isinstance(direction, Mapping)
        and provenance is not None
        and provenance.source_kind == "deterministic_adapter"
        and provenance.supervision == "strong"
        and provenance.student_inferable
        and provenance.loss_channel == "direction_choice"
    )


def _system1_available(source: ReasoningSourceV3) -> bool:
    provenance = source.label_provenance.get("system1.trajectory")
    return bool(
        source.model_input.system1_enabled
        and provenance is not None
        and provenance.source_kind == "dataset_gt"
        and provenance.supervision == "strong"
        and provenance.student_inferable
        and provenance.loss_channel == "system1_action"
    )


def _configure_teacher(
    args: argparse.Namespace,
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    """Configure a provider without putting key-file values into reports."""

    provider = (
        str(args.provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
    )
    if not provider:
        provider = "ark" if os.getenv("ARK_API_KEY", "").strip() else "openai"
    if provider not in _PROVIDER_DEFAULT_BASES:
        raise SystemExit(f"unsupported teacher provider: {provider}")
    model = str(
        args.model
        or os.getenv("LLM_MODEL")
        or os.getenv("VLM_MODEL")
        or os.getenv("ARK_MODEL")
        or ""
    ).strip()
    if not model:
        raise SystemExit(
            "teacher model is required via --model/LLM_MODEL; use --source-only "
            "to stop before paid annotation"
        )
    api_base = str(
        args.api_base
        or os.getenv("LLM_API_BASE_URL")
        or os.getenv("ARK_BASE_URL")
        or os.getenv("ARK_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or _PROVIDER_DEFAULT_BASES[provider]
    ).strip()
    file_keys = _read_api_keys(args.api_keys_file)
    environment_keys = tuple(
        value
        for value in (
            os.getenv("LLM_API_KEY", "").strip(),
            os.getenv("ARK_API_KEY", "").strip(),
            os.getenv("OPENAI_API_KEY", "").strip(),
        )
        if value
    )
    runtime_keys = file_keys or environment_keys[:1]
    if not runtime_keys:
        raise SystemExit(
            "teacher API key is required via --api-keys-file, LLM_API_KEY, "
            "ARK_API_KEY, or OPENAI_API_KEY"
        )
    os.environ.update(
        {
            "LLM_PROVIDER": provider,
            "LLM_MODEL": model,
            "LLM_API_BASE_URL": api_base,
            "LLM_API_STYLE": "chat_completions",
            "LLM_TEMPERATURE": "0",
            "LLM_TIMEOUT_S": "120",
            "LLM_VISION_TIMEOUT_S": "300",
            "LLM_MAX_RETRIES": "1",
            "LLM_CHAT_TEMPLATE_ENABLE_THINKING": "0",
        }
    )
    secret_values = tuple(dict.fromkeys((*file_keys, *environment_keys)))
    return provider, model, runtime_keys, secret_values


def _read_api_keys(path: Path | None) -> tuple[str, ...]:
    """Read raw or ``NAME=value`` keys without logging their contents."""

    if path is None:
        return ()
    value = path.expanduser().resolve()
    if not value.is_file():
        raise SystemExit(f"API key file is missing: {value}")
    keys: list[str] = []
    accepted_names = {"ARK_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"}
    for raw_line in value.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" in line:
            name, candidate = line.split("=", 1)
            if name.strip() not in accepted_names:
                continue
            line = candidate.strip()
        line = line.strip().strip("\"'")
        if line:
            keys.append(line)
    if not keys:
        raise SystemExit(f"API key file has no usable key entries: {value}")
    return tuple(dict.fromkeys(keys))


def _write_latest_annotations(
    source_jsonl: Path,
    annotation_jsonl: Path,
    output_jsonl: Path,
) -> None:
    """Compact append-only annotations to one latest row in source order."""

    latest: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(annotation_jsonl):
        annotation = ReasoningAnnotationV3.model_validate(row)
        latest[annotation.source_group_id] = annotation.model_dump(
            mode="json",
            exclude_none=True,
        )
    ordered: list[dict[str, Any]] = []
    for row in iter_jsonl(source_jsonl):
        source = ReasoningSourceV3.model_validate(row)
        source_id = source.model_input.source_group_id
        if source_id in latest:
            ordered.append(latest[source_id])
    write_jsonl_atomic(output_jsonl, ordered)


def _convert_ms_swift(
    *,
    ms_swift_root: Path,
    input_jsonl: Path,
    output_jsonl: Path,
    report_json: Path,
) -> None:
    """Convert the strong-sidecar view through ms-swift's versioned adapter."""

    env = os.environ.copy()
    for name in _PROVIDER_SECRET_ENV_NAMES:
        env.pop(name, None)
    env["PYTHONPATH"] = str(ms_swift_root) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "swift.sidewalk_nav.data.versioned_reasoning_manifest",
            "--input-jsonl",
            str(input_jsonl),
            "--output-jsonl",
            str(output_jsonl),
            "--report-json",
            str(report_json),
            "--expected-version",
            "v3",
            "--v3-training-profile",
            "strong_action_only",
        ],
        env=env,
        check=True,
    )


def _annotation_dataset_counts(
    source_jsonl: Path,
    annotation_jsonl: Path,
) -> dict[str, int]:
    """Count latest annotations by their frozen source family."""

    source_family: dict[str, str] = {}
    for row in iter_jsonl(source_jsonl):
        source = ReasoningSourceV3.model_validate(row)
        source_family[source.model_input.source_group_id] = _dataset_family(
            source.model_input.dataset_source
        )
    counts: Counter[str] = Counter()
    for row in iter_jsonl(annotation_jsonl):
        annotation = ReasoningAnnotationV3.model_validate(row)
        counts[source_family.get(annotation.source_group_id, "unknown")] += 1
    return dict(sorted(counts.items()))


def _provider_telemetry(
    annotations: Sequence[ReasoningAnnotationV3],
) -> dict[str, Any]:
    """Aggregate sanitized provider usage from latest annotations."""

    counts: Counter[str] = Counter()
    roundtrips: list[float] = []
    for annotation in annotations:
        calls = annotation.telemetry.get("calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            counts["successful_calls"] += 1
            roundtrip = call.get("roundtrip_ms")
            if isinstance(roundtrip, (int, float)):
                roundtrips.append(float(roundtrip))
            response = call.get("response")
            usage = response.get("usage") if isinstance(response, Mapping) else None
            if isinstance(usage, Mapping):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = usage.get(key)
                    if isinstance(value, (int, float)):
                        counts[key] += int(value)
    result: dict[str, Any] = dict(sorted(counts.items()))
    if roundtrips:
        ordered = sorted(roundtrips)
        result.update(
            {
                "roundtrip_ms_sum": round(sum(ordered), 3),
                "roundtrip_ms_median": round(statistics.median(ordered), 3),
                "roundtrip_ms_p95": round(_percentile(ordered, 0.95), 3),
            }
        )
    return result


def _weak_label_availability(
    annotations: Sequence[ReasoningAnnotationV3],
) -> dict[str, int]:
    """Count weak outputs without presenting them as semantic accuracy."""

    counts: Counter[str] = Counter()
    for annotation in annotations:
        if annotation.observation is not None:
            counts["observation_rows"] += 1
        rationale = annotation.decision_rationale
        if rationale is None:
            continue
        counts["decision_rationale_rows"] += 1
        counts[f"rationale_status:{rationale.rationale_status}"] += 1
        counts["interaction_items"] += len(
            rationale.interaction_and_rule_grounding.interactions
        )
        counts["traffic_rule_items"] += len(
            rationale.interaction_and_rule_grounding.traffic_rules
        )
    return dict(sorted(counts.items()))


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    """Return a linear-interpolated percentile for an already sorted sequence."""

    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _assert_no_key_material(run_root: Path, secrets: Iterable[str]) -> None:
    """Fail if any supplied API key leaked into a text artifact."""

    values = tuple(
        dict.fromkeys(str(value) for value in secrets if len(str(value)) >= 8)
    )
    if not values:
        return
    text_suffixes = {".html", ".json", ".jsonl", ".md", ".txt"}
    for path in run_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if any(secret in content for secret in values):
            raise RuntimeError(
                "API key material was found in a smoke artifact; refusing completion"
            )


def _count_jsonl_rows(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def _write_json_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _dataset_family(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    for family in _CANONICAL_FAMILIES:
        if normalized.startswith(family):
            return family
    raise ValueError(f"unsupported Reasoning V3.4 smoke dataset: {value}")


if __name__ == "__main__":
    main()
