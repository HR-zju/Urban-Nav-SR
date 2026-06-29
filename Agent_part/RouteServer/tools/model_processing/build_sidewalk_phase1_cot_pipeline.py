#!/usr/bin/env python3
"""Build calibrated SideWalk Phase1 source, CoT, archive, and multi-frame data.

This script is a thin orchestration layer around the existing Phase1 tools.  It
starts from prepared SideWalk clip directories and can stop at the source JSONL
stage for external CoT annotation, or continue through teacher generation,
ms-swift archive materialization, and multi-frame archive expansion.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROUTE_SERVER_ROOT = SCRIPT_DIR.parents[1]
DATASET_PROCESSING_DIR = ROUTE_SERVER_ROOT / "tools" / "dataset_processing"
for path in (SCRIPT_DIR, DATASET_PROCESSING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_phase1_source_review_html import compact_row, pick_balanced, read_jsonl, render_html  # noqa: E402
from build_sidewalk_phase1_multiframe_archive import process_manifest as build_multiframe_archive  # noqa: E402
from sidewalk_phase1_archive import ArchiveConfig, archive_cot_dataset  # noqa: E402
from sidewalk_phase1_cot_teacher import TeacherConfig, generate_cot_dataset  # noqa: E402
from sidewalk_phase1_dataset import Phase1Config, build_phase1_dataset  # noqa: E402


DEFAULT_CAMERA_PARAMS = ROUTE_SERVER_ROOT / "config" / "camera" / "camera_x001_camera_params.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidewalk-root", required=True, help="Prepared SideWalk clip root or chunk directory.")
    parser.add_argument("--output-root", required=True, help="Pipeline output root.")
    parser.add_argument("--split-name", default="train", help="Manifest split name.")
    parser.add_argument("--camera-params-path", default=str(DEFAULT_CAMERA_PARAMS))
    parser.add_argument("--odom-yaw-offset-deg", type=float, default=90.0)

    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--future-frames", type=int, default=50)
    parser.add_argument("--future-horizon-sec", type=float, default=2.5)
    parser.add_argument("--num-points", type=int, default=16)
    parser.add_argument("--min-future-distance-m", type=float, default=0.3)
    parser.add_argument("--frame-rate-hz", type=float, default=20.0)
    parser.add_argument("--max-bev-time-delta-ms", type=float, default=300.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--allow-missing-bev", action="store_true")
    parser.add_argument("--allow-heuristic-trajectory", action="store_true")
    parser.add_argument("--write-overlays", action="store_true")

    parser.add_argument("--write-source-review", action="store_true", default=True)
    parser.add_argument("--no-source-review", dest="write_source_review", action="store_false")
    parser.add_argument("--source-review-limit", type=int, default=600)
    parser.add_argument("--source-review-per-intent", type=int, default=150)

    parser.add_argument("--run-cot", action="store_true", help="Run the CoT teacher after source generation.")
    parser.add_argument("--cot-jsonl", default=None, help="Existing or desired CoT JSONL path.")
    parser.add_argument("--cot-report-json", default=None)
    parser.add_argument("--cot-fallback-policy", choices=("drop", "error", "template"), default="drop")
    parser.add_argument("--cot-max-retries", type=int, default=2)
    parser.add_argument("--cot-image-detail", choices=("auto", "low", "high"), default="auto")
    parser.add_argument("--cot-progress-interval", type=int, default=25)

    parser.add_argument("--run-archive", action="store_true", help="Archive a CoT JSONL into ms-swift format.")
    parser.add_argument("--archive-root", default=None)
    parser.add_argument("--archive-media-mode", choices=("reference", "hardlink", "copy", "symlink"), default="hardlink")
    parser.add_argument("--archive-path-mode", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--archive-write-review-html", action="store_true", default=True)
    parser.add_argument("--archive-review-limit", type=int, default=200)
    parser.add_argument("--archive-validate-images", action="store_true")
    parser.add_argument("--archive-manifest", default=None, help="Existing archive JSONL for --run-multiframe without --run-archive.")

    parser.add_argument("--run-multiframe", action="store_true", help="Build a 5-frame multi-frame archive JSONL.")
    parser.add_argument("--multiframe-sidewalk-root", default=None)
    parser.add_argument("--multiframe-output-manifest", default=None)
    parser.add_argument("--multiframe-output-media-root", default=None)
    parser.add_argument("--multiframe-num-workers", type=int, default=4)
    parser.add_argument("--multiframe-sampling-mode", choices=("frame_offsets", "time"), default="time")
    parser.add_argument("--multiframe-offsets", default="-20,-10,0")
    parser.add_argument("--multiframe-num-fpv-frames", type=int, default=5)
    parser.add_argument("--multiframe-target-fps", type=float, default=5.0)
    parser.add_argument("--multiframe-layout", choices=("fpv_bev_pairs", "fpv_history_current_bev"), default="fpv_bev_pairs")
    parser.add_argument("--multiframe-copy-mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
    parser.add_argument("--multiframe-path-mode", choices=("absolute", "relative"), default="relative")
    parser.add_argument("--multiframe-max-rows", type=int, default=None)
    parser.add_argument("--multiframe-max-output-rows", type=int, default=None)
    parser.add_argument("--multiframe-max-clips", type=int, default=None)
    parser.add_argument("--multiframe-max-rows-per-clip", type=int, default=None)
    parser.add_argument("--multiframe-require-no-padding", action="store_true")
    parser.add_argument("--multiframe-max-fpv-time-delta-ms", type=float, default=150.0)
    parser.add_argument(
        "--multiframe-max-bev-time-delta-ms",
        type=float,
        default=1000.0,
        help="Nearest BEV tolerance for multi-frame history matching; lower-frequency BEV often needs a wider window.",
    )
    parser.add_argument("--multiframe-overwrite", action="store_true")
    return parser.parse_args()


def write_source_review(source_jsonl: Path, output_html: Path, *, max_samples: int, per_intent: int) -> dict[str, Any]:
    """Write the static source review HTML.

    Args:
        source_jsonl: Phase1 source JSONL.
        output_html: Destination HTML path.
        max_samples: Maximum number of review samples.
        per_intent: Maximum samples per intent before balancing.

    Returns:
        Review summary.
    """

    rows = read_jsonl(source_jsonl)
    selected = pick_balanced(rows, max_samples=int(max_samples), per_intent=int(per_intent))
    samples = [compact_row(row, index=i) for i, row in enumerate(selected)]
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(render_html(samples, source_jsonl), encoding="utf-8")
    return {
        "source_jsonl": str(source_jsonl),
        "output_html": str(output_html),
        "input_rows": len(rows),
        "review_samples": len(samples),
        "intent_counts": {
            intent: sum(1 for row in samples if row["intent"] == intent)
            for intent in sorted({row["intent"] for row in samples})
        },
    }


def build_source(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    """Build calibrated Phase1 source JSONL and sidecars.

    Args:
        args: Parsed pipeline arguments.
        output_root: Pipeline output root.

    Returns:
        Source generation report.
    """

    config = Phase1Config(
        frame_step=max(int(args.frame_step), 1),
        future_frames=max(int(args.future_frames), 1),
        future_horizon_sec=(float(args.future_horizon_sec) if float(args.future_horizon_sec) > 0.0 else None),
        num_points=max(int(args.num_points), 2),
        min_future_distance_m=float(args.min_future_distance_m),
        frame_rate_hz=float(args.frame_rate_hz),
        max_bev_time_delta_ms=float(args.max_bev_time_delta_ms),
        odom_yaw_offset_deg=float(args.odom_yaw_offset_deg),
    )
    return build_phase1_dataset(
        args.sidewalk_root,
        str(output_root / "phase1"),
        split_name=args.split_name,
        config=config,
        limit=args.limit,
        max_clips=args.max_clips,
        allow_missing_bev=bool(args.allow_missing_bev),
        allow_heuristic_trajectory=bool(args.allow_heuristic_trajectory),
        camera_params_path=args.camera_params_path,
        write_overlays=bool(args.write_overlays),
    )


async def maybe_build_cot(args: argparse.Namespace, source_jsonl: Path, output_root: Path) -> dict[str, Any] | None:
    """Optionally build CoT teacher JSONL.

    Args:
        args: Parsed pipeline arguments.
        source_jsonl: Phase1 source JSONL path.
        output_root: Pipeline output root.

    Returns:
        CoT report, or None when disabled.
    """

    if not args.run_cot:
        return None
    cot_jsonl = Path(args.cot_jsonl) if args.cot_jsonl else output_root / "phase1" / "manifests" / f"phase1_cot_{args.split_name}.jsonl"
    cot_report = Path(args.cot_report_json) if args.cot_report_json else output_root / "phase1" / "manifests" / f"phase1_cot_report_{args.split_name}.json"
    config = TeacherConfig(
        expected_points=max(int(args.num_points), 2),
        max_retries=max(int(args.cot_max_retries), 0),
        fallback_policy=args.cot_fallback_policy,
        image_detail=args.cot_image_detail,
    )
    return await generate_cot_dataset(
        str(source_jsonl),
        str(cot_jsonl),
        report_json=str(cot_report),
        limit=args.limit,
        config=config,
        progress_interval=int(args.cot_progress_interval),
    )


def maybe_build_archive(args: argparse.Namespace, output_root: Path) -> dict[str, Any] | None:
    """Optionally build the ms-swift archive.

    Args:
        args: Parsed pipeline arguments.
        output_root: Pipeline output root.

    Returns:
        Archive stats, or None when disabled.
    """

    if not args.run_archive:
        return None
    cot_jsonl = Path(args.cot_jsonl) if args.cot_jsonl else output_root / "phase1" / "manifests" / f"phase1_cot_{args.split_name}.jsonl"
    archive_root = Path(args.archive_root) if args.archive_root else output_root / "phase1_archive"
    config = ArchiveConfig(
        split=args.split_name,
        media_mode=args.archive_media_mode,
        path_mode=args.archive_path_mode,
        expected_points=max(int(args.num_points), 2),
        write_review_html=bool(args.archive_write_review_html),
        review_limit=int(args.archive_review_limit),
        validate_images=bool(args.archive_validate_images),
    )
    return archive_cot_dataset(str(cot_jsonl), str(archive_root), config=config)


def maybe_build_multiframe(args: argparse.Namespace, output_root: Path) -> dict[str, Any] | None:
    """Optionally build the multi-frame archive manifest.

    Args:
        args: Parsed pipeline arguments.
        output_root: Pipeline output root.

    Returns:
        Multi-frame build stats, or None when disabled.
    """

    if not args.run_multiframe:
        return None
    archive_manifest = (
        Path(args.archive_manifest)
        if args.archive_manifest
        else Path(args.archive_root or output_root / "phase1_archive") / "manifests" / f"{args.split_name}.jsonl"
    )
    media_root = Path(args.multiframe_output_media_root) if args.multiframe_output_media_root else None
    output_manifest = Path(args.multiframe_output_manifest) if args.multiframe_output_manifest else None
    namespace = argparse.Namespace(
        manifest=archive_manifest,
        sidewalk_root=Path(args.multiframe_sidewalk_root or args.sidewalk_root),
        output_manifest=output_manifest,
        output_media_root=media_root,
        offsets=args.multiframe_offsets,
        sampling_mode=args.multiframe_sampling_mode,
        num_fpv_frames=int(args.multiframe_num_fpv_frames),
        target_fps=float(args.multiframe_target_fps),
        layout=args.multiframe_layout,
        max_rows=args.multiframe_max_rows,
        max_output_rows=args.multiframe_max_output_rows,
        max_clips=args.multiframe_max_clips,
        max_rows_per_clip=args.multiframe_max_rows_per_clip,
        start_row=0,
        max_bev_time_delta_ms=float(args.multiframe_max_bev_time_delta_ms),
        max_fpv_time_delta_ms=float(args.multiframe_max_fpv_time_delta_ms),
        missing_policy="pad_current",
        copy_mode=args.multiframe_copy_mode,
        absolute_paths=args.multiframe_path_mode == "absolute",
        overwrite=bool(args.multiframe_overwrite),
        no_prompt_update=False,
        require_no_padding=bool(args.multiframe_require_no_padding),
        dry_run=False,
        num_workers=max(int(args.multiframe_num_workers), 1),
    )
    return build_multiframe_archive(namespace)


async def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Run the configured pipeline.

    Args:
        args: Parsed pipeline arguments.

    Returns:
        Summary dictionary written to the pipeline report.
    """

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "schema_version": "sidewalk_phase1_cot_pipeline_v1",
        "sidewalk_root": str(Path(args.sidewalk_root).expanduser()),
        "output_root": str(output_root),
        "split_name": args.split_name,
        "odom_yaw_offset_deg": float(args.odom_yaw_offset_deg),
        "camera_params_path": str(Path(args.camera_params_path).expanduser()),
    }
    source_report = build_source(args, output_root)
    summary["source"] = source_report
    source_jsonl = Path(str(source_report["source_jsonl"]))

    if args.write_source_review:
        summary["source_review"] = write_source_review(
            source_jsonl,
            output_root / "phase1" / "review" / "source_review.html",
            max_samples=int(args.source_review_limit),
            per_intent=int(args.source_review_per_intent),
        )

    cot_report = await maybe_build_cot(args, source_jsonl, output_root)
    if cot_report is not None:
        summary["cot"] = cot_report

    archive_report = maybe_build_archive(args, output_root)
    if archive_report is not None:
        summary["archive"] = archive_report

    multiframe_report = maybe_build_multiframe(args, output_root)
    if multiframe_report is not None:
        summary["multiframe"] = multiframe_report

    report_path = output_root / "pipeline_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["report_path"] = str(report_path)
    return summary


def main() -> None:
    """Run the command-line pipeline.

    Returns:
        None.
    """

    summary = asyncio.run(run_pipeline(parse_args()))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
