#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def _normalize_client_args(raw_args: Sequence[str]) -> list[str]:
    args = list(raw_args)
    if args and args[0] == "--":
        args = args[1:]
    return args


def _drop_option(args: Sequence[str], option_name: str) -> list[str]:
    out: list[str] = []
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token == option_name:
            skip_next = True
            continue
        if token.startswith(f"{option_name}="):
            continue
        out.append(token)
    return out


def _upsert_option(args: Sequence[str], option_name: str, option_value: str) -> list[str]:
    updated = _drop_option(args, option_name)
    updated.extend([option_name, str(option_value)])
    return updated


def _find_first_existing_image(frames_dir: Path, frame_name: str, exts: Sequence[str]) -> Path:
    for ext in exts:
        candidate = frames_dir / f"{frame_name}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for frame '{frame_name}' under {frames_dir}")


def _collect_frame_names(
    *,
    frames_dir: Path,
    image_exts: Sequence[str],
    frame_names_csv: str | None,
    frame_start: int | None,
    frame_end: int | None,
    frame_step: int,
) -> list[str]:
    if frame_names_csv:
        return [item.strip() for item in str(frame_names_csv).split(",") if item.strip()]

    ext_set = {ext.lower() for ext in image_exts}
    names: list[str] = []
    for path in sorted(frames_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ext_set:
            continue
        stem = path.stem.strip()
        if not stem.isdigit():
            continue
        frame_id = int(stem)
        if frame_start is not None and frame_id < frame_start:
            continue
        if frame_end is not None and frame_id > frame_end:
            continue
        names.append(stem)

    if frame_step > 1:
        names = names[::frame_step]
    return names


def _load_ride_pose_reader_class(repo_root: Path):
    module_path = repo_root / "src" / "mcp_server" / "services" / "ride_pose_service.py"
    spec = importlib.util.spec_from_file_location("ride_pose_service", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load ride pose module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RidePoseReader


def _run_command(command: Sequence[str], *, log_path: Path | None = None) -> int:
    process = subprocess.run(command, text=True, capture_output=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        merged = (process.stdout or "") + "\n" + (process.stderr or "")
        log_path.write_text(merged.strip() + "\n", encoding="utf-8")
    if process.returncode != 0:
        tail_lines = ((process.stdout or "") + "\n" + (process.stderr or "")).strip().splitlines()[-20:]
        raise RuntimeError(
            "Command failed with exit code "
            f"{process.returncode}: {' '.join(command)}\n" + "\n".join(tail_lines)
        )
    return int(process.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run visual prompt generation for a full sequence by iterating frames, "
            "optionally using ride pose for each frame."
        )
    )
    parser.add_argument("--provider", default="sam3", help="Segmentation provider (default: sam3)")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]), help="RouteServer repo root")
    parser.add_argument("--client", default="yuquan", choices=["yuquan", "zero-shot"], help="Visual prompt client")
    parser.add_argument("--run-sam3", action="store_true", help="Run segmentation provider once before frame loop")
    parser.add_argument("--sam3-env-name", default="sam3", help="Conda env name for SAM3")
    parser.add_argument("--sam3-script", default="/home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/prepare_frames_and_run_batch_sam3.py", help="Path to SAM3 script")
    parser.add_argument("--rides-base-dir", required=True, help="Rides base directory")
    parser.add_argument("--frames-base-dir", required=True, help="Frames base directory")
    parser.add_argument("--sam3-output-base-dir", required=True, help="Segmentation outputs base directory")
    parser.add_argument("--sequence-name", required=True, help="Sequence name")
    parser.add_argument("--camera-view", default="camera_front", help="Camera view")
    parser.add_argument("--recordings-subdir", default="recordings", help="Forwarded to SAM3 pipeline")
    parser.add_argument("--stream-pattern", default="*_video.m3u8", help="Forwarded to SAM3 pipeline")
    parser.add_argument("--fps", type=float, default=1.0, help="Forwarded to SAM3 pipeline")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0], help="Forwarded to SAM3 pipeline")
    parser.add_argument("--min-score", type=float, default=0.3, help="Forwarded to SAM3 pipeline")
    parser.add_argument("--finetuned", action=argparse.BooleanOptionalAction, default=True, help="Forwarded to SAM3 pipeline")
    parser.add_argument("--checkpoint-path", default=None, help="Required when --run-sam3")
    parser.add_argument("--max-frames", type=int, default=None, help="Forwarded to SAM3 pipeline")
    parser.add_argument("--overwrite-frames", action="store_true", help="Forwarded to SAM3 pipeline")

    parser.add_argument("--frame-names", default=None, help="Comma-separated frame names, e.g. 000005,000006")
    parser.add_argument("--frame-start", type=int, default=None, help="Start frame id (inclusive)")
    parser.add_argument("--frame-end", type=int, default=None, help="End frame id (inclusive)")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame sampling step")
    parser.add_argument("--image-exts", nargs="+", default=[".jpg", ".png", ".jpeg"], help="Candidate image extensions")

    parser.add_argument("--pose-source", default="ride", choices=["ride", "image"], help="Pose source for per-frame run")
    parser.add_argument("--ride-dir", default=None, help="Ride directory for pose CSVs; defaults to rides-base-dir/sequence-name")

    parser.add_argument("--output-dir", required=True, help="Output directory for per-frame overlay/json/log and manifest")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue when a frame fails")
    parser.add_argument(
        "--alignment-mode",
        default="error",
        choices=["error", "warn"],
        help="Preflight behavior for segmentation/image alignment checks in wrapper",
    )

    parser.add_argument(
        "--client-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments forwarded to the chosen client (base template for all frames)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    wrapper_script = repo_root / "tools" / "run_sam3_visual_prompt_pipeline.py"
    if not wrapper_script.exists():
        raise RuntimeError(f"Wrapper script not found: {wrapper_script}")

    frames_dir = Path(args.frames_base_dir).expanduser() / args.sequence_name / args.camera_view
    if not frames_dir.exists():
        raise RuntimeError(f"Frames directory not found: {frames_dir}")

    frame_names = _collect_frame_names(
        frames_dir=frames_dir,
        image_exts=args.image_exts,
        frame_names_csv=args.frame_names,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_step=max(int(args.frame_step), 1),
    )
    if not frame_names:
        raise RuntimeError("No frames selected. Check frame filters or --frame-names.")

    if args.run_sam3 and not str(args.checkpoint_path or "").strip():
        raise RuntimeError("--checkpoint-path is required when --run-sam3 is enabled")

    output_dir = Path(args.output_dir).expanduser()
    overlays_dir = output_dir / "overlay"
    json_dir = output_dir / "json"
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_client_args = _normalize_client_args(args.client_args)

    pose_reader = None
    if args.pose_source == "ride":
        ride_dir = Path(args.ride_dir).expanduser() if args.ride_dir else (Path(args.rides_base_dir).expanduser() / args.sequence_name)
        RidePoseReader = _load_ride_pose_reader_class(repo_root)
        pose_reader = RidePoseReader.from_ride_directory(ride_dir, camera=args.camera_view.replace("camera_", ""))

    # Optionally run segmentation once for this sequence before looping frames.
    if args.run_sam3:
        seed_frame = frame_names[0]
        seed_image = _find_first_existing_image(frames_dir, seed_frame, args.image_exts)
        pre_cmd = [
            sys.executable,
            str(wrapper_script),
            "--provider",
            str(args.provider),
            "--repo-root",
            str(repo_root),
            "--client",
            str(args.client),
            "--run-sam3",
            "--sam3-env-name",
            str(args.sam3_env_name),
            "--sam3-script",
            str(args.sam3_script),
            "--rides-base-dir",
            str(args.rides_base_dir),
            "--frames-base-dir",
            str(args.frames_base_dir),
            "--sam3-output-base-dir",
            str(args.sam3_output_base_dir),
            "--sequence-name",
            str(args.sequence_name),
            "--camera-view",
            str(args.camera_view),
            "--recordings-subdir",
            str(args.recordings_subdir),
            "--stream-pattern",
            str(args.stream_pattern),
            "--fps",
            str(args.fps),
            "--gpus",
            *[str(gpu) for gpu in args.gpus],
            "--min-score",
            str(args.min_score),
            "--frame-name",
            str(seed_frame),
            "--alignment-mode",
            str(args.alignment_mode),
            "--print-only",
            "--client-args",
            "--image-path",
            str(seed_image),
        ]
        pre_cmd.extend(["--checkpoint-path", str(args.checkpoint_path)])
        if args.max_frames is not None:
            pre_cmd.extend(["--max-frames", str(args.max_frames)])
        if args.overwrite_frames:
            pre_cmd.append("--overwrite-frames")
        if not args.finetuned:
            pre_cmd.append("--no-finetuned")
        _run_command(pre_cmd, log_path=logs_dir / "_pre_run_sam3.log")

    manifest: dict[str, Any] = {
        "sequence_name": str(args.sequence_name),
        "camera_view": str(args.camera_view),
        "provider": str(args.provider),
        "client": str(args.client),
        "pose_source": str(args.pose_source),
        "frames_total": len(frame_names),
        "frames": [],
    }

    failures = 0
    for frame_name in frame_names:
        image_path = _find_first_existing_image(frames_dir, frame_name, args.image_exts)
        per_frame_client_args = list(base_client_args)
        per_frame_client_args = _upsert_option(per_frame_client_args, "--image-path", str(image_path))

        if pose_reader is not None:
            frame_id = int(frame_name)
            pose = pose_reader.get_frame_pose(frame_id).to_dict()
            origin = f"{pose['longitude']},{pose['latitude']}"
            per_frame_client_args = _upsert_option(per_frame_client_args, "--gps-source", "origin")
            per_frame_client_args = _upsert_option(per_frame_client_args, "--origin", origin)
            heading = pose.get("heading_deg")
            if heading is None:
                per_frame_client_args = _drop_option(per_frame_client_args, "--heading-deg")
            else:
                per_frame_client_args = _upsert_option(per_frame_client_args, "--heading-deg", str(float(heading)))

        overlay_path = overlays_dir / f"{frame_name}.svg"
        json_path = json_dir / f"{frame_name}.json"
        per_frame_client_args = _upsert_option(per_frame_client_args, "--save-overlay", str(overlay_path))
        per_frame_client_args = _upsert_option(per_frame_client_args, "--save-json", str(json_path))

        cmd = [
            sys.executable,
            str(wrapper_script),
            "--provider",
            str(args.provider),
            "--repo-root",
            str(repo_root),
            "--client",
            str(args.client),
            "--sam3-output-base-dir",
            str(args.sam3_output_base_dir),
            "--sequence-name",
            str(args.sequence_name),
            "--camera-view",
            str(args.camera_view),
            "--frame-name",
            str(frame_name),
            "--alignment-mode",
            str(args.alignment_mode),
            "--client-args",
            *per_frame_client_args,
        ]

        frame_item: dict[str, Any] = {
            "frame_name": str(frame_name),
            "image_path": str(image_path),
            "overlay_path": str(overlay_path),
            "json_path": str(json_path),
            "log_path": str(logs_dir / f"{frame_name}.log"),
            "status": "ok",
        }

        try:
            _run_command(cmd, log_path=logs_dir / f"{frame_name}.log")
        except Exception as exc:
            frame_item["status"] = "error"
            frame_item["error"] = str(exc)
            failures += 1
            if not args.continue_on_error:
                manifest["frames"].append(frame_item)
                break

        manifest["frames"].append(frame_item)

    manifest["frames_failed"] = failures
    manifest["frames_succeeded"] = max(len(manifest["frames"]) - failures, 0)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if failures > 0 and not args.continue_on_error:
        raise RuntimeError(f"Sequence pipeline failed early. See manifest: {manifest_path}")

    print(f"Sequence visual prompt pipeline done. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
