#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_SAM3_SCRIPT = "/home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/prepare_frames_and_run_batch_sam3.py"


def _normalize_client_args(raw_args: Sequence[str]) -> list[str]:
    args = list(raw_args)
    if args and args[0] == "--":
        args = args[1:]
    return args


def _find_option_value(args: Sequence[str], option_name: str) -> str | None:
    for index, token in enumerate(args):
        if token == option_name and index + 1 < len(args):
            return str(args[index + 1])
        if token.startswith(f"{option_name}="):
            return token.split("=", 1)[1]
    return None


def _has_option(args: Sequence[str], option_name: str) -> bool:
    for token in args:
        if token == option_name or token.startswith(f"{option_name}="):
            return True
    return False


def _client_script_path(repo_root: Path, client: str) -> Path:
    if client == "yuquan":
        return repo_root / "examples" / "yuquan_visual_prompt_agent_client.py"
    if client == "zero-shot":
        return repo_root / "examples" / "zero_shot_visual_prompt_agent_client.py"
    raise RuntimeError(f"Unsupported client: {client}")


def _run_sam3(args: argparse.Namespace) -> None:
    command = [
        "conda",
        "run",
        "-n",
        args.sam3_env_name,
        "sam3",
        args.sam3_script,
        "--rides_base_dir",
        args.rides_base_dir,
        "--frames_base_dir",
        args.frames_base_dir,
        "--inference_output_base_dir",
        args.sam3_output_base_dir,
        "--recordings_subdir",
        args.recordings_subdir,
        "--stream_pattern",
        args.stream_pattern,
        "--camera_view",
        args.camera_view,
        "--sequence_names",
        args.sequence_name,
        "--fps",
        str(args.fps),
        "--gpus",
        *[str(gpu) for gpu in args.gpus],
        "--min_score",
        str(args.min_score),
        "--finetuned",
        "true" if args.finetuned else "false",
        "--checkpoint_path",
        args.checkpoint_path,
    ]
    if args.max_frames is not None:
        command.extend(["--max_frames", str(args.max_frames)])
    if args.overwrite_frames:
        command.extend(["--overwrite_frames", "true"])
    print("Running SAM3:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def _segmentation_paths(
    *,
    sam3_output_base_dir: str,
    sequence_name: str,
    camera_view: str,
    frame_name: str,
) -> tuple[Path, Path, Path]:
    base_dir = Path(sam3_output_base_dir).expanduser() / sequence_name / camera_view
    segmentation_npy = base_dir / f"{frame_name}_segmentation.npy"
    class_mapping = base_dir / "class_mapping.json"
    visualization = base_dir / f"{frame_name}_visualization.jpg"
    return segmentation_npy, class_mapping, visualization


def _build_client_command(args: argparse.Namespace) -> list[str]:
    repo_root = Path(args.repo_root).expanduser()
    client_script = _client_script_path(repo_root, args.client)
    client_args = _normalize_client_args(args.client_args)

    image_path = _find_option_value(client_args, "--image-path")
    if not image_path and args.client == "yuquan":
        raise RuntimeError("yuquan client requires --image-path in --client-args")
    if args.frame_name:
        frame_name = str(args.frame_name)
    elif image_path:
        frame_name = Path(image_path).expanduser().stem
    else:
        raise RuntimeError("Unable to determine frame name. Set --frame-name or pass --image-path in --client-args.")

    segmentation_npy, class_mapping, visualization = _segmentation_paths(
        sam3_output_base_dir=args.sam3_output_base_dir,
        sequence_name=args.sequence_name,
        camera_view=args.camera_view,
        frame_name=frame_name,
    )
    for path in [segmentation_npy, class_mapping]:
        if not path.exists():
            raise RuntimeError(f"Expected SAM3 artifact not found: {path}")

    command = ["python", str(client_script), *client_args]
    if not _has_option(command, "--fpv-segmentation-npy"):
        command.extend(["--fpv-segmentation-npy", str(segmentation_npy)])
    if not _has_option(command, "--fpv-segmentation-class-mapping"):
        command.extend(["--fpv-segmentation-class-mapping", str(class_mapping)])
    if visualization.exists() and not _has_option(command, "--fpv-segmentation-visualization"):
        command.extend(["--fpv-segmentation-visualization", str(visualization)])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 segmentation and then launch the current visual prompt demo client "
            "with the generated fpv_segmentation artifacts injected automatically."
        )
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]), help="RouteServer/OpenStreetMCP repo root")
    parser.add_argument("--client", default="yuquan", choices=["yuquan", "zero-shot"], help="Visual prompt client to run after SAM3")
    parser.add_argument("--run-sam3", action="store_true", help="Run the external SAM3 pipeline before invoking the client")
    parser.add_argument("--sam3-env-name", default="sam3", help="Conda env name for SAM3")
    parser.add_argument("--sam3-script", default=DEFAULT_SAM3_SCRIPT, help="Path to prepare_frames_and_run_batch_sam3.py")
    parser.add_argument("--rides-base-dir", help="rides_base_dir for SAM3 preprocessing")
    parser.add_argument("--frames-base-dir", help="frames_base_dir for SAM3 preprocessing")
    parser.add_argument("--sam3-output-base-dir", required=True, help="Base directory containing SAM3 frame outputs")
    parser.add_argument("--sequence-name", required=True, help="Sequence name used by SAM3, e.g. ride_xxx")
    parser.add_argument("--camera-view", default="camera_front", help="Camera view used by SAM3")
    parser.add_argument("--recordings-subdir", default="recordings", help="recordings subdir forwarded to SAM3")
    parser.add_argument("--stream-pattern", default="*_video.m3u8", help="stream pattern forwarded to SAM3")
    parser.add_argument("--fps", type=float, default=1.0, help="Extraction fps forwarded to SAM3")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPU ids forwarded to SAM3")
    parser.add_argument("--min-score", type=float, default=0.3, help="SAM3 min score threshold")
    parser.add_argument("--finetuned", action=argparse.BooleanOptionalAction, default=True, help="Whether to use the finetuned SAM3 checkpoint")
    parser.add_argument("--checkpoint-path", default=None, help="SAM3 checkpoint path; required with --run-sam3")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max_frames forwarded to SAM3")
    parser.add_argument("--overwrite-frames", action="store_true", help="Overwrite extracted frames when re-running SAM3 preprocessing")
    parser.add_argument("--frame-name", default=None, help="Frame stem used to find segmentation artifacts, e.g. 000005")
    parser.add_argument("--print-only", action="store_true", help="Print the final client command without running it")
    parser.add_argument(
        "--client-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Arguments forwarded to the chosen client. Use '--client-args -- ...' so everything "
            "after the separator is passed through unchanged."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_sam3:
        missing = [
            name
            for name, value in [
                ("--rides-base-dir", args.rides_base_dir),
                ("--frames-base-dir", args.frames_base_dir),
                ("--checkpoint-path", args.checkpoint_path),
            ]
            if not str(value or "").strip()
        ]
        if missing:
            raise RuntimeError(f"Missing required arguments with --run-sam3: {', '.join(missing)}")
        _run_sam3(args)

    client_command = _build_client_command(args)
    print("Running visual prompt client:")
    print(" ".join(client_command))
    if args.print_only:
        return
    subprocess.run(client_command, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
