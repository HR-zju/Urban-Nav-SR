from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_SAM3_SCRIPT = "/home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/prepare_frames_and_run_batch_sam3.py"


def _find_option_value(args: Sequence[str], option_name: str) -> str | None:
    for index, token in enumerate(args):
        if token == option_name and index + 1 < len(args):
            return str(args[index + 1])
        if token.startswith(f"{option_name}="):
            return token.split("=", 1)[1]
    return None


def _coerce_frame_name(*, frame_name: Optional[str], image_path: Optional[str]) -> str:
    if str(frame_name or "").strip():
        return str(frame_name).strip()
    raw_image_path = str(image_path or "").strip()
    if raw_image_path:
        return Path(raw_image_path).expanduser().stem
    raise RuntimeError("Unable to determine frame name. Provide frame_name or image_path.")


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


def _build_sam3_command(
    *,
    sam3_env_name: str,
    sam3_script: str,
    rides_base_dir: str,
    frames_base_dir: str,
    sam3_output_base_dir: str,
    recordings_subdir: str,
    stream_pattern: str,
    camera_view: str,
    sequence_name: str,
    fps: float,
    gpus: List[int],
    min_score: float,
    finetuned: bool,
    checkpoint_path: str,
    max_frames: Optional[int],
    overwrite_frames: bool,
) -> list[str]:
    command = [
        "conda",
        "run",
        "-n",
        str(sam3_env_name),
        "python",
        str(sam3_script),
        "--rides_base_dir",
        str(rides_base_dir),
        "--frames_base_dir",
        str(frames_base_dir),
        "--inference_output_base_dir",
        str(sam3_output_base_dir),
        "--recordings_subdir",
        str(recordings_subdir),
        "--stream_pattern",
        str(stream_pattern),
        "--camera_view",
        str(camera_view),
        "--sequence_names",
        str(sequence_name),
        "--fps",
        str(float(fps)),
        "--gpus",
        *[str(gpu) for gpu in gpus],
        "--min_score",
        str(float(min_score)),
        "--finetuned",
        "true" if finetuned else "false",
        "--checkpoint_path",
        str(checkpoint_path),
    ]
    if max_frames is not None:
        command.extend(["--max_frames", str(int(max_frames))])
    if overwrite_frames:
        command.extend(["--overwrite_frames", "true"])
    return command


async def _run_command(command: Sequence[str]) -> Dict[str, Any]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await process.communicate()
    stdout_text = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr_text = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
    output_tail = "\n".join((stdout_text + "\n" + stderr_text).strip().splitlines()[-40:])
    if process.returncode != 0:
        raise RuntimeError(
            "SAM3 command failed with exit code "
            f"{process.returncode}. Command: {' '.join(command)}\n{output_tail}"
        )
    return {
        "command": [str(part) for part in command],
        "returncode": int(process.returncode),
        "output_tail": output_tail,
    }


async def segment_fpv_frame(
    *,
    sequence_name: str,
    sam3_output_base_dir: str,
    image_path: Optional[str] = None,
    frame_name: Optional[str] = None,
    camera_view: str = "camera_front",
    run_sam3: bool = False,
    sam3_env_name: str = "sam3",
    sam3_script: str = DEFAULT_SAM3_SCRIPT,
    rides_base_dir: Optional[str] = None,
    frames_base_dir: Optional[str] = None,
    recordings_subdir: str = "recordings",
    stream_pattern: str = "*_video.m3u8",
    fps: float = 1.0,
    gpus: Optional[List[int]] = None,
    min_score: float = 0.3,
    finetuned: bool = True,
    checkpoint_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    overwrite_frames: bool = False,
) -> Dict[str, Any]:
    resolved_frame_name = _coerce_frame_name(frame_name=frame_name, image_path=image_path)
    resolved_gpus = list(gpus or [0])

    sam3_run_info: Dict[str, Any] | None = None
    if run_sam3:
        missing = [
            name
            for name, value in [
                ("rides_base_dir", rides_base_dir),
                ("frames_base_dir", frames_base_dir),
                ("checkpoint_path", checkpoint_path),
            ]
            if not str(value or "").strip()
        ]
        if missing:
            raise RuntimeError(
                "segment_fpv_frame missing required fields when run_sam3=true: "
                + ", ".join(missing)
            )
        command = _build_sam3_command(
            sam3_env_name=sam3_env_name,
            sam3_script=sam3_script,
            rides_base_dir=str(rides_base_dir),
            frames_base_dir=str(frames_base_dir),
            sam3_output_base_dir=sam3_output_base_dir,
            recordings_subdir=recordings_subdir,
            stream_pattern=stream_pattern,
            camera_view=camera_view,
            sequence_name=sequence_name,
            fps=fps,
            gpus=resolved_gpus,
            min_score=min_score,
            finetuned=bool(finetuned),
            checkpoint_path=str(checkpoint_path),
            max_frames=max_frames,
            overwrite_frames=bool(overwrite_frames),
        )
        sam3_run_info = await _run_command(command)

    segmentation_npy, class_mapping, visualization = _segmentation_paths(
        sam3_output_base_dir=sam3_output_base_dir,
        sequence_name=sequence_name,
        camera_view=camera_view,
        frame_name=resolved_frame_name,
    )
    missing_artifacts = [
        str(path)
        for path in [segmentation_npy, class_mapping]
        if not path.exists()
    ]
    if missing_artifacts:
        raise RuntimeError(
            "SAM3 artifacts not found for frame "
            f"{resolved_frame_name}: {', '.join(missing_artifacts)}"
        )

    fpv_segmentation: Dict[str, Any] = {
        "segmentation_npy_path": str(segmentation_npy),
        "class_mapping_path": str(class_mapping),
    }
    if visualization.exists():
        fpv_segmentation["visualization_path"] = str(visualization)

    request_echo: Dict[str, Any] = {
        "sequence_name": str(sequence_name),
        "sam3_output_base_dir": str(sam3_output_base_dir),
        "image_path": str(image_path) if image_path is not None else None,
        "frame_name": str(frame_name) if frame_name is not None else None,
        "camera_view": str(camera_view),
        "run_sam3": bool(run_sam3),
        "sam3_env_name": str(sam3_env_name),
        "sam3_script": str(sam3_script),
        "rides_base_dir": str(rides_base_dir) if rides_base_dir is not None else None,
        "frames_base_dir": str(frames_base_dir) if frames_base_dir is not None else None,
        "recordings_subdir": str(recordings_subdir),
        "stream_pattern": str(stream_pattern),
        "fps": float(fps),
        "gpus": [int(gpu) for gpu in resolved_gpus],
        "min_score": float(min_score),
        "finetuned": bool(finetuned),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "max_frames": int(max_frames) if max_frames is not None else None,
        "overwrite_frames": bool(overwrite_frames),
    }

    return {
        "frame_name": resolved_frame_name,
        "sequence_name": str(sequence_name),
        "camera_view": str(camera_view),
        "fpv_segmentation": fpv_segmentation,
        "sam3_run": sam3_run_info,
        "request": request_echo,
    }