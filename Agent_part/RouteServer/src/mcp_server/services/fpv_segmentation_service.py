"""FPV segmentation artifact service and alignment contract builder.

This service is the boundary between external SAM3 inference and the internal
visual-prompt pipeline. It either resolves precomputed artifacts or launches the
configured SAM3 batch script, then returns a small contract that downstream
code can validate before using the mask.

Artifact contract invariants:

    mask_shape_hw = [H_mask, W_mask]
    image_shape_hw = [H_image, W_image]
    shape_aligned_with_image = (H_mask == H_image and W_mask == W_image)

Coordinate contract:

    image_uv origin = top-left
    u axis = right, v axis = down
    transform_to_image_uv = identity_3x3

The segmentation mask is never interpreted semantically here. Class policy,
safe-corridor priors, and prompt validation live in ``brain.fpv_segmentation``;
this module only guarantees that the artifact paths and coordinate metadata are
present, timed, and error-coded.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


DEFAULT_SAM3_SCRIPT = "/home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/prepare_frames_and_run_batch_sam3.py"
SEGMENTATION_SCHEMA_VERSION = 1
SUPPORTED_SEGMENTATION_PROVIDERS = {"sam3"}

ERR_INPUT_INVALID = "SEGMENTATION_INPUT_INVALID"
ERR_SCRIPT_NOT_FOUND = "SEGMENTATION_SCRIPT_NOT_FOUND"
ERR_COMMAND_FAILED = "SEGMENTATION_COMMAND_FAILED"
ERR_ARTIFACT_MISSING = "SEGMENTATION_ARTIFACT_MISSING"
ERR_PAYLOAD_INVALID = "SEGMENTATION_PAYLOAD_INVALID"
ERR_PROVIDER_UNSUPPORTED = "SEGMENTATION_PROVIDER_UNSUPPORTED"
CONTRACT_SCHEMA_VERSION = 1


class SegmentationServiceError(RuntimeError):
    """Typed service error returned as a structured API payload."""

    def __init__(self, *, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(str(message))
        self.code = str(code)
        self.message = str(message)
        self.retryable = bool(retryable)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _image_size_from_path(image_path: Optional[str]) -> tuple[int, int] | None:
    raw = str(image_path or "").strip()
    if not raw or Image is None:
        return None
    path = Path(raw).expanduser()
    if not path.exists() or not path.is_file():
        return None
    with Image.open(path) as image:
        width, height = image.size
    return int(width), int(height)


def _artifact_contract(
    *,
    segmentation_npy: Path,
    class_mapping: Path,
    sequence_name: str,
    camera_view: str,
    frame_name: str,
    image_path: Optional[str],
) -> Dict[str, Any]:
    """Build the image/mask alignment contract consumed by prompt validation."""
    mask = np.load(segmentation_npy, mmap_mode="r")
    if mask.ndim != 2:
        raise SegmentationServiceError(
            code=ERR_PAYLOAD_INVALID,
            message=f"Expected segmentation mask HxW, got shape={tuple(mask.shape)}",
            retryable=False,
        )
    mask_h = int(mask.shape[0])
    mask_w = int(mask.shape[1])
    image_size = _image_size_from_path(image_path)
    image_shape_hw = [int(image_size[1]), int(image_size[0])] if image_size else None
    include_hashes = str(os.getenv("SEGMENTATION_CONTRACT_INCLUDE_SHA256", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    contract: Dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "coordinate_space": {
            "name": "image_uv",
            "origin": "top_left",
            "axis": {"u": "right", "v": "down"},
            "units": "pixel",
        },
        "frame_identity": {
            "sequence_name": str(sequence_name),
            "camera_view": str(camera_view),
            "frame_name": str(frame_name),
        },
        "mask_shape_hw": [mask_h, mask_w],
        "image_shape_hw": image_shape_hw,
        "shape_aligned_with_image": (
            bool(image_shape_hw and image_shape_hw[0] == mask_h and image_shape_hw[1] == mask_w)
            if image_shape_hw
            else None
        ),
        "source_image_path": str(Path(str(image_path)).expanduser()) if str(image_path or "").strip() else None,
        "transform_to_image_uv": {
            "type": "identity",
            "matrix_3x3": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
    }
    if include_hashes:
        contract["sha256"] = {
            "segmentation_npy": _sha256_file(segmentation_npy),
            "class_mapping_json": _sha256_file(class_mapping),
            "source_image": _sha256_file(Path(str(image_path)).expanduser()) if str(image_path or "").strip() else None,
        }
    return contract


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
    """Serialize a SAM3 provider request into a reproducible subprocess command."""
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


def _validate_provider(provider_name: str) -> None:
    if provider_name not in SUPPORTED_SEGMENTATION_PROVIDERS:
        raise SegmentationServiceError(
            code=ERR_PROVIDER_UNSUPPORTED,
            message=(
                f"Unsupported segmentation provider '{provider_name}'. "
                f"Supported providers: {sorted(SUPPORTED_SEGMENTATION_PROVIDERS)}"
            ),
            retryable=False,
        )


async def _run_provider(
    provider_name: str,
    *,
    run_sam3: bool,
    sam3_env_name: str,
    sam3_script: str,
    rides_base_dir: Optional[str],
    frames_base_dir: Optional[str],
    sam3_output_base_dir: str,
    recordings_subdir: str,
    stream_pattern: str,
    camera_view: str,
    sequence_name: str,
    fps: float,
    gpus: List[int],
    min_score: float,
    finetuned: bool,
    checkpoint_path: Optional[str],
    max_frames: Optional[int],
    overwrite_frames: bool,
) -> Dict[str, Any] | None:
    if not run_sam3:
        return None
    if provider_name != "sam3":
        raise SegmentationServiceError(
            code=ERR_PROVIDER_UNSUPPORTED,
            message=(
                f"Provider '{provider_name}' does not support online inference in this service. "
                "Supported online providers: ['sam3']"
            ),
            retryable=False,
        )

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
        raise SegmentationServiceError(
            code=ERR_INPUT_INVALID,
            message=(
                "segment_fpv_frame missing required fields when run_sam3=true: "
                + ", ".join(missing)
            ),
            retryable=False,
        )

    script_path = Path(str(sam3_script)).expanduser()
    if not script_path.exists():
        raise SegmentationServiceError(
            code=ERR_SCRIPT_NOT_FOUND,
            message=f"sam3_script not found: {script_path}",
            retryable=False,
        )

    command = _build_sam3_command(
        sam3_env_name=sam3_env_name,
        sam3_script=str(script_path),
        rides_base_dir=str(rides_base_dir),
        frames_base_dir=str(frames_base_dir),
        sam3_output_base_dir=sam3_output_base_dir,
        recordings_subdir=recordings_subdir,
        stream_pattern=stream_pattern,
        camera_view=camera_view,
        sequence_name=sequence_name,
        fps=fps,
        gpus=gpus,
        min_score=min_score,
        finetuned=bool(finetuned),
        checkpoint_path=str(checkpoint_path),
        max_frames=max_frames,
        overwrite_frames=bool(overwrite_frames),
    )
    try:
        return await _run_command(command)
    except Exception as exc:
        raise SegmentationServiceError(
            code=ERR_COMMAND_FAILED,
            message=str(exc),
            retryable=True,
        ) from exc


def _resolve_artifacts(
    provider_name: str,
    *,
    sam3_output_base_dir: str,
    sequence_name: str,
    camera_view: str,
    frame_name: str,
    image_path: Optional[str],
    retriable: bool,
) -> Dict[str, Any]:
    if provider_name != "sam3":
        raise SegmentationServiceError(
            code=ERR_PROVIDER_UNSUPPORTED,
            message=(
                f"Provider '{provider_name}' does not support artifact resolution in this service. "
                "Supported providers: ['sam3']"
            ),
            retryable=False,
        )

    segmentation_npy, class_mapping, visualization = _segmentation_paths(
        sam3_output_base_dir=sam3_output_base_dir,
        sequence_name=sequence_name,
        camera_view=camera_view,
        frame_name=frame_name,
    )
    missing_artifacts = [
        str(path)
        for path in [segmentation_npy, class_mapping]
        if not path.exists()
    ]
    if missing_artifacts:
        raise SegmentationServiceError(
            code=ERR_ARTIFACT_MISSING,
            message=(
                "SAM3 artifacts not found for frame "
                f"{frame_name}: {', '.join(missing_artifacts)}"
            ),
            retryable=bool(retriable),
        )
    contract = _artifact_contract(
        segmentation_npy=segmentation_npy,
        class_mapping=class_mapping,
        sequence_name=sequence_name,
        camera_view=camera_view,
        frame_name=frame_name,
        image_path=image_path,
    )
    return {
        "segmentation_npy_path": str(segmentation_npy),
        "class_mapping_path": str(class_mapping),
        "visualization_path": str(visualization) if visualization.exists() else None,
        "alignment_contract": contract,
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
    provider: str = "sam3",
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve or generate segmentation artifacts for one FPV frame.

    The returned payload is stable across success and failure: both branches
    include request echo, timing fields, provider identity, and structured error
    metadata. This makes batch pipeline failures easy to aggregate by code.
    """
    started_at = perf_counter()
    provider_name = str(provider or "sam3").strip().lower() or "sam3"
    resolved_request_id = str(request_id or uuid4())
    resolved_gpus = [int(gpu) for gpu in list(gpus or [0])]
    run_started_at = 0.0
    run_elapsed_ms = 0
    artifact_started_at = 0.0
    artifact_elapsed_ms = 0
    resolved_frame_name = str(frame_name or "").strip()

    request_echo: Dict[str, Any] = {
        "request_id": resolved_request_id,
        "provider": provider_name,
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
        "gpus": resolved_gpus,
        "min_score": float(min_score),
        "finetuned": bool(finetuned),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "max_frames": int(max_frames) if max_frames is not None else None,
        "overwrite_frames": bool(overwrite_frames),
    }

    def _timing_payload() -> Dict[str, int]:
        total_ms = int(round((perf_counter() - started_at) * 1000.0))
        return {
            "total_ms": total_ms,
            "run_ms": int(run_elapsed_ms),
            "artifact_resolve_ms": int(artifact_elapsed_ms),
        }

    def _error_payload(*, code: str, message: str, retryable: bool) -> Dict[str, Any]:
        return {
            "status": "error",
            "provider": provider_name,
            "schema_version": SEGMENTATION_SCHEMA_VERSION,
            "request_id": resolved_request_id,
            "sequence_name": str(sequence_name),
            "camera_view": str(camera_view),
            "frame_name": resolved_frame_name or None,
            "timing_ms": _timing_payload(),
            "artifacts": None,
            "fpv_segmentation": None,
            "sam3_run": None,
            "request": request_echo,
            "error": {
                "code": str(code),
                "message": str(message),
                "retryable": bool(retryable),
            },
        }

    try:
        _validate_provider(provider_name)
    except SegmentationServiceError as exc:
        return _error_payload(code=exc.code, message=exc.message, retryable=exc.retryable)

    try:
        resolved_frame_name = _coerce_frame_name(frame_name=frame_name, image_path=image_path)
    except Exception as exc:
        return _error_payload(
            code=ERR_INPUT_INVALID,
            message=str(exc),
            retryable=False,
        )

    sam3_run_info: Dict[str, Any] | None = None
    if run_sam3:
        run_started_at = perf_counter()
        try:
            sam3_run_info = await _run_provider(
                provider_name,
                run_sam3=run_sam3,
                sam3_env_name=sam3_env_name,
                sam3_script=sam3_script,
                rides_base_dir=rides_base_dir,
                frames_base_dir=frames_base_dir,
                sam3_output_base_dir=sam3_output_base_dir,
                recordings_subdir=recordings_subdir,
                stream_pattern=stream_pattern,
                camera_view=camera_view,
                sequence_name=sequence_name,
                fps=fps,
                gpus=resolved_gpus,
                min_score=min_score,
                finetuned=bool(finetuned),
                checkpoint_path=checkpoint_path,
                max_frames=max_frames,
                overwrite_frames=bool(overwrite_frames),
            )
        except SegmentationServiceError as exc:
            run_elapsed_ms = int(round((perf_counter() - run_started_at) * 1000.0))
            return _error_payload(
                code=exc.code,
                message=exc.message,
                retryable=exc.retryable,
            )
        run_elapsed_ms = int(round((perf_counter() - run_started_at) * 1000.0))

    artifact_started_at = perf_counter()
    try:
        artifacts = _resolve_artifacts(
            provider_name,
            sam3_output_base_dir=sam3_output_base_dir,
            sequence_name=sequence_name,
            camera_view=camera_view,
            frame_name=resolved_frame_name,
            image_path=image_path,
            retriable=bool(run_sam3),
        )
    except SegmentationServiceError as exc:
        artifact_elapsed_ms = int(round((perf_counter() - artifact_started_at) * 1000.0))
        return _error_payload(
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
        )
    artifact_elapsed_ms = int(round((perf_counter() - artifact_started_at) * 1000.0))

    fpv_segmentation: Dict[str, Any] = {
        "segmentation_npy_path": artifacts["segmentation_npy_path"],
        "class_mapping_path": artifacts["class_mapping_path"],
        "alignment_contract": artifacts.get("alignment_contract"),
    }
    if artifacts["visualization_path"]:
        fpv_segmentation["visualization_path"] = str(artifacts["visualization_path"])

    return {
        "status": "ok",
        "provider": provider_name,
        "schema_version": SEGMENTATION_SCHEMA_VERSION,
        "request_id": resolved_request_id,
        "sequence_name": str(sequence_name),
        "camera_view": str(camera_view),
        "frame_name": resolved_frame_name,
        "timing_ms": _timing_payload(),
        "artifacts": artifacts,
        "fpv_segmentation": fpv_segmentation,
        "sam3_run": sam3_run_info,
        "request": request_echo,
        "error": None,
    }
