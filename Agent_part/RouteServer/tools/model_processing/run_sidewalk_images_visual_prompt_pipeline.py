#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Ensure `import mcp_server...` works even when the package is not installed.
_SRC_DIR = _repo_root() / "src"
if str(_SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(_SRC_DIR))

from mcp_server.utils.timing import (  # noqa: E402
    compact_visual_prompt_frame_timing,
    elapsed_ms,
    format_visual_prompt_frame_timing,
    summarize_visual_prompt_manifest_timings,
    timer_start,
)
from mcp_server.utils.async_batch import run_ordered_bounded  # noqa: E402


@dataclass(frozen=True)
class TrajectoryFrame:
    frame_name: str
    lat: float
    lng: float
    heading_deg: float | None
    image_path: Path
    roll_deg: float | None = None
    pitch_deg: float | None = None
    attitude_source: str | None = None


@dataclass(frozen=True)
class FrameJob:
    frame_name: str
    frame: TrajectoryFrame | None
    fpv_segmentation: dict[str, Any] | None
    frame_item: dict[str, Any]


def _wrap_180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def _forward_unit_from_heading(heading_deg: float) -> tuple[float, float]:
    # Heading convention: 0=North, 90=East.
    rad = math.radians(float(heading_deg) % 360.0)
    east = math.sin(rad)
    north = math.cos(rad)
    return east, north


def _offset_latlng_m(lat: float, lng: float, east_m: float, north_m: float) -> tuple[float, float]:
    # Small-angle approximation is fine for local synthetic straight routes.
    r = 6371000.0
    d_lat = (north_m / r) * (180.0 / math.pi)
    lat2 = float(lat) + float(d_lat)
    denom = max(1e-6, math.cos(math.radians((float(lat) + float(lat2)) * 0.5)))
    d_lng = (east_m / (r * denom)) * (180.0 / math.pi)
    lng2 = float(lng) + float(d_lng)
    return lat2, lng2


def build_straight_route_polyline(
    *,
    lat: float,
    lng: float,
    heading_deg: float,
    route_length_m: float,
    step_m: float,
) -> list[dict[str, float]]:
    east_u, north_u = _forward_unit_from_heading(heading_deg)
    route_length_m = max(float(route_length_m), 5.0)
    step_m = max(float(step_m), 1.0)
    points: list[dict[str, float]] = []
    traveled = 0.0
    while traveled < route_length_m + 1e-6:
        east = traveled * east_u
        north = traveled * north_u
        lat_i, lng_i = _offset_latlng_m(lat, lng, east, north)
        points.append({"latitude": float(lat_i), "longitude": float(lng_i)})
        traveled += step_m
    if len(points) < 2:
        lat2, lng2 = _offset_latlng_m(lat, lng, 5.0 * east_u, 5.0 * north_u)
        points.append({"latitude": float(lat2), "longitude": float(lng2)})
    return points


def build_straight_coarse_waypoints(
    *,
    route_polyline: list[dict[str, float]],
) -> list[dict[str, Any]]:
    # Minimal waypoint contract expected by route_context:
    # - coordinate.lat/lon and an action. Keep it "go_straight" to avoid arrival semantics.
    end = route_polyline[-1]
    return [
        {
            "coordinate": {"latitude": float(end["latitude"]), "longitude": float(end["longitude"])},
            "action": "go_straight",
            "instruction": "Continue forward.",
            "road": "",
            "distance_m": 0.0,
            "duration_s": 0.0,
        }
    ]


def build_navigation_path_gps(
    *,
    selected_frame_names: list[str],
    frames_by_name: dict[str, TrajectoryFrame],
) -> list[dict[str, float]]:
    """Collect the offline dataset trajectory for BEV map visualization."""
    path: list[dict[str, float]] = []
    for frame_name in selected_frame_names:
        frame = frames_by_name.get(frame_name)
        if frame is None:
            continue
        path.append({"lat": float(frame.lat), "lng": float(frame.lng)})
    return path


def _camera_attitude_for_frame(frame: TrajectoryFrame) -> dict[str, Any] | None:
    """Return optional frame attitude for FPV ground-geometry normalization."""
    if frame.roll_deg is None and frame.pitch_deg is None:
        return None
    return {
        "roll_deg": float(frame.roll_deg) if frame.roll_deg is not None else None,
        "pitch_deg": float(frame.pitch_deg) if frame.pitch_deg is not None else None,
        "source": str(frame.attitude_source or "trajectory_json"),
    }


def _load_trajectory_frames(*, trajectory_path: Path, images_dir: Path) -> dict[str, TrajectoryFrame]:
    raw = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError(f"trajectory.json must be a list, got: {type(raw)}")

    frames: dict[str, TrajectoryFrame] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        image_path_raw = str(item.get("image_path") or "").strip()
        if not image_path_raw:
            continue
        frame_name = Path(image_path_raw).stem
        if not frame_name:
            continue
        lat = float(item.get("lat"))
        lng = float(item.get("lng"))
        heading = item.get("heading_deg")
        heading_deg = float(heading) if heading is not None else None
        roll = item.get("roll_deg")
        pitch = item.get("pitch_deg")
        attitude_source = item.get("attitude_source")
        resolved_image = images_dir / f"{frame_name}.jpg"
        if not resolved_image.exists():
            # Try original extension if it's not jpg.
            resolved_image = images_dir / Path(image_path_raw).name
        frames[frame_name] = TrajectoryFrame(
            frame_name=str(frame_name),
            lat=lat,
            lng=lng,
            heading_deg=heading_deg,
            image_path=resolved_image,
            roll_deg=float(roll) if roll is not None else None,
            pitch_deg=float(pitch) if pitch is not None else None,
            attitude_source=str(attitude_source) if attitude_source else None,
        )
    if not frames:
        raise RuntimeError(f"No usable frames found in trajectory: {trajectory_path}")
    return frames


def _iter_selected_frame_names(
    *,
    images_dir: Path,
    frame_start: int | None,
    frame_end: int | None,
    frame_step: int,
) -> list[str]:
    names: list[str] = []
    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = path.stem.strip()
        if not stem.isdigit():
            continue
        idx = int(stem)
        if frame_start is not None and idx < int(frame_start):
            continue
        if frame_end is not None and idx > int(frame_end):
            continue
        names.append(stem)
    step = max(int(frame_step), 1)
    return names[::step]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_data_url_image(data_url: str, output_path_without_suffix: Path) -> Path | None:
    raw = str(data_url or "").strip()
    if not raw.startswith("data:image/") or "," not in raw:
        return None
    header, encoded = raw.split(",", 1)
    mime = header.split(";", 1)[0].replace("data:", "")
    ext = ".jpg" if mime in {"image/jpeg", "image/jpg"} else ".png" if mime == "image/png" else ".img"
    output_path = Path(str(output_path_without_suffix) + ext)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(encoded))
    return output_path


def _safe_artifact_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value or "").strip())
    return cleaned or "image"


def _set_env_if_value(name: str, value: Any | None) -> None:
    if value is None:
        return
    os.environ[str(name)] = str(value)


def _symlink_or_copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src, target_is_directory=True)
    except Exception:
        # Fallback: avoid copying the whole directory unless absolutely needed.
        raise RuntimeError(f"Unable to symlink images dir: {src} -> {dst}")


def _prepare_frame_subset_dir(*, images_dir: Path, subset_dir: Path, frame_names: Iterable[str]) -> None:
    subset_dir.mkdir(parents=True, exist_ok=True)
    # Important: keep the directory strictly as "the selected frames only".
    # If a previous run populated this dir with many images, SAM3 would process all of them.
    for child in list(subset_dir.iterdir()):
        if child.is_file() or child.is_symlink():
            try:
                child.unlink()
            except Exception:
                pass
    for name in frame_names:
        # Prefer jpg, then png/jpeg.
        candidates = [images_dir / f"{name}.jpg", images_dir / f"{name}.png", images_dir / f"{name}.jpeg"]
        src = next((p for p in candidates if p.exists()), None)
        if src is None:
            continue
        dst = subset_dir / src.name
        if dst.exists():
            continue
        try:
            dst.symlink_to(src)
        except Exception:
            # If symlink is not allowed in this environment, fall back to copying just the small subset.
            dst.write_bytes(src.read_bytes())


def _run_sam3_inference_on_images(
    *,
    images_dir: Path,
    selected_frame_names: list[str] | None,
    sam3_output_base_dir: Path,
    sequence_name: str,
    camera_view: str,
    sam3_env_name: str,
    sam3_inference_script: Path,
    gpus: list[int],
    min_score: float,
    finetuned: bool,
    checkpoint_path: Path | None,
    work_dir: Path,
) -> None:
    # Build a cocoa-style directory structure expected by batch_sam3_demo_260124.py:
    #   cocoa_base_dir/<sequence_name>/<camera_view>/*.jpg
    cocoa_base_dir = work_dir / "cocoa_frames"
    camera_dir = cocoa_base_dir / sequence_name / camera_view
    if selected_frame_names:
        # Important: batch_sam3_demo_260124.py processes ALL frames in the directory it sees.
        # For debug runs, only expose the chosen frames to SAM3.
        _prepare_frame_subset_dir(images_dir=images_dir, subset_dir=camera_dir, frame_names=selected_frame_names)
    else:
        _symlink_or_copy_tree(images_dir, camera_dir)

    script_path = sam3_inference_script.expanduser().resolve()
    if not script_path.exists():
        raise RuntimeError(f"SAM3 inference script not found: {script_path}")

    if finetuned and (checkpoint_path is None or not checkpoint_path.exists()):
        raise RuntimeError("--checkpoint-path is required (and must exist) when --finetuned is enabled")

    cmd: list[str] = [
        "conda",
        "run",
        "-n",
        str(sam3_env_name),
        "python",
        str(script_path),
        "--cocoa_base_dir",
        str(cocoa_base_dir),
        "--output_base_dir",
        str(sam3_output_base_dir),
        "--camera_views",
        str(camera_view),
        "--gpus",
        *[str(int(g)) for g in gpus],
        "--min_score",
        str(float(min_score)),
        "--sequence_names",
        str(sequence_name),
        "--finetuned",
        "true" if finetuned else "false",
    ]
    if finetuned and checkpoint_path is not None:
        cmd.extend(["--checkpoint_path", str(checkpoint_path)])

    print("[INFO] Running SAM3 inference (images -> segmentation artifacts):")
    if selected_frame_names:
        print(f"[INFO] SAM3 frame subset size: {len(selected_frame_names)}")
    print(" ".join(cmd))
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(work_dir / "mplconfig"))
    env.setdefault("XDG_CACHE_HOME", str(work_dir / "xdg_cache"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, env=env)


def _segmentation_artifacts_for_frame(
    *,
    sam3_output_base_dir: Path,
    sequence_name: str,
    camera_view: str,
    frame_name: str,
) -> dict[str, Any] | None:
    base_dir = sam3_output_base_dir / sequence_name / camera_view
    npy = base_dir / f"{frame_name}_segmentation.npy"
    mapping = base_dir / "class_mapping.json"
    vis = base_dir / f"{frame_name}_visualization.jpg"
    if not npy.exists() or not mapping.exists():
        return None
    payload: dict[str, Any] = {
        "segmentation_npy_path": str(npy),
        "class_mapping_path": str(mapping),
    }
    if vis.exists():
        payload["visualization_path"] = str(vis)
    return payload


async def _run_frame(
    *,
    frame: TrajectoryFrame,
    route_length_m: float,
    route_step_m: float,
    lookahead_m: float,
    off_route_threshold_m: float,
    front_extent_m: float,
    back_extent_m: float,
    side_extent_m: float,
    navigation_mode: str,
    llm_detail: str,
    enable_verifier: bool,
    enable_segmentation_overlay_input: bool,
    enable_som_overlay_input: bool | None,
    enable_sam_visualization_input: bool,
    enable_bev_map_provider: bool,
    bev_map_provider_session: Any | None,
    llm_primary_image_role: str,
    enforce_alignment: bool,
    emit_qa_bundle: bool,
    emit_llm_input_images: bool,
    fpv_segmentation: dict[str, Any] | None,
    navigation_path_gps: list[dict[str, float]] | None,
) -> dict[str, Any]:
    from mcp_server.services.visual_prompt_service import run_zero_shot_visual_prompt_pipeline

    heading = frame.heading_deg
    if heading is None:
        heading = 0.0

    route_polyline = build_straight_route_polyline(
        lat=frame.lat,
        lng=frame.lng,
        heading_deg=float(heading),
        route_length_m=route_length_m,
        step_m=route_step_m,
    )
    coarse_waypoints = build_straight_coarse_waypoints(route_polyline=route_polyline)

    return await run_zero_shot_visual_prompt_pipeline(
        gps={"lat": float(frame.lat), "lng": float(frame.lng)},
        route_polyline=route_polyline,
        coarse_waypoints=coarse_waypoints,
        image_path=str(frame.image_path),
        fpv_segmentation=fpv_segmentation,
        heading_deg=float(heading),
        navigation_path_gps=navigation_path_gps,
        navigation_path_source="offline_trajectory" if navigation_path_gps else None,
        camera_attitude=_camera_attitude_for_frame(frame),
        lookahead_m=float(lookahead_m),
        off_route_threshold_m=float(off_route_threshold_m),
        front_extent_m=float(front_extent_m),
        back_extent_m=float(back_extent_m),
        side_extent_m=float(side_extent_m),
        navigation_mode=str(navigation_mode),
        llm_detail=str(llm_detail),
        enable_verifier=bool(enable_verifier),
        enable_segmentation_overlay_input=bool(enable_segmentation_overlay_input),
        enable_som_overlay_input=enable_som_overlay_input,
        enable_sam_visualization_input=bool(enable_sam_visualization_input),
        enable_bev_map_provider=bool(enable_bev_map_provider),
        bev_map_provider_session=bev_map_provider_session,
        llm_primary_image_role=str(llm_primary_image_role),
        enforce_alignment=bool(enforce_alignment),
        emit_qa_bundle=bool(emit_qa_bundle),
        emit_llm_input_images=bool(emit_llm_input_images),
    )


def _render_overlay_from_prompt(*, image_path: Path, rendered_prompt: dict[str, Any], title: str) -> str:
    # Reuse the exact overlay rendering used by the service to keep visuals consistent.
    from mcp_server.services.visual_prompt_service import _image_href_for_overlay, _render_prompt_overlay_svg

    image_href = _image_href_for_overlay(image_path=str(image_path), image_url=None)
    # The service computes image size from file, but we can rely on the overlay helper.
    from PIL import Image as PILImage

    with PILImage.open(image_path) as im:
        w, h = im.size
    return _render_prompt_overlay_svg(
        image_href=image_href,
        image_width=int(w),
        image_height=int(h),
        rendered_prompt=rendered_prompt,
        title=title,
    )


def _build_frame_jobs(
    *,
    selected: list[str],
    frames_by_name: dict[str, TrajectoryFrame],
    sam3_output_base_dir: Path,
    sequence_name: str,
    camera_view: str,
    json_dir: Path,
    overlays_dir: Path,
    segmentation_policy: str,
) -> tuple[list[FrameJob], int]:
    jobs: list[FrameJob] = []
    preflight_failures = 0
    segmentation_policy = str(segmentation_policy or "optional").strip().lower()
    if segmentation_policy not in {"required", "optional", "disabled"}:
        raise RuntimeError(f"Unsupported segmentation policy: {segmentation_policy!r}")
    for frame_name in selected:
        frame = frames_by_name.get(frame_name)
        frame_item: dict[str, Any]
        if frame is None:
            preflight_failures += 1
            jobs.append(
                FrameJob(
                    frame_name=str(frame_name),
                    frame=None,
                    fpv_segmentation=None,
                    frame_item={"frame_name": frame_name, "status": "error", "error": "frame_missing_in_trajectory"},
                )
            )
            continue
        out_json = json_dir / f"{frame_name}.json"
        out_svg = overlays_dir / f"{frame_name}.svg"
        frame_item = {
            "frame_name": str(frame_name),
            "image_path": str(frame.image_path),
            "json_path": str(out_json),
            "overlay_path": str(out_svg),
            "status": "ok",
        }
        if not frame.image_path.exists():
            preflight_failures += 1
            frame_item["status"] = "error"
            frame_item["error"] = f"image_missing:{frame.image_path}"
            jobs.append(FrameJob(frame_name=str(frame_name), frame=frame, fpv_segmentation=None, frame_item=frame_item))
            continue
        fpv_seg = None
        if segmentation_policy != "disabled":
            fpv_seg = _segmentation_artifacts_for_frame(
                sam3_output_base_dir=sam3_output_base_dir,
                sequence_name=str(sequence_name),
                camera_view=str(camera_view),
                frame_name=str(frame_name),
            )
        if fpv_seg is None and segmentation_policy == "required":
            preflight_failures += 1
            frame_item["status"] = "error"
            frame_item["error"] = "missing_segmentation_artifacts"
        jobs.append(FrameJob(frame_name=str(frame_name), frame=frame, fpv_segmentation=fpv_seg, frame_item=frame_item))
    return jobs, preflight_failures


async def _process_frame_job(
    *,
    job: FrameJob,
    route_length_m: float,
    route_step_m: float,
    lookahead_m: float,
    off_route_threshold_m: float,
    front_extent_m: float,
    back_extent_m: float,
    side_extent_m: float,
    navigation_mode: str,
    llm_detail: str,
    enable_verifier: bool,
    enable_segmentation_overlay_input: bool,
    enable_som_overlay_input: bool | None,
    enable_sam_visualization_input: bool,
    enable_bev_map_provider: bool,
    bev_map_provider_session: Any | None,
    llm_primary_image_role: str,
    enforce_alignment: bool,
    emit_qa_bundle: bool,
    emit_llm_input_images: bool,
    navigation_path_gps: list[dict[str, float]] | None,
    seg_vis_dir: Path,
    llm_inputs_dir: Path,
    json_dir: Path,
    generator_json_dir: Path,
    overlays_dir: Path,
) -> dict[str, Any]:
    frame_item = dict(job.frame_item)
    frame_name = str(job.frame_name)
    frame = job.frame
    if frame is None or frame_item.get("status") == "error":
        return frame_item

    out_json = json_dir / f"{frame_name}.json"
    out_generator_json = generator_json_dir / f"{frame_name}.generator.json"
    out_svg = overlays_dir / f"{frame_name}.svg"
    out_generator_svg = overlays_dir / f"{frame_name}.generator.svg"
    out_rejected_svg = overlays_dir / f"{frame_name}.rejected.svg"
    out_seg_vis = seg_vis_dir / f"{frame_name}_visualization.jpg"

    frame_started_at = timer_start()
    payload: dict[str, Any] | None = None
    try:
        if job.fpv_segmentation and isinstance(job.fpv_segmentation.get("visualization_path"), str):
            src_vis = Path(str(job.fpv_segmentation.get("visualization_path"))).expanduser()
            if src_vis.exists():
                if out_seg_vis.exists() or out_seg_vis.is_symlink():
                    out_seg_vis.unlink()
                try:
                    out_seg_vis.symlink_to(src_vis)
                except Exception:
                    out_seg_vis.write_bytes(src_vis.read_bytes())

        payload = await _run_frame(
            frame=frame,
            route_length_m=float(route_length_m),
            route_step_m=float(route_step_m),
            lookahead_m=float(lookahead_m),
            off_route_threshold_m=float(off_route_threshold_m),
            front_extent_m=float(front_extent_m),
            back_extent_m=float(back_extent_m),
            side_extent_m=float(side_extent_m),
            navigation_mode=str(navigation_mode),
            llm_detail=str(llm_detail),
            enable_verifier=bool(enable_verifier),
            enable_segmentation_overlay_input=bool(enable_segmentation_overlay_input),
            enable_som_overlay_input=enable_som_overlay_input,
            enable_sam_visualization_input=bool(enable_sam_visualization_input),
            enable_bev_map_provider=bool(enable_bev_map_provider),
            bev_map_provider_session=bev_map_provider_session,
            llm_primary_image_role=str(llm_primary_image_role),
            enforce_alignment=bool(enforce_alignment),
            emit_qa_bundle=bool(emit_qa_bundle),
            emit_llm_input_images=bool(emit_llm_input_images),
            navigation_path_gps=navigation_path_gps,
            fpv_segmentation=job.fpv_segmentation,
        )

        artifact_started_at = timer_start()
        llm_input_artifacts: list[str] = []
        for item in list(payload.get("llm_input_images") or []):
            if not isinstance(item, dict):
                continue
            role = _safe_artifact_name(str(item.get("role") or item.get("image_role") or "llm_input"))
            output_path = _write_data_url_image(
                str(item.get("data_url") or ""),
                llm_inputs_dir / f"{frame_name}.{role}",
            )
            if output_path is not None:
                llm_input_artifacts.append(str(output_path))
                item["saved_path"] = str(output_path)
        if llm_input_artifacts:
            frame_item["llm_input_images"] = llm_input_artifacts
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        overlay_svg = payload.get("overlay_svg")
        out_svg.write_text(overlay_svg if isinstance(overlay_svg, str) and overlay_svg.strip() else "", encoding="utf-8")

        generator_overlay_svg = payload.get("generator_overlay_svg")
        if isinstance(generator_overlay_svg, str) and generator_overlay_svg.strip():
            out_generator_svg.write_text(generator_overlay_svg, encoding="utf-8")
            frame_item["generator_overlay_path"] = str(out_generator_svg)
        elif out_generator_svg.exists():
            out_generator_svg.unlink()

        arb = payload.get("arbiter_decision") or {}
        debug = arb.get("debug") or {}
        generator_payload = {
            "frame_name": str(frame_name),
            "image_path": str(frame.image_path),
            "generator_raw": debug.get("generator_raw"),
            "generator_decision": debug.get("generator_decision"),
            "generator_rejected_decision": debug.get("generator_rejected_decision"),
            "generator_overlay_path": str(out_generator_svg) if out_generator_svg.exists() else None,
        }
        has_generator_payload = any(
            generator_payload.get(key)
            for key in ("generator_raw", "generator_decision", "generator_rejected_decision", "generator_overlay_path")
        )
        if has_generator_payload:
            out_generator_json.write_text(json.dumps(generator_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            frame_item["generator_json_path"] = str(out_generator_json)
        elif out_generator_json.exists():
            out_generator_json.unlink()

        rejected_prompt = debug.get("rejected_rendered_prompt")
        if isinstance(rejected_prompt, dict) and rejected_prompt:
            try:
                rejected_svg = _render_overlay_from_prompt(
                    image_path=frame.image_path,
                    rendered_prompt=rejected_prompt,
                    title="Rejected Visual Prompt (Pre-Constraint)",
                )
                out_rejected_svg.write_text(rejected_svg, encoding="utf-8")
            except Exception:
                if out_rejected_svg.exists():
                    out_rejected_svg.unlink()
        elif out_rejected_svg.exists():
            out_rejected_svg.unlink()

        timing = compact_visual_prompt_frame_timing(
            payload=payload,
            runner_elapsed_ms=elapsed_ms(frame_started_at),
            artifact_elapsed_ms=elapsed_ms(artifact_started_at),
        )
        frame_item["timing_ms"] = timing
        print(format_visual_prompt_frame_timing(str(frame_name), timing, status=str(frame_item.get("status") or "ok")))
    except Exception as exc:
        frame_item["status"] = "error"
        frame_item["error"] = str(exc)
        timing = compact_visual_prompt_frame_timing(
            payload=payload if isinstance(payload, dict) else None,
            runner_elapsed_ms=elapsed_ms(frame_started_at),
            artifact_elapsed_ms=0,
        )
        frame_item["timing_ms"] = timing
        print(format_visual_prompt_frame_timing(str(frame_name), timing, status="error"))
    return frame_item



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SAM3 segmentation on a flat images directory (optional), then run the zero-shot visual prompt "
            "pipeline per frame using a synthetic straight-ahead route context derived from trajectory heading."
        )
    )
    parser.add_argument("--images-dir", required=True, help="Directory containing numbered images like 000001.jpg")
    parser.add_argument("--trajectory-json", required=True, help="Trajectory JSON with lat/lng/heading/image_path")
    parser.add_argument("--output-dir", required=True, help="Output directory for per-frame json/overlay and manifest")

    parser.add_argument("--sequence-name", default="sidewalk_he4_adapter", help="Sequence name used for SAM3 artifacts")
    parser.add_argument("--camera-view", default="camera_front", help="Camera view name used for SAM3 artifacts")
    parser.add_argument("--sam3-output-base-dir", required=True, help="Base dir for SAM3 artifacts output/lookup")

    parser.add_argument("--run-sam3", action="store_true", help="Run SAM3 inference on images before visual prompt")
    parser.add_argument("--sam3-env-name", default="sam3", help="Conda env name for SAM3")
    parser.add_argument(
        "--sam3-inference-script",
        default="/home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/batch_sam3_demo_260124.py",
        help="Path to batch_sam3_demo_260124.py",
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPU IDs for SAM3")
    parser.add_argument("--min-score", type=float, default=0.3, help="SAM3 min_score")
    parser.add_argument("--finetuned", action=argparse.BooleanOptionalAction, default=True, help="Use finetuned SAM3 checkpoint")
    parser.add_argument("--checkpoint-path", default=None, help="SAM3 checkpoint path (required when --finetuned)")

    parser.add_argument("--frame-start", type=int, default=None, help="Start frame id (inclusive)")
    parser.add_argument("--frame-end", type=int, default=None, help="End frame id (inclusive)")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame sampling step")

    parser.add_argument("--route-length-m", type=float, default=120.0, help="Synthetic straight route length")
    parser.add_argument("--route-step-m", type=float, default=5.0, help="Synthetic route polyline sampling step")
    parser.add_argument("--lookahead-m", type=float, default=20.0, help="Local segment lookahead distance")
    parser.add_argument("--off-route-threshold-m", type=float, default=20.0, help="Off-route threshold")
    parser.add_argument("--front-extent-m", type=float, default=30.0, help="BEV front extent")
    parser.add_argument("--back-extent-m", type=float, default=10.0, help="BEV back extent")
    parser.add_argument("--side-extent-m", type=float, default=15.0, help="BEV side extent")

    parser.add_argument("--mode", default="foot", choices=["foot", "bike", "car"], help="Navigation mode")
    parser.add_argument("--llm-detail", default="auto", choices=["auto", "low", "high"], help="Vision detail")
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=1,
        help="Max number of frames that may call the LVLM pipeline concurrently",
    )
    parser.add_argument("--enable-verifier", action=argparse.BooleanOptionalAction, default=False, help="Enable optional verifier LVLM call")
    parser.add_argument(
        "--enable-segmentation-overlay-input",
        action="store_true",
        help="Compatibility alias for --enable-som-overlay-input; does not enable raw SAM visualization",
    )
    parser.add_argument(
        "--enable-som-overlay-input",
        action="store_true",
        help="Append numbered SoM overlay as an extra LVLM image input",
    )
    parser.add_argument(
        "--enable-sam-visualization-input",
        action="store_true",
        help="Append raw SAM visualization as an extra LVLM image input for debugging/ablation",
    )
    parser.add_argument(
        "--enable-bev-map-provider",
        action="store_true",
        help="Query the configured map provider for OSM-style semantic BEV anchors around each GNSS frame",
    )
    parser.add_argument(
        "--llm-input-mode",
        default="auto",
        choices=["auto", "fpv_only", "fpv_som", "fpv_bev", "som_bev", "som_only"],
        help=(
            "Shortcut for LVLM image inputs. som_bev selects numbered SoM as the primary LVLM image and appends BEV as the extra image."
        ),
    )
    parser.add_argument(
        "--llm-primary-image-role",
        default="primary_fpv",
        choices=["primary_fpv", "numbered_som_overlay"],
        help=(
            "Primary image sent to the LVLM. Use numbered_som_overlay with --enable-bev-map-provider "
            "to upload only two images: SoM primary plus BEV extra."
        ),
    )
    parser.add_argument(
        "--llm-image-max-side",
        type=int,
        default=None,
        help="Override VISUAL_PROMPT_LLM_IMAGE_MAX_SIDE for FPV/SoM/SAM LVLM images",
    )
    parser.add_argument(
        "--llm-image-max-bytes",
        type=int,
        default=None,
        help="Override VISUAL_PROMPT_LLM_IMAGE_MAX_BYTES for encoded LVLM images",
    )
    parser.add_argument(
        "--llm-image-jpeg-quality",
        type=int,
        default=None,
        help="Override VISUAL_PROMPT_LLM_IMAGE_JPEG_QUALITY for LVLM image encoding",
    )
    parser.add_argument(
        "--bev-map-output-size-px",
        type=int,
        default=None,
        help="Override BEV_MAP_TILE_OUTPUT_SIZE_PX for the raster BEV LVLM image",
    )
    parser.add_argument(
        "--segmentation-policy",
        default="optional",
        choices=["required", "optional", "disabled"],
        help=(
            "How to treat SAM/SAM3 artifacts. required fails frames without masks; "
            "optional uses masks when present; disabled ignores segmentation artifacts."
        ),
    )
    parser.add_argument(
        "--enforce-alignment",
        dest="enforce_alignment",
        action="store_true",
        default=True,
        help="Require segmentation/image alignment validation (default: true)",
    )
    parser.add_argument(
        "--no-enforce-alignment",
        dest="enforce_alignment",
        action="store_false",
        help="Allow proceeding when segmentation alignment validation fails",
    )
    parser.add_argument("--emit-qa-bundle", action="store_true", help="Emit structured QA bundle in output json")
    parser.add_argument(
        "--emit-llm-input-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save exact LVLM input images for audit/debug (default: true)",
    )

    parser.add_argument(
        "--allow-missing-segmentation",
        action="store_true",
        help="Compatibility alias for --segmentation-policy optional.",
    )
    return parser.parse_args()


async def main_async() -> None:
    pipeline_started_at = timer_start()
    args = parse_args()
    images_dir = Path(args.images_dir).expanduser().resolve()
    trajectory_path = Path(args.trajectory_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    sam3_output_base_dir = Path(args.sam3_output_base_dir).expanduser().resolve()

    if not images_dir.exists():
        raise RuntimeError(f"images-dir not found: {images_dir}")
    if not trajectory_path.exists():
        raise RuntimeError(f"trajectory-json not found: {trajectory_path}")
    segmentation_policy = str(args.segmentation_policy or "optional").strip().lower()
    if bool(args.allow_missing_segmentation) and segmentation_policy == "required":
        segmentation_policy = "optional"
    resolved_enable_som_overlay_input = bool(args.enable_som_overlay_input or args.enable_segmentation_overlay_input)
    resolved_enable_bev_map_provider = bool(args.enable_bev_map_provider)
    resolved_enable_sam_visualization_input = bool(args.enable_sam_visualization_input)
    resolved_llm_primary_image_role = str(args.llm_primary_image_role)
    llm_input_mode = str(args.llm_input_mode or "auto").strip().lower()
    if llm_input_mode != "auto":
        if llm_input_mode == "fpv_only":
            resolved_enable_som_overlay_input = False
            resolved_enable_bev_map_provider = False
            resolved_enable_sam_visualization_input = False
            resolved_llm_primary_image_role = "primary_fpv"
        elif llm_input_mode == "fpv_som":
            resolved_enable_som_overlay_input = True
            resolved_enable_bev_map_provider = False
            resolved_enable_sam_visualization_input = False
            resolved_llm_primary_image_role = "primary_fpv"
        elif llm_input_mode == "fpv_bev":
            resolved_enable_som_overlay_input = False
            resolved_enable_bev_map_provider = True
            resolved_enable_sam_visualization_input = False
            resolved_llm_primary_image_role = "primary_fpv"
        elif llm_input_mode == "som_bev":
            resolved_enable_som_overlay_input = True
            resolved_enable_bev_map_provider = True
            resolved_enable_sam_visualization_input = False
            resolved_llm_primary_image_role = "numbered_som_overlay"
        elif llm_input_mode == "som_only":
            resolved_enable_som_overlay_input = True
            resolved_enable_bev_map_provider = False
            resolved_enable_sam_visualization_input = False
            resolved_llm_primary_image_role = "numbered_som_overlay"
        else:
            raise RuntimeError(f"Unsupported llm-input-mode: {args.llm_input_mode!r}")
    _set_env_if_value("VISUAL_PROMPT_LLM_IMAGE_MAX_SIDE", args.llm_image_max_side)
    _set_env_if_value("VISUAL_PROMPT_LLM_IMAGE_MAX_BYTES", args.llm_image_max_bytes)
    _set_env_if_value("VISUAL_PROMPT_LLM_IMAGE_JPEG_QUALITY", args.llm_image_jpeg_quality)
    _set_env_if_value("BEV_MAP_TILE_OUTPUT_SIZE_PX", args.bev_map_output_size_px)

    _ensure_dir(output_dir)
    overlays_dir = output_dir / "overlay"
    json_dir = output_dir / "json"
    generator_json_dir = output_dir / "generator_json"
    logs_dir = output_dir / "logs"
    seg_vis_dir = output_dir / "seg_vis"
    llm_inputs_dir = output_dir / "llm_inputs"
    for d in [overlays_dir, json_dir, generator_json_dir, logs_dir]:
        _ensure_dir(d)
    _ensure_dir(seg_vis_dir)
    _ensure_dir(llm_inputs_dir)

    frames_by_name = _load_trajectory_frames(trajectory_path=trajectory_path, images_dir=images_dir)
    selected = _iter_selected_frame_names(
        images_dir=images_dir,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_step=args.frame_step,
    )
    if not selected:
        raise RuntimeError("No frames selected from images-dir. Check --frame-start/--frame-end.")

    work_dir = output_dir / "_work"
    _ensure_dir(work_dir)

    sam3_inference_elapsed_ms: int | None = None
    if args.run_sam3:
        sam3_started_at = timer_start()
        _run_sam3_inference_on_images(
            images_dir=images_dir,
            selected_frame_names=selected,
            sam3_output_base_dir=sam3_output_base_dir,
            sequence_name=str(args.sequence_name),
            camera_view=str(args.camera_view),
            sam3_env_name=str(args.sam3_env_name),
            sam3_inference_script=Path(args.sam3_inference_script),
            gpus=[int(x) for x in (args.gpus or [0])],
            min_score=float(args.min_score),
            finetuned=bool(args.finetuned),
            checkpoint_path=Path(args.checkpoint_path).expanduser().resolve() if args.checkpoint_path else None,
            work_dir=work_dir,
        )
        sam3_inference_elapsed_ms = elapsed_ms(sam3_started_at)
        print(f"[TIME] sam3_inference={sam3_inference_elapsed_ms}ms frames={len(selected)}")

    manifest: dict[str, Any] = {
        "images_dir": str(images_dir),
        "trajectory_json": str(trajectory_path),
        "sam3_output_base_dir": str(sam3_output_base_dir),
        "sequence_name": str(args.sequence_name),
        "camera_view": str(args.camera_view),
        "frames_total": len(selected),
        "frames_failed": 0,
        "sam3_inference_elapsed_ms": sam3_inference_elapsed_ms,
        "segmentation_policy": segmentation_policy,
        "llm_primary_image_role": str(resolved_llm_primary_image_role),
        "llm_input_mode": str(llm_input_mode),
        "llm_image_overrides": {
            "max_side": args.llm_image_max_side,
            "max_bytes": args.llm_image_max_bytes,
            "jpeg_quality": args.llm_image_jpeg_quality,
            "bev_map_output_size_px": args.bev_map_output_size_px,
        },
        "frames": [],
    }

    bev_map_provider_session = None
    if resolved_enable_bev_map_provider:
        from mcp_server.route_context.bev_map_provider import BEVMapProviderSession

        bev_map_provider_session = BEVMapProviderSession.from_env()
        await bev_map_provider_session.connect()

    frame_jobs, preflight_failures = _build_frame_jobs(
        selected=selected,
        frames_by_name=frames_by_name,
        sam3_output_base_dir=sam3_output_base_dir,
        sequence_name=str(args.sequence_name),
        camera_view=str(args.camera_view),
        json_dir=json_dir,
        overlays_dir=overlays_dir,
        segmentation_policy=segmentation_policy,
    )
    navigation_path_gps = build_navigation_path_gps(
        selected_frame_names=selected,
        frames_by_name=frames_by_name,
    )
    manifest["navigation_path_gps_points"] = len(navigation_path_gps)
    llm_concurrency = max(1, int(args.llm_concurrency or 1))
    manifest["llm_concurrency"] = int(llm_concurrency)
    try:
        async def _worker(_index: int, job: FrameJob) -> dict[str, Any]:
            return await _process_frame_job(
                job=job,
                route_length_m=float(args.route_length_m),
                route_step_m=float(args.route_step_m),
                lookahead_m=float(args.lookahead_m),
                off_route_threshold_m=float(args.off_route_threshold_m),
                front_extent_m=float(args.front_extent_m),
                back_extent_m=float(args.back_extent_m),
                side_extent_m=float(args.side_extent_m),
                navigation_mode=str(args.mode),
                llm_detail=str(args.llm_detail),
                enable_verifier=bool(args.enable_verifier),
                enable_segmentation_overlay_input=bool(args.enable_segmentation_overlay_input),
                enable_som_overlay_input=bool(resolved_enable_som_overlay_input),
                enable_sam_visualization_input=bool(resolved_enable_sam_visualization_input),
                enable_bev_map_provider=bool(resolved_enable_bev_map_provider),
                bev_map_provider_session=bev_map_provider_session,
                llm_primary_image_role=str(resolved_llm_primary_image_role),
                enforce_alignment=bool(args.enforce_alignment),
                emit_qa_bundle=bool(args.emit_qa_bundle),
                emit_llm_input_images=bool(args.emit_llm_input_images),
                navigation_path_gps=navigation_path_gps,
                seg_vis_dir=seg_vis_dir,
                llm_inputs_dir=llm_inputs_dir,
                json_dir=json_dir,
                generator_json_dir=generator_json_dir,
                overlays_dir=overlays_dir,
            )

        results = await run_ordered_bounded(frame_jobs, _worker, concurrency=llm_concurrency)
        for result in results:
            if result.ok and isinstance(result.value, dict):
                manifest["frames"].append(result.value)
            else:
                manifest["frames"].append(
                    {
                        "frame_name": getattr(result.item, "frame_name", str(result.item)),
                        "status": "error",
                        "error": result.error or "frame_worker_failed",
                    }
                )
    finally:
        if bev_map_provider_session is not None:
            manifest["bev_map_provider_cache"] = bev_map_provider_session.cache_stats()
            await bev_map_provider_session.disconnect()

    failures = sum(1 for frame_item in manifest["frames"] if frame_item.get("status") != "ok")
    manifest["preflight_frames_failed"] = int(preflight_failures)
    manifest["frames_failed"] = failures
    manifest["frames_succeeded"] = max(len(manifest["frames"]) - failures, 0)
    manifest["pipeline_elapsed_ms"] = elapsed_ms(pipeline_started_at)
    manifest["timing_summary"] = summarize_visual_prompt_manifest_timings(manifest["frames"])
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = manifest.get("timing_summary") or {}
    runner_summary = summary.get("runner_elapsed_ms") if isinstance(summary, dict) else {}
    stages_summary = summary.get("stages") if isinstance(summary, dict) else {}
    if isinstance(runner_summary, dict):
        print(
            "[TIME] summary "
            f"frames={summary.get('frames_profiled')} "
            f"runner_avg={runner_summary.get('avg_ms')}ms "
            f"runner_p95={runner_summary.get('p95_ms')}ms "
            f"pipeline={manifest['pipeline_elapsed_ms']}ms"
        )
    if isinstance(stages_summary, dict):
        for stage_name in ("generator_llm_ms", "segmentation_ms", "bev_map_overlay_ms", "artifact_write_ms"):
            stage = stages_summary.get(stage_name)
            if isinstance(stage, dict):
                print(
                    "[TIME] stage "
                    f"{stage_name} avg={stage.get('avg_ms')}ms "
                    f"p95={stage.get('p95_ms')}ms "
                    f"max={stage.get('max_ms')}ms"
                )
    print(f"[OK] Sequence done. Manifest: {manifest_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
