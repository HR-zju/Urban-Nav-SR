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
    return Path(__file__).resolve().parents[1]


# Ensure `import mcp_server...` works even when the package is not installed.
_SRC_DIR = _repo_root() / "src"
if str(_SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(_SRC_DIR))


@dataclass(frozen=True)
class TrajectoryFrame:
    frame_name: str
    lat: float
    lng: float
    heading_deg: float | None
    image_path: Path


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
    enforce_alignment: bool,
    emit_qa_bundle: bool,
    emit_llm_input_images: bool,
    fpv_segmentation: dict[str, Any] | None,
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
        lookahead_m=float(lookahead_m),
        off_route_threshold_m=float(off_route_threshold_m),
        front_extent_m=float(front_extent_m),
        back_extent_m=float(back_extent_m),
        side_extent_m=float(side_extent_m),
        navigation_mode=str(navigation_mode),
        llm_detail=str(llm_detail),
        enable_verifier=bool(enable_verifier),
        enable_segmentation_overlay_input=bool(enable_segmentation_overlay_input),
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
    parser.add_argument("--enable-verifier", action=argparse.BooleanOptionalAction, default=True, help="Enable verifier LVLM call")
    parser.add_argument(
        "--enable-segmentation-overlay-input",
        action="store_true",
        help="Append segmentation visualization as extra image input to the arbiter when available",
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
        "--allow-missing-segmentation",
        action="store_true",
        help="If SAM3 artifacts for a frame are missing, proceed without segmentation instead of failing",
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir).expanduser().resolve()
    trajectory_path = Path(args.trajectory_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    sam3_output_base_dir = Path(args.sam3_output_base_dir).expanduser().resolve()

    if not images_dir.exists():
        raise RuntimeError(f"images-dir not found: {images_dir}")
    if not trajectory_path.exists():
        raise RuntimeError(f"trajectory-json not found: {trajectory_path}")

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

    if args.run_sam3:
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

    manifest: dict[str, Any] = {
        "images_dir": str(images_dir),
        "trajectory_json": str(trajectory_path),
        "sam3_output_base_dir": str(sam3_output_base_dir),
        "sequence_name": str(args.sequence_name),
        "camera_view": str(args.camera_view),
        "frames_total": len(selected),
        "frames_failed": 0,
        "frames": [],
    }

    failures = 0
    for frame_name in selected:
        frame = frames_by_name.get(frame_name)
        if frame is None:
            failures += 1
            manifest["frames"].append(
                {"frame_name": frame_name, "status": "error", "error": "frame_missing_in_trajectory"}
            )
            continue
        if not frame.image_path.exists():
            failures += 1
            manifest["frames"].append(
                {"frame_name": frame_name, "status": "error", "error": f"image_missing:{frame.image_path}"}
            )
            continue

        fpv_seg = _segmentation_artifacts_for_frame(
            sam3_output_base_dir=sam3_output_base_dir,
            sequence_name=str(args.sequence_name),
            camera_view=str(args.camera_view),
            frame_name=str(frame_name),
        )
        if fpv_seg is None and not args.allow_missing_segmentation:
            failures += 1
            manifest["frames"].append(
                {"frame_name": frame_name, "status": "error", "error": "missing_segmentation_artifacts"}
            )
            continue

        out_json = json_dir / f"{frame_name}.json"
        out_generator_json = generator_json_dir / f"{frame_name}.generator.json"
        out_svg = overlays_dir / f"{frame_name}.svg"
        out_generator_svg = overlays_dir / f"{frame_name}.generator.svg"
        out_rejected_svg = overlays_dir / f"{frame_name}.rejected.svg"
        out_seg_vis = seg_vis_dir / f"{frame_name}_visualization.jpg"

        frame_item: dict[str, Any] = {
            "frame_name": str(frame_name),
            "image_path": str(frame.image_path),
            "json_path": str(out_json),
            "overlay_path": str(out_svg),
            "status": "ok",
        }

        try:
            # Keep segmentation visualization close to outputs for quick review.
            if fpv_seg and isinstance(fpv_seg.get("visualization_path"), str):
                src_vis = Path(str(fpv_seg.get("visualization_path"))).expanduser()
                if src_vis.exists():
                    if out_seg_vis.exists() or out_seg_vis.is_symlink():
                        out_seg_vis.unlink()
                    try:
                        out_seg_vis.symlink_to(src_vis)
                    except Exception:
                        out_seg_vis.write_bytes(src_vis.read_bytes())
            payload = await _run_frame(
                frame=frame,
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
                enforce_alignment=bool(args.enforce_alignment),
                emit_qa_bundle=bool(args.emit_qa_bundle),
                emit_llm_input_images=True,
                fpv_segmentation=fpv_seg,
            )
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
            if isinstance(overlay_svg, str) and overlay_svg.strip():
                out_svg.write_text(overlay_svg, encoding="utf-8")
            else:
                out_svg.write_text("", encoding="utf-8")

            generator_overlay_svg = payload.get("generator_overlay_svg")
            if isinstance(generator_overlay_svg, str) and generator_overlay_svg.strip():
                out_generator_svg.write_text(generator_overlay_svg, encoding="utf-8")
                frame_item["generator_overlay_path"] = str(out_generator_svg)
            else:
                # Keep directory tidy; only write when available.
                if out_generator_svg.exists():
                    out_generator_svg.unlink()

            # If the final decision is a segmentation reject, also export the rejected prompt overlay
            # so that the user can inspect what the LVLM generated before constraints.
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
                out_generator_json.write_text(
                    json.dumps(generator_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
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
            else:
                if out_rejected_svg.exists():
                    out_rejected_svg.unlink()
        except Exception as exc:
            frame_item["status"] = "error"
            frame_item["error"] = str(exc)
            failures += 1

        manifest["frames"].append(frame_item)

    manifest["frames_failed"] = failures
    manifest["frames_succeeded"] = max(len(manifest["frames"]) - failures, 0)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Sequence done. Manifest: {manifest_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
