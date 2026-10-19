#!/usr/bin/env python3
"""Run SideWalk Phase1 CoT inference and render predicted FPV trajectories.

This runner is intentionally separate from the RouteServer visual-prompt
pipeline.  The visual-prompt service has its own online safety contract, while
Phase1 dataset evaluation needs the same input/output grammar as ms-swift SFT:

    FPV + BEV -> <think>...</think><finish>{"trajectory": ...}</finish>

Two BEV input modes are supported:

* ``phase1_source`` reads ``phase1_source_*.jsonl`` and uses the exact
  ``fpv_image_path`` / ``bev_image_path`` pair produced by the dataset exporter.
* ``online_bev`` keeps compatibility with older RouteServer smoke tests by
  rendering an online OSM BEV image from ``trajectory.json``.  Only the BEV
  source changes; the VLM prompt and CoT output contract remain Phase1 format.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image as PILImage


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODEL_PROCESSING_DIR = Path(__file__).resolve().parent
if str(MODEL_PROCESSING_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_PROCESSING_DIR))

from mcp_server.llm import VisionAnalyzer, local_image_path_to_data_url  # noqa: E402
from mcp_server.visualization.map_overlay import _build_bev_map_overlay_payload
from mcp_server.visualization.images import _image_href_for_overlay
from mcp_server.visualization.prompt_overlay import _render_prompt_overlay_svg
from sidewalk_phase1_cot_teacher import (  # noqa: E402
    THINK_SECTION_TAGS,
    build_training_prompt,
    extract_tag,
    parse_json_object,
    validate_tag_layout,
)
from run_sidewalk_images_visual_prompt_pipeline import (  # noqa: E402
    _iter_selected_frame_names,
    _load_trajectory_frames,
    _write_data_url_image,
    build_navigation_path_gps,
)


SCHEMA_VERSION = "sidewalk_phase1_cot_baseline_v1"


@dataclass(frozen=True)
class BaselineSample:
    row: dict[str, Any]
    fpv_image_path: Path
    bev_image_path: Path | None
    bev_data_url: str | None = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_baseline_system_prompt() -> str:
    return (
        "You are a sidewalk navigation assistant for Phase1 structured CoT inference. "
        "Use the FPV image and the aligned BEV image to infer safe local navigation. "
        "Fully use both FPV and BEV environmental information and understand their cross-view relationship. "
        "Return only the required XML-like tags. Put structured observation and planning JSON sections inside <think>; "
        "put only the executable normalized FPV trajectory JSON inside <finish>. "
        "The <FPVScene> section is mandatory: first identify navigable_support, conditional_support, and risk_objects, "
        "then ensure the final trajectory stays on the listed navigable support and avoids the listed risks. "
        "Do not use markdown, prose outside tags, path_corridor fields, left/right boundaries, masks, or region ids. "
        "The trajectory is a short local FPV visual prompt. Every point must stay on visible walkable ground support. "
        "The final point should normally remain in the near-field lower half of the FPV image unless the ground support is visibly continuous and unobstructed. "
        "Never extend trajectory points into sky, trees, walls, fences, building facades, vehicles, pedestrians, or far background. "
        "If the safe sidewalk support becomes occluded or blocked, shorten the path or stop before the unsafe region."
    )


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int | None,
    frame_start: int | None,
    frame_end: int | None,
    frame_step: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        frame_idx = row.get("frame_idx")
        try:
            idx = int(frame_idx)
        except Exception:
            idx = None
        if idx is not None:
            if frame_start is not None and idx < int(frame_start):
                continue
            if frame_end is not None and idx > int(frame_end):
                continue
        selected.append(row)
    step = max(int(frame_step), 1)
    selected = selected[::step]
    if limit is not None:
        selected = selected[: max(int(limit), 0)]
    return selected


def build_phase1_source_samples(
    *,
    source_jsonl: Path,
    limit: int | None,
    frame_start: int | None,
    frame_end: int | None,
    frame_step: int,
) -> list[BaselineSample]:
    rows = _select_rows(
        read_jsonl(source_jsonl),
        limit=limit,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_step=frame_step,
    )
    samples: list[BaselineSample] = []
    for row in rows:
        fpv = Path(str(row.get("fpv_image_path") or "")).expanduser()
        bev_raw = str(row.get("bev_image_path") or "").strip()
        bev = Path(bev_raw).expanduser() if bev_raw else None
        if not fpv.exists():
            raise FileNotFoundError(f"fpv_image_path not found for {row.get('id')}: {fpv}")
        if bev is not None and not bev.exists():
            raise FileNotFoundError(f"bev_image_path not found for {row.get('id')}: {bev}")
        samples.append(BaselineSample(row=dict(row), fpv_image_path=fpv, bev_image_path=bev))
    return samples


def build_online_bev_samples(
    *,
    images_dir: Path,
    trajectory_json: Path,
    output_dir: Path,
    limit: int | None,
    frame_start: int | None,
    frame_end: int | None,
    frame_step: int,
    instruction: str,
) -> list[BaselineSample]:
    frames_by_name = _load_trajectory_frames(trajectory_path=trajectory_json, images_dir=images_dir)
    selected = _iter_selected_frame_names(
        images_dir=images_dir,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_step=frame_step,
    )
    if limit is not None:
        selected = selected[: max(int(limit), 0)]
    navigation_path_gps = build_navigation_path_gps(selected_frame_names=selected, frames_by_name=frames_by_name)
    online_bev_dir = output_dir / "online_bev_inputs"
    online_bev_dir.mkdir(parents=True, exist_ok=True)
    samples: list[BaselineSample] = []
    for frame_name in selected:
        frame = frames_by_name.get(frame_name)
        if frame is None:
            continue
        with PILImage.open(frame.image_path) as im:
            width, height = im.size
        local_bev_context = {
            "_origin_gps": {"lat": float(frame.lat), "lng": float(frame.lng)},
            "origin_gps": {"lat": float(frame.lat), "lng": float(frame.lng)},
            "gps": {"lat": float(frame.lat), "lng": float(frame.lng)},
            "bev_rotation_deg": float(frame.heading_deg or 0.0),
            "navigation_path_gps": navigation_path_gps,
            "navigation_path_source": "offline_trajectory",
        }
        payload = _build_bev_map_overlay_payload(
            local_bev_context=local_bev_context,
            llm_max_bytes=int(os.getenv("VISUAL_PROMPT_LLM_IMAGE_MAX_BYTES", "2500000") or 2500000),
            llm_jpeg_quality=int(os.getenv("VISUAL_PROMPT_LLM_IMAGE_JPEG_QUALITY", "85") or 85),
        )
        bev_url = str((payload or {}).get("bev_map_overlay_image_data_url") or "").strip()
        bev_path = _write_data_url_image(bev_url, online_bev_dir / f"{frame_name}.online_bev") if bev_url else None
        row = {
            "id": f"{trajectory_json.stem}_{frame_name}",
            "clip_id": trajectory_json.parent.name,
            "frame_name": str(frame_name),
            "frame_idx": int(frame_name) if str(frame_name).isdigit() else frame_name,
            "fpv_image_path": str(frame.image_path),
            "bev_image_path": str(bev_path) if bev_path else "",
            "image_width": int(width),
            "image_height": int(height),
            "instruction": instruction,
            "route_context": {},
            "sidewalk_meta": {
                "bev_mapping_source": "online_bev_provider",
                "lat": float(frame.lat),
                "lng": float(frame.lng),
                "heading_deg": float(frame.heading_deg or 0.0),
            },
        }
        samples.append(BaselineSample(row=row, fpv_image_path=frame.image_path, bev_image_path=bev_path, bev_data_url=bev_url))
    return samples


def _round_float(value: float, digits: int = 6) -> float:
    return round(float(value), int(digits))


def _normalize_trajectory(value: Any) -> list[list[float]] | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    if arr.shape[0] < 2 or not np.all(np.isfinite(arr)):
        return None
    return [[_round_float(x), _round_float(y)] for x, y in arr.tolist()]


def parse_cot_answer(answer: str) -> tuple[dict[str, Any], list[str]]:
    errors = validate_tag_layout(answer)
    sections: dict[str, Any] = {}
    think_body = extract_tag(answer, "think")
    if think_body is None:
        errors.append("missing_tag:think")
    else:
        sections["think"] = think_body
    for tag in THINK_SECTION_TAGS:
        body = extract_tag(think_body or "", tag)
        if body is None:
            errors.append(f"missing_tag:{tag}")
            continue
        parsed = parse_json_object(body)
        if parsed is None:
            errors.append(f"invalid_json:{tag}")
        else:
            sections[tag] = parsed
    finish_body = extract_tag(answer, "finish")
    if finish_body is None:
        errors.append("missing_tag:finish")
    else:
        finish = parse_json_object(finish_body)
        if finish is None:
            errors.append("invalid_json:finish")
        else:
            sections["finish"] = finish
    finish = sections.get("finish") if isinstance(sections.get("finish"), dict) else {}
    trajectory = _normalize_trajectory((finish or {}).get("trajectory"))
    if trajectory is None:
        errors.append("missing_or_invalid_finish_trajectory")
    else:
        sections["trajectory_norm"] = trajectory
    return sections, sorted(set(errors))


def trajectory_to_rendered_prompt(
    trajectory_norm: list[list[float]],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    path_uv = [
        [round(float(x) * float(image_width), 2), round(float(y) * float(image_height), 2)]
        for x, y in trajectory_norm
    ]
    direction = None
    if len(path_uv) >= 2:
        sx, sy = path_uv[-2]
        ex, ey = path_uv[-1]
        dx = float(ex) - float(sx)
        dy = float(ey) - float(sy)
        norm = math.hypot(dx, dy)
        if norm > 1e-6:
            direction = [round(dx / norm, 4), round(dy / norm, 4)]
    return {
        "type": "arrow_path",
        "center_uv": path_uv[-1],
        "direction_vec": direction,
        "span_px": 72.0,
        "path_uv": path_uv,
        "termination_semantics": "continue_visible_corridor",
        "support_confidence": 0.0,
    }


def _resample_polyline(points: np.ndarray, num_samples: int) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points, int(num_samples), axis=0)
    seg = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumulative[-1])
    if total <= 1e-8:
        return np.repeat(points[:1], int(num_samples), axis=0)
    target = np.linspace(0.0, total, int(num_samples), dtype=np.float32)
    x = np.interp(target, cumulative, points[:, 0])
    y = np.interp(target, cumulative, points[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def compute_trajectory_metrics(
    *,
    pred_norm: list[list[float]] | None,
    gt_norm: Any,
    curve_samples: int = 100,
) -> dict[str, Any] | None:
    if pred_norm is None:
        return None
    try:
        pred = np.asarray(pred_norm, dtype=np.float32).reshape(-1, 2)
        gt = np.asarray(gt_norm, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    if pred.shape[0] < 2 or gt.shape[0] < 2:
        return None
    metrics: dict[str, Any] = {
        "pred_points": int(pred.shape[0]),
        "gt_points": int(gt.shape[0]),
        "point_count_match": bool(pred.shape == gt.shape),
    }
    if pred.shape == gt.shape:
        point_dist = np.linalg.norm(pred - gt, axis=1)
        metrics["point_ade_norm"] = _round_float(float(np.mean(point_dist)))
        metrics["point_fde_norm"] = _round_float(float(point_dist[-1]))
        metrics["max_abs_delta_norm"] = _round_float(float(np.max(np.abs(pred - gt))))
    pred_curve = _resample_polyline(pred, int(curve_samples))
    gt_curve = _resample_polyline(gt, int(curve_samples))
    curve_dist = np.linalg.norm(pred_curve - gt_curve, axis=1)
    metrics["curve_samples"] = int(curve_samples)
    metrics["curve_ade_norm"] = _round_float(float(np.mean(curve_dist)))
    metrics["curve_fde_norm"] = _round_float(float(np.linalg.norm(pred_curve[-1] - gt_curve[-1])))
    metrics["curve_max_error_norm"] = _round_float(float(np.max(curve_dist)))
    return metrics


def trajectory_sanity_warnings(
    *,
    pred_norm: list[list[float]] | None,
    gt_norm: Any,
    min_endpoint_y: float = 0.45,
    max_gt_endpoint_y_delta: float = 0.25,
) -> list[str]:
    """Return warnings for impossible local visual-prompt trajectories.

    These checks never repair the prediction. They only make failure modes such
    as "trajectory climbs into the sky" visible in the baseline artifacts.
    """

    if pred_norm is None:
        return []
    try:
        pred = np.asarray(pred_norm, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return ["trajectory_unreadable_for_sanity_check"]
    if pred.shape[0] < 2 or not np.all(np.isfinite(pred)):
        return ["trajectory_unreadable_for_sanity_check"]
    warnings: list[str] = []
    first_x, first_y = [float(v) for v in pred[0]]
    if not (0.35 <= first_x <= 0.65 and first_y >= 0.75):
        warnings.append(f"start_not_near_bottom_center:first=({first_x:.3f},{first_y:.3f})")
    endpoint_y = float(pred[-1, 1])
    min_y = float(np.min(pred[:, 1]))
    if min_y < float(min_endpoint_y):
        warnings.append(f"trajectory_extends_above_local_ground_band:min_y={min_y:.3f}<threshold={min_endpoint_y:.3f}")
    try:
        gt = np.asarray(gt_norm, dtype=np.float32).reshape(-1, 2)
        if gt.shape[0] >= 2 and np.all(np.isfinite(gt)):
            gt_endpoint_y = float(gt[-1, 1])
            if gt_endpoint_y - endpoint_y > float(max_gt_endpoint_y_delta):
                warnings.append(
                    "endpoint_much_higher_than_geometry_gt:"
                    f"pred_y={endpoint_y:.3f},gt_y={gt_endpoint_y:.3f},delta={gt_endpoint_y - endpoint_y:.3f}"
                )
    except Exception:
        pass
    return warnings


def _render_overlay(
    *,
    image_path: Path,
    rendered_prompt: dict[str, Any] | None,
    title: str,
) -> str:
    with PILImage.open(image_path) as im:
        width, height = im.size
    image_href = _image_href_for_overlay(image_path=str(image_path), image_url=None)
    return _render_prompt_overlay_svg(
        image_href=image_href,
        image_width=int(width),
        image_height=int(height),
        rendered_prompt=rendered_prompt,
        title=title,
    )


async def run_sample(
    sample: BaselineSample,
    *,
    output_dir: Path,
    detail: str,
    input_mode: str,
) -> dict[str, Any]:
    analyzer = VisionAnalyzer()
    if not analyzer.is_available():
        raise RuntimeError("VLM baseline is not configured. Set ARK_API_KEY or LLM_API_KEY.")
    row = dict(sample.row)
    fpv_url = local_image_path_to_data_url(str(sample.fpv_image_path))
    extra_urls: list[str] = []
    if sample.bev_data_url:
        extra_urls.append(str(sample.bev_data_url))
    elif sample.bev_image_path is not None:
        extra_urls.append(local_image_path_to_data_url(str(sample.bev_image_path)))
    prompt = build_training_prompt(row)
    system_prompt = build_baseline_system_prompt()
    result = await analyzer.analyze(
        image_url=fpv_url,
        extra_image_urls=extra_urls,
        question=prompt,
        system_prompt=system_prompt,
        detail=detail,
    )
    answer = str(result.get("answer") or "").strip()
    sections, errors = parse_cot_answer(answer)
    trajectory_norm = sections.get("trajectory_norm") if isinstance(sections.get("trajectory_norm"), list) else None
    rendered_prompt = (
        trajectory_to_rendered_prompt(
            trajectory_norm,
            image_width=int(row.get("image_width") or 0),
            image_height=int(row.get("image_height") or 0),
        )
        if trajectory_norm is not None
        else None
    )
    metrics = compute_trajectory_metrics(pred_norm=trajectory_norm, gt_norm=row.get("pixel_trajectory"))
    trajectory_warnings = trajectory_sanity_warnings(
        pred_norm=trajectory_norm,
        gt_norm=row.get("pixel_trajectory"),
    )
    status = (
        "ok"
        if trajectory_norm is not None and not errors and not trajectory_warnings
        else "parse_error"
        if trajectory_norm is None
        else "format_warning"
    )
    frame_name = str(row.get("frame_name") or Path(str(row.get("fpv_image_path") or "")).stem or row.get("id"))
    json_path = output_dir / "json" / f"{frame_name}.json"
    overlay_path = output_dir / "overlay" / f"{frame_name}.svg"
    overlay_svg = _render_overlay(
        image_path=sample.fpv_image_path,
        rendered_prompt=rendered_prompt,
        title=f"SideWalk Phase1 CoT Baseline {frame_name}",
    )
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(overlay_svg, encoding="utf-8")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "input_mode": input_mode,
        "id": row.get("id"),
        "frame_name": frame_name,
        "fpv_image_path": str(sample.fpv_image_path),
        "bev_image_path": str(sample.bev_image_path) if sample.bev_image_path is not None else None,
        "system_prompt": system_prompt,
        "prompt": prompt,
        "response": answer,
        "parsed_cot": {
            key: value
            for key, value in sections.items()
            if key != "think"
        },
        "validation_errors": errors,
        "prediction": {
            "trajectory_norm": trajectory_norm,
            "rendered_prompt": rendered_prompt,
            "intent": (sections.get("finish") or {}).get("intent") if isinstance(sections.get("finish"), dict) else None,
            "route_theta_deg": (
                (sections.get("BEVContext") or {}).get("route_theta_deg")
                if isinstance(sections.get("BEVContext"), dict)
                else ((sections.get("finish") or {}).get("route_theta_deg") if isinstance(sections.get("finish"), dict) else None)
            ),
            "confidence_level": (
                (sections.get("finish") or {}).get("confidence_level")
                if isinstance(sections.get("finish"), dict)
                else None
            ),
        },
        "ground_truth": {
            "trajectory_norm": row.get("pixel_trajectory"),
            "intent": (row.get("route_context") or {}).get("intent_gt") or (row.get("sidewalk_meta") or {}).get("intent_gt"),
            "route_theta_deg": (row.get("route_context") or {}).get("route_theta_deg") or (row.get("sidewalk_meta") or {}).get("route_theta_deg"),
        },
        "metrics": metrics,
        "trajectory_warnings": trajectory_warnings,
        "overlay_path": str(overlay_path),
        "llm": {
            "provider": result.get("provider"),
            "model": result.get("model"),
            "api_style": result.get("api_style"),
            "telemetry": result.get("llm_telemetry"),
        },
    }
    write_json(json_path, payload)
    return {
        "id": row.get("id"),
        "frame_name": frame_name,
        "json_path": str(json_path),
        "overlay_path": str(overlay_path),
        "status": status,
        "validation_errors": errors,
        "trajectory_warnings": trajectory_warnings,
        "metrics": metrics,
    }


async def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_mode = str(args.input_mode)
    if input_mode == "phase1_source":
        if not args.source_jsonl:
            raise RuntimeError("--source-jsonl is required for --input-mode phase1_source")
        samples = build_phase1_source_samples(
            source_jsonl=Path(args.source_jsonl).expanduser().resolve(),
            limit=args.limit,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            frame_step=args.frame_step,
        )
    else:
        if not args.images_dir or not args.trajectory_json:
            raise RuntimeError("--images-dir and --trajectory-json are required for --input-mode online_bev")
        samples = build_online_bev_samples(
            images_dir=Path(args.images_dir).expanduser().resolve(),
            trajectory_json=Path(args.trajectory_json).expanduser().resolve(),
            output_dir=output_dir,
            limit=args.limit,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            frame_step=args.frame_step,
            instruction=str(args.instruction),
        )
    manifest: dict[str, Any] = {
        "schema_version": f"{SCHEMA_VERSION}_manifest",
        "input_mode": input_mode,
        "output_dir": str(output_dir),
        "frames_total": len(samples),
        "frames_failed": 0,
        "frames": [],
    }
    for sample in samples:
        try:
            item = await run_sample(sample, output_dir=output_dir, detail=args.detail, input_mode=input_mode)
        except Exception as exc:
            item = {
                "id": sample.row.get("id"),
                "frame_name": sample.row.get("frame_name"),
                "status": "error",
                "error": f"{type(exc).__name__}:{exc}",
            }
        if item.get("status") not in {"ok", "format_warning"}:
            manifest["frames_failed"] += 1
        manifest["frames"].append(item)
        print(f"[{item.get('status')}] {item.get('frame_name')} -> {item.get('json_path')}")
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SideWalk Phase1 CoT baseline inference")
    parser.add_argument("--input-mode", default="phase1_source", choices=["phase1_source", "online_bev"])
    parser.add_argument("--source-jsonl", default=None, help="Phase1 source JSONL with FPV/BEV pairs")
    parser.add_argument("--images-dir", default=None, help="Online-BEV mode images directory")
    parser.add_argument("--trajectory-json", default=None, help="Online-BEV mode trajectory.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--detail", default="auto", choices=["auto", "low", "high"])
    parser.add_argument("--instruction", default="Navigate safely on the sidewalk and follow the visible route.")
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_baseline(parse_args()))


if __name__ == "__main__":
    main()
