#!/usr/bin/env python3
"""Evaluate SideWalk Phase1 dataset artifacts and optional model predictions.

The script runs before training to catch broken paths, malformed trajectories,
and weak BEV labels.  It can also score a prediction JSONL from ms-swift
inference if rows are aligned with the Phase1 source JSONL.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def evaluate_artifacts(
    source_jsonl: str,
    *,
    prediction_jsonl: str | None = None,
    qa_root: str | None = None,
    summary_json: str | None = None,
    output_jsonl: str | None = None,
    expected_points: int = 16,
) -> dict[str, Any]:
    references = list(read_jsonl(Path(source_jsonl)))
    predictions = list(read_jsonl(Path(prediction_jsonl))) if prediction_jsonl else None
    if predictions is not None and len(predictions) != len(references):
        raise ValueError(f"Prediction/reference length mismatch: {len(predictions)} != {len(references)}")

    rows: list[dict[str, Any]] = []
    for idx, ref in enumerate(references):
        pred = predictions[idx] if predictions is not None else None
        row = evaluate_one(ref, pred=pred, expected_points=expected_points)
        row["index"] = idx
        rows.append(row)

    summary = summarize(rows)
    if qa_root:
        summary["qa_bundle_summary"] = summarize_qa_bundles(Path(qa_root))
    if output_jsonl:
        write_jsonl(Path(output_jsonl), rows)
    if summary_json:
        path = Path(summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def evaluate_one(ref: dict[str, Any], *, pred: dict[str, Any] | None, expected_points: int) -> dict[str, Any]:
    meta = ref.get("sidewalk_meta") or {}
    image_width = int(ref.get("image_width") or meta.get("image_width") or 1920)
    image_height = int(ref.get("image_height") or meta.get("image_height") or 1080)
    target = np.asarray(ref.get("pixel_trajectory") or [], dtype=np.float32).reshape(-1, 2)
    missing_paths = []
    for key in ("fpv_image_path", "bev_image_path", "trajectory_label_path", "bev_patch_meta_path", "qa_bundle_path"):
        value = ref.get(key) or meta.get(key)
        if value and not Path(str(value)).expanduser().is_file():
            missing_paths.append(key)

    target_ok = target.shape == (int(expected_points), 2)
    in_bounds = bool(target.size > 0 and np.all(np.isfinite(target)) and np.min(target) >= -1e-4 and np.max(target) <= 1.0001)
    row: dict[str, Any] = {
        "source_id": ref.get("id") or meta.get("source_id"),
        "clip_id": ref.get("clip_id") or meta.get("clip_id"),
        "frame_idx": ref.get("frame_idx") or meta.get("frame_idx"),
        "missing_path_count": float(len(missing_paths)),
        "missing_paths": missing_paths,
        "target_shape_ok": float(target_ok),
        "target_in_bounds": float(in_bounds),
        "target_points": float(target.shape[0]),
        "intent_gt": meta.get("intent_gt") or (ref.get("route_context") or {}).get("intent_gt"),
        "route_theta_deg": _optional_float(meta.get("route_theta_deg") or (ref.get("route_context") or {}).get("route_theta_deg")),
        "image_width": float(image_width),
        "image_height": float(image_height),
    }
    if pred is not None:
        pred_arr = extract_prediction_trajectory(pred)
        parse_success = pred_arr is not None and pred_arr.size > 0
        if parse_success:
            pred_norm = normalize_prediction(pred_arr, image_width, image_height)
            ade, fde = ade_fde(pred_norm, target)
        else:
            ade = fde = float("nan")
        row.update(
            {
                "prediction_parse_success": float(parse_success),
                "ade": ade,
                "fde": fde,
            }
        )
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_samples": float(len(rows)),
        "missing_path_rate": mean(row["missing_path_count"] > 0 for row in rows),
        "target_shape_ok": mean(row["target_shape_ok"] for row in rows),
        "target_in_bounds": mean(row["target_in_bounds"] for row in rows),
    }
    if rows and "prediction_parse_success" in rows[0]:
        summary["prediction_parse_success"] = mean(row.get("prediction_parse_success") for row in rows)
        summary["ade"] = mean(row.get("ade") for row in rows)
        summary["fde"] = mean(row.get("fde") for row in rows)
    intent_counts: dict[str, int] = {}
    for row in rows:
        intent = str(row.get("intent_gt") or "unknown")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    summary["intent_counts"] = intent_counts
    return summary


def summarize_qa_bundles(root: Path) -> dict[str, Any]:
    paths = list(iter_qa_bundle_paths(root))
    task_counts: dict[str, int] = {}
    invalid = 0
    samples = 0
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            invalid += 1
            continue
        if data.get("schema_version") != "sidewalk_bev_fpv_qa_v1":
            invalid += 1
        for sample in data.get("samples") or []:
            if not isinstance(sample, dict):
                continue
            samples += 1
            task = str(sample.get("task_type") or "unknown")
            task_counts[task] = task_counts.get(task, 0) + 1
    return {
        "num_bundles": float(len(paths)),
        "invalid_bundles": float(invalid),
        "num_samples": float(samples),
        "task_counts": task_counts,
    }


def iter_qa_bundle_paths(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    yield from sorted(root.rglob("*_qa.json"))


def extract_prediction_trajectory(row: dict[str, Any]) -> np.ndarray | None:
    for key in ("trajectory_norm", "trajectory", "trajectory_pixel", "pred_norm", "points"):
        if row.get(key) is not None:
            return np.asarray(row[key], dtype=np.float32).reshape(-1, 2)
    text = str(row.get("response") or row.get("trajectory_text") or row.get("prediction") or "")
    if not text:
        messages = row.get("messages") or []
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                text = str(message.get("content") or "")
                break
    return parse_points(text)


def parse_points(text: str) -> np.ndarray | None:
    pairs = re.findall(r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?", text)
    if not pairs:
        return None
    return np.asarray([[float(x), float(y)] for x, y in pairs], dtype=np.float32).reshape(-1, 2)


def normalize_prediction(points: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 2).copy()
    if arr.size == 0:
        return arr
    if float(np.nanmax(arr)) > 2.0:
        arr[:, 0] /= float(image_width)
        arr[:, 1] /= float(image_height)
    return arr


def ade_fde(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float32).reshape(-1, 2)
    if pred.size == 0 or target.size == 0:
        return float("nan"), float("nan")
    n = min(pred.shape[0], target.shape[0])
    dist = np.linalg.norm(pred[:n] - target[:n], axis=1)
    return float(np.mean(dist)), float(dist[-1])


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(values: Iterable[Any]) -> float:
    vals = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except Exception:
            number = 1.0 if bool(value) else 0.0
        if not math.isnan(number):
            vals.append(number)
    return float(np.mean(vals)) if vals else float("nan")


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SideWalk Phase1 artifacts")
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--predictions", default=None, help="Optional aligned prediction JSONL")
    parser.add_argument("--qa-root", default=None, help="Optional QA bundle root")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--expected-points", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_artifacts(
        args.source_jsonl,
        prediction_jsonl=args.predictions,
        qa_root=args.qa_root,
        summary_json=args.summary_json,
        output_jsonl=args.output_jsonl,
        expected_points=int(args.expected_points),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
