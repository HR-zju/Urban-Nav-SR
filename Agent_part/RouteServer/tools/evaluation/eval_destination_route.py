#!/usr/bin/env python3
"""Evaluate destination recommendation and route quality.

The design follows CitySeeker's separation between:
  - destination / stop success: did the agent end at the intended target?
  - route quality: how many steps/meters did it spend compared with GT?

This evaluator is deliberately offline. It does not call map providers or LLMs.
It compares saved RouteServer predictions against a JSON/JSONL ground-truth file.

Ground truth can be either CitySeeker-like JSON:
  [
    {
      "question_idx": 0,
      "idx": 2,
      "question": "...",
      "from": "00001864.jpg",
      "to": "00002415.jpg",
      "total_weight": 131.16,
      "total_steps": 10,
      "complete_route": [{"filename": "...", "latitude": 39.9, "longitude": 116.3}]
    }
  ]

or UrbanNav-like JSONL/JSON:
  {"id":"case-1","origin":{"lat":...,"lng":...},"gt_destination":{"lat":...,"lng":...},"gt_route":[...]}

Predictions can be saved RouteServer outputs containing:
  - chosen_destination
  - candidate_destinations / recommendations
  - route_polyline
  - route_summary
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


EARTH_RADIUS_M = 6371000.0


@dataclass(frozen=True)
class Point:
    lat: float
    lng: float
    node_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class Case:
    case_id: str
    question: str | None
    category: str | None
    origin: Point | None
    gt_destination: Point | None
    gt_route: list[Point]
    gt_distance_m: float | None
    gt_steps: int | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class Prediction:
    case_id: str
    pred_destination: Point | None
    ranked_destinations: list[Point]
    route: list[Point]
    route_distance_m: float | None
    route_steps: int | None
    reached_flag: bool | None
    raw: dict[str, Any]


def _load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("data", "cases", "records", "predictions", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    raise RuntimeError(f"Unsupported JSON root in {path}: {type(data)}")


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _str_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _pick(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw and raw[key] not in (None, ""):
            return raw[key]
    return None


def _split_lnglat(text: str) -> tuple[float, float] | None:
    try:
        a, b = str(text).split(",", 1)
        lng = float(a.strip())
        lat = float(b.strip())
    except Exception:
        return None
    if not (math.isfinite(lat) and math.isfinite(lng)):
        return None
    return lat, lng


def point_from_any(value: Any) -> Point | None:
    if value is None:
        return None
    if isinstance(value, Point):
        return value
    if isinstance(value, str):
        parsed = _split_lnglat(value)
        if parsed is None:
            return None
        lat, lng = parsed
        return Point(lat=lat, lng=lng)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        # RouteServer route_polyline lists are usually [lng, lat].
        lng = _finite_float(value[0])
        lat = _finite_float(value[1])
        if lat is None or lng is None:
            return None
        return Point(lat=lat, lng=lng)
    if not isinstance(value, dict):
        return None

    coordinates = value.get("coordinates") if isinstance(value.get("coordinates"), dict) else {}
    location = value.get("location")
    if isinstance(location, str):
        parsed = _split_lnglat(location)
        if parsed is not None:
            lat, lng = parsed
            return Point(
                lat=lat,
                lng=lng,
                node_id=_str_or_none(_pick(value, "id", "provider_poi_id", "filename", "pano_id")),
                name=_str_or_none(_pick(value, "name", "display_name", "to")),
            )

    lat = _finite_float(_pick(value, "lat", "latitude", "y") or coordinates.get("latitude") or coordinates.get("lat"))
    lng = _finite_float(_pick(value, "lng", "lon", "longitude", "x") or coordinates.get("longitude") or coordinates.get("lng") or coordinates.get("lon"))
    if lat is None or lng is None:
        return None
    return Point(
        lat=lat,
        lng=lng,
        node_id=_str_or_none(_pick(value, "id", "provider_poi_id", "filename", "pano_id")),
        name=_str_or_none(_pick(value, "name", "display_name", "to")),
    )


def route_from_any(value: Any) -> list[Point]:
    if not isinstance(value, list):
        return []
    points: list[Point] = []
    for item in value:
        point = point_from_any(item)
        if point is not None:
            points.append(point)
    return points


def case_id_from_record(raw: dict[str, Any]) -> str:
    for key in ("id", "case_id", "sample_id", "qid", "question_id"):
        value = _str_or_none(raw.get(key))
        if value:
            return value
    question_idx = raw.get("question_idx")
    idx = raw.get("idx")
    round_id = raw.get("round")
    if question_idx is not None and idx is not None and round_id is not None:
        return f"{question_idx}:{idx}:{round_id}"
    if question_idx is not None and idx is not None:
        return f"{question_idx}:{idx}"
    if raw.get("question") and raw.get("from"):
        return f"{raw.get('question')}::{raw.get('from')}"
    raise RuntimeError(f"Cannot infer case id from record keys={sorted(raw.keys())}")


def _route_distance_m(points: list[Point]) -> float | None:
    if len(points) < 2:
        return None
    return sum(haversine_m(a, b) for a, b in zip(points, points[1:]))


def _summary_distance(raw: dict[str, Any]) -> float | None:
    route_summary = raw.get("route_summary") if isinstance(raw.get("route_summary"), dict) else {}
    for value in (
        raw.get("total_weight"),
        raw.get("distance_m"),
        route_summary.get("distance_m"),
        route_summary.get("distance"),
        route_summary.get("total_distance_m"),
    ):
        parsed = _finite_float(value)
        if parsed is not None:
            return parsed
    return None


def _summary_steps(raw: dict[str, Any], route: list[Point]) -> int | None:
    route_summary = raw.get("route_summary") if isinstance(raw.get("route_summary"), dict) else {}
    for value in (raw.get("total_steps"), raw.get("step_count"), route_summary.get("step_count"), route_summary.get("steps")):
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed >= 0:
            return parsed
    if route:
        return max(len(route) - 1, 0)
    return None


def parse_case(raw: dict[str, Any]) -> Case:
    route = route_from_any(_pick(raw, "gt_route", "complete_route", "route_polyline", "route"))
    gt_destination = point_from_any(_pick(raw, "gt_destination", "destination", "target"))
    if gt_destination is None and route:
        gt_destination = route[-1]
    origin = point_from_any(_pick(raw, "origin", "start"))
    if origin is None and route:
        origin = route[0]
    distance = _summary_distance(raw)
    if distance is None:
        distance = _route_distance_m(route)
    return Case(
        case_id=case_id_from_record(raw),
        question=_str_or_none(raw.get("question") or raw.get("prompt")),
        category=_str_or_none(raw.get("category") or raw.get("service") or raw.get("folder")),
        origin=origin,
        gt_destination=gt_destination,
        gt_route=route,
        gt_distance_m=distance,
        gt_steps=_summary_steps(raw, route),
        raw=raw,
    )


def _candidate_list(raw: dict[str, Any]) -> list[Any]:
    for key in ("candidate_destinations", "recommendations", "candidates", "ranked_destinations"):
        value = raw.get(key)
        if isinstance(value, list):
            return value
    # RouteServer result often stores chosen_destination separately only.
    chosen = raw.get("chosen_destination") or raw.get("pred_destination")
    return [chosen] if chosen else []


def _extract_prediction_payload(raw: dict[str, Any]) -> dict[str, Any]:
    for key in ("prediction", "result", "output", "recommendation"):
        value = raw.get(key)
        if isinstance(value, dict):
            merged = dict(value)
            for passthrough in ("id", "case_id", "question_idx", "idx", "round", "question", "prompt"):
                if passthrough in raw and passthrough not in merged:
                    merged[passthrough] = raw[passthrough]
            return merged
    return raw


def parse_prediction(raw_outer: dict[str, Any]) -> Prediction:
    raw = _extract_prediction_payload(raw_outer)
    route = route_from_any(_pick(raw, "route_polyline", "pred_route", "complete_route", "route"))
    candidates = [point for point in (point_from_any(item) for item in _candidate_list(raw)) if point is not None]
    pred_destination = point_from_any(_pick(raw, "chosen_destination", "pred_destination", "destination"))
    if pred_destination is None and candidates:
        pred_destination = candidates[0]
    if pred_destination is None and route:
        pred_destination = route[-1]
    return Prediction(
        case_id=case_id_from_record(raw),
        pred_destination=pred_destination,
        ranked_destinations=candidates,
        route=route,
        route_distance_m=_summary_distance(raw) or _route_distance_m(route),
        route_steps=_summary_steps(raw, route),
        reached_flag=raw.get("flag") if isinstance(raw.get("flag"), bool) else raw.get("round_success") if isinstance(raw.get("round_success"), bool) else None,
        raw=raw,
    )


def haversine_m(a: Point, b: Point) -> float:
    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    dlat = lat2 - lat1
    dlng = math.radians(b.lng - a.lng)
    x = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(x)))


def _text_match(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return False
    return " ".join(a.lower().split()) == " ".join(b.lower().split())


def destination_match(pred: Point | None, gt: Point | None, *, radius_m: float) -> tuple[bool, float | None]:
    if pred is None or gt is None:
        return False, None
    if pred.node_id and gt.node_id and pred.node_id == gt.node_id:
        return True, haversine_m(pred, gt)
    if _text_match(pred.name, gt.name):
        return True, haversine_m(pred, gt)
    distance = haversine_m(pred, gt)
    return distance <= radius_m, distance


def candidate_rank(pred: Prediction, gt: Point | None, *, radius_m: float) -> int | None:
    if gt is None:
        return None
    for idx, candidate in enumerate(pred.ranked_destinations, start=1):
        ok, _dist = destination_match(candidate, gt, radius_m=radius_m)
        if ok:
            return idx
    return None


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return float(num) / float(den)


def _interp(a: Point, b: Point, t: float) -> Point:
    return Point(lat=a.lat + (b.lat - a.lat) * t, lng=a.lng + (b.lng - a.lng) * t)


def resample_route(points: list[Point], n: int) -> list[Point]:
    if not points:
        return []
    if len(points) == 1 or n <= 1:
        return [points[0]]
    seg_lens = [haversine_m(a, b) for a, b in zip(points, points[1:])]
    total = sum(seg_lens)
    if total <= 1e-6:
        return [points[0] for _ in range(n)]
    targets = [total * i / (n - 1) for i in range(n)]
    out: list[Point] = []
    seg_idx = 0
    acc = 0.0
    for target in targets:
        while seg_idx < len(seg_lens) - 1 and acc + seg_lens[seg_idx] < target:
            acc += seg_lens[seg_idx]
            seg_idx += 1
        seg_len = max(seg_lens[seg_idx], 1e-9)
        t = max(0.0, min(1.0, (target - acc) / seg_len))
        out.append(_interp(points[seg_idx], points[seg_idx + 1], t))
    return out


def curve_errors_m(pred_route: list[Point], gt_route: list[Point], *, samples: int) -> dict[str, float | None]:
    if len(pred_route) < 2 or len(gt_route) < 2:
        return {"route_curve_ade_m": None, "route_curve_fde_m": None, "route_curve_mde_m": None}
    pred = resample_route(pred_route, samples)
    gt = resample_route(gt_route, samples)
    distances = [haversine_m(a, b) for a, b in zip(pred, gt)]
    return {
        "route_curve_ade_m": mean(distances),
        "route_curve_fde_m": distances[-1],
        "route_curve_mde_m": max(distances),
    }


def evaluate_pair(case: Case, pred: Prediction, *, destination_radius_m: float, route_samples: int) -> dict[str, Any]:
    dest_success, dest_error_m = destination_match(pred.pred_destination, case.gt_destination, radius_m=destination_radius_m)
    rank = candidate_rank(pred, case.gt_destination, radius_m=destination_radius_m)
    endpoint_error_m = None
    if pred.route and case.gt_destination:
        endpoint_error_m = haversine_m(pred.route[-1], case.gt_destination)

    route_success = bool(dest_success and pred.route)
    route_distance_ratio = _safe_div(pred.route_distance_m, case.gt_distance_m)
    route_step_ratio = _safe_div(float(pred.route_steps) if pred.route_steps is not None else None, float(case.gt_steps) if case.gt_steps is not None else None)
    spl = None
    if route_success and pred.route_distance_m is not None and case.gt_distance_m is not None:
        spl = case.gt_distance_m / max(pred.route_distance_m, case.gt_distance_m)
    elif route_success and pred.route_steps is not None and case.gt_steps is not None:
        spl = case.gt_steps / max(pred.route_steps, case.gt_steps)
    else:
        spl = 0.0

    row: dict[str, Any] = {
        "case_id": case.case_id,
        "question": case.question,
        "category": case.category,
        "destination_success": dest_success,
        "destination_error_m": dest_error_m,
        "destination_rank": rank,
        "hit_at_1": rank == 1 if rank is not None else dest_success,
        "route_success": route_success,
        "route_endpoint_error_m": endpoint_error_m,
        "gt_distance_m": case.gt_distance_m,
        "pred_distance_m": pred.route_distance_m,
        "route_distance_ratio": route_distance_ratio,
        "gt_steps": case.gt_steps,
        "pred_steps": pred.route_steps,
        "route_step_ratio": route_step_ratio,
        "spl": spl,
        "cityseeker_flag": pred.reached_flag,
    }
    row.update(curve_errors_m(pred.route, case.gt_route, samples=route_samples))
    return row


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return mean(clean) if clean else None


def _rate(values: Iterable[bool]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(1 for v in vals if v) / len(vals)


def _round_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _round_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_value(v) for v in value]
    return value


def summarize(rows: list[dict[str, Any]], *, hit_ks: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case_count": len(rows),
        "destination": {
            "success_rate": _rate(bool(r.get("destination_success")) for r in rows),
            "hit_at_1": _rate(bool(r.get("hit_at_1")) for r in rows),
            "mean_error_m": _mean(r.get("destination_error_m") for r in rows),
            "median_error_m": None,
            "mrr": _mean((1.0 / r["destination_rank"]) if r.get("destination_rank") else 0.0 for r in rows),
        },
        "route": {
            "success_rate": _rate(bool(r.get("route_success")) for r in rows),
            "mean_endpoint_error_m": _mean(r.get("route_endpoint_error_m") for r in rows),
            "mean_distance_ratio": _mean(r.get("route_distance_ratio") for r in rows),
            "mean_step_ratio": _mean(r.get("route_step_ratio") for r in rows),
            "spl": _mean(r.get("spl") for r in rows),
            "curve_ade_m": _mean(r.get("route_curve_ade_m") for r in rows),
            "curve_fde_m": _mean(r.get("route_curve_fde_m") for r in rows),
            "curve_mde_m": _mean(r.get("route_curve_mde_m") for r in rows),
        },
    }
    errors = sorted(float(r["destination_error_m"]) for r in rows if isinstance(r.get("destination_error_m"), (int, float)))
    if errors:
        summary["destination"]["median_error_m"] = median(errors)
    for k in hit_ks:
        summary["destination"][f"hit_at_{k}"] = _rate(
            bool(r.get("destination_rank") and int(r["destination_rank"]) <= k) for r in rows
        )
    return _round_value(summary)


def load_cases(path: Path) -> dict[str, Case]:
    cases = [parse_case(raw) for raw in _load_records(path)]
    return {case.case_id: case for case in cases}


def load_predictions(path: Path) -> dict[str, Prediction]:
    preds = [parse_prediction(raw) for raw in _load_records(path)]
    return {pred.case_id: pred for pred in preds}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate destination recommendation and route metrics.")
    parser.add_argument("--ground-truth", required=True, help="Ground-truth JSON or JSONL.")
    parser.add_argument("--predictions", required=True, help="Prediction JSON or JSONL.")
    parser.add_argument("--output-json", help="Optional summary JSON output path.")
    parser.add_argument("--output-jsonl", help="Optional per-case metrics JSONL output path.")
    parser.add_argument("--destination-radius-m", type=float, default=50.0, help="Destination success radius in meters.")
    parser.add_argument("--route-samples", type=int, default=64, help="Number of points for curve route metrics.")
    parser.add_argument("--hit-k", type=int, nargs="*", default=[1, 3, 5], help="Destination Hit@K cutoffs.")
    parser.add_argument("--require-all-predictions", action="store_true", help="Fail when a GT case has no prediction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(Path(args.ground_truth).expanduser())
    predictions = load_predictions(Path(args.predictions).expanduser())

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for case_id, case in cases.items():
        pred = predictions.get(case_id)
        if pred is None:
            missing.append(case_id)
            continue
        rows.append(
            evaluate_pair(
                case,
                pred,
                destination_radius_m=float(args.destination_radius_m),
                route_samples=max(2, int(args.route_samples)),
            )
        )

    if missing and args.require_all_predictions:
        raise SystemExit(f"Missing predictions for {len(missing)} cases, first missing={missing[:5]}")

    summary = summarize(rows, hit_ks=sorted(set(args.hit_k)))
    summary["matched_case_count"] = len(rows)
    summary["missing_prediction_count"] = len(missing)
    summary["missing_prediction_ids_preview"] = missing[:20]

    text = json.dumps(summary, indent=2, ensure_ascii=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    if args.output_jsonl:
        out = Path(args.output_jsonl)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(_round_value(row), ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
