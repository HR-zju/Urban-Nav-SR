from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_capture(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _valid_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def _series(frames: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for frame in frames:
        debug = frame.get("debug", {}) or {}
        value = debug.get(field)
        if _valid_number(value):
            values.append(float(value))
    return values


def compute_metrics(capture: dict[str, Any]) -> dict[str, Any]:
    frames = capture.get("frames", []) or []
    monotonicity_violations = 0
    corridor_switches = 0
    distance_inconsistencies = 0
    off_route_frames = 0
    false_positive_like_frames = 0

    prior_segment_index: int | None = None
    prior_corridor: str | None = None
    prior_distance: float | None = None

    for frame in frames:
        debug = frame.get("debug", {}) or {}
        segment_index = debug.get("matched_segment_index")
        corridor_type = str(debug.get("corridor_type") or "")
        off_route = bool(debug.get("off_route"))
        cross_track = debug.get("cross_track_error_m")
        turn_semantics = str(debug.get("turn_semantics") or "")
        distance_to_turn = debug.get("distance_to_next_turn_m")

        if off_route:
            off_route_frames += 1
            if _valid_number(cross_track) and float(cross_track) < 25.0 and turn_semantics != "approach_route_start":
                false_positive_like_frames += 1

        if isinstance(segment_index, int) and prior_segment_index is not None and segment_index + 2 < prior_segment_index:
            monotonicity_violations += 1
        if isinstance(segment_index, int):
            prior_segment_index = segment_index

        if corridor_type and prior_corridor is not None and corridor_type != prior_corridor:
            corridor_switches += 1
        if corridor_type:
            prior_corridor = corridor_type

        if _valid_number(distance_to_turn):
            current_distance = float(distance_to_turn)
            if prior_distance is not None and current_distance > prior_distance + 8.0:
                distance_inconsistencies += 1
            prior_distance = current_distance

    frame_count = len(frames)
    transition_count = max(frame_count - 1, 0)

    cross_track_values = _series(frames, "cross_track_error_m")
    heading_delta_values = _series(frames, "heading_delta_deg")
    distance_values = _series(frames, "distance_to_next_turn_m")

    return {
        "frame_count": frame_count,
        "segment_index_monotonicity_violation_rate": round(_safe_rate(monotonicity_violations, transition_count), 4),
        "corridor_switching_rate": round(_safe_rate(corridor_switches, transition_count), 4),
        "distance_to_turn_inconsistency_rate": round(
            _safe_rate(distance_inconsistencies, max(len(distance_values) - 1, 0)), 4
        ),
        "off_route_rate": round(_safe_rate(off_route_frames, frame_count), 4),
        "off_route_false_positive_like_rate": round(_safe_rate(false_positive_like_frames, frame_count), 4),
        "cross_track_error_m": {
            "mean": round(mean(cross_track_values), 2) if cross_track_values else None,
            "max": round(max(cross_track_values), 2) if cross_track_values else None,
        },
        "heading_delta_deg": {
            "mean": round(mean(heading_delta_values), 2) if heading_delta_values else None,
            "max": round(max(heading_delta_values), 2) if heading_delta_values else None,
        },
        "distance_to_next_turn_m": {
            "min": round(min(distance_values), 2) if distance_values else None,
            "max": round(max(distance_values), 2) if distance_values else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute RouteContext stability metrics from a capture JSON file")
    parser.add_argument("--capture", required=True, help="Path to RouteContext capture JSON")
    parser.add_argument("--output", help="Optional output JSON path for metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture = _load_capture(args.capture)
    metrics = compute_metrics(capture)
    text = json.dumps(metrics, indent=2, ensure_ascii=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote RouteContext metrics to {args.output}")
        return
    print(text)


if __name__ == "__main__":
    main()
