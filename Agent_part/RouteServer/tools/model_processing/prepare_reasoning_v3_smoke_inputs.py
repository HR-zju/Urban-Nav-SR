#!/usr/bin/env python3
"""Select a deterministic four-dataset Reasoning V3.4 smoke set.

The selector uses only existing manifest fields.  It invokes the approved
ms-swift direction adapter for sampling and, except for FrodoBots, writes the
adapter's complete profile stamp into each selected row.  It never reads route
theta, GPS, pose, route matching, or projected geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SELECTION_RULE_VERSION = "reasoning-v3-smoke-v2"
DIRECTION_PROFILES = {
    "sidewalk": "fpv_degree10_terminal_centroid_v2",
    "foxglove": "fpv_degree10_terminal_centroid_v2",
    "frodobots": "fpv_degree10_terminal_centroid_v2",
    "egowalk": "ego_degree10_12way_terminal_centroid_v2",
}
DEFAULT_BUCKETS = {
    "sidewalk": ("left", "straight", "right"),
    "foxglove": ("left", "straight", "stop"),
    "frodobots": ("left", "straight", "right"),
    "egowalk": ("left", "right", "stop"),
}


def _dataset_arg(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    normalized = name.strip().lower()
    if not separator or normalized not in DEFAULT_BUCKETS or not path.strip():
        raise argparse.ArgumentTypeError(
            "dataset must be one of sidewalk|foxglove|frodobots|egowalk=JSONL"
        )
    return normalized, Path(path).expanduser()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, type=_dataset_arg)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--ms-swift-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_paths = dict(args.dataset)
    missing = sorted(set(DEFAULT_BUCKETS) - set(dataset_paths))
    if missing:
        raise SystemExit(f"missing required datasets: {missing}")

    ms_swift_root = args.ms_swift_root.expanduser().resolve()
    sys.path.insert(0, str(ms_swift_root))
    direction_module = importlib.import_module("swift.sidewalk_nav.direction_choice")
    derive_direction = direction_module.derive_direction_choice_from_row

    output_root = args.output_root.expanduser().resolve()
    curated_root = output_root / "curated_inputs"
    curated_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": "urbannav_reasoning_v3_4_smoke_selection_report_v1",
        "selection_rule_version": SELECTION_RULE_VERSION,
        "direction_profiles": DIRECTION_PROFILES,
        "datasets": {},
    }

    for dataset, buckets in DEFAULT_BUCKETS.items():
        manifest = dataset_paths[dataset].resolve()
        direction_profile = DIRECTION_PROFILES[dataset]
        candidates: dict[str, list[tuple[str, str, int, dict[str, Any], Any]]] = defaultdict(list)
        scanned = 0
        direction_errors = 0
        with manifest.open("r", encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if not line.strip():
                    continue
                scanned += 1
                row = json.loads(line)
                if dataset == "sidewalk" and (
                    (row.get("instruction_meta") or {}).get("status") != "success"
                ):
                    continue
                try:
                    label = derive_direction(
                        row,
                        profile=direction_profile,
                        existing_label_policy="rederive",
                    )
                except (TypeError, ValueError):
                    direction_errors += 1
                    continue
                bucket = _direction_bucket(label.name)
                if bucket not in buckets:
                    continue
                clip_id = _clip_id(row)
                source_id = _source_id(row, clip_id=clip_id)
                if not clip_id or not source_id:
                    continue
                digest = hashlib.sha256(
                    (
                        f"{SELECTION_RULE_VERSION}|{dataset}|"
                        f"{clip_id}|{source_id}"
                    ).encode("utf-8")
                ).hexdigest()
                candidates[bucket].append(
                    (digest, clip_id, line_index, row, label)
                )

        selected: list[tuple[str, str, int, dict[str, Any], Any, str]] = []
        used_clips: set[str] = set()
        for bucket in buckets:
            for digest, clip_id, line_index, row, label in sorted(candidates[bucket]):
                if clip_id in used_clips:
                    continue
                media = _media_paths(row, dataset=dataset)
                if media is None or not all(path.is_file() for path in media):
                    continue
                selected.append((digest, clip_id, line_index, row, label, bucket))
                used_clips.add(clip_id)
                break
            else:
                raise RuntimeError(f"no valid {dataset}/{bucket} smoke candidate")

        output_path = curated_root / f"{dataset}.jsonl"
        output_rows: list[dict[str, Any]] = []
        selected_report: list[dict[str, Any]] = []
        for digest, clip_id, line_index, row, label, bucket in selected:
            output = (
                _recover_sidewalk_base_row(row)
                if dataset == "sidewalk"
                else dict(row)
            )
            if dataset == "sidewalk":
                label = derive_direction(
                    output,
                    profile=direction_profile,
                    existing_label_policy="rederive",
                )
            for key in tuple(output):
                if key.startswith("direction_") or key == "canonical_direction":
                    output.pop(key, None)
            if dataset != "frodobots":
                output.update(label.to_manifest_fields(include_candidates=False))
            output["reasoning_v3_smoke_selection"] = {
                "selection_rule_version": SELECTION_RULE_VERSION,
                "source_manifest": str(manifest),
                "source_line_index_zero_based": line_index,
                "source_record_id": _source_id(output, clip_id=clip_id),
                "clip_id": clip_id,
                "selection_hash": digest,
                "stratum": bucket,
                "canonical_direction_promoted": dataset != "frodobots",
            }
            output_rows.append(output)
            selected_report.append(
                {
                    **output["reasoning_v3_smoke_selection"],
                    "direction_name": label.name,
                    "direction_label_index": label.index,
                    "direction_source": label.source,
                }
            )
        _write_jsonl(output_path, output_rows)
        report["datasets"][dataset] = {
            "source_manifest": str(manifest),
            "direction_profile": direction_profile,
            "output_jsonl": str(output_path),
            "scanned_rows": scanned,
            "direction_errors": direction_errors,
            "selected_rows": selected_report,
        }

    report_path = output_root / "selection_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _direction_bucket(name: str) -> str:
    value = str(name)
    if value.startswith("left_"):
        return "left"
    if value.startswith("right_"):
        return "right"
    if value in {"straight", "stop"}:
        return value
    return "other"


def _clip_id(row: dict[str, Any]) -> str:
    meta = row.get("sidewalk_meta") if isinstance(row.get("sidewalk_meta"), dict) else {}
    return str(row.get("clip_id") or meta.get("clip_id") or "").strip()


def _source_id(row: dict[str, Any], *, clip_id: str) -> str:
    if row.get("id"):
        return str(row["id"])
    meta = row.get("sidewalk_meta") if isinstance(row.get("sidewalk_meta"), dict) else {}
    frame = row.get("frame_idx")
    if frame is None:
        frame = meta.get("frame_idx")
    return f"{clip_id}:{frame}" if clip_id and frame is not None else ""


def _media_paths(row: dict[str, Any], *, dataset: str) -> list[Path] | None:
    history_padding = row.get("history_padding")
    if isinstance(history_padding, list) and any(bool(item) for item in history_padding):
        return None

    if dataset == "sidewalk":
        images = [Path(str(item)).expanduser() for item in row.get("images") or []]
        if len(images) < 10:
            return None
        fpv = images[0::2][-5:]
        bev = images[1::2][-1:]
    elif dataset in {"foxglove", "frodobots"}:
        fpv = [
            Path(str(item)).expanduser()
            for item in (
                row.get("fpv_image_paths")
                or [*(row.get("history_fpv_image_paths") or []), row.get("fpv_image_path")]
            )
            if item
        ][-5:]
        bev_value = row.get("bev_image_path")
        bev = [Path(str(bev_value)).expanduser()] if bev_value else []
    else:
        fpv = [
            Path(str(item)).expanduser()
            for item in [
                *(row.get("history_fpv_image_paths") or []),
                row.get("fpv_image_path"),
            ]
            if item
        ][-5:]
        bev = []

    if len(fpv) != 5 or len(set(fpv)) != 5:
        return None
    if dataset != "egowalk" and len(bev) != 1:
        return None
    return [*fpv, *bev]


def _recover_sidewalk_base_row(row: dict[str, Any]) -> dict[str, Any]:
    """Recover the pre-instruction-goal row to avoid action-text shortcuts."""

    meta = row.get("sidewalk_meta") if isinstance(row.get("sidewalk_meta"), dict) else {}
    source_manifest = meta.get("direction_choice_source_manifest")
    source_line = meta.get("direction_choice_source_line")
    if not source_manifest or not isinstance(source_line, int) or source_line < 1:
        raise ValueError("SideWalk smoke row lacks its pre-instruction-goal source pointer")
    with Path(str(source_manifest)).expanduser().open("r", encoding="utf-8") as handle:
        for one_based_index, line in enumerate(handle, start=1):
            if one_based_index == source_line:
                base = json.loads(line)
                break
        else:
            raise ValueError(
                f"SideWalk base source line is missing: {source_manifest}:{source_line}"
            )
    if _clip_id(base) != _clip_id(row):
        raise ValueError("SideWalk base-row clip identity mismatch")
    base_meta = base.get("sidewalk_meta") if isinstance(base.get("sidewalk_meta"), dict) else {}
    row_meta = row.get("sidewalk_meta") if isinstance(row.get("sidewalk_meta"), dict) else {}
    if base_meta.get("frame_idx") != row_meta.get("frame_idx"):
        raise ValueError("SideWalk base-row frame identity mismatch")
    return base


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
