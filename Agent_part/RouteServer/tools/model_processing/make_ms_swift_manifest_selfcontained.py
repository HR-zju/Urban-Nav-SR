#!/usr/bin/env python3
"""Materialize ms-swift multimodal JSONL media into a portable directory.

This utility is intentionally independent from the CoT archive stage.  It
accepts any ms-swift-style JSONL whose rows contain ``images`` and rewrites
those image paths to be relative to the output ``manifests`` directory, while
copying the image files under ``media/``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_name(value: Any, fallback: str = "unknown") -> str:
    text = str(value or fallback).strip()
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    out = "".join(keep).strip("._")
    return out or fallback


def resolve_input_path(path: str, *, input_manifest_dir: Path, root_image_dir: Path | None) -> Path:
    raw = Path(path).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(input_manifest_dir / raw)
        if root_image_dir is not None:
            candidates.append(root_image_dir / raw)
        candidates.append(Path.cwd() / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve image path: {path}")


def materialize_manifest(
    input_jsonl: Path,
    output_root: Path,
    *,
    split: str | None = None,
    root_image_dir: Path | None = None,
) -> dict[str, Any]:
    input_jsonl = input_jsonl.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    manifest_dir = output_root / "manifests"
    media_root = output_root / "media" / "clips"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_jsonl)
    out_rows: list[dict[str, Any]] = []
    copied: dict[str, str] = {}
    copy_count = 0

    for index, row in enumerate(rows):
        meta = dict(row.get("sidewalk_meta") or {})
        clip_id = safe_name(meta.get("clip_id") or row.get("clip_id"), fallback="unknown_clip")
        frame_idx = meta.get("frame_idx") if meta.get("frame_idx") is not None else row.get("frame_idx")
        frame_name = safe_name(meta.get("frame_name") or (f"{int(frame_idx):06d}" if frame_idx is not None else index))
        new_images: list[str] = []
        for image_index, image_value in enumerate(row.get("images") or []):
            source = resolve_input_path(str(image_value), input_manifest_dir=input_jsonl.parent, root_image_dir=root_image_dir)
            kind = "fpv" if image_index == 0 else "bev"
            suffix = source.suffix or ".jpg"
            name = f"{frame_name}{suffix}" if kind == "fpv" else f"{frame_name}_bev{suffix}"
            dest = media_root / clip_id / kind / name
            source_key = str(source)
            if source_key not in copied:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                copied[source_key] = str(dest)
                copy_count += 1
            rel = os.path.relpath(Path(copied[source_key]).resolve(), manifest_dir.resolve())
            new_images.append(rel)
        out_row = dict(row)
        out_row["images"] = new_images
        out_meta = dict(out_row.get("sidewalk_meta") or {})
        out_meta.update(
            {
                "selfcontained_root": str(output_root),
                "selfcontained_source_jsonl": str(input_jsonl),
                "selfcontained_media_relative_to": "manifest_dir",
            }
        )
        out_row["sidewalk_meta"] = out_meta
        out_rows.append(out_row)

    split_name = split or input_jsonl.stem
    output_jsonl = manifest_dir / f"{safe_name(split_name)}.jsonl"
    write_jsonl(output_jsonl, out_rows)

    report = {
        "schema_version": "ms_swift_selfcontained_manifest_v1",
        "input_jsonl": str(input_jsonl),
        "output_root": str(output_root),
        "output_jsonl": str(output_jsonl),
        "rows": len(out_rows),
        "unique_images_copied": copy_count,
        "path_mode": "relative_to_manifest_dir",
        "root_image_dir_for_ms_swift": str(manifest_dir),
    }
    report_path = manifest_dir / f"{safe_name(split_name)}_media_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a portable ms-swift JSONL + media directory")
    parser.add_argument("--input-jsonl", required=True, help="ms-swift JSONL with messages/images")
    parser.add_argument("--output-root", required=True, help="Portable output root")
    parser.add_argument("--split", default=None, help="Output manifest basename")
    parser.add_argument(
        "--root-image-dir",
        default=None,
        help="Optional base dir for resolving relative input images, matching ms-swift ROOT_IMAGE_DIR.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = materialize_manifest(
        Path(args.input_jsonl),
        Path(args.output_root),
        split=args.split,
        root_image_dir=Path(args.root_image_dir).expanduser().resolve() if args.root_image_dir else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
