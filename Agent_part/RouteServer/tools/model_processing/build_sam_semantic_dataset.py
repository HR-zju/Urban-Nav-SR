#!/usr/bin/env python3
"""Build compact ms-swift semantic segmentation data from SAM outputs.

The compact layout keeps only:
  clip/images/*.jpg
  clip/semantic_masks/*.png
  root/sam_semantic_manifest.jsonl
  root/sam_semantic_labelmap.json

Bulky intermediate directories can be removed with --cleanup-bulky after the
PNG masks and manifest have been verified.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


BULKY_DIR_NAMES = {
    "fpv_segmentation",
    "som",
    "route_support",
    "debug_video",
    "sam3_intermediate",
    "sam3_work",
}


SYSTEM_PROMPT = (
    "You are a grounded pedestrian accessibility assistant. Identify visible "
    "sidewalk navigation regions and align them with segmentation supervision."
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_clips(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.glob("*/front.mp4"))


def _find_class_mapping(root: Path) -> dict[str, Any]:
    candidates = sorted(root.glob("*/sam3_intermediate/**/class_mapping.json"))
    if candidates:
        return _load_json(candidates[0])
    return {"class_to_idx": {"background": 0}}


def _id_to_name(mapping: dict[str, Any]) -> dict[int, str]:
    class_to_idx = mapping.get("class_to_idx")
    if isinstance(class_to_idx, dict):
        return {int(idx): str(name) for name, idx in class_to_idx.items()}
    try:
        return {int(k): str(v) for k, v in mapping.items()}
    except Exception:
        return {0: "background"}


def _write_labelmap(root: Path, mapping: dict[str, Any]) -> Path:
    path = root / "sam_semantic_labelmap.json"
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _png_to_label_ids(path: Path) -> list[int]:
    with Image.open(path) as image:
        arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return sorted(int(x) for x in np.unique(arr) if int(x) != 0)


def _convert_mask_npy(npy_path: Path, png_path: Path) -> list[int]:
    arr = np.load(npy_path)
    if arr.ndim == 3:
        if arr.shape[0] in {1, 3}:
            arr = arr[0] if arr.shape[0] == 1 else arr.argmax(axis=0)
        elif arr.shape[-1] in {1, 3}:
            arr = arr[..., 0] if arr.shape[-1] == 1 else arr.argmax(axis=-1)
    arr = arr.astype(np.uint16, copy=False)
    max_id = int(arr.max(initial=0))
    if max_id > 255:
        raise ValueError(f"{npy_path} has class id {max_id}; uint8 PNG cannot represent it")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(png_path, optimize=True)
    return sorted(int(x) for x in np.unique(arr) if int(x) != 0)


def _assistant_text(class_ids: list[int], names: dict[int, str]) -> str:
    if not class_ids:
        return "No confident foreground navigation regions are visible."
    parts = []
    for class_id in class_ids[:16]:
        name = names.get(class_id, f"class_{class_id}")
        parts.append(f"[p] {name} [/p][SEG]")
    return "Visible semantic regions: " + " ".join(parts)


def _build_row(
    *,
    clip: Path,
    image_path: Path,
    mask_path: Path,
    class_ids: list[int],
    names: dict[int, str],
    labelmap_path: Path,
) -> dict[str, Any]:
    frame_idx = image_path.stem
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "<image>\nIdentify accessible and harmful sidewalk navigation regions in this frame.",
            },
            {"role": "assistant", "content": _assistant_text(class_ids, names)},
        ],
        "images": [str(image_path.resolve())],
        "semantic_segmentation_path": str(mask_path.resolve()),
        "semantic_label_format": "label_png_uint8",
        "segmentation_valid_mask": True,
        "grounding_classes": class_ids,
        "seg_labelmap_path": str(labelmap_path.resolve()),
        "sidewalk_meta": {
            "source_dataset": "urbannav_sam",
            "clip_id": clip.name,
            "frame_idx": frame_idx,
            "image_width": None,
            "image_height": None,
        },
    }


def _cleanup_bulky(root: Path) -> None:
    for clip in _find_clips(root):
        for name in BULKY_DIR_NAMES:
            path = clip / name
            if path.exists() and path.is_dir():
                shutil.rmtree(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact SAM semantic dataset for ms-swift")
    parser.add_argument("--sidewalk-root", required=True, help="Chunk/root directory containing clip folders")
    parser.add_argument("--output-jsonl", default=None, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit")
    parser.add_argument("--cleanup-bulky", action="store_true", help="Delete bulky intermediate dirs after building")
    args = parser.parse_args()

    root = Path(args.sidewalk_root).expanduser().resolve()
    output_jsonl = Path(args.output_jsonl).expanduser().resolve() if args.output_jsonl else root / "sam_semantic_manifest.jsonl"
    mapping = _find_class_mapping(root)
    labelmap_path = _write_labelmap(root, mapping)
    names = _id_to_name(mapping)

    rows_written = 0
    converted = 0
    skipped = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as f:
        for clip in _find_clips(root):
            images_dir = clip / "images"
            if not images_dir.exists():
                continue
            semantic_dir = clip / "semantic_masks"
            for image_path in sorted(images_dir.glob("*.jpg")):
                if args.limit is not None and rows_written >= args.limit:
                    break
                frame = image_path.stem
                mask_png = semantic_dir / f"{frame}.png"
                if mask_png.exists():
                    class_ids = _png_to_label_ids(mask_png)
                else:
                    mask_npy = clip / "fpv_segmentation" / f"{frame}_mask.npy"
                    if not mask_npy.exists():
                        skipped += 1
                        continue
                    class_ids = _convert_mask_npy(mask_npy, mask_png)
                    converted += 1
                row = _build_row(
                    clip=clip,
                    image_path=image_path,
                    mask_path=mask_png,
                    class_ids=class_ids,
                    names=names,
                    labelmap_path=labelmap_path,
                )
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1
            if args.limit is not None and rows_written >= args.limit:
                break

    print(f"wrote_rows={rows_written}")
    print(f"converted_masks={converted}")
    print(f"skipped_missing_masks={skipped}")
    print(f"output_jsonl={output_jsonl}")
    print(f"labelmap={labelmap_path}")

    if args.cleanup_bulky:
        _cleanup_bulky(root)
        print("cleanup_bulky=done")


if __name__ == "__main__":
    main()
