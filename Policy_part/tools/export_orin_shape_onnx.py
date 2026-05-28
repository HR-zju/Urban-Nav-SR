#!/usr/bin/env python3
"""Generate Orin/TensorRT-friendly ONNX files with inferred shapes."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import shape_inference


DEFAULT_MODELS = (
    "vint_goal_conditioned.onnx",
    "vint_goal_free.onnx",
    "vint_goal_less.onnx",
)


def export_shape_model(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] {src} -> {dst}")
    shape_inference.infer_shapes_path(str(src), str(dst))
    onnx.checker.check_model(str(dst))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export *_orin_shape.onnx files for Jetson Orin TensorRT."
    )
    default_dir = Path(__file__).resolve().parents[1] / "onnx_ckpts"
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=default_dir,
        help="Directory containing the original ViNT ONNX files.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model filenames to convert.",
    )
    args = parser.parse_args()

    for model_name in args.models:
        src = args.onnx_dir / model_name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = src.with_name(f"{src.stem}_orin_shape{src.suffix}")
        export_shape_model(src, dst)


if __name__ == "__main__":
    main()
