#!/usr/bin/env python3
"""Export Orin-specialized ViNT ONNX graphs.

The original export keeps the training image size (288x512) as the ONNX input
and performs the final 224x224 backbone resize inside the graph. For Jetson
TensorRT this makes the graph larger than necessary. This exporter bakes the
fixed runtime constants used by the ROS node and expects pre-resized 224x224
frames by default.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import yaml


DEFAULT_VINT_ROOT = Path("/home/zjy/code/myk/vint")
if str(DEFAULT_VINT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_VINT_ROOT))

from method_2_ViNT import gen_sineembed_for_position  # noqa: E402


def load_yaml_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_class_from_str(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def parse_metric_spacing(values: Sequence[float] | None, config: dict) -> list[float]:
    if values is not None:
        if len(values) != 4:
            raise ValueError("--metric-spacing requires exactly 4 floats")
        return [float(v) for v in values]

    for dataset_key in ("train_dataset", "eval_dataset", "test_dataset"):
        dataset_cfg = config.get(dataset_key, {})
        spacing = dataset_cfg.get("metric_spacing")
        if spacing is not None:
            if len(spacing) != 4:
                raise ValueError(f"{dataset_key}.metric_spacing must contain 4 values")
            return [float(v) for v in spacing]
    raise ValueError("No metric_spacing found in config; pass --metric-spacing explicitly")


def choose_dataset_config(config: dict) -> dict:
    for dataset_key in ("train_dataset", "eval_dataset", "test_dataset"):
        dataset_cfg = config.get(dataset_key)
        if dataset_cfg:
            return dataset_cfg
    return {}


def load_checkpoint_weights(model: nn.Module, ckpt_path: Path, strict: bool = False) -> dict:
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    skipped_shape_keys: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    if not strict:
        model_state = model.state_dict()
        compatible_state = {}
        for key, value in state_dict.items():
            if key in model_state and tuple(model_state[key].shape) != tuple(value.shape):
                skipped_shape_keys.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            compatible_state[key] = value
        state_dict = compatible_state

    result = model.load_state_dict(state_dict, strict=strict)
    summary = {
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "missing_keys": list(getattr(result, "missing_keys", [])),
        "unexpected_keys": list(getattr(result, "unexpected_keys", [])),
        "skipped_shape_keys": skipped_shape_keys,
    }

    if skipped_shape_keys:
        print(f"[export] skipped shape-mismatched keys: {len(skipped_shape_keys)}")
        print(f"[export] first skipped keys: {skipped_shape_keys[:10]}")
    if summary["missing_keys"]:
        print(f"[export] missing checkpoint keys: {len(summary['missing_keys'])}")
        print(f"[export] first missing keys: {summary['missing_keys'][:10]}")
    if summary["unexpected_keys"]:
        print(f"[export] unexpected checkpoint keys: {len(summary['unexpected_keys'])}")
        print(f"[export] first unexpected keys: {summary['unexpected_keys'][:10]}")
    print(f"[export] checkpoint loaded (epoch={summary['epoch']}, global_step={summary['global_step']})")
    return summary


class OrinVintBaseWrapper(nn.Module):
    """Fixed-shape, TensorRT-friendly ViNT inference path."""

    def __init__(
        self,
        lightning_model: nn.Module,
        past_length: int,
        metric_spacing: Sequence[float],
        input_h: int,
        input_w: int,
        resize_inside_graph: bool,
    ):
        super().__init__()
        self.core = lightning_model.model
        self.past_length = int(past_length)
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.resize_inside_graph = bool(resize_inside_graph)
        self.register_buffer(
            "imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "metric_spacing_const",
            torch.tensor(metric_spacing, dtype=torch.float32).view(1, 4),
            persistent=False,
        )
        time_embedding = gen_sineembed_for_position(
            torch.arange(self.past_length, dtype=torch.float32).reshape(self.past_length, 1).repeat(1, 2),
            hidden_dim=64,
        )
        self.register_buffer(
            "time_embedding_const",
            time_embedding.unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "position_ids_const",
            torch.arange(self.past_length + 2, dtype=torch.long),
            persistent=False,
        )

    def _encode_observation(self, observation: torch.Tensor) -> torch.Tensor:
        batch_size = observation.shape[0]
        total_frames = self.past_length + 1
        dtype = observation.dtype
        mean = self.imagenet_mean.to(dtype=dtype)
        std = self.imagenet_std.to(dtype=dtype)

        obs_all_norm = observation.reshape(batch_size * total_frames, 3, self.input_h, self.input_w)
        obs_all_norm = (obs_all_norm - mean) / std
        if self.resize_inside_graph:
            obs_all_norm = torch.nn.functional.interpolate(
                obs_all_norm, size=(224, 224), mode="bilinear", align_corners=False
            )

        features = self.core.backbone(obs_all_norm)
        feat = features.reshape(batch_size, total_frames, -1)
        feat = self.core.token_projector(feat)

        time_embedding = self.time_embedding_const.to(dtype=dtype).expand(batch_size, -1, -1)
        past_feat = feat[:, : self.past_length] + self.core.time_embeeding(time_embedding)
        current_feat = feat[:, self.past_length :]
        return torch.cat((past_feat, current_feat), dim=1)

    def _decode(self, feat: torch.Tensor, goal_point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = feat.shape[0]
        dtype = feat.dtype
        goal_point_encoded = self.core.goal_point_encoder(goal_point.to(dtype=dtype)).reshape(batch_size, 1, -1)
        x_context = torch.cat([feat, goal_point_encoded], dim=1)
        pos_ids = self.position_ids_const[: x_context.shape[1]].to(device=x_context.device)
        pos_embs = self.core.position_embedding(pos_ids).unsqueeze(0).to(dtype=dtype)
        x_context = self.core.vlln(x_context) + pos_embs
        x_context = self.core.vl_self_attention(x_context)

        context = x_context.mean(dim=1)
        raw_action = self.core.afg_free_action_decoder(context).reshape(batch_size, 20, 3)
        metric_spacing = self.metric_spacing_const.to(device=raw_action.device, dtype=dtype)
        x = (
            (raw_action[..., 0] + 1.0)
            * 0.5
            * (metric_spacing[:, 1] - metric_spacing[:, 0]).unsqueeze(-1)
            + metric_spacing[:, 0].unsqueeze(-1)
        )
        y = (
            (raw_action[..., 1] + 1.0)
            * 0.5
            * (metric_spacing[:, 3] - metric_spacing[:, 2]).unsqueeze(-1)
            + metric_spacing[:, 2].unsqueeze(-1)
        )
        xy_action = torch.stack((x, y), dim=-1)
        trajectory = torch.cumsum(xy_action, dim=1).unsqueeze(1)
        score = torch.ones(batch_size, 1, device=trajectory.device, dtype=trajectory.dtype)
        return trajectory, score


class OrinVintGoalFreeWrapper(OrinVintBaseWrapper):
    def forward(self, observation: torch.Tensor):
        feat = self._encode_observation(observation)
        goal_point = torch.zeros(feat.shape[0], 3, device=feat.device, dtype=feat.dtype)
        return self._decode(feat, goal_point)


class OrinVintGoalConditionedWrapper(OrinVintBaseWrapper):
    def forward(self, observation: torch.Tensor, goal_point: torch.Tensor):
        feat = self._encode_observation(observation)
        return self._decode(feat, goal_point)


def export_one(args: argparse.Namespace, config: dict, mode: str, output_path: Path) -> None:
    config_path = Path(args.config).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt_path}")

    dataset_cfg = choose_dataset_config(config)
    past_length = args.past_length or int(dataset_cfg.get("past_length", 20))
    total_frames = past_length + 1
    metric_spacing = parse_metric_spacing(args.metric_spacing, config)

    model_cls = get_class_from_str(config["model"]["model_path"])
    lightning_model = model_cls(config["model"])
    checkpoint_summary = load_checkpoint_weights(lightning_model, ckpt_path, strict=args.strict_ckpt)
    lightning_model.eval()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[export] CUDA was requested but is not available; falling back to CPU.")
    if args.dtype == "fp16" and device.type != "cuda":
        raise ValueError("FP16 export requires --device cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    wrapper_cls = OrinVintGoalConditionedWrapper if mode == "goal_conditioned" else OrinVintGoalFreeWrapper
    wrapper = wrapper_cls(
        lightning_model=lightning_model,
        past_length=past_length,
        metric_spacing=metric_spacing,
        input_h=args.input_h,
        input_w=args.input_w,
        resize_inside_graph=args.resize_inside_graph,
    ).to(device).eval()
    if dtype == torch.float16:
        wrapper = wrapper.half()

    observation = torch.randn(
        args.batch_size,
        total_frames,
        3,
        args.input_h,
        args.input_w,
        device=device,
        dtype=dtype,
    )
    goal_point = torch.tensor(args.goal_point, device=device, dtype=dtype).unsqueeze(0).repeat(args.batch_size, 1)

    if mode == "goal_conditioned":
        export_inputs = (observation, goal_point)
        input_names = ["observation", "goal_point"]
    else:
        export_inputs = (observation,)
        input_names = ["observation"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        trajectory, score = wrapper(*export_inputs)
        torch.onnx.export(
            wrapper,
            export_inputs,
            str(output_path),
            input_names=input_names,
            output_names=["trajectory", "score"],
            dynamic_axes=None,
            opset_version=args.opset,
            do_constant_folding=True,
        )

    if args.shape_inference:
        import onnx

        inferred_path = output_path.with_suffix(".shape.onnx")
        onnx.shape_inference.infer_shapes_path(str(output_path), str(inferred_path))
        inferred_path.replace(output_path)

    metadata = {
        "mode": mode,
        "config": str(config_path),
        "checkpoint": str(ckpt_path),
        "output": str(output_path.resolve()),
        "opset": args.opset,
        "dynamic_batch": False,
        "input_h": args.input_h,
        "input_w": args.input_w,
        "resize_inside_graph": args.resize_inside_graph,
        "metric_spacing_baked": metric_spacing,
        "sample_shapes": {
            "observation": list(observation.shape),
            **({"goal_point": list(goal_point.shape)} if mode == "goal_conditioned" else {}),
        },
        "output_shapes": {"trajectory": list(trajectory.shape), "score": list(score.shape)},
        "checkpoint_summary": checkpoint_summary,
    }
    with open(output_path.with_suffix(output_path.suffix + ".json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"[export] wrote {output_path.resolve()}")
    print(f"[export] mode={mode} inputs={input_names}")
    print(f"[export] trajectory: shape={tuple(trajectory.shape)} dtype={trajectory.dtype}")
    print(f"[export] score: shape={tuple(score.shape)} dtype={score.dtype}")

    if args.verify:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"[export] onnx.checker passed for {mode}")

    if args.ort_verify:
        import numpy as np
        import onnxruntime as ort

        ort_session = ort.InferenceSession(str(output_path), providers=args.ort_providers)
        ort_inputs = {"observation": observation.detach().cpu().numpy().astype(np.float32)}
        if mode == "goal_conditioned":
            ort_inputs["goal_point"] = goal_point.detach().cpu().numpy().astype(np.float32)
        ort_outputs = ort_session.run(None, ort_inputs)
        print(f"[export] ort providers for {mode}: {ort_session.get_providers()}")
        print(f"[export] ort output shapes for {mode}: {[tuple(x.shape) for x in ort_outputs]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Orin-specialized ViNT ONNX.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["goal_free", "goal_conditioned"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--past-length", type=int, default=None)
    parser.add_argument("--input-h", type=int, default=224)
    parser.add_argument("--input-w", type=int, default=224)
    parser.add_argument(
        "--resize-inside-graph",
        action="store_true",
        help="Keep the final 224x224 resize inside the graph for compatibility testing.",
    )
    parser.add_argument("--metric-spacing", type=float, nargs=4, default=None)
    parser.add_argument("--goal-point", type=float, nargs=3, default=[0.03, 1.0, 0.0])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--strict-ckpt", action="store_true")
    parser.add_argument("--shape-inference", action="store_true", default=True)
    parser.add_argument("--no-shape-inference", dest="shape_inference", action="store_false")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--ort-verify", action="store_true")
    parser.add_argument("--ort-providers", nargs="+", default=["CPUExecutionProvider"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(Path(args.config).expanduser().resolve())
    export_one(args, config, args.mode, Path(args.output).expanduser())


if __name__ == "__main__":
    main()
