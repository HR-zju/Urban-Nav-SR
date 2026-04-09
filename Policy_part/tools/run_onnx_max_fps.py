#!/usr/bin/env python3
"""Ultra-light ONNX FPS benchmark runner.

Goal:
- Measure near-peak model inference FPS on a powerful GPU (e.g. RTX 3090).
- Minimize non-essential overhead (no visualization, no ROS, no PD control).
- Keep runtime path as close as possible to pure ONNX inference.

Default strategy:
- CUDAExecutionProvider only (no CPU fallback) to avoid cross-device memcpy.
- Optional IO binding to reduce per-iteration host<->device transfer overhead.
- Reuse static input tensors for deterministic and lightweight benchmarking.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import List

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal ONNX max-FPS benchmark")
    p.add_argument("--onnx", required=True, help="Path to ONNX model")
    p.add_argument("--iters", type=int, default=500, help="Measured iterations")
    p.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    p.add_argument("--frames", type=int, default=21, help="Temporal T dimension")
    p.add_argument("--height", type=int, default=288, help="Input height")
    p.add_argument("--width", type=int, default=512, help="Input width")
    p.add_argument(
        "--input_mode",
        choices=["zeros", "random"],
        default="zeros",
        help="zeros is most stable for pure engine benchmarking",
    )
    p.add_argument(
        "--use_io_binding",
        action="store_true",
        help="Use IO binding to minimize host-device transfer overhead",
    )
    p.add_argument(
        "--no_cuda_only",
        action="store_true",
        help="Allow CPU fallback providers (less strict for max-FPS tests)",
    )
    p.add_argument(
        "--gpu_mem_limit_gb",
        type=float,
        default=10.0,
        help="CUDA provider gpu_mem_limit in GB (used when CUDA is enabled)",
    )
    return p.parse_args()


def choose_providers(cuda_only: bool, gpu_mem_limit_gb: float):
    available = ort.get_available_providers()

    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is not available in current onnxruntime build/environment"
        )

    cuda_opt = {
        "gpu_mem_limit": str(int(gpu_mem_limit_gb * (1024**3))),
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "DEFAULT",
    }

    if cuda_only:
        return ["CUDAExecutionProvider"], [cuda_opt]

    providers = ["CUDAExecutionProvider"]
    provider_options = [cuda_opt]
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
        provider_options.append({})
    return providers, provider_options


def make_inputs(args: argparse.Namespace):
    shape = (1, args.frames, 3, args.height, args.width)
    if args.input_mode == "random":
        observation = np.random.rand(*shape).astype(np.float32)
    else:
        observation = np.zeros(shape, dtype=np.float32)

    embodiment_id = np.array([[0]], dtype=np.int64)
    metric_spacing = np.array([[0.0, 0.51, -0.26, 0.26]], dtype=np.float32)
    return observation, embodiment_id, metric_spacing


def benchmark_with_session_run(
    session: ort.InferenceSession,
    observation: np.ndarray,
    embodiment_id: np.ndarray,
    metric_spacing: np.ndarray,
    warmup: int,
    iters: int,
) -> List[float]:
    feed = {
        "observation": observation,
        "embodiment_id": embodiment_id,
        "metric_spacing": metric_spacing,
    }

    for _ in range(warmup):
        session.run(None, feed)

    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, feed)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
    return latencies


def benchmark_with_iobinding(
    session: ort.InferenceSession,
    observation: np.ndarray,
    embodiment_id: np.ndarray,
    metric_spacing: np.ndarray,
    warmup: int,
    iters: int,
) -> List[float]:
    io = session.io_binding()

    obs_gpu = ort.OrtValue.ortvalue_from_numpy(observation, "cuda", 0)
    emb_gpu = ort.OrtValue.ortvalue_from_numpy(embodiment_id, "cuda", 0)
    met_gpu = ort.OrtValue.ortvalue_from_numpy(metric_spacing, "cuda", 0)

    io.bind_ortvalue_input("observation", obs_gpu)
    io.bind_ortvalue_input("embodiment_id", emb_gpu)
    io.bind_ortvalue_input("metric_spacing", met_gpu)

    for out in session.get_outputs():
        io.bind_output(out.name, "cuda", 0)

    for _ in range(warmup):
        session.run_with_iobinding(io)

    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run_with_iobinding(io)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
    return latencies


def summarize(latencies: List[float]) -> None:
    if not latencies:
        print("No latencies captured")
        return

    ms = [x * 1000.0 for x in latencies]
    avg_ms = statistics.mean(ms)
    med_ms = statistics.median(ms)
    p90_ms = np.percentile(ms, 90)
    p95_ms = np.percentile(ms, 95)
    best_ms = min(ms)
    worst_ms = max(ms)

    avg_fps = 1000.0 / avg_ms
    med_fps = 1000.0 / med_ms
    best_fps = 1000.0 / best_ms

    print("\n===== ONNX Benchmark Result =====")
    print(f"avg latency : {avg_ms:.3f} ms")
    print(f"median      : {med_ms:.3f} ms")
    print(f"p90         : {p90_ms:.3f} ms")
    print(f"p95         : {p95_ms:.3f} ms")
    print(f"best/worst  : {best_ms:.3f} / {worst_ms:.3f} ms")
    print(f"avg FPS     : {avg_fps:.2f}")
    print(f"median FPS  : {med_fps:.2f}")
    print(f"peak FPS    : {best_fps:.2f}")


def main() -> None:
    args = parse_args()

    providers, provider_options = choose_providers(
        cuda_only=(not args.no_cuda_only),
        gpu_mem_limit_gb=args.gpu_mem_limit_gb,
    )

    print(f"ONNX providers: {providers}")
    print(f"provider_options: {provider_options}")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True

    session = ort.InferenceSession(
        args.onnx,
        sess_options=so,
        providers=providers,
        provider_options=provider_options,
    )

    observation, embodiment_id, metric_spacing = make_inputs(args)

    print(
        f"Input shape: observation={observation.shape}, "
        f"embodiment_id={embodiment_id.shape}, metric_spacing={metric_spacing.shape}"
    )
    print(
        f"Benchmark config: warmup={args.warmup}, iters={args.iters}, "
        f"mode={args.input_mode}, io_binding={args.use_io_binding}"
    )

    if args.use_io_binding:
        latencies = benchmark_with_iobinding(
            session=session,
            observation=observation,
            embodiment_id=embodiment_id,
            metric_spacing=metric_spacing,
            warmup=args.warmup,
            iters=args.iters,
        )
    else:
        latencies = benchmark_with_session_run(
            session=session,
            observation=observation,
            embodiment_id=embodiment_id,
            metric_spacing=metric_spacing,
            warmup=args.warmup,
            iters=args.iters,
        )

    summarize(latencies)


if __name__ == "__main__":
    main()
