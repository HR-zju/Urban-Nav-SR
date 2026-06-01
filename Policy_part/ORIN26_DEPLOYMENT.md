# Orin26 aarch64 Deployment Notes

Last updated: 2026-05-28

This note records the current Orin26 setup for running the policy part of
`Urban-Nav-SR`. The local workstation remains the x86/RTX 3090 development
environment; Orin26 uses the aarch64 Jetson stack and a separate runner.

## Target Machine

- Host: `orin26@10.12.120.102`
- Remote repo: `/home/orin26/code/Urban-Nav-SR`
- Policy directory: `/home/orin26/code/Urban-Nav-SR/Policy_part`
- Conda/mamba env: `foundation_nav`
- Jetson release observed on Orin26: JetPack/L4T `R36.5.0`
- ONNX Runtime in env: `1.23.0`
- Available ONNX Runtime providers:
  - `TensorrtExecutionProvider`
  - `CUDAExecutionProvider`
  - `CPUExecutionProvider`
- Power/clock mode used for the benchmark:
  - `nvpmodel`: `MAXN`
  - `jetson_clocks`: enabled
  - GPU clock observed by TensorRT: about `918 MHz`

Do not install ROS1 for this deployment. The repo may still contain old ROS1
GPS/VINS sources, but the Orin launch path uses ROS2 Humble and Python
replacement nodes.

## Normal Startup

Run the Orin-specific script on Orin26:

```bash
cd /home/orin26/code/Urban-Nav-SR/Policy_part
./run_LLM_policy_orin.sh
```

The policy pane in `run_LLM_policy_orin.sh` currently starts:

```bash
export NAVFLOW_DISABLE_TRT=0
export NAVFLOW_TRT_FP16=1
export NAVFLOW_TRT_BUILDER_OPT_LEVEL=0
export NAVFLOW_TRT_ENGINE_CACHE=/home/orin26/.cache/navflow_ort_trt_static224_fp16

python run_navflow_realsense_vis_goal_orin.py \
  --onnx onnx_ckpts/vint_goal_conditioned_orin_static224.onnx \
  --targets ./topomaps/LLM_Policy/LLM_UTM_odom_poses.txt \
  --no_show_window
```

This is the fastest path that is currently integrated into the ROS2 Python
node: **static224 ONNX + ONNX Runtime TensorRT EP FP16**, about **24.23 Hz**
for model forward in the standalone benchmark.

The Orin runner defaults to raw RealSense input and raw overlay output to avoid
unnecessary JPEG decode/encode:

- input image default: `/camera/camera/color/image_raw`
- overlay default: `/navflow/overlay/image_raw`
- add `--no_publish_overlay` to skip overlay rendering/publishing work.

## Model and FPS Mapping

All paths below are relative to
`/home/orin26/code/Urban-Nav-SR/Policy_part`.

| Mode | File | Measured speed |
| --- | --- | ---: |
| Original ONNX + ORT CUDA EP, no provider options | `onnx_ckpts/vint_goal_conditioned.onnx` | `11.64 Hz` |
| Original ONNX + ORT TensorRT EP | `onnx_ckpts/vint_goal_conditioned.onnx` | failed: missing intermediate shape for `/ScatterND_output_0` |
| Original ONNX + standalone TensorRT engine | `onnx_ckpts/vint_goal_conditioned_original_fp16_fast.engine` | `24.23 Hz` |
| Shape-inferred ONNX + ORT TensorRT EP FP16 | `onnx_ckpts/vint_goal_conditioned_orin_shape.onnx` | `21.25 Hz` |
| Static224 ONNX + ORT TensorRT EP FP16 | `onnx_ckpts/vint_goal_conditioned_orin_static224.onnx` | `24.23 Hz` |
| Static224 standalone TensorRT engine | `onnx_ckpts/vint_goal_conditioned_orin_static224_fp16_fast.engine` | `25.75 Hz` |

The standalone `.engine` path is slightly faster, but the current ROS2 node does
not load TensorRT engines directly. It uses ONNX Runtime. Direct engine runtime
would require adding TensorRT Python runtime support plus CUDA buffer binding to
`run_navflow_realsense_vis_goal_orin.py`.

As of the benchmark, neither the mamba env nor system Python on Orin26 had these
Python modules available:

- `tensorrt`
- `cuda` / `cuda-python`
- `pycuda`

Therefore, do not switch the normal launch script to a `.engine` file until the
ROS node has a tested engine backend.

## Benchmark Commands

Original ONNX with plain ORT CUDA EP:

```bash
cd /home/orin26/code/Urban-Nav-SR/Policy_part
source /home/orin26/miniforge3/etc/profile.d/conda.sh
conda activate foundation_nav

python tools/run_onnx_max_fps.py \
  --onnx onnx_ckpts/vint_goal_conditioned.onnx \
  --provider cuda \
  --height 288 \
  --width 512 \
  --warmup 20 \
  --iters 100 \
  --input_mode zeros
```

Static224 ONNX with ORT TensorRT EP FP16:

```bash
cd /home/orin26/code/Urban-Nav-SR/Policy_part
source /home/orin26/miniforge3/etc/profile.d/conda.sh
conda activate foundation_nav
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH

python tools/run_onnx_max_fps.py \
  --onnx onnx_ckpts/vint_goal_conditioned_orin_static224.onnx \
  --provider trt \
  --trt_fp16 \
  --trt_cache /home/orin26/.cache/navflow_ort_trt_static224_fp16 \
  --trt_builder_opt_level 0 \
  --warmup 20 \
  --iters 120 \
  --input_mode zeros
```

Static224 standalone TensorRT engine:

```bash
cd /home/orin26/code/Urban-Nav-SR/Policy_part
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:$LD_LIBRARY_PATH

/usr/src/tensorrt/bin/trtexec \
  --loadEngine=onnx_ckpts/vint_goal_conditioned_orin_static224_fp16_fast.engine \
  --duration=8 \
  --warmUp=500 \
  --useCudaGraph \
  --useSpinWait
```

## Important Performance Notes

The old `~1.8 Hz` result for the original ONNX was not a fair GPU baseline. It
was caused by passing:

```text
cudnn_conv_algo_search=DEFAULT
```

to ONNX Runtime CUDA EP. On this Jetson stack that setting triggered many Conv
fallbacks. With no provider options, or with `EXHAUSTIVE`/`HEURISTIC`, the same
original ONNX runs at about `11.64 Hz`.

Current Orin-specific runners avoid that bad default:

- `Policy_part/tools/run_onnx_max_fps.py`
- `Policy_part/run_navflow_realsense_vis_goal_orin.py`
- `Policy_part/run_navflow_realsense_vis_orin.py`

The TensorRT gain should be compared against the corrected CUDA baseline:

- corrected ORT CUDA EP: about `11.64 Hz`
- static224 ORT TensorRT EP FP16: about `24.23 Hz`
- static224 standalone engine: about `25.75 Hz`

So the real TensorRT improvement is about `2.1x` to `2.2x`. The apparent
`10x+` improvement only appears if comparing against the broken Conv fallback
case.

## Power and Monitoring

Check Jetson mode:

```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

Set clocks for benchmarking:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Monitor CPU/GPU usage and power:

```bash
tegrastats
```

Expected sign that GPU is active during inference:

- `GR3D_FREQ` near `90-99%`
- `VDD_CPU_GPU_CV` rises during inference
- CPU usage remains relatively low compared with GPU load

## Barrier Note

Barrier details are intentionally stored outside the repo in:

```text
/home/zjy/.codex/memories/orin26-barrier.md
```

Check that memory first before rediscovering the Barrier client name, server IP,
and `--disable-crypto` setting.
