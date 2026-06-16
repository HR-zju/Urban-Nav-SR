# Orin26 aarch64 Deployment Notes

Last updated: 2026-05-28

This note records the current Orin26 setup for running the policy part of
`Urban-Nav-SR`. The local workstation remains the x86/RTX 3090 development
environment; Orin26 uses the aarch64 Jetson stack and a separate runner.

## Target Machine

- Host: `orin26@192.168.1.102`
- Conda/mamba env: `foundation_nav`
- Jetson release observed on Orin26: JetPack/L4T `R36.5.0`
- ONNX Runtime in env: `1.23.0`
- Power/clock mode used for the benchmark:
  - `nvpmodel`: `MAXN`
  - `jetson_clocks`: enabled

## environment install
```
 # create mamba/conda env and activate
cd /home/orin26/code/Urban-Nav-SR && mamba env create -f Policy_part/environment.orin26-aarch64.yml
```

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

- input image default: `image_raw`
- overlay default: `/navflow/overlay/image_raw`
- add `--no_publish_overlay` to skip overlay rendering/publishing work.

## Model and FPS Mapping

All paths below are relative to
`/home/orin26/code/Urban-Nav-SR/Policy_part`.

| Mode | File | Measured speed |
| --- | --- | ---: |
| Original ONNX + ORT CUDA EP, no provider options | `onnx_ckpts/vint_goal_conditioned.onnx` | `11.64 Hz` |
| Original ONNX + standalone TensorRT engine | `onnx_ckpts/vint_goal_conditioned_original_fp16_fast.engine` | `24.23 Hz` |
| Shape-inferred ONNX + ORT TensorRT EP FP16 | `onnx_ckpts/vint_goal_conditioned_orin_shape.onnx` | `21.25 Hz` |
| Static224 ONNX + ORT TensorRT EP FP16 | `onnx_ckpts/vint_goal_conditioned_orin_static224.onnx` | `24.23 Hz` |
| Static224 standalone TensorRT engine | `onnx_ckpts/vint_goal_conditioned_orin_static224_fp16_fast.engine` | `25.75 Hz` |

The standalone `.engine` path is slightly faster, but the current ROS2 node does
not load TensorRT engines directly. It uses ONNX Runtime. Direct engine runtime
would require adding TensorRT Python runtime support plus CUDA buffer binding to
`run_navflow_realsense_vis_goal_orin.py`.

do not switch the normal launch script to a `.engine` file until the
ROS node has a tested engine backend.
