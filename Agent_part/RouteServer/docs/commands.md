# RouteServer 常用运行指令

本文档记录 RouteServer 侧数据抽取、批处理和调试常用命令。所有命令默认在以下目录执行：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer
```

## 从 MCAP 抽取 Sidewalk Pipeline 输入

用途：把单个 ROS2 MCAP 数据包转换成 `tools/run_sidewalk_images_visual_prompt_pipeline.py` 可直接消费的最小输入格式：

```text
<WORK_DIR>/
  images/
    000001.jpg
    000002.jpg
    ...
  trajectory.json
  frames_manifest.json
  mcap_extract_meta.json
```

依赖说明：很多 MCAP chunk 使用 `lz4` 压缩。如果当前 Python 环境缺少 `lz4`，会出现 `ModuleNotFoundError: No module named 'lz4'` 或脚本提示安装 `lz4`。在当前环境中安装即可：

```bash
python -m pip install lz4
```

如果使用 conda，也可以：

```bash
conda install -c conda-forge lz4
```

当时针对 `/home/ubuntu/Downloads/viobot_sidewalk_zju_bag_20260510_081025_15.mcap` 的 1 FPS smoke 抽取命令如下：

```bash
WORK_DIR=/tmp/viobot_sidewalk_zju_081025_15_1fps

python tools/mcap_to_sidewalk_inputs.py \
  --mcap-path /home/ubuntu/Downloads/viobot_sidewalk_zju_bag_20260510_081025_15.mcap \
  --output-dir "$WORK_DIR" \
  --image-topic /camera_x001/image_raw/compressed \
  --gps-topic /baton/gnss/spp_fix \
  --heading-topic /baton/gnss/heading \
  --imu-topic auto \
  --pose-topic auto \
  --attitude-policy optional \
  --sample-fps 1.0 \
  --max-sync-delta-s 0.5 \
  --image-start-index 1
```

参数说明：

- `--mcap-path`：输入 MCAP 文件。
- `--output-dir`：输出目录，后续 pipeline 的 `WORK_DIR`。
- `--image-topic`：`sensor_msgs/msg/CompressedImage` 图像 topic。
- `--gps-topic`：`sensor_msgs/msg/NavSatFix` GNSS topic。
- `--heading-topic`：`std_msgs/msg/Float64` heading topic。
- `--imu-topic`：可选 `sensor_msgs/msg/Imu` 姿态 topic；`auto` 表示按 schema 自动选择，`none` 表示禁用。
- `--pose-topic`：可选 `geometry_msgs/msg/PoseStamped` 姿态 topic；`auto` 表示按 schema 自动选择，`none` 表示禁用。
- `--attitude-policy`：姿态抽取策略；`optional` 表示有则同步 roll/pitch、无则继续，`required` 表示缺姿态时跳过帧，`disabled` 表示完全关闭。
- `--sample-fps 1.0`：按 1 FPS 抽帧；设为 `0` 或负数表示保留全部图像帧。
- `--max-sync-delta-s 0.5`：图像帧与最近 GNSS/heading 样本的最大允许时间差。
- `--image-start-index 1`：输出图像从 `000001.jpg` 开始编号。

如果不确定 topic 名称，可以先使用 `auto` 让脚本按消息 schema 自动选择：

```bash
python tools/mcap_to_sidewalk_inputs.py \
  --mcap-path /path/to/input.mcap \
  --output-dir /tmp/sidewalk_inputs_auto \
  --image-topic auto \
  --gps-topic auto \
  --heading-topic auto \
  --sample-fps 1.0
```

抽取完成后，终端会打印：

```text
Done.
trajectory_json=<WORK_DIR>/trajectory.json
images_dir=<WORK_DIR>/images
meta_json=<WORK_DIR>/mcap_extract_meta.json
```

其中 `mcap_extract_meta.json` 会记录实际选中的 topic、原始消息数量、写出的帧数、跳过帧原因预览，以及后续 pipeline 的输入路径提示。

## 接 Visual Prompt Pipeline

抽取完成后，可以继续用同一个 `WORK_DIR` 跑 visual prompt pipeline。若 SAM3 产物已经存在，可以不加 `--run-sam3`：

推荐先设置低 reasoning / 小图输入，减少 LVLM 长尾：

```bash
export LLM_REASONING_EFFORT=minimal
export VISUAL_PROMPT_LLM_IMAGE_MAX_SIDE=960
export VISUAL_PROMPT_LLM_IMAGE_JPEG_QUALITY=70
export LLM_MAX_RETRIES=1
export LLM_VISION_TIMEOUT_S=60
```

如果要拆分 LVLM 首 token 前耗时和流式解码耗时，可以额外开启 streaming timing：

```bash
export LLM_VISION_STREAM_TIMING=1
```

开启后 `manifest.json` 会记录：

- `generator_llm_time_to_first_chunk_ms`
- `generator_llm_time_to_first_token_ms`
- `generator_llm_stream_decode_ms`
- `generator_llm_stream_chunks`

如果要让 LVLM 只输出导航必需字段，并由本地恢复完整 `VisualPromptDecision`，可以开启极简输出协议：

```bash
export LVLM_MINIMAL_OUTPUT=1
```

极简输出协议要求模型只返回：

```json
{"r":0,"t":"c","p":[[[800,827],[631,827],[969,827]]],"c":0.91}
```

字段含义：

- `r`：是否 reject，`0|1`。
- `t`：终止语义，`c=continue_visible_corridor`，`u=reach_turn_entry`，`o=stop_before_occlusion`。
- `p`：3-6 个 corridor samples，每个 sample 是 `[center,left,right]` 三个 FPV 像素点。
- `c`：置信度。

后处理会把极简输出恢复成完整的 `path_corridor` 结构，并继续执行 segmentation validation / repair。

```bash
WORK_DIR=/tmp/viobot_sidewalk_zju_081025_15_1fps

python tools/run_sidewalk_images_visual_prompt_pipeline.py \
  --images-dir "$WORK_DIR/images" \
  --trajectory-json "$WORK_DIR/trajectory.json" \
  --output-dir "$WORK_DIR/output_all" \
  --sequence-name viobot_sidewalk_zju_081025_15 \
  --camera-view camera_front \
  --sam3-output-base-dir "$WORK_DIR/sam3" \
  --segmentation-policy optional \
  --enable-segmentation-overlay-input \
  --no-emit-llm-input-images \
  --enforce-alignment \
  --mode foot

python tools/run_sidewalk_images_visual_prompt_pipeline.py \
  --images-dir "$WORK_DIR/images" \
  --trajectory-json "$WORK_DIR/trajectory.json" \
  --output-dir "$WORK_DIR/output_all" \
  --sequence-name viobot_sidewalk_zju_081025_16 \
  --camera-view camera_front \
  --sam3-output-base-dir "$WORK_DIR/sam3" \
  --segmentation-policy optional \
  --sam3-env-name sam3 \
  --sam3-inference-script /home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/batch_sam3_demo_260124.py \
  --checkpoint-path /media/ubuntu/data1/checkpoints/checkpoint_35_merged.pt \
  --gpus 0 \
  --min-score 0.3 \
  --enable-segmentation-overlay-input \
  --enable-bev-map-provider \
  --enforce-alignment \
  --llm-concurrency 2 \
  --mode foot
```

如果 API rate limit 允许，可以打开 LVLM 并发。推荐先从 `2` 开始，观察失败率、限流和平均耗时：

```bash
python tools/run_sidewalk_images_visual_prompt_pipeline.py \
  --images-dir "$WORK_DIR/images" \
  --trajectory-json "$WORK_DIR/trajectory.json" \
  --output-dir "$WORK_DIR/output_all_concurrency2" \
  --sequence-name viobot_sidewalk_zju_081025_15 \
  --camera-view camera_front \
  --sam3-output-base-dir "$WORK_DIR/sam3" \
  --enable-segmentation-overlay-input \
  --no-emit-llm-input-images \
  --llm-concurrency 2 \
  --enforce-alignment \
  --mode foot

python tools/run_sidewalk_images_visual_prompt_pipeline.py \
  --images-dir "$WORK_DIR/images" \
  --trajectory-json "$WORK_DIR/trajectory.json" \
  --output-dir "$WORK_DIR/output_debug_one" \
  --sequence-name viobot_sidewalk_zju_081025_16 \
  --camera-view camera_front \
  --sam3-output-base-dir "$WORK_DIR/sam3" \
  --enable-segmentation-overlay-input \
  --enable-bev-map-provider \
  --enforce-alignment \
  --llm-concurrency 1 \
  --llm-image-max-side 1024 \
  --bev-map-output-size-px 512 \
  --frame-start 1 \
  --frame-end 60 \
  --frame-step 1 \
  --mode foot
```

说明：

- `--llm-concurrency 1`：默认串行，最稳。
- `--llm-concurrency 2`：推荐首选加速设置。
- `--llm-concurrency 4`：只有在模型 API 额度、QPS 和稳定性允许时再使用。
- 输出 `manifest.json` 会记录 `llm_concurrency`，并保持 `frames` 按输入帧顺序回填。
- 如果想只上传两张 LVLM 图片（numbered SoM 主图 + BEV extra），使用：
  `--llm-primary-image-role numbered_som_overlay --enable-bev-map-provider`。
  该模式会自动生成 SoM overlay 作为主图，不再把 raw FPV 作为单独图片上传；坐标仍对应 FPV/SoM 同一图像平面。
- 可用 `--llm-image-max-side` / `--llm-image-max-bytes` / `--llm-image-jpeg-quality` 调整 FPV、SoM、SAM visualization 的 LVLM 输入压缩；可用 `--bev-map-output-size-px` 调整 BEV 截图分辨率。
- `--segmentation-policy optional`：推荐缺省模式；有 SAM/SAM3 时启用 SoM、safe corridor 和 validation，没有 mask 时仍允许 FPV+BEV zero-shot baseline 输出。
- `--segmentation-policy required`：严格实验模式，缺少 segmentation artifacts 的帧直接失败。
- `--segmentation-policy disabled`：纯 FPV/BEV 消融模式，完全忽略 segmentation artifacts。

如果需要同时运行 SAM3，则追加 `--run-sam3`、`--sam3-env-name`、`--sam3-inference-script` 和 `--checkpoint-path` 等参数。

## 估算 LLM 延时拆分

如果你已经有一次 pipeline 的 `manifest.json`，可以直接做离线估算，不需要重新跑 API：

```bash
python tools/analyze_visual_prompt_llm_latency.py \
  --manifest /tmp/viobot_sidewalk_zju_081025_16_1fps/output_one/manifest.json \
  --upload-mbps 20 \
  --download-mbps 100 \
  --rtt-ms 80 \
  --top-n 5
```

常用参数说明：

- `--upload-mbps`：本地到 API 区域的假设上行带宽。
- `--download-mbps`：返回文本的假设下行带宽。
- `--rtt-ms`：假设的网络 RTT。
- `--probe-url`：可选，做一个粗粒度 HTTP 探测。
- `--json`：输出机器可读 JSON，便于后续自动化分析。

该脚本会基于现有 telemetry 估算：

- `network_est_ms`
- `service_est_ms`
- `time_to_first_token_ms`
- `stream_decode_ms`
- `payload_bytes`
- `prompt_tokens`
- `completion_tokens`

其中 `service_est_ms` 仍然包含 Ark 网关排队、视觉编码、模型 prefill 和解码。
