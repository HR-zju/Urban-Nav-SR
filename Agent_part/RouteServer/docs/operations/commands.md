# RouteServer 常用运行指令

本文档记录 RouteServer 侧数据抽取、批处理和调试常用命令。所有命令默认在以下目录执行：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer
```

## 从 MCAP 抽取 Sidewalk Pipeline 输入

用途：把单个 ROS2 MCAP 数据包转换成
`mcp_server.visualization.frame_io` 可读取的帧与轨迹格式：

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

## 已退役的 Visual Prompt Pipeline

旧生成/验证流水线及 CLI 已移除，当前默认 server 和 teacher 均不提供该能力。
历史复现应使用隔离 checkout `4879be4`，不要把旧命令指向当前 checkout。
地图渲染使用 `mcp_server.visualization.map_overlay`；当前闭环参见 [architecture.md](../design/architecture.md)。

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
