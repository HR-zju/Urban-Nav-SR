# RouteServer

> 重构状态：默认 MCP server 已移除 `run_zero_shot_visual_prompt_agent`，且不再加载旧 visual-prompt/brain 管线。下方相关示例属于待退役的历史说明，不适用于当前默认服务；历史复现请使用 Git 提交 `4879be4` 的隔离 checkout。地图、RouteSession 和 episode runtime 保留。

RouteServer 是 UrbanNav 的后端导航与视觉提示服务。它通过 MCP tools 对外提供：

- 自然语言目的地推荐
- 多目标 itinerary 拆解
- 动态途经点 route session
- RouteContext / LocalBEVContext
- 本地 FPV 图像理解
- zero-shot visual prompt overlay
- AMap 静态 BEV 路线可视化

历史包名仍是 `osm-mcp-server`，但当前已经支持多地图 provider。

## Quick Start

### 安装

```bash
cd Agent_part/RouteServer
pip install -e .
```

### 启动

OSM 默认模式：

```bash
export MAP_PROVIDER=osm
osm-mcp-server
```

中国大陆建议使用 AMap：

```bash
export MAP_PROVIDER=amap
export AMAP_KEY=your_amap_key
osm-mcp-server
```

Volcengine ARK / Doubao:

```bash
export LLM_PROVIDER=ark
export ARK_API_KEY=your_ark_api_key
export LLM_MODEL=doubao-seed-2-0-pro-260215
export LLM_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
export LLM_API_STYLE=chat_completions
```

通用 OpenAI-compatible endpoint:

```bash
export LLM_API_KEY=your_api_key
export LLM_MODEL=gpt-4o-mini
export LLM_API_BASE_URL=https://api.openai.com/v1
```

## 主链路

当前推荐使用的在线主链路是：

```text
user instruction + GPS + heading + FPV image
-> Route Session / Route Plan
-> RouteContext
-> mode-aware VisualPromptAgent
-> SVG visual prompt overlay
```

BEV proposal / PoV candidates 仍然保留，但定位为：

- teacher/debug prior
- 可解释中间结果
- 后续蒸馏数据来源

它们不再是在线 visual prompt 的强制决策路径。

更详细的数学与模块关系见：

- [docs/architecture.md](docs/architecture.md)
- [docs/FRAMEWORK_MATH_MODEL.md](docs/FRAMEWORK_MATH_MODEL.md)
- [docs/URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md](docs/URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md)
- [docs/EVALUATION.md](docs/EVALUATION.md)
- [docs/README.md](docs/README.md)

UrbanNav Module 1 的标准 benchmark 入口是：

```bash
PYTHONPATH=src python tools/evaluation/run_urbannav_module1_benchmark.py \
  --dataset data/benchmark/urbannav_global_osm_local_full_en_v1/task_gt.jsonl \
  --runner external \
  --predictions /path/to/model_predictions.jsonl \
  --output-dir data/benchmark/urbannav_module1_runs/<model_name>
```

调试工具、测试建议和性能参数见：

- [docs/debugging.md](docs/debugging.md)

## Provider

| Provider | 环境变量 | 能力 |
| --- | --- | --- |
| `osm` | `MAP_PROVIDER=osm` | Nominatim / OSRM / Overpass |
| `amap` | `MAP_PROVIDER=amap`, `AMAP_KEY` | POI、路线、静态图，中国大陆推荐 |
| `baidu` | `MAP_PROVIDER=baidu`, `BAIDU_AK` | 百度地图基础能力 |
| `google` | `MAP_PROVIDER=google`, `GOOGLE_MAPS_API_KEY` | Geocoding / Places / Routes |

## Core Tools

### Recommendation / Planning

- `recommend_destination_and_waypoints`
  - 自然语言需求 + 当前位置 -> 目的地推荐 + 路线

- `decompose_itinerary_goals`
  - 多目标需求 -> 有序 goals

- `recommend_itinerary_and_waypoints`
  - 多目标需求 -> via points + final destination + combined route plan

- `plan_route_with_via_points`
  - origin + via points + destination -> multi-leg route

### Route Session

- `create_route_session`
- `get_route_session`
- `list_route_sessions`
- `add_via_point_to_session`
- `remove_via_point_from_session`
- `replace_destination_in_session`
- `replan_route_session_from_origin`

session 默认持久化到：

```text
~/.urban_nav/route_sessions.json
```

可配置：

```bash
export ROUTE_SESSION_STORE=file
export ROUTE_SESSION_STORE_PATH=/path/to/route_sessions.json
```

如果只想使用内存态 demo：

```bash
export ROUTE_SESSION_STORE=memory
```

`get_route_session` 默认返回缓存的 `route_plan`，不会重新请求地图 API。需要强制刷新路线时传：

```json
{"force_replan": true}
```

### Route Context / BEV

- `get_route_context`
- `get_local_bev_context`
- `get_route_state`
- `render_local_bev_patch`
- `render_bev_route_context_snapshot`
- `render_itinerary_bev_snapshot`

### Prompt / Vision

- `run_zero_shot_visual_prompt_agent`
- `analyze_image`
- `analyze_local_image`

## Examples

当前只保留主线 examples。

### FPV 图像理解

```bash
python examples/image_understanding_client.py \
  --image-path /home/ubuntu/WorkSpace/dataset/urban.jpeg \
  --question "Describe the navigable directions and obstacles in this FPV image."
```

### RouteContext / BEV 调试

```bash
python examples/route_context_debug_client.py \
  --prompt "我想去西湖" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --heading-deg 35 \
  --lookahead-m 20 \
  --open
```

带 FPV 图像时会额外调用当前 visual prompt agent：

```bash
python examples/route_context_debug_client.py \
  --prompt "我想去西湖" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --heading-deg 35 \
  --lookahead-m 20 \
  --image-path /home/ubuntu/WorkSpace/dataset/urban.jpeg
```

### Zero-shot Visual Prompt

```bash
python examples/zero_shot_visual_prompt_agent_client.py \
  --prompt "我想去西湖" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --heading-deg 35 \
  --lookahead-m 20 \
  --image-path /home/ubuntu/WorkSpace/dataset/urban.jpeg \
  --save-overlay /tmp/visual_prompt.svg
```

使用 iPhone 原图 EXIF pose 测试固定路线：

```bash
python examples/yuquan_visual_prompt_agent_client.py \
  --image-path /home/ubuntu/WorkSpace/project/SR/test4.jpg \
  --save-overlay /tmp/yuquan_visual_prompt_overlay.svg
```

默认路线是 `浙江大学玉泉校区南门` 到 `浙江大学玉泉校区北门`。脚本会从图片 EXIF 读取 `lat/lon/GPSImgDirection`，并用它作为当前 FPV pose；如果只想排查路线起点附近的 prompt，可加 `--gps-source origin`。

给 `yuquan_visual_prompt_agent_client.py` 接入 SAM3 可通行区域约束：

```bash
python examples/yuquan_visual_prompt_agent_client.py \
  --image-path /home/ubuntu/WorkSpace/project/SR/test4.jpg \
  --fpv-segmentation-npy /path/to/000005_segmentation.npy \
  --fpv-segmentation-class-mapping /path/to/class_mapping.json \
  --fpv-segmentation-visualization /path/to/000005_visualization.jpg \
  --enable-segmentation-overlay-input \
  --enforce-alignment \
  --save-overlay /tmp/yuquan_visual_prompt_overlay.svg \
  --save-json /tmp/yuquan_visual_prompt_result.json
```

当 `fpv_segmentation` 存在时，主链会同时：

- 把 segmentation summary 提供给 visual prompt generator / verifier
- 对最终 `rendered_prompt` 做基于可通行类别的硬校验
- 在输出 JSON 中返回 `agent.fpv_segmentation_summary`

可选三路输入与对齐参数：

- `--enable-segmentation-overlay-input`：让 A/V arbiter 在主图之外再接收 segmentation overlay（默认关闭）
- `--fpv-segmentation-overlay-image-path` / `--fpv-segmentation-overlay-image-url`：显式指定 overlay 图像输入（会透传到 MCP 参数 `fpv_segmentation_overlay_image_path/url`）
- `--enforce-alignment` / `--no-enforce-alignment`：控制 segmentation 与图像平面对齐失败时是否继续（默认强制）

如果想把 “运行 SAM3” 和 “运行 visual prompt demo” 合并成一个命令，可用：

```bash
WORK_DIR=/home/ubuntu/WorkSpace/dataset/sidewalk_he4_adapter
OUT_DIR=/home/ubuntu/WorkSpace/dataset/sidewalk_visual_prompt_55_200_rerun
SAM3_DIR=/home/ubuntu/WorkSpace/dataset/sidewalk_he4_adapter_sam3

cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer

python tools/model_processing/run_sidewalk_images_visual_prompt_pipeline.py \
  --images-dir "$WORK_DIR/images" \
  --trajectory-json "$WORK_DIR/trajectory.json" \
  --output-dir "$OUT_DIR" \
  --sequence-name sidewalk_he4_adapter \
  --camera-view camera_front \
  --sam3-output-base-dir "$SAM3_DIR" \
  --run-sam3 \
  --sam3-env-name sam3 \
  --sam3-inference-script /home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/batch_sam3_demo_260124.py \
  --checkpoint-path /media/ubuntu/data1/checkpoints/checkpoint_35_merged.pt \
  --gpus 0 \
  --min-score 0.3 \
  --frame-start 55 \
  --frame-end 200 \
  --frame-step 5 \
  --enable-segmentation-overlay-input \
  --enforce-alignment \
  --mode foot

```

按下面这一版参数也可以直接运行（与上面同用途，保留你当前使用的参数写法）：

```bash
python tools/run_sam3_visual_prompt_pipeline.py \
  --repo-root /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer \
  --client yuquan \
  --run-sam3 \
  --sam3-env-name sam3 \
  --sam3-script prepare_frames_and_run_batch_sam3.py \
  --rides-base-dir /home/ubuntu/WorkSpace/dataset/output_rides_0 \
  --frames-base-dir /home/ubuntu/WorkSpace/dataset/output_rides_0_frames \
  --sam3-output-base-dir /home/ubuntu/WorkSpace/dataset/output_rides_0_sam3_frame_results \
  --sequence-name ride_16446_20240115041752 \
  --camera-view camera_front \
  --fps 1.0 \
  --gpus 0 \
  --min-score 0.3 \
  --checkpoint-path checkpoint_35_merged.pt \
  --frame-name 000005 \
  --alignment-mode error \
  --client-args --image-path /home/ubuntu/WorkSpace/project/SR/test4.jpg --gps-source image --enable-segmentation-overlay-input --enforce-alignment --save-overlay /tmp/yuquan_visual_prompt_overlay.svg --save-json /tmp/yuquan_visual_prompt_result.json \
  2>&1 | tee sam3_pipeline.log
```

如果 SAM3 结果已经存在，只想直接跑 demo，可去掉 `--run-sam3` 和 SAM3 预处理参数，保留 `--sam3-output-base-dir`、`--sequence-name`、`--camera-view`、`--frame-name` 即可。

如果想对整个序列逐帧输出 visual prompt（每帧读取 RGB 和 ride pose、每帧落盘 overlay/json），可用：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer
python tools/run_sequence_visual_prompt_pipeline.py \
  --provider sam3 \
  --repo-root /home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer \
  --client yuquan \
  --run-sam3 \
  --sam3-env-name sam3 \
  --sam3-script /home/ubuntu/WorkSpace/project/SR/Code/sam3/scripts/prepare_frames_and_run_batch_sam3.py \
  --rides-base-dir /home/ubuntu/WorkSpace/dataset/output_rides_0 \
  --frames-base-dir /home/ubuntu/WorkSpace/dataset/output_rides_0_frames \
  --sam3-output-base-dir /home/ubuntu/WorkSpace/dataset/output_rides_0_sam3_frame_results \
  --sequence-name ride_16446_20240115041752 \
  --camera-view camera_front \
  --frame-start 1 \
  --frame-end 50 \
  --frame-step 1 \
  --pose-source ride \
  --checkpoint-path /media/ubuntu/data1/checkpoints/checkpoint_35_merged.pt \
  --output-dir /tmp/sequence_visual_prompt \
  --continue-on-error \
  --client-args --destination-name 浙江大学玉泉校区北门 杭州 --mode foot
```
```bash
conda run -n UrbanNav python tools/model_processing/run_sidewalk_images_visual_prompt_pipeline.py \
  --images-dir /home/ubuntu/WorkSpace/dataset/sidewalk_he4_adapter/images \
  --trajectory-json /home/ubuntu/WorkSpace/dataset/sidewalk_he4_adapter/trajectory.json \
  --sequence-name sidewalk_he4_adapter \
  --camera-view camera_front \
  --sam3-output-base-dir /home/ubuntu/WorkSpace/dataset/sidewalk_he4_adapter_sam3 \
  --output-dir /home/ubuntu/WorkSpace/dataset/sidewalk_visual_prompt \
  --frame-start 50 --frame-end 52 --frame-step 1 \
  --enable-segmentation-overlay-input \
  --mode foot
```

输出目录会包含：

- `overlay/{frame}.svg`: 每帧叠加图
- `json/{frame}.json`: 每帧结构化结果
- `logs/{frame}.log`: 每帧执行日志
- `manifest.json`: 序列级汇总（成功/失败帧统计）

### 动态途经点

直接追加坐标途经点：

```bash
python examples/route_session_client.py \
  --origin 120.1180,30.2645 \
  --destination 120.116555,30.270429 \
  --mode foot \
  --add-via 120.118376,30.257515 \
  --add-via 120.12601,30.26017
```

自然语言追加途经点：

```bash
python examples/realtime_instruction_session_client.py \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --init-instruction "我想去西湖" \
  --append-instruction "我有点累，先找个地方坐下休息" \
  --append-instruction "再找个地方吃晚餐" \
  --radius 3000
```

### 多目标 Itinerary

```bash
python examples/itinerary_client.py \
  --prompt "我今天累了，想先休息，吃顿晚餐，再看个电影" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --render-map
```

### macOS 交互式推荐

```bash
python examples/macos_prompt_recommendation_client.py \
  --prompt "我想去西湖" \
  --mode foot \
  --radius 3000 \
  --lat 30.2645 \
  --lon 120.1180
```

该脚本是 macOS 专用入口，用于读取/确认当前位置；Ubuntu 环境可以忽略，但保留。

## Runtime Tuning

地图 HTTP：

```bash
export MAP_HTTP_TOTAL_TIMEOUT_S=60
export MAP_HTTP_CONNECT_TIMEOUT_S=12
export MAP_HTTP_SOCK_READ_TIMEOUT_S=30
export MAP_HTTP_TRUST_ENV=1
```

LLM：

```bash
export LLM_TIMEOUT_S=30
export LLM_VISION_TIMEOUT_S=60
export LLM_ITINERARY_TIMEOUT_S=60
export LLM_FEW_SHOT_MAX_EXAMPLES=4
```

AMap 静态图采样：

```bash
export STATIC_ROUTE_MAX_POINTS=240
export AMAP_STATIC_ROUTE_MAX_POINTS=240
export AMAP_ITINERARY_STATIC_MAX_POINTS=120
```

## macOS Client

SwiftUI 客户端位于：

- [apps/nav-assistant/](apps/nav-assistant/)

说明见：

- [apps/nav-assistant/README.md](apps/nav-assistant/README.md)

## Development

```bash
python -m py_compile src/mcp_server/server.py examples/*.py
```

MCP Inspector：

```bash
npx @modelcontextprotocol/inspector uv --directory /path/to/RouteServer run osm-mcp-server
```

构建：

```bash
uv sync
uv build
```

## Known Limits

- `render_bev_route_snapshot` / `render_itinerary_bev_snapshot` 当前只支持 `MAP_PROVIDER=amap`。
- AMap 静态图是静态 URL，不是原生导航 SDK。
- `RouteSession` 的 JSON file store 适合单机后端；多实例部署建议替换成 Redis / DB。
- 多目标 itinerary 当前是逐 goal 选择 + multi-leg route，尚未实现全局组合优化。
- zero-shot visual prompt 是 teacher/agent 原型，仍需真实 FPV 数据回放验证。
