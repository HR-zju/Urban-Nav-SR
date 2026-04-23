# Debugging And Operations

## 1. Recommended Smoke Tests

### Syntax

```bash
python -m py_compile src/mcp_server/server.py examples/*.py
```

### Image Understanding

```bash
python examples/image_understanding_client.py \
  --image-path /home/ubuntu/WorkSpace/dataset/urban.jpeg \
  --question "Describe the navigable directions and obstacles in this FPV image."
```

### RouteContext / BEV / Proposal

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

### Visual Prompt

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

带 SAM3 segmentation 约束：

```bash
python examples/yuquan_visual_prompt_agent_client.py \
  --image-path /home/ubuntu/WorkSpace/project/SR/test4.jpg \
  --fpv-segmentation-npy /path/to/000005_segmentation.npy \
  --fpv-segmentation-class-mapping /path/to/class_mapping.json \
  --fpv-segmentation-visualization /path/to/000005_visualization.jpg \
  --save-overlay /tmp/yuquan_visual_prompt_overlay.svg \
  --save-json /tmp/yuquan_visual_prompt_result.json
```

单命令串联 SAM3 和 visual prompt demo：

```bash
python tools/run_sam3_visual_prompt_pipeline.py \
  --client yuquan \
  --run-sam3 \
  --rides-base-dir /path/to/rides \
  --frames-base-dir /path/to/frames \
  --sam3-output-base-dir /path/to/sam3_outputs \
  --sequence-name ride_16446_20240115041752 \
  --camera-view camera_front \
  --fps 1.0 \
  --checkpoint-path /media/ubuntu/data1/checkpoints/checkpoint_35_merged.pt \
  --frame-name 000005 \
  --client-args -- \
  --image-path /home/ubuntu/WorkSpace/project/SR/test4.jpg \
  --save-overlay /tmp/yuquan_visual_prompt_overlay.svg \
  --save-json /tmp/yuquan_visual_prompt_result.json
```

### Dynamic Via Points

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

## 2. RouteContext Checks

关键字段：

- `pose_match.matched_segment_index`
- `pose_match.matched_segment_ratio`
- `pose_match.cross_track_error_m`
- `pose_match.heading_delta_deg`
- `pose_match.off_route`
- `progress.distance_to_next_turn_m`
- `progress.next_turn_type`
- `local_semantics.turn_semantics`
- `local_semantics.corridor_type`
- `local_semantics.short_action_summary`

正常轨迹里：

- `matched_segment_index` 应大体单调递增
- `distance_to_next_turn_m` 应平滑下降，经过 turn 后跳到下一个关键点
- `off_route` 不应在正常直行时频繁误报
- `corridor_type` 不应在连续帧高频抖动

## 3. Capture / Replay

采集：

```bash
python tools/route_context_capture.py \
  --trajectory /tmp/trajectory.jsonl \
  --prompt "我想去西湖" \
  --mode foot \
  --radius 3000 \
  --lookahead-m 20 \
  --output /tmp/route_state_capture.json
```

生成回放页面：

```bash
python tools/route_context_debugger.py \
  --capture /tmp/route_state_capture.json \
  --output /tmp/route_context_debugger.html
```

指标：

```bash
python tools/route_context_metrics.py \
  --capture /tmp/route_state_capture.json
```

## 4. AMap Static Map

AMap 静态图相关工具：

- `render_bev_route_snapshot`
- `render_bev_route_context_snapshot`
- `render_itinerary_bev_snapshot`

常见问题：

- `INVALID_USER_KEY`: AMap key 错误或服务未开通
- `CUQPS_HAS_EXCEEDED_THE_LIMIT`: QPS 超限
- `UNKNOWN_ERROR / 20003`: 静态图 URL 参数过长或 paths/markers 不兼容

调优：

```bash
export STATIC_ROUTE_MAX_POINTS=240
export AMAP_STATIC_ROUTE_MAX_POINTS=240
export AMAP_ITINERARY_STATIC_MAX_POINTS=120
```

如果 itinerary 静态图失败，优先打开返回里的：

- `route_only_image_url`
- `markers_only_image_url`
- `leg_overlay_image_url`

## 5. Route Session Store

默认文件持久化：

```text
~/.urban_nav/route_sessions.json
```

配置：

```bash
export ROUTE_SESSION_STORE=file
export ROUTE_SESSION_STORE_PATH=/path/to/route_sessions.json
```

纯内存：

```bash
export ROUTE_SESSION_STORE=memory
```

QPS 防护：

- `create/add/remove/replace/replan` 会重新规划并缓存 `route_plan`
- `get_route_session` 默认返回缓存
- 只有 `force_replan=true` 才重新请求地图 API

## 6. Timeout Tuning

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

代理：

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
```

## 7. MCP Inspector

```bash
npx @modelcontextprotocol/inspector uv --directory /path/to/RouteServer run osm-mcp-server
```
