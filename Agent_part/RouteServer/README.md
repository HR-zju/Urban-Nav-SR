# RouteServer

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

- `generate_bev_prompt_proposals`
- `generate_pov_prompt_candidates`
- `generate_pov_prompt_candidates_v2`
- `run_zero_shot_visual_prompt_agent`
- `run_zero_shot_direct_arrow_agent`
- `run_visual_prompt_only_from_context`
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

### RouteContext / BEV / Proposal 调试

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

带 FPV 图像时会额外生成 RGB-aware PoV candidates：

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
