# UrbanNav MCP Server

`osm-mcp-server` 是一个面向位置理解、目的地推荐、路线规划和地图可视化的 MCP server。  
虽然包名历史上沿用了 `osm-mcp-server`，但当前代码已经支持多地图 provider：

- `osm`
- `amap`
- `baidu`
- `google`

项目重点不是“单纯查地图”，而是把自然语言需求转换成：

- 候选目的地召回
- 语义重排
- 路线规划
- BEV 路线可视化

同时仓库还包含一个 macOS SwiftUI 客户端 `apps/nav-assistant/`，用于语音输入、候选选择和地图展示。

## 核心能力

- 自然语言目的地推荐：`recommend_destination_and_waypoints`
- 语义 POI 检索：`search_nearby_pois`
- 粗粒度路线步骤：`get_coarse_waypoints`
- AMap 静态 BEV 路线图：`render_bev_route_snapshot`
- 图像理解：`analyze_image`
- 地址正反解码：`geocode_address` / `reverse_geocode`
- 会议点推荐、停车、充电、通勤、学区、社区分析等辅助工具

## 当前推荐链路

当前推荐流程已经改成两阶段：

1. 候选阶段  
   只做意图解析、候选召回、语义排序、轻量距离/时长估计

2. 路线阶段  
   只对最终 `chosen_destination` 计算一次真实路线  
   用户在交互里改选其他候选时，再单独请求该候选的路线

这样做的目的是避免对每个候选都调用一次地图 route API，降低延迟。

## Provider 支持

### OSM

默认 provider。

- Geocoding: Nominatim
- Routing: OSRM
- Nearby/category search: Overpass

启动：

```bash
export MAP_PROVIDER=osm
osm-mcp-server
```

### AMap

适合中国大陆场景。

- Geocoding / reverse geocoding
- POI around / text search
- Walking / driving coarse route
- Static map BEV snapshot

启动：

```bash
export MAP_PROVIDER=amap
export AMAP_KEY=your_amap_key
osm-mcp-server
```

### Baidu

```bash
export MAP_PROVIDER=baidu
export BAIDU_AK=your_baidu_ak
osm-mcp-server
```

### Google Maps

```bash
export MAP_PROVIDER=google
export GOOGLE_MAPS_API_KEY=your_google_maps_api_key
export GOOGLE_MAPS_LANGUAGE=en
export GOOGLE_MAPS_REGION=CN
osm-mcp-server
```

Google provider 当前使用：

- Geocoding API
- Places API (New)
- Routes API

注意：

- 需要在对应 Google Cloud project 里启用这些 API
- 在受限网络环境下，通常还需要设置 `HTTPS_PROXY` / `HTTP_PROXY`

## LLM 能力

LLM 主要用于：

- 自然语言意图解析
- 候选语义重排
- 英文指令到中文 provider-facing query 的本地化
- 中文结果回英文
- 图像理解

支持 OpenAI-compatible endpoint，也支持 Volcengine ARK。

### OpenAI-compatible

```bash
export LLM_API_KEY=your_api_key
export LLM_MODEL=gpt-4o-mini
export LLM_API_BASE_URL=https://api.openai.com/v1
```

### Volcengine ARK / Doubao

```bash
export LLM_PROVIDER=ark
export ARK_API_KEY=your_ark_api_key
export LLM_MODEL=doubao-seed-2-0-pro-260215
export LLM_API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
export LLM_API_STYLE=chat_completions
```

常用调优参数：

```bash
export LLM_TIMEOUT_S=20
export LLM_FEW_SHOT_MAX_EXAMPLES=6
```

如果意图解析经常超时，可以先降到：

```bash
export LLM_FEW_SHOT_MAX_EXAMPLES=2
```

## 安装

### 本地开发安装

```bash
pip install -e .
```

如需 macOS CoreLocation：

```bash
pip install -e ".[macos]"
```

### 在 MCP Host 中使用

```json
{
  "mcpServers": {
    "osm-mcp-server": {
      "command": "uvx",
      "args": ["osm-mcp-server"]
    }
  }
}
```

本地未发布开发版本可用：

```json
{
  "mcpServers": {
    "osm-mcp-server": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/open-streetmap-mcp",
        "run",
        "osm-mcp-server"
      ]
    }
  }
}
```

## 主要工具

### 推荐与规划

- `recommend_destination_and_waypoints`
  - 输入自然语言需求和当前位置
  - 返回候选目的地、选中目的地、路线摘要、关键点、细 polyline

- `get_coarse_waypoints`
  - 输入起终点
  - 返回 provider route 的语义步骤和 coarse waypoints

- `get_route_context`
  - 输入当前 `gps`、`heading_deg`、`route_polyline`、`coarse_waypoints`
  - 输出统一 `RouteContext`
  - 适合后续 PromptProposal / LVLM-Arbiter / TemporalRefine 消费
  - 它不重新规划路线，只做当前时刻的局部运行时上下文整理

- `render_bev_route_snapshot`
  - 仅在 `MAP_PROVIDER=amap` 下可用
  - 根据起终点和路线 polyline 生成高德静态 BEV 图 URL
  - 可按固定频率轮询刷新

- `render_bev_route_context_snapshot`
  - 仅在 `MAP_PROVIDER=amap` 下可用
  - 在静态 BEV 图上叠加 `RouteContext`
  - 可视化 full route、local segment、connector polyline、current pose、matched point

### 检索与解析

- `search_nearby_pois`
- `geocode_address`
- `reverse_geocode`
- `find_nearby_places`
- `search_category`

### 其他分析工具

- `analyze_image`
- `suggest_meeting_point`
- `find_schools_nearby`
- `analyze_commute`
- `find_ev_charging_stations`
- `analyze_neighborhood`
- `find_parking_facilities`

## 路线与 BEV 输出说明

### 1. 推荐返回的路线信息

`recommend_destination_and_waypoints` 现在会返回：

- `recommendations[*].route_summary`
  候选阶段默认是轻量估计值

- `chosen_destination`
  只对它计算一次真实路线

- `key_waypoints`
  粗粒度关键路径点

- `route_polyline`
  更细的路线 polyline，适合地图渲染

### 1.5 RouteContext 运行时上下文

`get_route_context` 会把：

- 当前 GPS / heading
- `route_polyline`
- `coarse_waypoints`

压缩成一个稳定的局部导航上下文对象：

- `pose_match`
  - `matched_point`
  - `matched_segment_index`
  - `matched_segment_ratio`
  - `cross_track_error_m`
  - `heading_delta_deg`
  - `off_route`
  - `route_confidence`

- `progress`
  - `current_waypoint_index`
  - `next_waypoint_index`
  - `remaining_distance_m`
  - `distance_to_next_waypoint_m`
  - `distance_to_next_turn_m`
  - `next_turn_type`
  - `progress_ratio`

- `local_semantics`
  - `turn_type`
  - `turn_semantics`
  - `corridor_type`
  - `short_action_summary`
  - `instruction_hint`
  - `road_name`

- `geometry`
  - `local_segment_polyline`
  - `connector_polyline`
  - `connector_active`
  - `lookahead_m`

它的设计目标不是替代地图 provider，而是为后续具身导航 Brain 提供一个轻量、连续、结构化的 route runtime state。

其中：

- `local_segment_polyline`
  表示已经并入规划路径后的局部路线段

- `connector_polyline`
  表示从当前 GPS 到 provider 路线起点/并入点的几何引导段

- `connector_active`
  为 `true` 时，说明当前仍处于“接近路线起点、准备并入正式路径”的启动阶段

### 2. AMap 静态 BEV 图

`render_bev_route_snapshot` 返回：

- `image_url`
- `image_url_cache_busted`
- `route_polyline`
- `refresh_interval_s`

它不是视频流，而是“静态图按频率刷新”。

### 2.5 RouteContext BEV Overlay

`render_bev_route_context_snapshot` 会把 `RouteContext` 中可视化的部分直接叠加到 AMap 静态图：

- 绿色：full route
- 蓝色：local segment
- 橙色：connector polyline
- `C`：current pose
- `M`：matched point
- `D`：destination

当 `connector_active = true` 时，说明当前仍在接近 provider 路线起点，橙色 connector 会显式展示从当前 GPS 到路线并入点的引导段。

### 3. macOS app 本地渲染

`apps/nav-assistant/` 现在优先使用服务端返回的 `route_polyline` 在 `MapKit` 里本地渲染，只有拿不到细 polyline 时才退回 `key_waypoints`。

这意味着当前仓库里有两种 BEV 展示方式：

- AMap static map URL
- macOS app 本地 `MapPolyline`

## 示例客户端

### 基础示例

```bash
python examples/client.py
```

### Prompt 推荐

```bash
python examples/prompt_recommendation_client.py \
  --prompt "Find a quiet cafe nearby with fast driving access" \
  --lat 37.7749 \
  --lon -122.4194 \
  --mode car \
  --max-candidates 3 \
  --radius 2500
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

这个脚本会：

- 用当前位置生成候选
- 先返回候选列表
- 只在用户确认选项后再补该候选路线

### RouteContext

```python
{
  "gps": {"lat": 30.2645, "lng": 120.1180},
  "heading_deg": 35.0,
  "route_polyline": [...],
  "coarse_waypoints": [...],
  "lookahead_m": 20.0
}
```

你可以先调用 `recommend_destination_and_waypoints` 或 `get_coarse_waypoints` 拿到 `route_polyline` 和 `coarse_waypoints`，再把它们送进 `get_route_context`。

最直接的测试方法是运行：

```bash
python examples/route_context_client.py \
  --prompt "我想去西湖" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --heading-deg 35 \
  --lookahead-m 20
```

它会先调用 `recommend_destination_and_waypoints`，再自动调用 `get_route_context`，最后打印标准 `RouteContext` JSON。

### 图像理解

```bash
python examples/image_understanding_client.py \
  --image-url "https://ark-project.tos-cn-beijing.ivolces.com/images/view.jpeg" \
  --question "What place is shown in this image?"
```

### AMap BEV 快照轮询

```bash
export MAP_PROVIDER=amap
export AMAP_KEY=your_amap_key

python examples/amap_bev_snapshot_client.py \
  --prompt "I want to go to the WestLake" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --refresh-interval-s 3 \
  --iterations 5
```

## 测试建议

### 1. 静态语法检查

如果你刚同步了 `RouteContext` 相关代码，先跑：

```bash
python -m py_compile \
  src/mcp_server/route_context/__init__.py \
  src/mcp_server/route_context/common.py \
  src/mcp_server/route_context/schema.py \
  src/mcp_server/route_context/match_pose.py \
  src/mcp_server/route_context/progress.py \
  src/mcp_server/route_context/describe_segment.py \
  src/mcp_server/server.py
```

### 2. 推荐链路测试

先确认推荐输出已经包含：

- `route_polyline`
- `coarse_waypoints` 或 `key_waypoints`

可以直接运行：

```bash
python examples/macos_prompt_recommendation_client.py \
  --prompt "我想去西湖" \
  --mode foot \
  --radius 3000 \
  --lat 30.2645 \
  --lon 120.1180
```

### 3. RouteContext 测试

建议直接跑新的示例客户端：

```bash
python examples/route_context_client.py \
  --prompt "我想去西湖" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --heading-deg 35
```

重点检查输出里的这些字段：

- `pose_match.cross_track_error_m`
- `pose_match.heading_delta_deg`
- `pose_match.off_route`
- `progress.distance_to_next_turn_m`
- `progress.next_turn_type`
- `local_semantics.turn_semantics`
- `local_semantics.corridor_type`
- `local_semantics.short_action_summary`

### 4. 手工偏航测试

把 `gps` 人为改到路线外 20 到 50 米，再调用一次 `get_route_context`，确认：

- `pose_match.cross_track_error_m` 变大
- `pose_match.off_route` 变成 `true`
- `local_semantics.turn_semantics` 会切到 `off_route_recovery`

### 5. Heading 测试

只改 `heading_deg`，不改 GPS，验证 `corridor_type` 是否随朝向变化：

- 朝向基本对齐路线时，应接近 `front`
- 朝向偏左时，应接近 `slight_left_front` 或 `left_front`
- 朝向偏右时，应接近 `slight_right_front` 或 `right_front`

### 6. MCP Inspector 测试

如果你想直接在 MCP 层验证 tool 输入输出：

```bash
npx @modelcontextprotocol/inspector \
  uv --directory /path/to/open-streetmap-mcp run osm-mcp-server
```

然后手动调用：

- `recommend_destination_and_waypoints`
- `get_route_context`

这对检查 JSON schema 和字段稳定性最直接。

## macOS 客户端

SwiftUI 客户端位于：

- [`apps/nav-assistant/`](apps/nav-assistant/)

主要功能：

- 语音输入 / 文本输入
- 候选目的地选择
- 规划结果桥接
- MapKit 路线可视化

运行说明见：

- [`apps/nav-assistant/README.md`](apps/nav-assistant/README.md)

## 性能与超时调优

### 地图 HTTP

```bash
export MAP_HTTP_TOTAL_TIMEOUT_S=60
export MAP_HTTP_CONNECT_TIMEOUT_S=12
export MAP_HTTP_SOCK_READ_TIMEOUT_S=30
export MAP_HTTP_TRUST_ENV=1
```

### 代理

如果地图或 Google API 走代理：

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
```

### 推荐链路

```bash
export GEOCODE_CONCURRENCY=4
export REGIONAL_EXPANSION_CONCURRENCY=3
export ROUTE_EVAL_CONCURRENCY=2
```

虽然代码里保留了这些并发参数，但当前推荐主链路已经改成“只对最终 chosen destination 算真实路线”，所以真正的热点不再是“多候选逐个 route planning”。

### 静态图路线采样

```bash
export STATIC_ROUTE_MAX_POINTS=240
export AMAP_STATIC_ROUTE_MAX_POINTS=240
```

如果你想让静态 BEV 图尽量保留更多原始 polyline 点，可以把它们再调高。

## 英文输入与中文地图 provider

对 `amap` / `baidu`：

- 英文输入会先翻译成中文用于 provider 检索
- 输出结果会尝试再翻回英文

如果结果翻译失败，系统会退回 provider 原始语言文本，但检索仍然会继续。

## 已知限制

- `render_bev_route_snapshot` 目前只支持 `MAP_PROVIDER=amap`
- AMap 静态图是“静态图轮询刷新”，不是原生导航 SDK 视频流
- 英文地名在大陆 provider 上通常不如中文稳定
- Google provider 依赖额外 API 开通和网络可达性
- Swift app 的本地 polyline 渲染更适合交互式地图，静态图更适合截图或外部展示

## 开发与调试

### 运行 server

```bash
osm-mcp-server
```

### 构建分发

```bash
uv sync
uv build
```

### MCP Inspector

```bash
npx @modelcontextprotocol/inspector uv --directory /path/to/open-streetmap-mcp run osm-mcp-server
```
