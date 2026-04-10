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

如果多模态图像分析经常超时，可以单独提高视觉超时：

```bash
export LLM_VISION_TIMEOUT_S=60
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

- `get_local_bev_context`
  - 输入当前 `gps`、`heading_deg`、`route_polyline`、`coarse_waypoints`
  - 输出显式 `LocalBEVContext`
  - 提供局部 BEV 坐标系下的 route/local segment/connector/next turn 几何
  - 适合 PromptProposal 和 BEV-to-PoV 对齐

- `get_route_state`
  - 同时输出：
    - `route_context`
    - `local_bev_context`
  - 适合后续 Brain 一次性消费两路输入

- `render_local_bev_patch`
  - 输入当前 `gps`、`heading_deg`、`route_polyline`、`coarse_waypoints`
  - 返回局部姿态对齐的 `LocalBEVContext` SVG patch
  - 输出：
    - `local_bev_context`
    - `route_context`
    - `prompt_proposals`
    - `svg`
    - `svg_data_url`
  - 当前会把 BEV proposal sectors 直接叠加到 patch 上
  - 适合 PromptProposal 试验和可视化调试

- `generate_bev_prompt_proposals`
  - 输入：
    - `RouteContext` 所需字段
    - `LocalBEVContext` 所需字段
  - 输出 route-prior-only 的 `PromptProposal v1`
  - 当前不读取 RGB
  - 适合作为：
    - PoV projection 的前置 teacher
    - LVLM-Arbiter 的候选先验

- `generate_pov_prompt_candidates`
  - 输入：
    - `RouteContext` 所需字段
    - `LocalBEVContext` 所需字段
    - `image_width`
    - `image_height`
  - 输出 `PoV Prompt Projection v1`
  - 当前是弱几何投影：
    - `BEV prompt proposals -> image-plane prompt candidates`
  - 不读取 RGB
  - 适合作为后续 LVLM-Arbiter 的图像平面候选先验

- `generate_pov_prompt_candidates_v2`
  - 输入：
    - `RouteContext` 所需字段
    - `LocalBEVContext` 所需字段
    - `image_path` 或 `image_url`
  - 输出 **RGB-aware PromptProposal v2**
  - 会把：
    - route-prior teacher
    - image-space visual region candidates
    融合成更贴近当前 FPV 的 PoV candidates
  - 适合作为后续：
    - LVLM-Arbiter
    - TemporalRefine
    - teacher dataset logging

- `run_zero_shot_visual_prompt_agent`
  - 输入：
    - `RouteContext` 所需字段
    - `LocalBEVContext` 所需字段
    - `image_path` 或 `image_url`
  - 输出：
    - `prompt_proposals`
    - `visual_regions`
    - `pov_prompt_candidates`
    - `arbiter_decision`
    - `selected_candidate`
  - 工作方式：
    - 先生成 RGB-aware PoV candidates
    - 再调用多模态 LLM 在候选中做裁决
  - 如果 LLM 超时、返回非法 JSON、选择了不存在的 candidate，或拒绝全部候选
      则自动退回 rule-based fallback

- `run_zero_shot_direct_arrow_agent`
  - 输入：
    - `RouteContext` 所需字段
    - `image_path` 或 `image_url`
  - 输出：
    - `direct_decision`
    - `rendered_prompt`
  - 这是 direct-arrow baseline：
    - 让 LVLM 直接生成一个 visual prompt
    - 仍然读取 route context 和 visual regions
    - 如果输出非法或不稳定，则退回结构化 candidate fallback

- `compare_zero_shot_agents`
  - 同时运行：
    - `run_zero_shot_direct_arrow_agent`
    - `run_zero_shot_visual_prompt_agent`
  - 输出两边的决策结果和简洁对比摘要

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
  - 支持公共 `image_url`
  - 适合公网可访问图像
- `analyze_local_image`
  - 支持本地 `image_path`
  - 会把本地图片编码成 inline data URL 后发送给多模态 LLM
  - 适合 FPV / PoV 本地样例调试
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

### 1.6 LocalBEVContext 显式分支

当前版本除了 `RouteContext`，还新增了并行的 `LocalBEVContext` 分支。

它不是整张地图截图，而是围绕当前位姿裁切、并按当前 heading 对齐的局部 BEV 几何上下文。

它当前包含：

- `current_pose_bev`
- `matched_point_bev`
- `route_polyline_local`
- `local_segment_bev`
- `connector_bev`
- `next_turn_point_bev`
- `next_waypoint_bev`
- `bev_bbox`
- `bev_rotation_deg`
- `corridor_type`
- `turn_type`
- `turn_semantics`
- `corridor_sector_deg`
- `turn_sector_deg`
- `semantic_bev_layers`

其中：

- `RouteContext`
  更适合：
  - progress tracking
  - off-route detection
  - instruction generation
  - TemporalRefine

- `LocalBEVContext`
  更适合：
  - PromptProposal
  - BEV-to-PoV 几何对齐
  - 候选 prompt 搜索空间约束

### 1.7 Local BEV Patch 渲染

`render_local_bev_patch` 会把 `LocalBEVContext` 直接渲染成局部 pose-aligned SVG patch。

这个 patch 会画出：

- 绿色：`route_polyline_local`
- 蓝色：`local_segment_bev`
- 橙色虚线：`connector_bev`
- 蓝色扇区：`corridor_sector_deg`
- 橙色扇区：`turn_sector_deg`
- 半透明 proposal sectors：
  - teal: `corridor_sector`
  - orange: `turn_sector`
  - purple: `connector_merge_sector`
  - cyan: `entrance_candidate_sector`
  - red: `crossing_candidate_sector`
- `C`：current pose
- `M`：matched point
- `W`：next waypoint
- `T`：next turn

它不是 provider 的地图截图，而是由 `LocalBEVContext` 的几何和语义字段直接渲染出来的局部 BEV patch。

### 1.8 PromptProposal v1

当前新增的是 **route-prior-only / BEV PromptProposal v1**。

它不会直接输出 RGB 上的最终箭头，而是先把：

- `RouteContext`
- `LocalBEVContext`

压成一组结构化候选 proposal，供后续：

- PoV projection
- LVLM-Arbiter
- unified model distillation

使用。

当前支持的 proposal type：

- `corridor_sector`
- `turn_sector`
- `connector_merge_sector`
- `entrance_candidate_sector`
- `crossing_candidate_sector`

每个 proposal 至少包含：

- `proposal_id`
- `proposal_type`
- `frame`
- `sector_center_deg`
- `sector_span_deg`
- `range_m`
- `priority`
- `geometry_source`
- `semantic_tag`
- `rationale`
- `prompt_text`
- `debug`

也就是说，现在这一步输出的是：

- BEV 可解释候选先验

而不是最终 PoV prompt。

### 1.9 PoV Prompt Projection v1

当前新增的是 **weak geometric PoV Prompt Projection v1**。

它会把：

- `RouteContext`
- `PromptProposal v1`

映射成一组 image-plane 候选 prompt，而不是直接输出唯一最终箭头。

每个 candidate 至少包含：

- `proposal_id`
- `source_bev_proposal_id`
- `prompt_type`
- `frame="image_plane"`
- `center_uv`
- `direction_vec`
- `span_px`
- `priority`
- `projection_confidence`
- `debug`

这一步仍然不读取 RGB，它只是把：

- `BEV proposal sectors`

翻译成：

- `PoV 粗候选视觉提示`

供后续：

- RGB-aware PromptProposal v2
- LVLM-Arbiter
- unified student distillation

使用。

### 1.10 PromptProposal v2 (RGB-aware)

当前新增的是 **RGB-aware PromptProposal v2**。

它不是直接做最终裁决，而是把：

- `RouteContext`
- `LocalBEVContext`
- `PromptProposal v1`
- 当前 RGB 的视觉区域候选

融合成一组更合理的 image-plane prompt candidates。

这一步新增了两个对象：

- `VisualRegionBundle`
  - `navigability_heatmap`
  - `candidate_regions`

- `PoVPromptCandidate v2`
  - 在 v1 基础上新增：
    - `mask`
    - `visual_confidence`

当前 `VisualRegionExtractor` 是 heuristic v1：

- 支持 `image_path`
- 支持 `data:image/...` URL
- 输出 top-k `candidate_regions`
- 每个 region 都带：
  - `bbox_uv`
  - `polygon_uv`
  - `score`
  - `debug`

然后 `PromptProposal v2` 会把 route-prior-only 的候选和这些 visual regions 做融合，得到：

- 更贴近图像可通行区域的 `center_uv`
- region-aware `mask`
- 融合后的 `priority`
- `visual_confidence`

也就是说，这一步已经从：

- `route-prior-only PoV candidates`

推进到了：

- `route-prior + RGB region-aware PoV candidates`

### 1.11 Zero-shot Visual Prompt Agent

当前新增的是 **zero-shot visual prompt agent v1**。

它的设计原则不是让 LVLM 自由回归图像坐标，而是：

1. 先生成一小组 `PoVPromptCandidateBundle v2`
2. 再让 LVLM 只在这些候选中做选择
3. 最后输出：
   - `arbiter_decision`
   - `selected_candidate`

也就是说，这一层的核心是：

- `RouteContext`
- `BEV PromptProposal`
- `RGB-aware PoV Prompt candidates`
- `LVLM candidate arbitration`

同时保留必要 fallback：

- 如果 LLM 不可用
- 如果 LLM 超时
- 如果 LLM 返回非 JSON
- 如果 LLM 选择了不存在的 candidate
- 如果 LLM 直接 `reject_all`

系统会自动退回规则排序，保证 zero-shot agent 始终能产出一个稳定的可用结果。

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

## RouteContext 调试工具

`RouteContext` 是中间运行时状态，单看日志不容易定位问题到底出在：

- 当前匹配段是否吸附错路
- `turn_type` 是否和几何形状一致
- `heading_delta_deg` 是否合理
- `corridor_type` 是否抖动
- `distance_to_next_turn_m` 是否稳定递减

因此仓库里新增了一套 `tools/` 调试工具，建议按“采集 -> 指标 -> 可视化回放”三步来校验。

### 1. 采集调试序列

`tools/route_context_capture.py` 会：

1. 调 `recommend_destination_and_waypoints`
2. 必要时 fallback 调 `get_coarse_waypoints`
3. 对轨迹文件中的每一帧调用 `get_route_context`
4. 输出一份完整 capture JSON

轨迹文件支持：

- JSON 数组
- JSONL，每行一帧

每一帧至少需要：

```json
{"lat": 30.2645, "lng": 120.1180, "heading_deg": 35.0, "timestamp": "2026-04-07T10:00:00"}
```

示例：

```bash
python tools/route_context_capture.py \
  --trajectory /tmp/trajectory.jsonl \
  --prompt "我想去西湖" \
  --mode foot \
  --radius 3000 \
  --lookahead-m 20 \
  --output /tmp/route_context_capture.json
```

如果你只是想快速验证工具链，不想先准备轨迹文件，也可以直接做单帧 capture：

```bash
python tools/route_context_capture.py \
  --lat 30.2645 \
  --lon 120.1180 \
  --heading-deg 35 \
  --prompt "我想去西湖" \
  --mode foot \
  --radius 3000 \
  --lookahead-m 20 \
  --output /tmp/route_state_capture.json
```

capture 中会保存：

- 推荐元数据
- `route_polyline`
- `coarse_waypoints`
- 每一帧的标准 debug 字段
- 每一帧的完整 `RouteContext`
- 每一帧的 `LocalBEVContext`
- 每一帧的 `prompt_proposals`
- 每一帧的 `pov_prompt_candidates`
- 可选的 `image_path / image_url / image_data_url`

如果你想做三联面板回放，也可以在 capture 时直接带当前 RGB：

```bash
python tools/route_context_capture.py \
  --lat 30.2645 \
  --lon 120.1180 \
  --heading-deg 35 \
  --image-path /home/ubuntu/WorkSpace/dataset/urban.jpeg \
  --image-width 1280 \
  --image-height 720 \
  --prompt "我想去西湖" \
  --mode foot \
  --radius 3000 \
  --lookahead-m 20 \
  --output /tmp/route_state_capture.json
```

### 2. 计算稳定性指标

`tools/route_context_metrics.py` 会从 capture 中计算一组时间一致性指标：

- `segment_index_monotonicity_violation_rate`
- `corridor_switching_rate`
- `distance_to_turn_inconsistency_rate`
- `off_route_rate`
- `off_route_false_positive_like_rate`

示例：

```bash
python tools/route_context_metrics.py \
  --capture /tmp/route_context_capture.json
```

也可以输出成 JSON：

```bash
python tools/route_context_metrics.py \
  --capture /tmp/route_context_capture.json \
  --output /tmp/route_context_metrics.json
```

### 3. 生成可回放调试页面

`tools/route_context_debugger.py` 会生成一个自包含 HTML 页面，里面有：

- BEV 视图
  - full route
  - current GPS
  - matched point
  - local segment
  - connector polyline
  - heading vector
  - route tangent vector
- 时间曲线
  - segment index
  - cross-track
  - heading delta
  - distance-to-turn
  - corridor type
- 当前帧完整 `RouteContext` JSON
- `LocalBEVContext` patch
  - corridor / turn priors
  - BEV prompt proposals
- 当前 RGB + PoV candidates overlay
  - 会把 `pov_prompt_candidates` 直接叠加到当前图像平面
  - top-1 candidate 会高亮显示

示例：

```bash
python tools/route_context_debugger.py \
  --capture /tmp/route_context_capture.json \
  --output /tmp/route_context_debugger.html
```

然后直接打开：

```bash
xdg-open /tmp/route_context_debugger.html
```

### 4. 推荐的校验顺序

先看几何正确性：

- `matched_segment_index` 大体应单调递增
- `distance_to_next_turn_m` 应平滑下降，接近 turn 后跳到下一个关键点
- `heading_delta_deg` 在直行段通常较小
- `local_segment_polyline` 不应跳到错误支路

再看语义正确性：

- `turn_type`
- `turn_semantics`
- `corridor_type`
- `short_action_summary`

最后看时间稳定性：

- `corridor_type` 不应在连续几帧里高频抖动
- `off_route` 不应在正常直行时频繁误报

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

### Local BEV Patch

最直接的测试方法是运行：

```bash
python examples/local_bev_patch_client.py \
  --prompt "我想去西湖" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --heading-deg 35 \
  --lookahead-m 20 \
  --open
```

它会先调用 `recommend_destination_and_waypoints`，必要时 fallback 到 `get_coarse_waypoints`，然后自动调用 `render_local_bev_patch`，最后打印 `LocalBEVContext` 并打开局部 SVG patch。

### PromptProposal v1

最直接的测试方法是运行：

```bash
python examples/bev_prompt_proposal_client.py \
  --prompt "我想去西湖" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --heading-deg 35 \
  --lookahead-m 20
```

它会先生成路线，再自动调用 `generate_bev_prompt_proposals`，最后打印结构化 proposal 列表。

### PoV Prompt Projection v1

最直接的测试方法是运行：

```bash
python examples/pov_prompt_projection_client.py \
  --prompt "我想去西湖" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --heading-deg 35 \
  --lookahead-m 20 \
  --image-width 1280 \
  --image-height 720
```

它会先生成路线，再自动调用：

- `generate_bev_prompt_proposals`
- `generate_pov_prompt_candidates`

最后打印：

- `prompt_proposals`
- `pov_prompt_candidates`

### PromptProposal v2 (RGB-aware)

最直接的测试方法是运行：

```bash
python examples/pov_prompt_projection_v2_client.py \
  --prompt "我想去西湖" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --heading-deg 35 \
  --lookahead-m 20 \
  --image-path /home/ubuntu/WorkSpace/dataset/urban.jpeg
```

它会先生成路线，再自动调用：

- `generate_bev_prompt_proposals`
- `generate_pov_prompt_candidates_v2`

最后打印：

- `visual_regions`
- `pov_prompt_candidates`

如果你用 `tools/route_context_capture.py` 且提供了 `--image-path` 或轨迹帧里的 `image_path`，capture 会优先走 v2，并把：

- `visual_regions`
- `pov_prompt_candidates_v2`

一起保存给 `route_context_debugger.py` 做三联面板回放。

### Zero-shot Visual Prompt Agent

最直接的测试方法是运行：

```bash
python examples/zero_shot_visual_prompt_agent_client.py \
  --prompt "我想去西湖" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --heading-deg 35 \
  --lookahead-m 20 \
  --image-path /home/ubuntu/WorkSpace/dataset/urban.jpeg
```

它会先生成：

- `RouteContext`
- `PromptProposal v1`
- `PromptProposal v2`

然后再调用 zero-shot LVLM arbiter，并最终打印：

- `arbiter_decision`
- `selected_candidate`

### Zero-shot Direct-Arrow Baseline

最直接的测试方法是运行：

```bash
python examples/compare_zero_shot_agents.py \
  --prompt "我想去西湖" \
  --lat 30.2645 \
  --lon 120.1180 \
  --mode foot \
  --radius 3000 \
  --heading-deg 35 \
  --lookahead-m 20 \
  --image-path /home/ubuntu/WorkSpace/dataset/urban.jpeg
```

它会在同一帧上同时比较：

- direct-arrow zero-shot baseline
- candidate-arbiter zero-shot agent

并输出两边的：

- decision source
- fallback usage
- reject_all
- rendered prompt / selected candidate

### 图像理解

```bash
python examples/image_understanding_client.py \
  --image-url "https://ark-project.tos-cn-beijing.ivolces.com/images/view.jpeg" \
  --question "What place is shown in this image?"
```

也支持本地图片路径：

```bash
python examples/image_understanding_client.py \
  --image-path /home/ubuntu/WorkSpace/dataset/urban.jpeg \
  --question "Describe the navigable directions and obstacles in this FPV image."
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
