# RouteServer 变更记录

本文档用于记录影响 FPV/BEV visual prompt 主链路、QA teachers、调试产物和批处理工具的工程变更。记录目标是方便后续审计：为什么改、改了什么、对运行行为有什么影响。

## 2026-05-12

### 拆分 SoM 与 SAM Visualization 输入

- 在 visual prompt service 中新增显式开关：
  - `enable_som_overlay_input`
  - `enable_sam_visualization_input`
- 在批处理 CLI 中新增参数：
  - `--enable-som-overlay-input`
  - `--enable-sam-visualization-input`
- 保留兼容参数：
  - `--enable-segmentation-overlay-input`
- 兼容参数现在等价于 `--enable-som-overlay-input`，不再自动把原始 SAM visualization 图送入 LVLM。
- 推荐的 LVLM 多图输入组合变为：
  - 原始 FPV 图像
  - numbered SoM overlay
  - 真实 OSM BEV screenshot
- 原始 SAM visualization 只建议在 debug 或 ablation 时开启；SAM mask 本身仍然用于 deterministic validation、repair 和 safe corridor 生成。

### 增加 BEV Heading 校准

- 为 BEV tile 截图增加 heading convention 和 offset 配置：
  - `BEV_MAP_HEADING_CONVENTION=compass|enu_yaw`
  - `BEV_MAP_HEADING_OFFSET_DEG=<degrees>`
- `compass` 表示：
  - `0=N`
  - `90=E`
  - 顺时针为正
- `enu_yaw` 表示 ROS/ENU yaw：
  - `0=E`
  - `90=N`
  - 逆时针为正
- BEV tile metadata 会记录：
  - `raw_heading_deg`
  - `heading_convention`
  - `heading_offset_deg`
  - `compass_heading_deg`
  - `heading_rotation_deg`
- 已更新 `docs/debugging.md`，补充推荐多图输入策略和 BEV 方向校准流程。

### 将伪 BEV 草图替换为真实 OSM 截图

- 撤回任务专用的 BEV 点线草图输入。
- 改为真实 OpenStreetMap raster tile 截图：
  - GNSS 转 Web-Mercator slippy tile 坐标
  - 下载并缓存 OSM tile
  - 拼接 tile mosaic
  - 以 ego 位置为中心，根据 heading 先旋转完整 mosaic 为 ego-forward-up
  - 再从旋转后的 mosaic 中心裁剪/缩放为 LVLM 输入图
- 修正早期“先裁剪再旋转”的实现偏差，避免 BEV 截图在旋转后出现视野边角不规则、前向区域被裁剪框限制的问题。
- 新增 tile 相关环境变量：
  - `BEV_MAP_TILE_ZOOM`
  - `BEV_MAP_TILE_MOSAIC_TILES`
  - `BEV_MAP_TILE_OUTPUT_SIZE_PX`
  - `BEV_MAP_TILE_CACHE_DIR`
  - `BEV_MAP_TILE_URL_TEMPLATE`
  - `BEV_MAP_TILE_TIMEOUT_S`
- `llm_inputs/*.bev_map_overlay.jpg` 现在是真实 OSM raster screenshot，不再是语义点线渲染图。

## 2026-05-11

### 增加导航级 BEV Map Provider Cache

- 新增 `BEVMapProviderSession`，用于在一次 batch/navigation run 内复用 map provider 连接。
- 按量化后的 GNSS 网格和查询半径 bucket 缓存 provider 原始 map features。
- cache 命中后，仍然按每帧的 GNSS/heading 重新投影到 ego-BEV，保证坐标正确。
- 批处理脚本现在每个 sequence 创建一个 BEV map session，并在 `manifest.json` 中记录 cache 统计。

### 增加 OSM-Derived BEV Semantic Provider

- 新增 `route_context/bev_map_provider.py`。
- 基于当前 GNSS 查询地图 provider，并投影成 ego-aligned BEV point anchors：
  - sidewalk
  - crossing
  - road
  - building
  - barrier
  - entrance
- QA teachers 可以消费 `semantic_bev_layers` 来回答 BEV topology 相关问题。
- 该 provider 默认关闭，通过 `--enable-bev-map-provider` 开启。

### 增加 MCAP Sidewalk 输入抽取工具

- 新增 `tools/mcap_to_sidewalk_inputs.py`。
- 从 MCAP 中抽取并同步：
  - FPV image
  - GNSS
  - heading
- 输出 batch pipeline 所需的最小输入格式：
  - `images/000001.jpg`
  - `trajectory.json`
- 支持 1 FPS smoke extraction 和最近邻时间同步。

## 2026-05-10

### 增加 Obstacle-Aware Safe Corridor Prior

- 将 safe corridor 从逐行水平 midpoint 选择升级为 obstacle-aware corridor planning。
- 在采样行上引入 clearance-aware 选择逻辑。
- 增加横向平滑约束，减少 row-to-row 的突兀跳变。
- 当 LVLM 输出违反 segmentation 约束时，增加 deterministic repair/replan。

### 改进 Conditional Connector 规则

- 放宽短距离 conditional connector 的验证规则。
- 允许底部区域短暂使用 road/curb/gutter 作为 connector，只要路径能快速接入 preferred sidewalk/crosswalk。
- 避免将“底部 conditional connector -> 前方 sidewalk”的合理场景误判为 reject。

### 改进 Prompt 几何 Guardrails

- 增加或强化以下检查：
  - forbidden surface hit
  - corridor width ratio
  - centerline offset from traversable run midpoint
  - edge margin from traversable boundaries
  - lower-middle start constraint
- 在 segmentation geometry 可用时，优先执行 deterministic repair/fallback，再决定是否最终 reject。

## 2026-05-09

### 增加 Cross-View QA Stage 1

- 新增 Stage-1 cross-view topology grounding QA task。
- 增加对应的 schema、task、teacher 和 bundle 逻辑。
- QA bundle 现在携带：
  - local BEV context
  - FPV segmentation summary
  - SoM hints
  - safe corridor prior
- 目标是支持 BEV-FPV 对齐、拓扑锚定和后续评估。

### 增加核心注释与数学公式文档

- 为 segmentation、visual prompt、route context、QA teachers 等核心模块补充文件级和函数级注释。
- 在 `docs/architecture.md` 中整理主要数学抽象，包括：
  - route matching
  - local BEV projection
  - visual prompt coordinate scaling
  - segmentation validation / repair constraints

## 操作约定

- 当前 SR 仓库 RouteServer 路径：
  `/home/ubuntu/WorkSpace/project/SR/Code/Urban-Nav-SR/Agent_part/RouteServer`
- 目标 MCP 仓库路径：
  `/home/ubuntu/WorkSpace/research/code/UrbanNav/open-streetmap-mcp`
- 影响 RouteServer runtime 的改动需要同步到两个仓库。
- 目标 MCP 仓库中的 `api_http.py` 是目标侧分叉文件，除非明确要求，否则不要覆盖。
