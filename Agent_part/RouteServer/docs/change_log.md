# RouteServer 变更记录

本文档用于记录影响 FPV/BEV visual prompt 主链路、QA teachers、调试产物和批处理工具的工程变更。记录目标是方便后续审计：为什么改、改了什么、对运行行为有什么影响。

## 2026-05-19

### 优化 LVLM 输入缓存与两图模式

- 新增 `--llm-input-mode som_bev` 快捷档位，显式固定为：
  - 主图使用 numbered SoM overlay
  - 额外图使用真实 OSM BEV screenshot
- 新增 `LVLM_COMPACT_INPUT_CONTEXT=1`，默认对 LVLM 上下文进行更紧凑的压缩：
  - 缩短 SoM/segmentation/legend/candidate 列表
  - 把稳定 contract 前缀前置，尽量让跨帧前缀保持一致
- 在 A/V prompt builder 中新增稳定 `prompt_prefix`，用于提升 provider 侧 prompt caching 命中概率。
- `manifest.json` / timing telemetry 新增 `generator_llm_prompt_cached_tokens`，用于观察缓存命中情况。
- 保留 `LLM_REASONING_EFFORT=minimal`、`LVLM_MINIMAL_OUTPUT=1`、`VISUAL_PROMPT_LLM_IMAGE_MAX_SIDE` 和 `VISUAL_PROMPT_LLM_IMAGE_JPEG_QUALITY` 作为降耗主旋钮。

## 2026-05-12

### 增加 LVLM 调用异常诊断

- 新增 `LLM_DEBUG_TRACEBACK=1` / `VISUAL_PROMPT_DEBUG_TRACEBACK=1` 调试开关。
- 当视觉 LLM 调用失败时，per-frame JSON 会记录：
  - 完整 Python traceback
  - provider / model / api_style / reasoning_effort
  - FPV、SoM、BEV 输入图片数量和估算 payload 大小
  - system prompt 与 question 的 ASCII 探针结果
- 目的：定位请求装配、编码、网关或 SDK 层异常，避免只看到 `str(exc)`。

### 增加 LVLM 极简输出协议

- 新增 `LVLM_MINIMAL_OUTPUT=1` / `VISUAL_PROMPT_MINIMAL_OUTPUT=1`。
- 开启后 A/V 阶段要求 LVLM 只输出导航必需字段：
  - `r`：是否 reject
  - `t`：终止语义短码
  - `p`：3-6 个 `[center,left,right]` FPV 像素采样
  - `c`：置信度
- 本地 `coerce_direct_prompt_decision` 会把短键输出恢复成完整 `VisualPromptDecision`：
  - `selected_prompt_type=path_corridor`
  - `rendered_prompt.path_samples`
  - `path_uv`
  - `coverage_polygon_uv`
  - `termination_semantics`
  - `support_confidence`
- 后续 segmentation validation / repair / render 保持不变，确保质量护栏不被绕过。
- 该模式的目标是减少 completion tokens 和 JSON 解析失败，同时保留 LVLM 对 FPV/SoM/BEV 的全帧判断。

### 增加 streaming 首 token 耗时拆分

- `VisionAnalyzer` 新增可选 streaming timing：
  - `LLM_VISION_STREAM_TIMING=1`
  - `LLM_STREAM_TIMING=1`
- 当开启后，chat-completions 调用会记录：
  - `time_to_first_chunk_ms`
  - `time_to_first_token_ms`
  - `stream_decode_ms`
  - `stream_chunks`
  - `stream_content_chunks`
- 这样可以把黑盒 roundtrip 再拆成两段：
  - 首 token 前：上传、排队、视觉编码、prefill
  - 首 token 后：流式解码与返回
- `LLM_STREAM_INCLUDE_USAGE=1` 默认开启；如果 provider 拒绝该参数，会自动回退到纯 streaming。

### 新增 LLM 延时拆分估算脚本

- 新增 `tools/analyze_visual_prompt_llm_latency.py`。
- 支持直接读取 `manifest.json`，离线估算：
  - `network_est_ms`
  - `service_est_ms`
  - `payload_bytes`
  - `prompt_tokens`
  - `completion_tokens`
- 支持可选 `--probe-url`，用于粗粒度 HTTP 探测。
- 该脚本不改主流程，适合对历史产物做延时分析和回归比较。

### LLM 请求统计继续细化

- 当前 telemetry 已记录：
  - `generator_llm_roundtrip_ms`
  - `generator_llm_request_payload_chars`
  - `generator_llm_request_text_chars`
  - `generator_llm_response_text_chars`
  - `generator_llm_image_count`
  - `generator_llm_approx_image_bytes`
  - `generator_llm_prompt_tokens`
  - `generator_llm_completion_tokens`
  - `generator_llm_total_tokens`
- `reasoning_effort=minimal` 仍然生效，便于压缩 Ark/Doubao 的推理开销。

### 降低 LVLM 推理负担与 Prompt 体积

- `LLMSettings` 新增 `reasoning_effort`，默认对 Ark/Doubao 取 `minimal`，优先让模型直接回答，减少深度 thinking 带来的长尾。
- 视觉 prompt 请求会把 `reasoning_effort` 透传给 Ark 兼容接口。
- A/V prompt builder 改为 compact JSON 序列化：
  - `json.dumps(..., separators=(",", ":"))`
  - 去除 `None` / 空字段
  - 把长列表规则压缩成 DSL 风格字符串
- `som_spatial_constraints` 新增紧凑 DSL 视图，例如：
  - `start(...)`
  - `corridor(...)`
  - `center(...)`
  - `edge(...)`
  - `support(...)`
- VLM 输入现在既保留可审计的结构化上下文，也减少了冗余 token 开销，便于对比不同 `reasoning_effort` 设置下的响应时延。

### 增加 LLM 调用细粒度耗时与请求体统计

- `VisionAnalyzer` 现在会把每次 LVLM 调用的详细 telemetry 回传到 pipeline：
  - `roundtrip_ms`
  - `request_text_chars`
  - `request_payload_chars`
  - `response_text_chars`
  - `image_payload.approx_image_bytes`
  - `usage.prompt_tokens` / `completion_tokens` / `total_tokens`（provider 返回时）
- `visual_prompt_service` 会把 generator / verifier 的 LLM telemetry 记录到每帧 `manifest.json`。
- `mcp_server.utils.timing` 新增对 LLM call telemetry 的扁平化输出，便于快速定位：
  - 是请求体太大
  - 还是模型推理时间高
  - 还是 provider roundtrip 明显偏慢
- 说明：
  - 端到端网络往返时间可以直接记录。
  - provider 内部排队 / 推理 / 解码时间只有在返回头或 body 暴露时才能进一步拆分，否则只能作为 roundtrip 观测。

### 增加缺省 Segmentation 策略与可选 IMU 姿态接入

- 批处理主流程新增 `--segmentation-policy required|optional|disabled`：
  - `required`：缺少 SAM/SAM3 mask 的帧直接失败。
  - `optional`：有 mask 时启用 SoM、safe corridor、validation/repair；没有 mask 时仍允许 LVLM+BEV 生成 visual prompt。
  - `disabled`：完全忽略 segmentation artifacts，便于纯 FPV/BEV baseline 或消融实验。
- `--allow-missing-segmentation` 保留为兼容参数，等价于 optional 模式。
- MCAP 输入抽取工具新增可选姿态同步：
  - `--imu-topic auto|none|<topic>`
  - `--pose-topic auto|none|<topic>`
  - `--attitude-policy optional|required|disabled`
- 当 MCAP 中存在 `sensor_msgs/Imu` 或 `geometry_msgs/PoseStamped` 四元数时，`trajectory.json` 每帧会写入：
  - `roll_deg`
  - `pitch_deg`
  - `attitude_yaw_deg`
  - `attitude_source`
  - `attitude_sync_delta_s`
- 四元数转换公式：
  - `roll = atan2(2(wx + yz), 1 - 2(x^2 + y^2))`
  - `pitch = asin(clamp(2(wy - zx), -1, 1))`
  - `yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))`
- visual prompt service 会把 frame attitude 传入 `segmentation_context.camera_attitude`。
- safe corridor prior 新增 roll-aware gravity alignment：
  - 先按 `-roll_deg` 将 semantic mask 旋转到近似重力对齐空间。
  - 在对齐后的 mask 中运行原有 obstacle-aware corridor planning。
  - 再按 `+roll_deg` 把 path、boundary 和 polygon 映射回源图像坐标。
- 新增环境变量：
  - `VISUAL_PROMPT_ATTITUDE_RECTIFY=1|0`
  - `VISUAL_PROMPT_ATTITUDE_RECTIFY_MIN_ROLL_DEG=<degrees>`
- 没有 IMU/Pose 或 roll 小于阈值时，自动回退到原水平 scanline prior，不影响现有数据集。

### BEV 输入增加当前位置与导航路径

- OSM BEV screenshot 现在会绘制导航参考路径：
  - 蓝色折线表示完整导航/轨迹路径。
  - 橙色圆点表示当前 GNSS 位置。
  - 白色短线表示当前朝向，即 BEV 图像上方方向。
  - 绿色圆点表示路径起点，红色方块表示路径终点。
- 离线 batch runner 会从 `trajectory.json` 的选中帧构造 `navigation_path_gps`，并传入 visual prompt service。
- 在线模式后续可以直接传入 `planner_path_gps` 或 `navigation_path_gps`，复用同一 BEV 视觉输入协议。
- `bev_map_overlay_legend` 和 payload debug 中新增 `navigation_path_overlay`，用于确认路径点数量和来源。

### 更新 Visual Prompt 调试视频脚本

- `tools/make_visual_prompt_seg_video.py` 改为适配当前 pipeline 产物。
- 输出视频从原来的双栏/四宫格视图调整为 3x1 横向布局：
  - FPV SoM / SAM numbered overlay
  - 最终 visual prompt overlay
  - OSM BEV map screenshot
- 脚本优先读取 `manifest.json` 中记录的 `llm_input_images`，保证视频展示的是实际送入 LVLM 的多图证据。
- visual prompt 面板现在直接读取 runner 产出的 `overlay_path` SVG，而不是在视频脚本里二次绘制 prompt。
- 原始 FPV 面板被移除，避免与 SoM/最终 overlay 重复占用版面。
- 中间面板语义明确为 `Visual Prompt on FPV`：优先使用 runner 产出的 FPV+prompt overlay SVG；若 overlay 缺失或解析失败，则回退到 raw FPV 并根据 JSON 中的 `arbiter_decision.rendered_prompt` 现场绘制。
- 修复 overlay SVG 栅格化缺少 `polygon`、`polyline`、`circle` 支持的问题，确保 corridor 填充、边界线、中心线和目标点能正常出现在视频中。
- 修复半透明 SVG 蒙层在 Pillow 中直接写 alpha 导致 FPV 底图被压黑的问题，改为在透明 layer 上绘制后再做 alpha composite。
- 修复移动/复制产物目录后的路径解析问题：当 manifest 中仍保存旧绝对路径时，脚本会优先读取当前 `--run-dir` 下的同名 `json/`、`overlay/`、`llm_inputs/` 文件，以及 `--run-dir/../images/` 中的原始帧。
- 增加固定 tile 尺寸和 letterbox 逻辑，避免 BEV、SoM、overlay 图像比例不同导致视频尺寸跳变。

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

### 增加 Pipeline 耗时统计

- 新增 `mcp_server.utils.timing`，集中维护耗时记录、telemetry flatten、manifest summary 和单帧日志格式化逻辑。
- 批处理主流程只调用 timing 工具函数，不再在 orchestration 代码里内联实现统计细节。
- 新增 `mcp_server.utils.async_batch`，提供有界并发执行和稳定顺序回填能力。
- `run_sidewalk_images_visual_prompt_pipeline.py` 新增 `--llm-concurrency`，用于并发处理多帧 LVLM visual prompt 生成。
- 单帧处理被封装为 frame job，manifest 仍按输入帧顺序写入，避免并发完成顺序影响产物审计。
- 在 visual prompt service 的 telemetry 中补充关键阶段耗时：
  - `image_preprocess`
  - `multi_image_inputs`
  - `bev_map_overlay`
- 批处理脚本现在会在每帧结束后打印 `[TIME]` 摘要，包含：
  - 单帧 runner 总耗时
  - service 内部耗时
  - generator LLM 耗时
  - segmentation/SOM 耗时
  - BEV tile overlay 耗时
  - 输出产物写入耗时
- `manifest.json` 中新增：
  - `sam3_inference_elapsed_ms`
  - `pipeline_elapsed_ms`
  - 每帧 `timing_ms`
  - 聚合 `timing_summary`
- 批处理 CLI 新增 `--no-emit-llm-input-images`，用于速度测试时关闭 LVLM 输入图落盘，减少 debug artifact 写入开销。
- 该统计用于判断慢点来自 LVLM 请求、SAM3、OSM tile/BEV、segmentation geometry，还是大量 debug artifact 写入。

### 增加常用运行指令文档

- 新增 `docs/commands.md`。
- 记录从 MCAP 抽取 sidewalk pipeline 输入的完整命令，包含：
  - `mcap_to_sidewalk_inputs.py` 1 FPS smoke 抽取命令
  - 图像、GNSS、heading topic 参数
  - 输出目录结构
  - 接续 visual prompt pipeline 的运行示例
- 补充 MCAP lz4 压缩依赖说明，并将 `lz4` 加入项目依赖。
- `mcap_to_sidewalk_inputs.py` 改为 lazy import `lz4`，只有遇到 lz4 压缩 chunk 且环境缺依赖时才给出明确安装提示。

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
