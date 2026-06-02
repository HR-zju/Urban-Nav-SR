# SideWalkNav

SideWalkNav 是 Agent 侧的 SideWalk System2 数据生成、标注和评估层。它负责把原始 sidewalk clip 转成 ms-swift 可训练的 FPV/BEV 成对样本，保留 InternNav 兼容的轨迹标签，并生成 QA/CoT artifact 供训练、审计和人工抽检使用。

## Phase1 边界

Phase1 不训练 segmentation，也不依赖 SAM mask 作为监督。必须具备的数据是：

```text
front.mp4 或 FPV frames
frame_positions.npy
frame_orientations.npy
camera_params.json
bev_images/
bev_index.json
```

`bev_images/` 可以低于 FPV 帧率。当前 exporter 会对每个 FPV/pose frame 查找最近时间戳的 BEV frame，并把匹配信息写入 `sidewalk_meta`：

```json
{
  "bev_frame_idx": 0,
  "bev_timestamp": 1752432872210405376,
  "bev_time_delta_ms": 46.36,
  "bev_mapping_source": "nearest_timestamp"
}
```

默认 `FRAME_STEP=1`，即每个 FPV/pose frame 都生成一个训练样本。只在调试或控制数据规模时增大 `FRAME_STEP`。

## QA 口径

仓库里现在有两条相关 QA 链路：

1. `Agent_part/RouteServer/src/mcp_server/qa`
   - 在线 RouteServer visual-prompt pipeline 使用。
   - 负责 `bev_route_theta`、`bev_navigation_stage`、`fpv_scene_stage`、`crossview_topology_grounding`、`point_prompt`、`prompt_confidence`、`prompt_validity` 等确定性 teacher。
   - 输入来自运行时 artifact：`route_context`、`local_bev_context`、segmentation summary、candidate bundle、`rendered_prompt`。

2. `Agent_part/RouteServer/tools/model_processing/sidewalk_phase1_dataset.py`
   - 离线 Phase1 ms-swift 数据集 exporter 使用。
   - 生成轻量 QA bundle，当前与在线核心 task 对齐：`bev_route_theta`、`bev_navigation_stage`、`crossview_topology_grounding`、`point_prompt`，并保留 `trajectory_intent_check` 作为 Phase1 轨迹一致性审计项。
   - 这些是几何/模板标签，不是高质量自然语言 CoT。

高质量自然语言 CoT 必须再运行 LLM teacher：

```bash
tools/model_processing/run_sidewalk_phase1_cot_teacher.sh
```

LLM teacher 使用 FPV、BEV 和几何证据生成。`<think>` 内部只保留结构化观察和规划 section，不再额外生成叙述性 reasoning，避免 token 冗余：

```text
<think>
<Scene>{...}</Scene>
<FPV>{...}</FPV>
<BEV>{"grid_size":[16,16],"ego_cell":[8,12],"target_cell":[8,9],...}</BEV>
<CrossView>{...}</CrossView>
<Plan>{...}</Plan>
</think>
<finish>{"trajectory":[[x,y],...],"intent":"...","route_theta_deg":...,"confidence_level":"..."}</finish>
```

validator 会强制 `<BEV>.ego_cell/target_cell`、`<finish>.trajectory`、`intent`、`route_theta_deg` 与几何标签一致。LLM 只能生成解释文本，不允许改写几何真值。

其中 `<BEV>.ego_cell` 和 `<BEV>.target_cell` 是 Stage1 拓扑锚定的模型输出字段；对应真值来自 `bev_patch_meta.ego_cell_gt` 和 `bev_patch_meta.target_cell_gt`。后续 GridAcc@1 和 k-hop Accuracy 就是用模型解析出的 cell 与这两个 GT cell 比较。

## GT 轨迹与 Agent Baseline 轨迹

CoT teacher answer 中的轨迹，例如：

```json
{"trajectory": [[0.5, 0.9], [0.50058, 0.87508]]}
```

不是 Agent baseline 推理结果，而是从 `trajectory_labels/*.json` 或 `phase1_source_*.jsonl` 复制出来的 GT。它用于 SFT token-level 监督；拿它和 GT 再算误差，只能检查 teacher 是否正确复制标签。

真正评估 Agent baseline 时，应使用真实预测来源：

- RouteServer zero-shot visual-prompt 输出中的 `rendered_prompt.path_uv`。
- ms-swift text inference 输出中的 `<finish>{"trajectory": ...}</finish>`。
- Phase1 trajectory head prediction JSONL。

然后与 Phase1 几何 GT `trajectory_labels/*.json` 对齐计算指标。

## RouteServer Baseline 输出约定

RouteServer 的内部 visual prompt 仍可能在 A/V 阶段使用 corridor samples、segmentation validation 或 safe-corridor prior 做安全检查。但对 SideWalk Phase1 baseline 来说，公开导出的预测只认中心轨迹：

```json
{
  "type": "path_corridor",
  "path_uv": [[960.0, 1015.0], [970.0, 860.0], [990.0, 720.0]],
  "trajectory_only": true
}
```

`path_samples`、`left_edge_uv`、`right_edge_uv`、`coverage_polygon_uv`、`mask` 不参与 Phase1 轨迹评估，也不会在 baseline overlay 中绘制。这样 RouteServer baseline、离线 QA、ms-swift 训练标签使用同一条“FPV center trajectory”口径。

## Debug 可视化

Phase1 exporter 可以画 GT 轨迹 overlay：

```bash
WRITE_OVERLAYS=1 tools/model_processing/run_sidewalk_phase1_dataset.sh
```

这些 overlay 只是 GT 标签可视化，不是模型预测。正式造数据时建议：

```bash
WRITE_OVERLAYS=0
```

RouteServer prediction overlay 应在跑完 visual-prompt pipeline 后，从 manifest / prediction JSON 中读取 `rendered_prompt.path_uv` 后处理生成。当前 RouteServer overlay 只画橙色中心轨迹，不画左右 corridor 边界。

## 曲线级轨迹指标

不要只评估 16 个离散点。应把 GT 与预测都看作连续曲线。

给定 GT 点列：

```text
P* = {p*_0, p*_1, ..., p*_{N-1}}
```

预测点列：

```text
P_hat = {p_hat_0, p_hat_1, ..., p_hat_{M-1}}
```

先做弦长参数化：

```text
u_0 = 0
u_i = sum_{k=0}^{i-1} ||p_{k+1} - p_k||_2
      / sum_{k=0}^{N-2} ||p_{k+1} - p_k||_2
```

理论上可以拟合平滑三次曲线：

```text
C*(u) = argmin_C sum_i ||C(u_i) - p*_i||_2^2
        + lambda * integral_0^1 ||C''(u)||_2^2 du
```

预测曲线 `C_hat(u)` 同理。推荐报告：

```text
cADE = integral_0^1 ||C_hat(u) - C*(u)||_2 du
cFDE = ||C_hat(1) - C*(1)||_2
cMDE = max_u ||C_hat(u) - C*(u)||_2
```

数值实现时可在曲线上等距采样 `K` 个点：

```text
cADE ~= (1 / K) * sum_j ||C_hat(s_j) - C*(s_j)||_2
```

安全性还需要 lateral error。设 GT 切向量和法向量为：

```text
t(u) = C*'(u) / ||C*'(u)||_2
n(u) = (-t_y(u), t_x(u))
```

则：

```text
E_lateral = integral_0^1 |(C_hat(u) - C*(u)) dot n(u)| du
```

视觉重合可使用 TubeIoU。半径 `r` 像素下：

```text
Omega_r(C) = {q | min_u ||q - C(u)||_2 <= r}
TubeIoU = |Omega_r(C_hat) intersect Omega_r(C*)|
          / |Omega_r(C_hat) union Omega_r(C*)|
```

建议同时报告归一化坐标和像素坐标指标：

```text
cADE_norm
cFDE_norm
cMDE_norm
cADE_px
cFDE_px
lateral_error_px
tube_iou_r16
parse_success
```

ms-swift 侧 `swift.sidewalk_nav.eval` 应消费 prediction JSONL 与 reference manifest，统一按以上指标输出 summary。
