# SideWalk System1 V2 Contract

本文档固定 `assistant_suffix_v2` contract，用于把 System2 agent tools 与
System1 CFM 训练连接起来。v2 不替代已经跑通的 v1 open-loop；RouteServer
exporter 默认仍输出 v1，只有显式选择 `assistant_suffix_v2` 时才导出本格式。

## 1. 核心因果顺序

v1 contract 是：

```text
tool_response / route state
-> user final readout prompt + [TRAJ_0] ... [TRAJ_N]
-> assistant final CoT + <finish>{...}</finish>
```

v2 contract 是：

```text
tool_response / route state
-> user final navigation prompt
-> assistant final CoT + <finish>{NavigationDecisionAction}</finish>
-> <readout>[SYS1_0] ... [SYS1_N]</readout>
-> System1 CFM reads H([SYS1_i])
```

数学约束来自 causal LM：

```text
H_i = F_theta(x_{<=i})
```

因此 System1 readout token 必须位于最终 `<finish>` 之后，才能看到工具结果、
`ToolResultUse`、用户交互状态和最终导航语义决策。

## 2. Finish Schema

v2 的 `<finish>` 是 System2 文本动作，不再承担连续轨迹输出。

推荐字段：

```json
{
  "action_type": "navigation_decision",
  "route_action": "follow_route",
  "intent": "turn_right",
  "tool_used": true,
  "tool": "map_tool",
  "op": "get_route_context",
  "route_session_id": "route_001",
  "tool_result_use": {
    "from_tool_response_id": "tr_002",
    "used_fields": ["next_maneuver", "route_summary"]
  },
  "instruction_text": "Follow the sidewalk and prepare to turn right."
}
```

v2 `<finish>` 不写入：

```text
pixel_trajectory
trajectory_grid1000
trajectory_fpv_t
trajectory_ego_t
action_ego_t
action_mask
segmentation labels
ground-truth route labels
teacher private fields
```

兼容说明：v1/legacy rows 仍可在 `<finish>` 中保留
`trajectory_grid1000 + trajectory_coord_frame`，validator 继续按等价比较检查它
是否与 top-level `pixel_trajectory` 一致。v2 CFM 主线应把连续监督留在
top-level `system1_supervision` 和 flat fields。

## 3. Readout Schema

v2 readout 固定为 assistant suffix：

```xml
<finish>{...}</finish>
<readout>[SYS1_0] [SYS1_1] ... [SYS1_15]</readout>
```

规则：

- `[SYS1_i]` 是 latent readout anchor，不是轨迹 token。
- `<readout>` 必须出现在 `</finish>` 之后。
- `[SYS1_i]` 必须 `attention_mask=1`，让 head 可以读取 hidden states。
- `[SYS1_i]` 必须 `labels=-100`，不参与 LM cross entropy。
- `[TRAJ_i]` 保留为 v1 legacy alias，不作为 v2 新数据的主 token。

## 4. system1_supervision

RouteServer row 保留 flat fields 以兼容当前 ms-swift trainer，同时提供 grouped
view：

```json
{
  "system1_supervision": {
    "target_type": "system1_head_supervision",
    "trajectory_ego_t": [[0.0, 0.0, 0.0], [0.3, 0.0, 0.01]],
    "action_ego_t": [[0.25, 0.0, 0.0], [0.25, 0.0, 0.01]],
    "action_mask": [true, true],
    "action_stats": {"space": "ego_delta_xyt"}
  },
  "contract_meta": {
    "system1_readout_placement": "assistant_suffix_v2",
    "system1_readout_tokens": ["[SYS1_0]", "[SYS1_1]"],
    "system1_readout_lm_label": -100,
    "system1_target_field": "trajectory_ego_t",
    "system1_target_format": "ego_xy_heading_delta"
  }
}
```

训练优先级：

1. `trajectory_ego_t` 作为当前推荐 CFM primary target，v2 固定列为
   `[x_right_m, y_front_m, heading_delta_rad]`。
2. `action_ego_t` 作为 fallback，兼容已有 CFM action head。
3. `pixel_trajectory` / `trajectory_grid1000` 只用于 v1 text/head 兼容和评测。

历史 Phase1 archive 中的 `trajectory_ego_t=[x_right_m,y_front_m,dt_s,speed_mps]`
不能直接作为 v2 CFM target。RouteServer 在 `assistant_suffix_v2` 导出时会确定性转换为
`ego_xy_heading_delta`；validator 会拒绝四维 v2 `trajectory_ego_t`。

## 5. Loss Mask

System2 LM loss：

```text
L_LM = - sum_{i in assistant_target and i notin readout} log p_theta(y_i | y_<i, o)
```

计入 LM loss：

```text
assistant <AgentState>
assistant <ToolDecision>
tool_call role content
assistant <ToolResultUse>
assistant <finish>
```

不计入 LM loss：

```text
system / user
tool_response
ContextAsset
<readout>[SYS1_i]</readout>
pixel_trajectory / trajectory_ego_t / action_ego_t / segmentation labels
```

System1 CFM loss：

```text
x_1 = trajectory_ego_t
x_0 ~ N(0, I)
lambda ~ U(0, 1)
x_lambda = (1 - lambda) x_0 + lambda x_1
v* = x_1 - x_0
L_CFM = || f_phi(x_lambda, lambda, Fuse(H_sys1, V_fpv)) - v* ||_2^2
```

当前阶段 CFM 条件支持三种可配置模式，训练和推理必须使用同名配置：

```text
traj_query_only:
  z = H([SYS1_i])
  memory = None

vlm_memory:
  z = MeanPool(prompt-only H)
  memory = prompt-only H

hybrid:
  z = H([SYS1_i])
  memory = prompt-only H
```

不显式输入：

```text
optional_proprio_state
optional_geometry_condition
route_polyline
waypoint
tool_response JSON
dialogue text
```

这些信息必须先由 System2 读入并通过 `H_sys1` 间接传递。

`traj_query_only` 对齐 InternNav latent-query System1 的解耦方式：System2 低频输出
readout latent，System1 CFM 高频采样 action chunk，不再 attend 到完整文本上下文。
`vlm_memory` 和 `hybrid` 用于消融，验证 prompt-visible VLM memory 是否提升 ADE/FDE、
collision/social risk 和 closed-loop recovery。

## 6. Export Modes

RouteServer exporter 支持：

```text
user_prompt_v1
assistant_suffix_v2
```

默认：

```text
user_prompt_v1
```

显式 v2 导出时：

```text
trace_to_ms_swift_stage2_row(trace, case, readout_placement="assistant_suffix_v2")
```

导出行为：

- 保留原有 message role sequence，避免破坏当前 agent-template 结构。
- final user turn 只保留导航触发文本，不再包含 `[TRAJ_i]`。
- final assistant 在 `<finish>` 后追加 `<readout>[SYS1_i]...</readout>`。
- `contract_meta.system1_readout_lm_label=-100` 作为 ms-swift label mask 约束。
