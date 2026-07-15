# UrbanNav LVLM-Agent-CFM 闭环架构

本文档描述 UrbanNav 当前代码中的统一训练与推理框架。主线由三个模块组成：

1. **System2 / LVLM Agent**：理解多模态观测和用户任务，决定工具调用、用户交互、全局路线动作和局部符号策略。
2. **System1 / Continuous Policy**：读取 LVLM latent，以 Direct Action 或 Conditional Flow Matching（CFM）生成 ego-frame 连续轨迹块。
3. **Episode Runtime / Environment Loop**：执行工具、维护 RouteSession、组织 HITL、验证连续轨迹、推进环境并把事件回流训练和评测。

本文档以当前代码为准，并明确区分：

- **已实现**：已有可执行代码和测试覆盖的合同或路径。
- **可选/兼容**：保留用于消融、旧数据或 teacher 的路径，不是推荐主线。
- **未完成**：目前只有接口、测试桩或设计，不能表述为生产能力。

---

## 1. 系统边界

### 1.1 三模块总体结构

```text
User instruction / dialogue
FPV history / current FPV / optional BEV / optional goal image
optional coordinate goal / route state / previous tool results
                          |
                          v
                System2: LVLM Agent
         multimodal understanding + Plan & Execute
       tool_call | ask_user | navigate | finish | stop
              /             |                 \
             /              |                  \
      typed tools          navigate             terminal
       + HITL                |                    |
             \               v                    |
              ------> [SYS1_i] latent readout     |
                              |                    |
                              v                    |
                 System1: Direct / CFM policy     |
                 trajectory_ego_t [T,3]           |
                              |                    |
                              v                    |
                     Safety Validator             |
                              |                    |
                              v                    |
                   controller / Env.step <--------
                              |
                              v
                   next observation + reward
                              |
                              v
                     EpisodeTraceV2
                              |
               +--------------+---------------+
               |                              |
        frozen evaluation          session SFT / DAgger / RL
```

System2 是低频认知与任务策略，System1 是高频局部连续策略。两者通过 LVLM hidden state 和稳定的 `[SYS1_i]` readout token 连接，不通过额外 RouteBelief 神经模块连接。

### 1.2 两个仓库的职责

```text
Urban-Nav-SR / RouteServer
  Pydantic contracts
  ToolSpecRegistry + live/replay tool execution
  RouteSession and map providers
  LangGraph episode runtime + HITL
  EpisodeTraceV2 logging
  frozen episode benchmark
  Stage2 trace validation/export compatibility

ms-swift / sidewalk_nav
  multimodal prompt and agent-template encoding
  LVLM/LoRA training and text generation
  [TRAJ_i]/[SYS1_i] isolated latent queries
  optional coordinate Goal Encoder
  Direct Action / CFM System1 heads
  multi-tick trace exporter
  UrbanNavEnvClient rollout bridge
```

核心边界是：

\[
\text{RouteServer}=\text{environment + tools + rollout + trace},
\]

\[
\text{ms-swift}=\text{model + encoding + loss + checkpoint inference}.
\]

RouteServer 不实现训练器；ms-swift 不复制地图、RouteSession 或冻结环境状态机。

### 1.3 不再作为主干的旧链路

以下模块仍有代码价值，但不是 E2E 模型必须消费的中间状态：

```text
RouteContext
LocalBEVContext
BEV/PoV proposals
VisualPromptAgent / overlay
```

它们可以作为 typed tool result、teacher prior、可视化调试结果或消融输入。主闭环不能强制依赖：

```text
pose -> RouteContext -> handcrafted belief -> local policy
```

否则训练数据会绑定特定定位、地图和手工状态压缩流程。

---

## 2. 闭环问题定义

在 tick \(t\)，runtime 状态写为：

\[
s_t=(o_t,h_t,q_t,e_t,b_t),
\]

其中：

- \(o_t\)：当前多模态观测引用。
- \(h_t\)：对话、工具和动作历史。
- \(q_t\)：RouteSession、路线修订号和外部工具状态。
- \(e_t\)：环境信息，包括 replay key、到达和碰撞信息。
- \(b_t\)：工具次数、安全重试次数和最大 tick 等 runtime budget。

System2 产生离散层级动作：

\[
a_t^{(2)}=\pi_{\theta}^{S2}(s_t),
\]

\[
a_t^{(2)}\in
\{\texttt{tool\_call},\texttt{ask\_user},\texttt{navigate},
\texttt{finish},\texttt{stop}\}.
\]

只有当 \(a_t^{(2)}=\texttt{navigate}\) 时才调用 System1：

\[
X_t^{ego}=\pi_{\phi}^{S1}(H_t,C_t),
\qquad X_t^{ego}\in\mathbb R^{T\times3}.
\]

轨迹通过执行前验证后推进环境：

\[
v_t=V_{safe}(X_t^{ego},o_t),
\]

\[
(o_{t+1},r_t,d_t,i_t)=Env.step(X_t^{ego}),\quad v_t=1.
\]

若 \(v_t=0\)，轨迹不应执行，结构化拒绝结果返回 System2 进行重规划或停止。

---

## 3. 输入合同

### 3.1 模型可见观测

当前 `EpisodeGraphState.observation` 和 `ObservationResult.observation` 使用 JSON object 承载模型可见引用。推荐稳定语义如下：

```json
{
  "instruction": "Go to the library and stop by the coffee shop.",
  "images": {
    "fpv_history": ["fpv_t-2.jpg", "fpv_t-1.jpg"],
    "fpv": "fpv_t.jpg",
    "bev": "bev_t.jpg",
    "goal_image": "goal.jpg"
  },
  "goal_geometry": [
    {
      "x": 1.2,
      "y": 6.8,
      "heading": 0.1,
      "kind": "goal",
      "coordinate_frame": "ego"
    }
  ],
  "session_id": "route-session-001",
  "route_revision": 3
}
```

监督标签不得混入 observation。以下字段只允许位于训练 row 顶层或 trace supervision：

```text
trajectory_ego_t
action_ego_t
action_mask
pixel_trajectory
semantic_segmentation_path
teacher correction
ground-truth route label
```

### 3.2 支持的输入组合

ms-swift 当前数据管线可编码：

| 输入组合 | 当前表示 | 状态 |
| --- | --- | --- |
| FPV-only | `images=[current_fpv]` | 已支持 |
| FPV + instruction | user text + FPV | 已支持 |
| 多帧 FPV | `history_fpv` oldest-to-newest，再接 current FPV | 已支持 |
| FPV + BEV | current FPV 后接 current BEV | 已支持 |
| FPV + image goal | ordered image list 中包含 `goal_image` | 图像编码可支持，仍需数据侧固定 role/order |
| FPV + coordinate/GPS goal | 顶层 goal geometry + `[GOAL_i]` | 已支持可选 Goal Encoder |
| 上述组合 + tools | agent-template messages + top-level `tools` | 已支持 |

多帧的 canonical 顺序为：

```text
historical FPV oldest -> newest
current FPV
current BEV optional
goal image optional, only when the dataset explicitly declares its role
```

所有局部轨迹坐标都以 current ego/current FPV 为参考，历史帧不改变监督坐标系。

当前 `dataset_builder.py` 主要通过 row 中的 `prompt_template` 和 image ordering 构造输入，尚未实现一个覆盖所有 instruction/image/coordinate goal 组合的自动 mode resolver。因此训练数据必须显式保存 image role 和 prompt template，不能仅按图片数量猜测语义。

### 3.3 Coordinate Goal Encoder

数值目标不要求 LVLM 在文本中复述。`SidewalkGoalEncoder` 把规范化几何特征映射到 LVLM hidden space：

\[
g_j=[\bar x_j,\bar y_j,\sin\theta_j,\cos\theta_j,
\bar d_j,\tau_j,k_j,f_j]\in\mathbb R^8,
\]

\[
e_j^{goal}=MLP_{goal}(g_j)+p_j.
\]

其中 \(k_j\) 和 \(f_j\) 分别编码 goal kind 与 coordinate frame，\(p_j\) 是有序目标位置 embedding。无效 slot 使用 learnable `null_goal`。

输入位置由 `[GOAL_i]` token 标识，但实际 embedding 会被 \(e_j^{goal}\) 替换：

\[
E([GOAL_j])\leftarrow e_j^{goal}.
\]

GPS 目标先近似转换到当前 ego frame。设局部北向、东向位移为 \((n,e)\)，机器人朝向为 \(\psi\)：

\[
y_{front}=n\cos\psi+e\sin\psi,
\]

\[
x_{right}=e\cos\psi-n\sin\psi.
\]

该转换只构造条件输入，不能替代连续轨迹 GT。

---

## 4. 分层 Action Space

### 4.1 System2 符号动作

LVLM 监督的是符号决策，而不是控制器数值：

```text
tool_call
route_action
intent / direction_choice / motion_primitive
speed_policy / social_action / traffic_action
trajectory_validity
finish / stop
```

Runtime 级 `System2ActionType` 固定为：

```text
tool_call | ask_user | navigate | finish | stop
```

Prompt 内的 `route_action` 进一步描述路线语义，例如：

```text
continue_navigation
use_map_context
create_route
update_route
add_via
replan
request_user_input
inspect_visual_context
stop
unknown
```

`ask_user`、`ask-user`、`clarify` 只是 `request_user_input` 的兼容 alias，不是多个不同动作。

### 4.2 Readout anchor 不是动作

`[TRAJ_i]` 和 `[SYS1_i]` 是 hidden-state 查询位置：

\[
H_{query}=H[\operatorname{index}([SYS1_0]),\dots,
\operatorname{index}([SYS1_{Q-1}])].
\]

它们满足：

```text
attention_mask = 1
labels = -100
```

因此：

- 它们参与 Transformer 前向。
- LM cross entropy 不要求模型生成这些 token 的语义。
- CFM/Direct Action loss 通过对应 hidden state 回传。

在 isolated latent-query 模式中，token id 只负责定位，词表 embedding 被独立参数表替换：

\[
e_i=Q_i\quad\text{而不是}\quad E_{vocab}([SYS1_i]),
\]

\[
h_i=F_{\theta}(I,U,e_i).
\]

### 4.3 System1 连续输出

推荐规划合同是：

```text
trajectory_ego_t [T,3]
columns = [x_right_m, y_front_m, heading_delta_rad]
frame = current_ego
```

其中每个 \((x_i,y_i)\) 是相对当前 ego 原点的轨迹点，heading 是该点局部方向。

兼容控制合同是：

```text
action_ego_t [T,3]
columns = [dx_right_m, dy_front_m, dtheta_rad]
space = ego_delta_xyt
```

二者不能仅凭维度混用。绝对 ego 轨迹转控制增量必须显式计算：

\[
\Delta x_i=x_i-x_{i-1},\quad
\Delta y_i=y_i-y_{i-1},
\]

\[
\Delta\theta_i=wrap(\theta_i-\theta_{i-1}),
\qquad (x_{-1},y_{-1},\theta_{-1})=(0,0,0).
\]

当前 trainer 优先使用 `trajectory_ego_t`，缺失时使用 `action_ego_t`。当前 `ActionRuntimeAdapter` 的执行空间仍是 `ego_delta_xyt`；因此当 checkpoint 输出绝对 `trajectory_ego_t` 时，部署前必须增加上述显式转换。该转换目前不是 runtime adapter 中已完成的统一主路径。

System1 head 本身只检查张量宽度，不会推断坐标语义，也不会自动把绝对轨迹转换成 delta action。当前两类 target 的尺度约定也不同：

```text
trajectory_ego_t: RouteServer v2 exporter 输出 metric ego trajectory
action_ego_t:     dataset action pipeline 通常按 action_stats 归一化
```

对 `action_ego_t`：

\[
a_{norm}=\frac{a_{raw}-\mu}{\sigma},\qquad
a_{raw}=a_{norm}\sigma+\mu.
\]

因此 checkpoint 和 inference row 必须保存 `system1_target_format`、`action_stats` 和 `normalized`。禁止只根据 `[T,3]` shape 决定是否反归一化或累计积分。

### 4.4 FPV 文本轨迹

Phase A 局部 CoT 数据可在 assistant `<finish>` 中使用：

```json
{
  "trajectory_grid1000": [[500, 900], [510, 820]],
  "trajectory_coord_frame": "norm1000"
}
```

坐标转换为：

\[
x_{px}=\frac{x_{grid}}{1000}W,\qquad
y_{px}=\frac{y_{grid}}{1000}H.
\]

这是图像平面可视化/文本监督，不是 metric ego control。

### 4.5 Stage2 v1 与 v2 finish

两种数据视图必须区分：

**v1 / local or compatibility row**

```text
assistant <finish> contains trajectory_grid1000
```

**v2 / assistant_suffix_v2 System1 row**

```text
assistant symbolic <finish>
<readout>[SYS1_0] ... [SYS1_15]</readout>
top-level trajectory_ego_t [T,3]
```

v2 exporter 会从 `<finish>` 删除数值轨迹、action 和 segmentation 字段。高频监督只保留在 row 顶层，避免让 LVLM 文本同时承担连续控制回归。

---

## 5. System2：LVLM Agent

### 5.1 Multimodal encoder

设文本 token 为 \(U\)，视觉输入为 \(I^{1:N}\)，可选几何目标为 \(G\)。视觉编码和 embedding 替换后：

\[
Z_I=VisionEncoder(I^{1:N}),
\]

\[
E_0=Embed(U,Z_I),
\]

\[
E=ReplaceGoal(ReplaceQuery(E_0,Q),G),
\]

\[
H=LVLM_{\theta}(E,A_{causal}).
\]

多帧输入当前由 VLM 原生 multimodal token 序列联合编码，尚未增加独立 async RGB encoder/resampler。时间顺序主要通过 image ordering、prompt 描述和位置编码学习。

### 5.2 Plan & Execute

System2 每一轮只选择一个可执行动作族：

1. 当前信息不足：生成 `tool_call`。
2. 需要人类确认：生成 `ask_user` 或 user-interaction tool call。
3. 已有足够上下文：生成 `navigate` 和符号局部策略。
4. 已到达或无法继续：生成 `finish/stop`。

工具返回后必须重新进入 System2，而不是直接把工具结果硬编码成轨迹：

\[
a_{t,k+1}^{S2}=\pi_{\theta}^{S2}
(o_t,h_t,c_{t,1:k},r_{t,1:k}),
\]

其中 \(c\) 是 tool call，\(r\) 是 tool response。这样支持多工具串联、路线修改和失败恢复。

### 5.3 社会导航语义

主干 prompt 使用同构结构：

```text
<FPVScene>
  navigable_support / conditional_support
  risk_objects / blocked_by
  agents / controls
  visibility / local_risk / social_risk

<BEVContext>
  available / route_summary / intent / target_direction

<CrossView>
  available / route_to_fpv_direction / safe_corridor / alignment

<Plan>
  motion_primitive / lateral_preference / avoidance_side
  speed_policy / social_action / safety_policy
  contingency / validity

<finish>
  route_action / intent / confidence / social_action / ...
```

FPV-only row 仍保留相同 schema，只把：

```text
BEVContext.available = false
CrossView.available = false
CrossView.alignment = fpv_only
```

这样 FPV-only 与 FPV-BEV 可以混合训练，不需要两套互斥输出语法。

---

## 6. Typed Tools 与 RouteSession

### 6.1 ToolCallV2

```json
{
  "tool": "map_tool",
  "arguments": {
    "op": "add_via_and_replan",
    "session_id": "route-session-001",
    "via_point": {"lat": 39.9, "lng": 116.4}
  },
  "call_id": "call-001",
  "schema_version": "urbannav_tool_contract_v2"
}
```

### 6.2 ToolResponseV2

```json
{
  "tool": "map_tool",
  "op": "add_via_and_replan",
  "success": true,
  "result": {},
  "error": null,
  "artifacts": [],
  "call_id": "call-001",
  "executor": "live",
  "latency_ms": 42.0,
  "state_revision": 4,
  "state_delta": {"via_added": true},
  "idempotency_key": null,
  "replay_key": null,
  "schema_version": "urbannav_tool_contract_v2"
}
```

`tool_response` 是环境上下文，必须满足：

```text
LM label = -100
```

模型只学习在后续 assistant 中使用其决策相关结果，不学习复述原始 provider payload。

### 6.3 ToolSpecRegistry

`ToolSpecRegistry` 是 schema、参数验证和 executor capabilities 的唯一来源。当前模型可见元工具为：

```text
map_tool
image_tool
user_interaction_tool
```

主要 map operations 包括：

```text
geocode / reverse_geocode
search_destination / search_poi / search_category
plan_route / get_route / get_route_context
create_route_session / get_route_session
add_via / add_via_and_replan / remove_via_and_replan
replace_destination / replan_from_current_pose
```

image operations：

```text
crop / zoom_in
```

user interaction operations：

```text
ask_user / request_clarification / present_options
confirm_route_change / notify_issue / answer_query
```

对 executor profile \(e\)，conformance 条件为：

\[
C_e^{actual}=C_e^{registry}.
\]

因此 live 与 replay 的差异通过 capability profile 显式声明，而不是维护另一份 op 常量。

### 6.4 RouteSession 的定位

RouteSession 是工具管理的外部状态，不是 LVLM 内部 belief：

\[
q_t=(origin_t,destination_t,V_t,mode,route_t,revision_t).
\]

路线修改工具产生 `state_delta` 和新的 `state_revision`，随后写入 `route_state_delta` 事件。模型通过 tool response 和后续 observation 获取变化，不直接读取某个隐式 Python session object。

---

## 7. System1：Direct Action 与 CFM

### 7.1 条件提取

最终 LVLM hidden state 为：

\[
H\in\mathbb R^{B\times S\times D}.
\]

当前实现支持三种 System1 condition mode。

**traj_query_only**

\[
Z=H_{query}\in\mathbb R^{B\times Q\times D},
\qquad C=\varnothing.
\]

**vlm_memory**

\[
C=MaskPrompt(H),
\qquad Z=MaskedMean(C)\in\mathbb R^{B\times D}.
\]

**hybrid**

\[
Z=H_{query},\qquad C=MaskPrompt(H).
\]

`MaskPrompt` 只保留有效 prompt/context token：

\[
M_i=A_i\land [label_i=-100].
\]

因此 teacher-forced assistant answer 不会进入 System1 cross-attention memory。训练默认 detach memory，避免 action loss 通过整个 prompt memory 更新全部 token hidden state；query latent 是否回传由配置控制。

### 7.2 DirectQueryActionHead

Direct baseline 与 CFM 使用相同 \((Z,C)\)、相同 target 和 decoder 宽度，只去掉 flow time 与中间噪声状态：

\[
q_i=e_i^{step}+W_zPool(Z),
\]

\[
\hat X=Decoder_{direct}(q,C),
\]

\[
\mathcal L_{direct}=
\frac{\sum_{b,t}m_{b,t}\|\hat X_{b,t}-X_{b,t}\|_2^2}
{\sum_{b,t}m_{b,t}}.
\]

它是 CFM 的公平确定性消融，不再使用旧 FPV curve heads 比较 metric ego control。

### 7.3 基础 Conditional Flow Matching

目标轨迹为：

\[
x_1=X^{ego}\in\mathbb R^{B\times T\times3}.
\]

基础起点：

\[
x_0\sim\mathcal N(0,I),\qquad t\sim U(0,1).
\]

线性 probability path：

\[
x_t=(1-t)x_0+tx_1,
\]

目标速度场：

\[
v^*=x_1-x_0.
\]

每个未来 step 的 decoder query 组合为：

\[
q_i=e_i^{step}+W_zPool(Z)+W_xx_{t,i}+W_t\gamma(t),
\]

其中 \(\gamma(t)\) 是 sinusoidal time embedding。Transformer decoder 对 prompt memory \(C\) 做 cross-attention，输出：

\[
\hat v_{\phi}=Decoder_{CFM}(q,C).
\]

基础损失：

\[
\mathcal L_{CFM}=
\frac{\sum_{b,t}m_{b,t}
\|\hat v_{\phi}(x_t,t,Z,C)-(x_1-x_0)\|_2^2}
{\sum_{b,t}m_{b,t}}.
\]

### 7.4 AURA-lite anchor CFM

可选固定轨迹 anchor bank：

\[
A_k\in\mathbb R^{T\times3},
\]

包含 stop、slow-forward、straight、left/right turn 和 detour 等 motion primitives。

训练时通过 GT 匹配 anchor：

\[
D(A_k,X)=w_aADE(A_k,X)+w_fFDE(A_k,X)
+w_h\operatorname{Mean}(wrap(A_{k,\theta}-X_{\theta})^2),
\]

\[
k^*=\arg\min_kD(A_k,X).
\]

起点变为：

\[
x_0=A_{k^*}+\sigma\epsilon.
\]

推理时通过分类器：

\[
\hat k=\arg\max_k p_{\phi}(k\mid Z),
\qquad x_0=A_{\hat k}.
\]

组合损失为：

\[
\mathcal L_{S1}=
\mathcal L_{CFM}
+\lambda_{anchor}\mathcal L_{CE}(\hat k,k^*)
+\lambda_{end}\mathcal L_{endpoint}
+\lambda_{smooth}\mathcal L_{smooth}
+\lambda_{valid}\mathcal L_{valid}.
\]

其中 smooth loss 约束二阶差分，valid loss 软惩罚过大单步位移与 heading。

### 7.5 CFM 推理

从 \(x^{(0)}=x_0\) 开始，用 \(K\) 步 Euler 积分：

\[
x^{(k+1)}=x^{(k)}+\Delta t\,
v_{\phi}(x^{(k)},t_k,Z,C),
\]

\[
\Delta t=\frac1K,\qquad t_k=\frac{k}{K}.
\]

最终输出：

```json
{
  "system1_chunk_ego": [[0.0, 0.4, 0.0], [0.1, 0.9, 0.08]],
  "action_chunk_metadata": {
    "head_type": "flow_action",
    "system1_condition_mode": "hybrid",
    "flow_steps": 8,
    "anchor_index": 2,
    "anchor_name": "straight_center"
  }
}
```

System1 checkpoint sidecar包括：

```text
sidewalk_action_head.pt/config
sidewalk_traj_latent_queries.pt/config
sidewalk_goal_encoder.pt/config, when enabled
```

---

## 8. Episode Runtime

### 8.1 LangGraph 节点图

```text
START
  -> observe
  -> system2
       tool_call -> tool -> system2 / hitl
       ask_user  -> hitl interrupt -> resume -> system2
       navigate  -> system1 -> safety
       finish    -> finish
       stop      -> finish
  -> safety
       accepted -> environment_step -> observe
       rejected -> system2
       retry exhausted -> finish(failed)
  -> END
```

Runtime budget 当前包括：

```text
max_ticks
max_tool_calls
max_safety_retries
```

### 8.2 HITL

`ask_user` 或 blocking user-interaction 会触发 LangGraph `interrupt()`。checkpointer 保存 graph state，用户回答后从相同 thread id 恢复。工具执行发生在 interrupt 之前的独立节点，避免恢复时重复副作用。

### 8.3 Safety Validator

接口输入输出已经固定：

```text
input:  System1Output + EpisodeGraphState
output: SafetyResult
  accepted
  reason
  metrics
  intervention optional
```

原则上它应验证：

\[
V_{safe}=C_{numeric}\land C_{kinematic}\land
C_{collision}\land C_{social}\land C_{freshness}.
\]

但当前代码只有 `SafetyValidator` Protocol、graph 路由和测试 fake，尚无生产级 occupancy/footprint/dynamic-agent validator。`SafetyResult.intervention` 当前也只记录在 trace 中，没有替换原始 System1 output。文档中的 safety node 因此是**已接线的接口边界**，不是已经完成的物理安全能力。

ms-swift 的 `kinematic_validity_loss` 是训练软约束，offline collision metric 是评测量；二者都不能替代运行时 hard gate。

### 8.4 Environment

统一环境协议：

```text
reset(initial_context) -> ObservationResult
step(System1Output, state) -> EnvironmentStepResult
```

step 返回：

```text
observation
reward: float
done: bool
info: arrival/collision/recovery diagnostics
replay_key optional/live, required/frozen RL
```

当前已有冻结 transition environment 和外部 HTTP episode service。真实机器人/仿真器 controller 与统一 `trajectory_ego_t` 的生产绑定仍需按坐标合同完成。

---

## 9. EpisodeTraceV2

### 9.1 Event-sourced contract

`EpisodeTraceV2` 是 runtime、训练和评测的唯一 session-level 因果记录：

```text
episode_started
system_message
observation_received
model_output
tool_call
tool_response
readout_requested
route_state_delta
system1_output
safety_validation
environment_step
user_interrupt
user_resume
episode_finished
```

每个 event 包含：

```text
episode_id
tick_id
event_id
parent_event_id
timestamp
source
payload
metadata
schema_version
```

因果条件：

\[
parent(e_i)\in\{e_0,\dots,e_{i-1}\}.
\]

Trace 是日志和数据合同，不是额外模型 belief。

### 9.2 单 tick 因果顺序

工具路径必须保持：

```text
observation/user
assistant tool reasoning optional
tool_call
tool_response
final navigation request
assistant symbolic <finish>
assistant <readout>[SYS1_i]</readout>
top-level System1 target
```

关键约束是：

\[
tool\_response \prec finish \prec [SYS1_i].
\]

因此 `[SYS1_i]` hidden state 能看到工具结果和最终符号决策。把 query 放在 tool response 或 finish 之前会破坏因果条件。

### 9.3 Loss mask

| 内容 | LVLM 可见 | LM CE | System1 loss |
| --- | --- | --- | --- |
| system/user prompt | 是 | 否 | 作为条件 |
| images | 是 | 否 | 通过 LVLM latent |
| assistant reasoning/finish | 自回归 | 是 | query 可读取之前部分 |
| tool_call role | 是 | 是 | 作为历史条件 |
| tool_response role | 是 | 否，`-100` | 作为历史条件 |
| `[SYS1_i]` | 是 | 否，`-100` | query hidden state |
| `trajectory_ego_t` | 否 | 否 | CFM/direct target |
| reward/replay key | 否 | 否 | rollout/RL metadata |

---

## 10. 训练数据与目标函数

### 10.1 Canonical v2 row

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image>..."},
    {"role": "tool_call", "content": "{...}"},
    {"role": "tool_response", "content": "{...}"},
    {"role": "user", "content": "Produce the final navigation decision."},
    {
      "role": "assistant",
      "content": "<think>...</think><finish>{...}</finish><readout>[SYS1_0] ... [SYS1_15]</readout>"
    }
  ],
  "tools": "[...]",
  "images": ["fpv.jpg", "bev.jpg"],
  "trajectory_ego_t": [[0.0, 0.4, 0.0], [0.1, 0.9, 0.08]],
  "action_mask": [true, true],
  "contract_meta": {
    "system1_readout_placement": "assistant_suffix_v2",
    "system1_readout_lm_label": -100,
    "system1_target_format": "ego_xy_heading_delta"
  }
}
```

### 10.2 联合损失

当前推荐主线：

\[
\mathcal L=
\mathcal L_{LM}
+\mathbf 1_{S1}\lambda_{S1}\mathcal L_{S1}.
\]

其中：

\[
\mathcal L_{LM}=-\sum_{i:m_i^{LM}=1}
\log p_{\theta}(y_i\mid y_{<i},o),
\]

\[
\mathcal L_{S1}=\mathcal L_{direct}
\quad\text{或}\quad
\mathcal L_{CFM+anchor}.
\]

旧 FPV pixel trajectory head 和 segmentation head 仍可由显式 flag 开启：

\[
\mathcal L_{compat}=\lambda_{traj}\mathcal L_{pixel}
+\lambda_{seg}\mathcal L_{seg}.
\]

但它们不是 CFM 主线的必要组成。大规模 System1 训练不应要求 `pixel_trajectory`。

### 10.3 训练课程

**Phase A：local multimodal SFT**

```text
FPV-only / FPV-BEV / instruction / social navigation / optional goal
train LVLM symbolic reasoning and finish
optional local auxiliary heads
```

**Phase B：tool cold start**

```text
tool_call + tool_response + final answer
LM CE supervises assistant/tool_call
tool_response remains context-only
```

**Phase C：joint System2 + System1**

```text
assistant_suffix_v2
LM/tool/finish CE
+ trajectory_ego_t CFM or direct loss
```

**Phase D：session SFT / DAgger**

`EpisodeTraceV2` 按 tick 展开成 causal prefix row。DAgger 只使用显式 teacher correction 替换错误 assistant action 或 System1 target，不能把失败 rollout 自身当作正确标签。

**Phase E：GRPO/RL**

只有 episode 同时满足以下条件才允许进入 RL：

```text
every environment_step has numeric reward
every environment_step has stable replay_key
episode has terminal episode_finished event
environment transition is reproducible
```

RL 不是 SFT 失败时的 fallback。

### 10.4 Multi-tick exporter

ms-swift 的 episode exporter 对每个 tick 构造历史前缀：

\[
row_t=Export(e_{0:t}).
\]

可通过 `window_ticks` 截断上下文：

\[
row_t=Export(e_{\max(0,t-W+1):t}).
\]

环境 observation 和 tool response 保持 context role，只有 assistant action 是 LM target。System1 target 从同 tick 的 `system1_output` 或显式 correction 读取。

---

## 11. 推理路径

### 11.1 Single open-loop

```text
one observation
-> System2 text/tool decision
-> optional real tool execution
-> final finish + [SYS1_i]
-> System1 head
-> one trajectory chunk
```

该路径用于 checkpoint smoke 和 Stage2 协议验证，不构成多 tick 到达闭环。

### 11.2 RouteServer-owned closed loop

```text
EpisodeRuntime.run()
-> LangGraph controls tools/HITL/System1/safety/env
-> EpisodeTraceV2
```

适合部署和冻结 benchmark，System2Policy/System1Policy 通过 Protocol 注入。

### 11.3 ms-swift-owned rollout scheduler

`UrbanNavEnvClient` 通过严格 HTTP 合同连接 RouteServer：

```text
POST /v1/episodes/reset
POST /v1/episodes/{episode_id}/step
```

它要求 numeric reward，并在 replay/RL 模式要求 `replay_key`。该路径用于 session rollout、DAgger 数据采集和后续 RL，而不是在 ms-swift 内复制环境。

---

## 12. 评测

### 12.1 System2 / tools

```text
action type accuracy
tool name/op accuracy
argument validity
tool success rate
unnecessary tool-call rate
route update / add-via / replan success
ask-user precision and resolution success
```

### 12.2 System1

在统一 ego target 上比较 Direct 与 CFM：

\[
ADE=\frac1T\sum_t\|\hat p_t-p_t\|_2,
\]

\[
FDE=\|\hat p_T-p_T\|_2.
\]

同时报告：

```text
heading error
endpoint error
smoothness / kinematic validity
collision proxy / off-grid rate
social risk
sampling latency
```

### 12.3 Episode level

冻结 benchmark 当前主指标：

```text
arrival_success
collision_free_success
recovery_success
normalized_efficiency
total_reward
```

成功加权 step efficiency：

\[
E=S_{arrival}\frac{N^*}{\max(N^*,\hat N)}.
\]

tool calls、safety rejections 和 environment steps 作为 diagnostics，不自动提升为主分数。

---

## 13. 核心代码索引

### 13.1 RouteServer

```text
src/mcp_server/contracts/tool.py
  ToolCallV2 / ToolResponseV2

src/mcp_server/tools/specs.py
  Pydantic tool argument models and aliases

src/mcp_server/tools/registry.py
  ToolSpecRegistry and capability profiles

src/mcp_server/contracts/episode.py
  EpisodeEvent / EpisodeTraceV2

src/mcp_server/runtime/interfaces.py
  System2Action / System1Output / SafetyResult / environment protocols

src/mcp_server/runtime/state.py
  checkpoint-safe EpisodeGraphState

src/mcp_server/runtime/graph.py
  LangGraph closed-loop runtime

src/mcp_server/runtime/tool_adapters.py
  live and replay executor adapters

src/mcp_server/runtime/service.py
  external episode HTTP contracts

src/mcp_server/eval/episode/
  frozen cases, environment, runner, metrics and service

src/mcp_server/eval/exporters.py
  Stage2 v1/v2 row exporter

src/mcp_server/agent/finish_schema.py
  legacy/canonical finish compatibility

src/mcp_server/agent/stage2.py
  final assistant target and readout placement
```

### 13.2 ms-swift

```text
swift/sidewalk_nav/prompt.py
  multimodal/social/agentic prompt schemas

swift/sidewalk_nav/tokens.py
  [TRAJ_i]/[SYS1_i] registration, indexing and LM masking

swift/sidewalk_nav/models/latent_query.py
  InternNav-style isolated latent queries

swift/sidewalk_nav/models/goal_encoder.py
  optional coordinate/GPS goal projection

swift/sidewalk_nav/system1_condition.py
  traj_query_only / vlm_memory / hybrid routing

swift/sidewalk_nav/models/action_head_heads.py
  DirectQueryActionHead and common System1 interface

swift/sidewalk_nav/models/flow_action_head.py
  CFM velocity field and Euler sampling

swift/sidewalk_nav/models/action_anchors.py
  AURA-lite anchors and auxiliary losses

swift/sidewalk_nav/models/action_head_io.py
  System1 sidecar checkpoint save/load

swift/sidewalk_nav/trainers/trainer.py
  joint LM + System1 loss and prompt-only memory mask

swift/sidewalk_nav/cli/infer.py
  text/head inference and system1_chunk_ego output

swift/sidewalk_nav/adapters/action_runtime_adapter.py
  ego_delta_xyt command adapter

swift/sidewalk_nav/data/agentic_episode_manifest.py
  EpisodeTraceV2 -> session SFT/DAgger rows

swift/sidewalk_nav/rollout/urbannav_env.py
  strict RouteServer environment client
```

---

## 14. 当前实现状态

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| typed System2/System1/runtime interfaces | 已实现 | Pydantic + Protocol |
| ToolSpecRegistry 和 capability conformance | 已实现 | live/replay profile |
| RouteSession map operations | 已实现 | 通过 map tool 暴露 |
| single open-loop Stage2 | 已实现 | live API teacher 和 ms-swift backend 路径 |
| LangGraph 多工具/HITL/System1/env 图 | 已实现框架 | 依赖注入具体 policy/env |
| EpisodeTraceV2 | 已实现 | event-sourced contract |
| v2 assistant-suffix readout | 已实现 | `[SYS1_i]` 位于 finish 后且 LM mask |
| isolated latent queries | 已实现 | 独立 query table |
| coordinate Goal Encoder | 已实现可选项 | 顶层 geometry，不进入 LM target |
| Direct Action 与 CFM 消融 | 已实现 | 相同 condition/target/output |
| AURA-lite anchor CFM | 已实现可选项 | fixed anchor + classifier + flow init |
| System1 checkpoint save/load/head inference | 已实现 | sidecar files |
| multi-tick session SFT/DAgger exporter | 已实现 | causal prefix rows |
| UrbanNavEnvClient | 已实现 | strict reset/step HTTP client |
| frozen episode benchmark | 已实现 | deterministic transition metrics |
| 自动多模态 input-mode resolver | 未完成 | 当前依赖显式 prompt/image roles |
| 生产级 Safety Validator | 未完成 | 只有 Protocol、graph node 和测试 fake |
| safety intervention execution | 未完成 | intervention 当前未替换轨迹 |
| System1 target 坐标/归一化统一 | 部分实现 | exporter 可固定 trajectory `[T,3]`，head 尚不区分 absolute/delta 语义 |
| absolute trajectory -> delta controller 统一桥接 | 未完成 | 当前 adapter 仍以 ego_delta_xyt 为输入 |
| live simulator/robot multi-tick rollout validation | 需要部署验证 | frozen service 不等于真实控制闭环 |
| GRPO 生产训练 | 未完成 | 必须先满足 replay/reward readiness |
| 独立 async RGB memory branch | 未实现 | 当前使用 LVLM multimodal hidden memory |

---

## 15. 不可破坏的框架约束

1. `tool_response` 必须是上下文，不能计算 LM CE。
2. `[TRAJ_i]/[SYS1_i]` 必须 `labels=-100`，且 v2 readout 位于工具结果和 finish 之后。
3. `trajectory_ego_t/action_ego_t` 只能作为顶层 head supervision，不能写入 assistant target。
4. `trajectory_ego_t` 与 `action_ego_t` 必须通过 format metadata 区分，禁止按 `[T,3]` 静默混用。
5. RouteSession、provider cache 和 safety diagnostics 属于外部 runtime state，不是 LVLM 必须预测的 belief。
6. live、replay、training exporter 必须共享同一 tool name/op/response contract。
7. DAgger 必须使用显式 teacher correction，不能把失败动作回写成监督。
8. RL 必须建立在可复现 transition、numeric reward、stable replay key 和 terminal trace 上。
9. Safety Validator 缺少观测证据时必须拒绝或停止，不能通过不透明 fallback 默认接受。
10. 所有闭环指标必须从环境事件计算，不能只用文本格式正确率替代 arrival、collision 和 recovery。
