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
initial instruction / reset context
out-of-band user events                         observation stream
new / modify / pause / resume / cancel          FPV history / BEV / goal
user_response / emergency_stop                  controller feedback
              |                                         |
              +--------------------+--------------------+
                                   v
                    Event Gate + Revision Reducer
              TaskState / DialogueState / ControlState
                                   |
                                   v
                         System2: LVLM Agent
                  multimodal Plan & Execute policy
       +----------------+----------------+----------------+
       |                |                |                |
   tool_call         ask_user         navigate   complete_task/abort_task
       |                |                |                |
       v                v                v                v
 ToolSpecRegistry    HITL wait       [SYS1_i]     terminal_control
 MCP/live/replay     interrupt           |
 RouteSession CAS    + resume            v
       |                |       System1: Direct / CFM
       +-------> System2 re-entry  trajectory_ego_t [T,3]
                                        |
                                  revision binding
                       task / route / control / observation
                                        |
                                 Event Gate + Safety
                              accept | replace | reject
                                        |
                                 ExecutionLease
                            execute first M points, M << T
                                        |
                         controller-feedback / event race
                              /                       \
                     next observation            preempt + replan
                              \                       /
                               +---------+-----------+
                                         v
                                  EpisodeTraceV2
                                         |
                      +------------------+------------------+
                      |                  |                  |
              frozen benchmark   session SFT/DAgger   replay-ready RL
```

System2 是低频认知与任务策略，System1 是高频局部连续策略。两者通过 LVLM hidden state 和稳定的 `[SYS1_i]` readout token 连接，不通过额外 RouteBelief 神经模块连接。

总体框架进一步分为三个互相解耦的执行平面：

| 平面 | 核心职责 | 不负责 |
| --- | --- | --- |
| Task / Agent plane | 多模态理解、Plan & Execute、工具选择、HITL、任务修订 | 直接修改 RouteSession 内部对象、直接驱动底盘 |
| Tool / State plane | typed tool 校验与执行、RouteSession CAS、地图/图像/用户交互外部状态 | 生成连续轨迹、训练 LVLM |
| Execution plane | System1 连续轨迹、安全 gate、短时域执行、抢占、环境反馈 | 解释用户自然语言、维护对话语义 |

三个平面只通过 Pydantic contract 和 append-only event 交换数据。`EpisodeTraceV2` 是事实记录，不是第四个模型模块。

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

在 tick \(t\)，runtime 的 checkpoint-safe 状态写为：

\[
s_t=(o_t,\tau_t,d_t,c_t,q_t,h_t,e_t,b_t),
\]

其中：

- \(o_t\)：当前多模态观测引用。
- \(\tau_t\)：`TaskState`，包括当前指令、goal、约束和 `task_revision`。
- \(d_t\)：`DialogueState`，包括有限窗口对话和待回答 interaction。
- \(c_t\)：`ControlState`，包括 `control_epoch`、活动 action lease 和暂停状态。
- \(q_t\)：RouteSession、`route_revision` 和外部工具状态。
- \(h_t\)：因果有序的模型、工具、交互和执行历史。
- \(e_t\)：环境信息，包括 replay key、到达和碰撞信息。
- \(b_t\)：单决策/全 episode 工具次数、安全重试次数、最大 tick 等 runtime budget。

任意时刻到达的外部事件记为：

\[
u_j=(type,content,r_{expected},interaction\_id,event\_id).
\]

它先经过纯 reducer，而不是直接拼接到 prompt：

\[
(\tau',d',c',g)=R_{event}(\tau,d,c,u_j),
\]

其中 \(g\) 决定继续 System2、等待、终止或立即抢占。`event_id` 提供幂等性，`expected_task_revision` 提供 optimistic concurrency control。

System2 产生离散层级动作：

\[
a_t^{(2)}=\pi_{\theta}^{S2}(s_t),
\]

\[
a_t^{(2)}\in
\{\texttt{tool\_call},\texttt{ask\_user},\texttt{navigate},
\texttt{complete\_task},\texttt{abort\_task}\}.
\]

只有当 \(a_t^{(2)}=\texttt{navigate}\) 时才调用 System1：

\[
X_t^{ego}=\pi_{\phi}^{S1}(H_t,C_t),
\qquad X_t^{ego}\in\mathbb R^{T\times3}.
\]

每个 System1 output 必须绑定生成时的任务、路线、控制代和观测：

\[
bind(X_t)=(r_t^{task},r_t^{route},e_t^{control},o_t^{id},a_t^{id}).
\]

执行前 freshness 条件为：

\[
fresh(X_t)=
[r_X^{task}=r_{now}^{task}]
\land[r_X^{route}=r_{now}^{route}]
\land[e_X^{control}=e_{now}^{control}]
\land[o_X^{id}=o_{latest}^{id}].
\]

任何一项不一致都必须拒绝，不能执行旧任务、旧路线或旧观测生成的 chunk。

`<finish>...</finish>` 只是 assistant 本轮输出结束的文本容器，不是任务成功信号。模型请求完成时还必须通过确定性到达验证：

\[
Success_t=
[a_t^{S2}=\texttt{complete\_task}]
\land ArrivalVerifier(o_t,r_t)=1.
\]

验证失败产生 `terminal_validation(accepted=false)` 并重新进入 System2；超过终态拒绝预算才以 policy-contract failure 结束。`abort_task` 表示结构化失败终止，`cancel` 只来自外部事件；二者都不调用 System1。

轨迹通过执行前验证后，只执行前 \(M\) 个点：

\[
v_t=V_{safe}(X_t^{ego},o_t),
\]

\[
(o_{t+1},r_t,done_t,i_t)=Env.step(X_{t,1:M}^{ego}),
\quad M\ll T,\quad v_t=1.
\]

执行期间 controller feedback 与外部事件并发竞争。若外部事件先到达，则 active lease 被抢占，`control_epoch` 递增，旧 chunk 失效，再由 reducer 决定暂停、终止或回到 System2。若 \(v_t=0\)，原轨迹不执行；validator 可以给出显式 replacement trajectory，否则结构化拒绝结果返回 System2。

`tick_id` 只在成功完成一次环境 transition 后递增。一个 tick 内可能因多次工具调用、安全拒绝或任务修改产生多个 `decision_id`，因此训练样本的主切分单位是 `decision_id`，不是 `tick_id`。

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
      "x_right": 1.2,
      "y_front": 6.8,
      "heading": 0.1,
      "kind": "goal",
      "coordinate_frame": "ego"
    }
  ],
  "route_session_id": "route-session-001",
  "route_revision": 3
}
```

注意：当前可选 Goal Encoder 的 `ego` 几何输入仍使用 `x_right/y_front` v1 语义；它只作为条件 latent。生产 System1 target 在 exporter 边界独立统一为 FLU `x_forward/y_left/yaw_ccw`，两者不能按同名 `x/y` 隐式混用。

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
decision_type
route_action
intent / direction_choice / motion_primitive
speed_policy / social_action / traffic_action
trajectory_validity
```

Runtime 级 `System2ActionType` 固定为：

```text
tool_call | ask_user | navigate | complete_task | abort_task
```

Prompt 内的 `decision_type` 与 Runtime 动作一一对应；`route_action` 只进一步描述路线/上下文操作，例如：

```text
continue_navigation
use_map_context
create_route
update_route
add_via
replan
inspect_visual_context
unknown
```

历史 `request_user_input/finish/stop` 只在输入 adapter 映射为 `ask_user/complete_task/abort_task`，新版 row 不再把交互或终态写入 `route_action`。

临时物理停车属于连续导航而非任务终止：

```text
decision_type = navigate
intent = stop
speed_policy = stop
trajectory_type = stop_hold
```

该路径仍调用 System1，生成短时保持/减速轨迹并经过 Safety 与 execution。`abort_task` 则禁止 readout/CFM，由 execution layer 直接执行确定性停车。

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
system1_target_format = ego_pose_xyt_flu_v1
columns = [x_forward_m, y_left_m, yaw_rel_ccw_rad]
representation = cumulative_relative_pose
frame = current_ego_flu
normalized = false
```

其中

\[
q_i=[x_i,y_i,\psi_i]
\]

表示未来第 \(i\) 个时刻相对当前观测时刻 \(t_0\) 的累计相对位姿。所有点都在
\(t_0\) 的固定 FLU 参考帧中：\(x\) 向前，\(y\) 向左，yaw 逆时针为正。
它不是世界坐标绝对位姿，也不是相邻 waypoint 增量。

历史 RouteServer 的 `x_right/y_front` 数据只在 exporter 边界转换一次：

\[
x_{forward}=y_{front},\qquad y_{left}=-x_{right}.
\]

第三维优先使用 odometry：

\[
\psi_i=wrap(\psi_i^{world}-\psi_0^{world}).
\]

缺少可靠 yaw 时，exporter 才从路径切线估计并记录 `yaw_source=path_tangent`：

\[
\psi_i=atan2(y_i-y_{i-1},x_i-x_{i-1}).
\]

#### 4.3.1 原始数据进入 CFM 前的固定转换闸门

原始 FPV、BEV、instruction 和 tool context 属于模型观测，可以按各自的多模态
adapter 进入 LVLM。原始 GPS、odometry、历史轨迹和 delta action 属于运动监督，
**不能直接作为生产 CFM target**。生产数据必须在离线 source adapter/exporter 边界
完成下面的确定性转换：

```text
raw GPS / world odometry / legacy trajectory
  -> 解析数据源坐标轴、单位、采样时刻和 yaw 定义
  -> GPS/global pose 投影到局部米制平面
  -> 以当前观测时刻 t0 建立固定 ego FLU 参考帧
  -> 对所有未来 pose 计算 (T_e0^w)^-1 T_i^w
  -> 构造 metric cumulative trajectory_ego_t [T,3]
  -> 补齐并校验 trajectory_policy
  -> 写入 canonical JSONL/replay row
  -> ms-swift dataset/collator
  -> Direct/CFM System1 training
```

不同来源必须按显式规则处理：

| 原始监督 | 固定处理 | 是否可直接训练 |
| --- | --- | --- |
| world/GPS pose + odometry yaw | 投影到米制平面，再做 current-ego SE(2) 相对变换 | 否，转换后才可用 |
| `[x_right,y_front,dt,speed]` | `x_forward=y_front`、`y_left=-x_right`；yaw 使用 odometry 或路径切线；`dt/speed` 不进入三维 target | 否，转换后才可用 |
| rolling body delta | 仅在语义明确时通过 SE(2) 累积重建轨迹；否则保留为独立 `action_ego_t` 消融 | 否，不允许隐式混用 |
| normalized trajectory/action | 使用该数据集绑定的统计量反归一化为物理量 | 否，转换后才可用 |
| canonical FLU `[T,3]` | 检查 finite、字段、format、frame、单位、yaw source 和 `normalized=false` | 校验通过后可直接训练 |

canonical row 至少必须声明：

```yaml
trajectory_policy:
  system1_target_field: trajectory_ego_t
  system1_target_format: ego_pose_xyt_flu_v1
  representation: cumulative_relative_pose
  coordinate_frame: current_ego_flu
  axis_names: [x_forward_m, y_left_m, yaw_rel_ccw_rad]
  yaw_source: odometry  # or path_tangent
  normalized: false
  dt_sec: 0.2
```

该转换不得放进 trainer、CFM head 或 runtime 中按 shape 动态推断。这样可以保证同一条
row 在训练、离线评测和 replay 中具有完全相同的数值语义。当前 CityWalker/EgoWalk
旧 exporter 仍可生成四维历史标签；这些 row 必须先经过 v2 canonical exporter 或专用
migration，再进入生产训练。

兼容控制合同是：

```text
action_ego_t [T,3]
columns = [dx_forward_body_m, dy_left_body_m, dtheta_ccw_rad]
space = ego_delta_xyt_flu_v1
```

二者不能仅凭维度混用。累计 ego 轨迹转 rolling body delta 必须显式计算：

\[
\Delta p_i^{ref}=p_i-p_{i-1},
\]

\[
\Delta p_i^{body}=R(-\psi_{i-1})\Delta p_i^{ref},
\]

\[
\Delta\psi_i=wrap(\psi_i-\psi_{i-1}),
\qquad (p_{-1},\psi_{-1})=(0,0).
\]

当前 `ActionRuntimeAdapter` 已实现该旋转转换，并把结果交给离散动作映射或
`v/omega` 控制层。真实差速底盘仍推荐用 Pure Pursuit/MPC 跟踪累计轨迹，adapter
的逐步 `v/omega` 主要用于统一 smoke/replay 接口。

生产 trainer 和 checkpoint 固定：

```text
sidewalk_system1_target_field = trajectory_ego_t
sidewalk_system1_target_format = ego_pose_xyt_flu_v1
```

当 row 缺少 `trajectory_ego_t` 时，该 row 的 System1 loss 关闭，不允许自动回退到
`action_ego_t`。`action_ego_t` 只用于独立 delta-action 消融和旧数据兼容，必须使用
单独配置与 checkpoint。

对 `action_ego_t`：

\[
a_{norm}=\frac{a_{raw}-\mu}{\sigma},\qquad
a_{raw}=a_{norm}\sigma+\mu.
\]

生产 row 始终保存真实物理量 `normalized=false`。CFM 内部如需标准化，统计量绑定在
checkpoint System1 config 中，推理后必须反归一化。checkpoint 和 inference row
必须保存 `system1_target_field`、`system1_target_format` 和 `normalized`，禁止只根据
`[T,3]` shape 决定是否反归一化或累计积分。

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
assistant <finish> contains trajectory_coord_frame=norm1000
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
4. 请求完成：生成 `complete_task`，但只有 ArrivalVerifier 通过后才能成功。
5. 无法继续：生成带 `reason_code` 的 `abort_task`，作为 failed 终态。

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
  decision_type / route_action / intent / confidence / social_action / ...
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
  "system1_chunk_ego": [[0.4, 0.0, 0.0], [0.9, 0.1, 0.08]],
  "action_chunk_metadata": {
    "head_type": "flow_action",
    "system1_condition_mode": "hybrid",
    "flow_steps": 8,
    "anchor_index": 2,
    "anchor_name": "straight_center"
  }
}
```

`system1_chunk_ego` 是 ms-swift inference artifact 字段；进入 Episode Runtime 时必须由 `System1Policy` adapter 映射为 `System1Output.trajectory_ego_t`，校验 checkpoint coordinate config，并绑定 `task_revision/route_revision/control_epoch/observation_id/action_id`。该映射不做坐标猜测或隐式累计。当前 graph 已固定 Protocol 和绑定逻辑，但仓库内尚无一个面向生产 checkpoint service 的统一 `System1Policy` adapter。

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
  -> event_before_system2
       deterministic pause/cancel/emergency -> wait or terminal_control
       task update/user response             -> reduce -> system2
       no event                              -> system2
  -> system2
       tool_call -> tool -> event_before_system2
       ask_user  -> prepare_hitl -> await_hitl -> event_before_system2
       navigate  -> system1 -> event_before_safety
       complete_task/abort_task -> event_before_terminal -> terminal_control
  -> safety
       accepted/replacement -> event_before_dispatch -> dispatch
       rejected             -> system2
       retry exhausted      -> terminal_control(failed)
  -> dispatch -> await_execution
       controller feedback -> environment_step -> observe
       external event      -> preempt -> event_before_system2
  -> terminal_control
       complete verified   -> halt -> finish(succeeded)
       complete rejected   -> system2
       abort/cancel/failure -> halt -> finish(failed/cancelled)
  -> END
```

`event_before_*` 不是模型节点，而是统一事件闸门。它在 System2、Safety 和 Dispatch 等副作用边界前 claim 外部事件，调用纯 reducer，完成后才 acknowledge；异常时 release，保证事件可重试。

Runtime budget 包括：

```text
max_ticks
max_tool_calls              # 一个 physical step 前的连续工具预算
max_total_tool_calls        # 全 episode 工具预算
max_safety_retries
max_terminal_rejections    # 未验证 complete_task 的重试预算
execute_steps               # 每次实际执行的轨迹前缀 M
max_dialogue_turns
```

### 8.2 外部事件与任务状态

当前允许任意时刻进入的事件为：

```text
new_instruction | modify_task | pause | resume
cancel | user_response | emergency_stop
```

处理规则是确定性的：

| 事件 | TaskState | ControlState | 后续路由 |
| --- | --- | --- | --- |
| `pause` / `emergency_stop` | `status=paused` | `control_epoch+1`，立即抢占 | `external_wait` |
| `cancel` | `status=cancelled` | `control_epoch+1`，幂等 halt | terminal |
| `resume` | `status=active` | `control_epoch+1` | 重新进入 System2，不续跑旧 chunk |
| `new_instruction` | 替换 active instruction，`task_revision+1` | 旧 lease 失效 | System2 |
| `modify_task` | 合并 goal/constraints，`task_revision+1` | 旧 lease 失效 | System2，再通过 typed tool 修改路线 |
| correlated `user_response` | 清除匹配的 pending interaction | 保持安全等待策略 | System2 |
| stale/unmatched `user_response` | 作为新指令处理 | 旧 lease 失效 | System2 |

`expected_task_revision` 不匹配时产生 `revision_conflict`，不改变任务。事件队列当前提供 `append/claim/acknowledge/release/wait_for_pending` Protocol；内置 `InMemoryExternalEventStore` 只适用于测试和单进程 smoke，生产多进程部署必须替换为事务型持久队列。

执行层明确区分两个接口：`preempt()` 只取消当前 active chunk，并调用控制器抢占；`halt()` 无论是否存在 active lease 都下发一次幂等物理停车。`complete_task/abort_task/cancel` 统一经过 `halt()`，不能用零轨迹或调用 CFM 代替确定性停车。

### 8.3 HITL

`ask_user` 或 blocking user-interaction 采用两阶段协议：

```text
prepare_hitl
  -> 先持久化 PendingInteraction、interaction_id、waiting trace
await_hitl
  -> interrupt()
resume / external user_response
  -> 以 interaction_id 关联
  -> reducer 清除 pending interaction
  -> System2
```

工具执行和 waiting-state 写入都发生在 `interrupt()` 之前的独立节点，避免恢复时重复副作用。`user_response` 消费后会从瞬时 graph 字段清除，长期对话只保存在有界 `DialogueState` 中。

### 8.4 Safety Validator

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

当前 graph 支持三种结果：

```text
accepted=true                   -> 执行原 System1Output
intervention.system1_output     -> 校验 canonical contract 后执行 replacement
accepted=false, no replacement  -> 返回 System2；超过重试预算后失败
```

replacement 会写入 `safety_intervention` event 并成为 `executable_system1_output`，不会再执行被拒绝的原轨迹。但当前仓库仍只有 `SafetyValidator` Protocol、graph 接线和 deterministic test validator；尚无生产级 occupancy、robot footprint、dynamic-agent prediction 和真实 emergency controller。因此“intervention 已可执行”不等于“物理安全系统已完成”。

ms-swift 的 `kinematic_validity_loss` 是训练软约束，offline collision metric 是评测量；二者都不能替代运行时 hard gate。

### 8.5 Receding-horizon 执行与抢占

安全通过后，runtime 不直接把完整 \(T\) 点 chunk 交给环境，而是建立版本绑定的 `ExecutionLease`，只 dispatch 前 \(M\) 点：

\[
X_{1:T}\rightarrow lease(X_{1:M}),\qquad M=execute\_steps\ll T.
\]

`await_execution` 同时等待：

```text
controller.wait(lease)
external_event_store.wait_for_pending(episode_id)
```

外部事件先到达时调用 `preempt()`，记录 `action_preempted`，并使该 decision 的 System1 supervision 无效。环境先完成时记录 `controller_feedback` 和 `environment_step`，更新 observation 后进入下一 tick。

内置 `EnvironmentExecutionManager` 已实现异步 dispatch、短前缀、freshness 检查和 preempt-before-wait 竞态处理。真实 robot/simulator adapter 必须同时实现 `EpisodeEnvironment.preempt(reason)` 与幂等 `EpisodeEnvironment.halt(reason)`：前者取消当前 active chunk，后者无论是否存在 active action 都向控制器下发确定性停车。取消 Python task 不能替代底盘停车或 emergency stop。

### 8.6 Environment

统一环境协议：

```text
reset(initial_context) -> ObservationResult
step(System1Output, state) -> EnvironmentStepResult
preempt(reason) -> cancel active controller command
halt(reason) -> idempotent deterministic physical stop
```

step 返回：

```text
observation
reward: float
done: bool
info: arrival/collision/recovery diagnostics
replay_key optional/live, required/frozen RL
external_events optional
```

当前已有冻结 transition environment、HTTP episode service 和统一 cumulative-FLU-to-controller adapter。冻结环境能够验证状态机、版本、trace 和 metric；真实机器人/仿真器的 observation timing、轨迹跟踪误差、hard preemption 和安全传感器仍需部署验证。

---

## 9. EpisodeTraceV2

### 9.1 Event-sourced contract

`EpisodeTraceV2` 是 runtime、训练和评测的唯一 session-level 因果记录：

```text
lifecycle:
  episode_started / system_message / episode_finished

model and tools:
  observation_received / model_output
  tool_call / tool_response / route_state_delta / readout_requested

external task and dialogue:
  external_event_received / external_event_rejected / revision_conflict
  task_state_updated / interaction_requested / interaction_resolved
  user_interrupt / user_resume

continuous execution:
  system1_output / safety_validation / safety_intervention
  action_dispatched / action_preempted
  controller_feedback / environment_step
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

### 9.2 决策级因果顺序

一个物理 tick 可以包含多个 `decision_id`。完整导航 decision 的因果顺序为：

```text
observation_received
external_event_received + task_state_updated optional
model_output(decision_id)
  -> tool_call -> tool_response -> next model_output
  -> interaction_requested -> interrupt -> interaction_resolved -> next model_output
  -> navigate finish -> readout_requested -> system1_output
system1_output
event gate / freshness check
safety_validation -> optional safety_intervention
action_dispatched(ExecutionLease)
controller_feedback | action_preempted
environment_step, only when execution completed
```

训练视图中工具路径必须保持：

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

因此 `[SYS1_i]` hidden state 能看到工具结果和最终符号决策。把 query 放在 tool response 或 finish 之前会破坏因果条件。外部事件和 runtime state 也必须先于受其影响的 assistant decision，不能从未来 event 反向泄漏。

只有满足下式的 System1 output 才可成为当前 decision 的训练 target：

\[
valid\_supervision=
has(system1\_output)\land has(action\_dispatched)
\land\neg has(action\_preempted).
\]

### 9.3 Loss mask

| 内容 | LVLM 可见 | LM CE | System1 loss |
| --- | --- | --- | --- |
| system/user prompt | 是 | 否 | 作为条件 |
| images | 是 | 否 | 通过 LVLM latent |
| assistant reasoning/finish | 自回归 | 是 | query 可读取之前部分 |
| tool_call role | 是 | 是 | 作为历史条件 |
| tool_response role | 是 | 否，`-100` | 作为历史条件 |
| external event / runtime state | 是 | 否，`-100` | 作为状态条件 |
| terminal validation | 是 | 否，`-100` | 被拒绝的完成请求作为下一轮 System2 条件 |
| `[SYS1_i]` | 是 | 否，`-100` | query hidden state |
| `trajectory_ego_t` | 否 | 否 | CFM/direct target |
| preempted System1 output | 可在 trace 中审计 | 否 | 不作为 target |
| reward/replay key | 否 | 否 | rollout/RL metadata |

---

## 10. 训练数据与目标函数

### 10.1 Canonical v2 row

```json
{
  "case_id": "episode-001:decision:3",
  "episode_id": "episode-001",
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
  "trajectory_ego_t": [[0.4, 0.0, 0.0], [0.9, 0.1, 0.08]],
  "trajectory_policy": {
    "system1_target_field": "trajectory_ego_t",
    "system1_target_format": "ego_pose_xyt_flu_v1",
    "representation": "cumulative_relative_pose",
    "coordinate_frame": "current_ego_flu",
    "axis_names": ["x_forward_m", "y_left_m", "yaw_rel_ccw_rad"],
    "normalized": false,
    "yaw_source": "odometry",
    "dt_sec": 0.2
  },
  "action_mask": [true, true],
  "contract_meta": {
    "episode_trace_schema": "urbannav_episode_trace_v2",
    "system1_readout_placement": "assistant_suffix_v2",
    "system1_readout_lm_label": -100,
    "environment_observation_lm_label": -100,
    "external_event_lm_label": -100,
    "runtime_state_lm_label": -100,
    "tool_response_lm_label": -100
  },
  "sidewalk_meta": {
    "prompt_template": "agentic_stage2_cot_trajectory",
    "episode_training_view": "urbannav_episode_per_decision_v3",
    "session_training_objective": "session_sft",
    "decision_id": "episode-001:decision:3",
    "decision_index": 3,
    "tick_start": 0,
    "tick_end": 1,
    "system1_supervision_valid": true
  }
}
```

`trajectory_policy` 是 active System1 row 的强合同，不是说明性 metadata。当前 trainer 会校验它与 checkpoint 的 `system1_target_field/format` 一致。事件级 exporter 目前已经按 `decision_id` 选择有效 `trajectory_ego_t/action_mask`，但尚未把 `System1Output.metadata` 中的 canonical policy 完整复制为顶层 `trajectory_policy`；因此该 exporter 直接产出的 active System1 row 在进入生产 trainer 前仍需补齐此字段。这是当前实现缺口，不能依赖 trainer 默认值绕过。

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

`EpisodeTraceV2` 对新 trace 按 `decision_id` 展开 causal prefix row；仅对没有 `decision_id` 的 legacy trace 按 tick 兼容导出。DAgger 只使用显式 teacher correction 替换错误 assistant action 或 System1 target，不能把失败 rollout 自身当作正确标签。

**Phase E：GRPO/RL**

只有 episode 同时满足以下条件才允许进入 RL：

```text
every environment_step has numeric reward
every environment_step has stable replay_key
episode has terminal episode_finished event
environment transition is reproducible
```

RL 不是 SFT 失败时的 fallback。

### 10.4 Multi-tick / per-decision exporter

对新 event-driven trace，ms-swift exporter 为每个 System2 decision 构造历史前缀：

\[
row_j=Export(e_{0:end(decision_j)}).
\]

同一 tick 的 `tool_call -> tool_response -> next model_output` 会产生多个 row，分别学习工具选择和工具结果后的新决策。可通过 `window_ticks` 截断物理 tick 历史，但不能打乱保留窗口内 event 顺序：

\[
row_j=Export(e_{tick\ge t_j-W+1:\,end(decision_j)}).
\]

环境 observation、external event、runtime state 和 tool response 保持 context role，只有 assistant/tool-call action 是 LM target。System1 target 从当前 `decision_id` 的 `system1_output` 读取，并且必须存在 `action_dispatched` 且不存在 `action_preempted`。DAgger correction 的主键为：

```text
trace.metadata.dagger_corrections[decision_id]
```

仅 legacy trace 才允许使用 tick id correction key。

---

## 11. 完整数据流与字段字典

本节列出当前代码在各边界实际使用的字段。它不是要求把全部状态塞进一个 prompt，而是固定每类数据的 owner、序列化位置和监督可见性。

### 11.1 全链路主键与版本字段

| 字段 | 作用 | 生命周期 |
| --- | --- | --- |
| `episode_id` | 一次闭环 rollout 的主键 | reset 到 terminal |
| `session_id` / `route_session_id` | 可跨多个任务事件延续的 RouteSession 主键 | route session |
| `task_id` | 当前用户导航任务主键 | task |
| `task_revision` | 用户任务语义版本 | 每次 new/modify task 递增 |
| `route_revision` | 已提交路线版本 | 每次 route mutation 成功后更新 |
| `control_epoch` | 连续控制代，用于使旧 action 失效 | 每次抢占/任务变化递增 |
| `observation_id` | System1 所依据的观测主键 | 每个 observation |
| `decision_id` | 一次 System2 决策主键 | 同一 tick 可有多个 |
| `call_id` | tool call/response 关联主键 | 一次工具调用 |
| `interaction_id` | ask-user/request-response 关联主键 | 一次 HITL |
| `action_id` | System1 output 与 execution lease 关联主键 | 一次 dispatch |
| `event_id` | external event 或 trace event 幂等/审计主键 | append-only |
| `parent_event_id` | trace 因果父事件 | append-only |
| `replay_key` | 冻结 observation/transition/tool result 主键 | offline replay |
| `schema_version` | contract 版本 | 每个序列化对象 |

这些字段不能用自然语言位置或 list index 替代。训练切分以 `decision_id` 为主，环境推进以 `tick_id` 为主，二者语义不同。

### 11.2 Reset 与模型观测字段

`initial_context` 当前是开放 JSON object，runtime 明确消费或透传的常用字段为：

```text
episode_id optional
task_id optional
instruction | request
goal_spec {}
constraints {}
route_session_id optional
route_revision optional
environment-specific reset fields
```

`ObservationResult`：

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `observation` | `dict` | 模型/runtime 可见的结构化观测引用 |
| `info` | `dict` | deterministic 环境 metadata，不默认作为 LM target |
| `replay_key` | `str|null` | frozen reset/observation key |
| `external_events` | `list[dict]` | 与该观测同时产生的冻结/仿真事件 |

`observation` 本身是可扩展 object，推荐稳定结构：

```json
{
  "instruction": "Go to the library.",
  "images": {
    "fpv_history": ["fpv_t-2.jpg", "fpv_t-1.jpg"],
    "fpv": "fpv_t.jpg",
    "bev": "bev_t.jpg",
    "goal_image": "goal.jpg"
  },
  "goal_geometry": [],
  "route_session_id": "route-session-001",
  "tool_context": {},
  "controller_state": {}
}
```

图像、instruction、goal 和 tool context 是 LVLM 条件；`info/replay_key` 是 runtime/eval metadata。GT trajectory、reward 和 teacher correction 不属于 observation。

### 11.3 Task、Dialogue、Control 与外部事件

#### TaskState

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `task_id` | `str` | 当前任务主键 |
| `active_instruction` | `str` | 当前有效自然语言指令 |
| `goal_spec` | `dict` | image/coordinate/semantic goal 描述 |
| `constraints` | `dict` | 当前任务约束和策略约束 |
| `route_session_id` | `str|null` | 绑定 RouteSession |
| `task_revision` | `int>=0` | 任务语义 revision |
| `route_revision` | `int>=0` | committed route revision |
| `status` | `active|paused|cancelled|completed` | 任务生命周期 |
| `last_event_id` | `str|null` | 最后接受的外部事件 |

#### DialogueState

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `recent_turns` | `list[DialogueTurn]` | 有界结构化对话窗口 |
| `summary` | `str|null` | 可选旧对话摘要 |
| `pending_interaction` | `PendingInteraction|null` | 当前阻塞问题 |

`DialogueTurn` 字段为 `role/content/event_id/interaction_id`；`PendingInteraction` 字段为 `interaction_id/request/motion_policy`。

#### ControlState

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `control_epoch` | `int>=0` | action freshness generation |
| `active_action_id` | `str|null` | 当前执行 lease |
| `status` | `idle|executing|paused|stopped` | controller 状态 |
| `last_preempt_reason` | `str|null` | 最近一次确定性抢占原因 |

#### ExternalUserEvent

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `episode_id` | `str` | 目标 episode |
| `event_type` | enum | `new_instruction/modify_task/pause/resume/cancel/user_response/emergency_stop` |
| `content` | any | 文本或结构化 task update |
| `expected_task_revision` | `int|null` | optimistic concurrency guard |
| `interaction_id` | `str|null` | user response 关联 id |
| `event_id` | `str` | 幂等键 |
| `timestamp` | datetime | 来源时间 |
| `metadata` | `dict` | transport diagnostics，不进入监督 |
| `schema_version` | `urbannav_task_contract_v1` | task contract 版本 |

#### EpisodeGraphState

LangGraph checkpoint 内部字段完整集合为：

```text
episode_id / session_id / tick_id / initial_context
observation / observation_id / observation_info
history / episode_trace
task_state / dialogue_state / control_state
latest_external_event / event_route
decision_id / system2_action / last_tool_response
pending_user_request / user_response
system1_output / executable_system1_output
safety_result / execution_lease / execution_interrupted
environment_result
tool_call_count / total_tool_call_count / safety_retry_count
terminal_rejection_count / terminal_validation / terminal_rejected
done / terminal_status / stop_reason
```

这些是 runtime checkpoint 字段，不是一个模型输入 schema。System2 policy adapter 只应渲染当前决策所需的 observation、task/dialogue 摘要、工具结果和版本信息；ground truth、reward 和未来 event 不能从 `EpisodeGraphState` 泄漏进 prompt。

### 11.4 System2 输出与 finish

`System2Action`：

| 字段 | 类型 | 条件 |
| --- | --- | --- |
| `action_type` | `tool_call|ask_user|navigate|complete_task|abort_task` | 必填 |
| `tool_call` | `ToolCallV2|null` | `tool_call` 时必填 |
| `user_request` | `dict|null` | `ask_user` 时必填 |
| `finish` | `dict|null` | assistant `<finish>` 符号字段；标签本身不表示完成 |
| `terminal` | `TerminalDecision|null` | complete/abort 时必填 |
| `metadata` | `dict` | assistant_text、teacher/model diagnostics |

`TerminalDecision` 是强类型 Pydantic contract：

```text
outcome: succeeded | failed
reason_code: non-empty string
reason optional
evidence_refs: list[str]
retryable: bool
```

`complete_task` 必须绑定 `outcome=succeeded`，`abort_task` 必须绑定 `outcome=failed`。模型给出的 `evidence_refs` 只用于审计，不能替代 runtime 到达验证。

assistant `<finish>` 主干 superset 为：

```text
decision_type
route_action
intent
confidence_level
trajectory_type
trajectory_validity
map_context_available
social_action
tool_used optional
teacher_source optional
```

Phase A/v1 文本轨迹还可包含：

```text
trajectory_grid1000
trajectory_coord_frame = norm1000
```

仅 `decision_type=navigate` 的 v2 System1 row 在 assistant `finish` 后追加：

```text
<readout>[SYS1_0] ... [SYS1_15]</readout>
```

`trajectory_ego_t`、`action_ego_t`、action label、segmentation label 和 teacher 私有判断不得写入 `finish`。

`ask_user/complete_task/abort_task` row 禁止 `[SYS1_i]/[TRAJ_i]` readout，也不得携带 System1 continuous supervision。旧 `finish/stop` runtime 值和旧 `route_action=request_user_input` 由双读 adapter 转换，canonical exporter 只输出新格式。

### 11.5 ToolCall、ToolResponse 与参数字段

`ToolCallV2`：

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `tool` | `str` | 稳定学生可见 meta-tool 名 |
| `arguments` | `dict` | 由 registry 对应 Pydantic model 校验 |
| `call_id` | `str|null` | 调用关联/幂等 id |
| `schema_version` | `urbannav_tool_contract_v2` | contract 版本 |

`ToolResponseV2`：

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `tool` / `op` | `str` / `str|null` | 实际工具与 operation |
| `success` | `bool` | 执行是否成功 |
| `result` | any | 模型可见确定性结果 |
| `error` | `dict|null` | 结构化失败 |
| `artifacts` | `list[dict]` | 图像、路线或缓存 artifact 引用 |
| `call_id` | `str|null` | 原 tool call id |
| `executor` | `str|null` | live/replay executor profile |
| `latency_ms` | `float|null` | 执行延迟 |
| `state_revision` | `str|int|null` | mutation 后 route/environment revision |
| `base_state_revision` | `str|int|null` | CAS 基础 revision |
| `state_delta` | `dict` | 显式状态变化 |
| `idempotency_key` | `str|null` | 副作用去重键 |
| `replay_key` | `str|null` | 冻结结果键 |
| `schema_version` | `urbannav_tool_contract_v2` | contract 版本 |

`map_tool.arguments` 是 operation superset，完整字段为：

```text
op
query / address / location
origin / destination / current_pose / via_point
destination_id / candidate_id / route_id / session_id
latitude / longitude
bbox / min_latitude / min_longitude / max_latitude / max_longitude
category / subcategories / keywords / radius / sortrule / limit
mode / steps / via_points / session_name / force_replan
insert_before_index / via_point_index / via_point_name / stop_index
maneuver / next_maneuver
distance_m / eta_sec / next_maneuver_distance_m
instruction_text / waypoint_ego / next_waypoint / image_ref
asset_types / horizon_m / frame
expected_state_revision / idempotency_key / source_event_id
```

`image_tool.arguments`：

```text
op = crop | zoom_in
image optional
image_path optional
bbox = [x1,y1,x2,y2]
```

`user_interaction_tool.arguments`：

```text
op
question / message
choices / options / recommended_option_id
requires_user_response / interaction_mode
motion_policy_while_waiting
evidence_refs / proposal / interaction_id
```

兼容 alias 只在 registry normalization 边界转换；trace、replay 和训练 row 统一保存 canonical name/op/arguments。

RouteSession 当前是持久化 JSON object，核心字段为：

```text
session_id / session_name / provider
origin / destination / via_points / mode
created_at / updated_at
route_revision
route_plan_cache / route_plan_cached_at
```

mutation result 还会返回 `session/route_plan/route_plan_cache_hit/base_state_revision/state_revision/idempotency_key/idempotent_replay/state_delta/session_store`。这些字段位于 tool result/runtime state，不是 assistant 必须复述的 LM target。

当前 CAS commit lock、JSON/memory store 和 mutation replay cache 适用于单进程 runtime。多 worker 或多机器人共享 RouteSession 时必须迁移到支持事务、唯一幂等键和 compare-and-swap 的持久 store，不能把进程内 `RLock` 当作分布式一致性。

### 11.6 System1、安全与执行字段

`System1Output`：

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `trajectory_ego_t` | `list[list[float]]` | finite、非空 `[T,3]` cumulative FLU trajectory |
| `action_mask` | `list[bool]|null` | 有效未来 step |
| `runtime_commands` | `list[dict]` | 可选 controller-ready 派生命令 |
| `task_revision` | `int|null` | 生成时 task revision |
| `route_revision` | `int|null` | 生成时 route revision |
| `control_epoch` | `int|null` | 生成时 controller generation |
| `observation_id` | `str|null` | 生成所依据观测 |
| `action_id` | `str|null` | execution lease id |
| `metadata` | `dict` | head、sampling 与 canonical coordinate contract |

`metadata` 会强制绑定：

```yaml
system1_target_field: trajectory_ego_t
system1_target_format: ego_pose_xyt_flu_v1
representation: cumulative_relative_pose
coordinate_frame: current_ego_flu
axis_names: [x_forward_m, y_left_m, yaw_rel_ccw_rad]
normalized: false
```

`SafetyResult` 字段：

```text
accepted: bool
reason: stable reason code
metrics: collision/social/kinematic/freshness diagnostics
intervention: optional replacement System1Output wrapper
```

`ExecutionLease` 字段：

```text
episode_id / action_id
task_revision / route_revision / control_epoch
observation_id
execute_steps
```

`ExecutionFeedback` 字段：

```text
lease
result: EnvironmentStepResult optional
preempted: bool
reason
```

### 11.7 环境反馈与 HTTP 字段

`EnvironmentStepResult`：

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| `observation` | `dict` | 下一观测 |
| `reward` | `float` | deterministic per-step reward |
| `done` | `bool` | terminal flag |
| `info` | `dict` | arrival/collision/recovery/controller diagnostics |
| `replay_key` | `str|null` | frozen transition key |
| `external_events` | `list[dict]` | transition 同时产生的事件 |

RouteServer HTTP contract：

```text
POST /v1/episodes/reset
  request:  {"episode": initial_context}
  response: EpisodeResetResponse

POST /v1/episodes/{episode_id}/step
  request:  {"action": latest_assistant, "messages": full_history}
  response: EpisodeStepResponse

POST /v1/episodes/{episode_id}/events
  request:  {"event": ExternalUserEvent, "resume_if_waiting": bool}
  response: EpisodeEventSubmitResponse
```

`EpisodeResetResponse`：

```text
episode_id / observation / replay_key / info / observation_text optional
```

`EpisodeStepResponse`：

```text
episode_id / observation / reward / done / replay_key
reward_components / info / observation_text optional
episode_trace optional / stop_reason optional
```

`EpisodeEventSubmitResponse`：

```text
episode_id / event / accepted / duplicate / info
```

### 11.8 EpisodeTraceV2 字段

`EpisodeEvent` 固定 envelope：

```text
episode_id / tick_id / event_type
payload
event_id / parent_event_id
timestamp / source / metadata
schema_version = urbannav_episode_trace_v2
```

主要 event payload：

| event_type | payload 核心字段 |
| --- | --- |
| `episode_started` | `task_state/dialogue_state/control_state` |
| `observation_received` | `observation/observation_id/info/replay_key` |
| `external_event_received` | `event` |
| `external_event_rejected` | `event_id/reason/task_state/dialogue_state/control_state` |
| `revision_conflict` | `event_id/expected_task_revision` |
| `task_state_updated` | `task_state/dialogue_state/control_state/reason` |
| `model_output` | `decision_id/action` |
| `terminal_validation` | `decision_id/request/result.accepted/reason_code/evidence_refs` |
| `tool_call` | `decision_id/call` |
| `tool_response` | `decision_id/response` |
| `route_state_delta` | `decision_id/state_delta/state_revision` |
| `interaction_requested` | `decision_id/interaction` |
| `interaction_resolved` | event/interaction id |
| `system1_output` | `decision_id/output` |
| `safety_validation` | `decision_id/result/retry_count` |
| `safety_intervention` | `decision_id/reason/output` |
| `action_dispatched` | `decision_id/lease` |
| `action_preempted` | execution race 时 `decision_id/lease/feedback`；event gate 时 `event_id/feedback` |
| `controller_feedback` | `decision_id/feedback` |
| `control_halted` | `decision_id/feedback(episode_id/reason/had_active_action/action_id)` |
| `environment_step` | `decision_id/result` |
| `episode_finished` | terminal status/reason |

`EpisodeTraceV2`：

```text
episode_id
session_id optional
events: ordered list[EpisodeEvent]
status: running|waiting_for_user|succeeded|failed|truncated|cancelled
stop_reason optional
metadata
schema_version = urbannav_episode_trace_v2
```

`metadata` 常用内容包括 tool schemas/capability profile、model/checkpoint id、frozen supervision、`dagger_corrections` 和 benchmark provenance。它不应承载会改变在线行为但未写 event 的隐式状态。

### 11.9 ms-swift trainable row 字段

event-driven exporter 的 row 顶层字段：

```text
case_id
episode_id
messages
tools                  # 当前 exporter 为 JSON string
images
trajectory_ego_t optional
action_ego_t optional, legacy/ablation
action_mask optional
trajectory_policy required when System1 supervision is active
contract_meta
sidewalk_meta
```

`messages` role：

| role | 来源 | LM label |
| --- | --- | --- |
| `system` | system policy | `-100` |
| `user` observation | `observation_received` | `-100` |
| `user` external/runtime | `external_event_received/task_state_updated` | `-100` |
| `user` terminal validation | `terminal_validation` | `-100` |
| `assistant` | `model_output` / correction | CE target |
| `tool_call` | `tool_call` | CE target |
| `tool_response` | `tool_response` | `-100` |

`contract_meta`：

```text
episode_trace_schema
action_space_contract = sidewalk_action_space_v4
system1_readout_placement = assistant_suffix_v2
system1_readout_lm_label = -100
environment_observation_lm_label = -100
external_event_lm_label = -100
runtime_state_lm_label = -100
terminal_validation_lm_label = -100
tool_response_lm_label = -100
```

`sidewalk_meta`：

```text
prompt_template
episode_training_view
session_training_objective
episode_id
decision_id / decision_index
tick_start / tick_end
system1_supervision_valid
```

### 11.10 字段可见性与数据血缘

| 数据 | RouteServer state | LVLM context | LM CE | System1 loss | Eval/RL |
| --- | --- | --- | --- | --- | --- |
| instruction/images/goal | 是 | 是 | prompt 不计 | 条件 latent | slice metadata |
| Task/Dialogue/Control snapshot | 是 | 以 masked runtime message 暴露必要部分 | 否 | 条件 | diagnostics |
| tool call | trace | 是 | 是 | 间接条件 | tool metrics |
| tool response | trace | 是 | 否 | 间接条件 | tool metrics |
| symbolic finish | trace | 自回归 | 是 | query 前置条件 | text/action metrics |
| `[SYS1_i]` | row | 是 | 否 | latent query | query diagnostics |
| `trajectory_ego_t` | trace/row top-level | 否 | 否 | 是 | ADE/FDE/safety |
| safety/intervention/controller | trace | 后续 decision 可见必要摘要 | 否 | preempted 不监督 | closed-loop metrics |
| reward/replay key | trace/HTTP info | 否 | 否 | 否 | RL/eval |

完整数据血缘固定为：

```text
raw sensors/user events/provider state
  -> RouteServer typed contracts + event reducer
  -> System2 context
  -> tool/HITL loop or symbolic finish
  -> masked [SYS1_i] latent
  -> canonical trajectory_ego_t
  -> freshness + safety + short execution lease
  -> controller/environment feedback
  -> EpisodeTraceV2
  -> per-decision exporter
  -> ms-swift messages + top-level System1 supervision
```

---

## 12. 推理路径

### 12.1 Single open-loop

```text
one observation
-> System2 text/tool decision
-> optional real tool execution
-> final finish + [SYS1_i]
-> System1 head
-> one trajectory chunk
```

该路径用于 checkpoint smoke 和 Stage2 协议验证，不构成多 tick 到达闭环。未经过 Stage2 tool/finish 监督的本地 checkpoint 可能生成不合法 action protocol；runtime 必须显式拒绝，不能通过 fake/fallback 修补成成功结果。

### 12.2 RouteServer-owned closed loop

```text
EpisodeRuntime.run()
-> event gates control external events and revisions
-> LangGraph controls tools/HITL/System1/safety/execution/env
-> EpisodeTraceV2
```

适合部署和冻结 benchmark，`System2Policy/System1Policy/SafetyValidator/EpisodeEnvironment` 通过 Protocol 注入。外部事件通过 `EpisodeRuntime.submit_event()` 独立进入，不能伪装成下一次普通 `Env.step`。

### 12.3 ms-swift-owned rollout scheduler

`UrbanNavEnvClient` 通过严格 HTTP 合同连接 RouteServer：

```text
POST /v1/episodes/reset
POST /v1/episodes/{episode_id}/step
POST /v1/episodes/{episode_id}/events
```

`UrbanNavEnvClient.step()` 要求 numeric reward，并在 replay/RL 模式要求 `replay_key`。`submit_event()` 支持主动任务修改、HITL 回答和暂停/恢复，`resume_if_waiting` 只用于唤醒已经持久化 waiting state 的 graph。该路径用于 session rollout、DAgger 数据采集和后续 RL，而不是在 ms-swift 内复制环境。

---

## 13. 评测

### 13.1 System2 / tools

```text
action type accuracy
tool name/op accuracy
argument validity
tool success rate
unnecessary tool-call rate
route update / add-via / replan success
ask-user precision and resolution success
task-update interpretation success
event acceptance / duplicate / revision-conflict rate
```

### 13.2 System1

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

### 13.3 Episode level

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

tool calls、safety rejections 和 environment steps 作为 diagnostics，不自动提升为主分数。当前 event-driven diagnostics 还包括：

```text
external_events
action_preemptions
revision_conflicts
interaction_resolution_rate
stale_action_dispatches
stale_action_execution_rate
terminal_validation_requests
rejected_completion_requests
abort_requests
control_halts
```

其中生产闭环必须满足：

\[
stale\_action\_execution\_rate=0.
\]

---

## 14. 核心代码索引

### 14.1 RouteServer

```text
src/mcp_server/contracts/tool.py
  ToolCallV2 / ToolResponseV2

src/mcp_server/tools/specs.py
  Pydantic tool argument models and aliases

src/mcp_server/tools/registry.py
  ToolSpecRegistry and capability profiles

src/mcp_server/contracts/episode.py
  EpisodeEvent / EpisodeTraceV2

src/mcp_server/contracts/task.py
  ExternalUserEvent / TaskState / DialogueState / ControlState

src/mcp_server/runtime/interfaces.py
  System2Action / TerminalDecision / System1Output / environment protocols

src/mcp_server/runtime/system1_contract.py
  canonical cumulative FLU constants and conversion helpers

src/mcp_server/runtime/state.py
  checkpoint-safe EpisodeGraphState

src/mcp_server/runtime/events.py
  event store protocol and deterministic task reducer

src/mcp_server/runtime/execution.py
  revision-bound dispatch, preemption and idempotent terminal halt

src/mcp_server/runtime/terminal.py
  authoritative arrival verification and terminal decision validation

src/mcp_server/runtime/graph.py
  event-driven LangGraph closed-loop runtime

src/mcp_server/runtime/tool_adapters.py
  live and replay executor adapters

src/mcp_server/runtime/service.py
  reset/step/event HTTP contracts and runtime registration

src/mcp_server/api_http.py
  /v1/episodes reset/step/events endpoints

src/mcp_server/services/route_session_service.py
  route mutation CAS and idempotency boundary

src/mcp_server/eval/episode/
  frozen cases, environment, runner, metrics and service

src/mcp_server/eval/episode/zeroshot_samples.py
  registry-derived event-driven diagnostic samples

src/mcp_server/eval/exporters.py
  Stage2 v1/v2 row exporter

src/mcp_server/agent/finish_schema.py
  legacy/canonical finish compatibility

src/mcp_server/agent/stage2.py
  final assistant target and readout placement
```

### 14.2 ms-swift

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
  cumulative FLU trajectory -> rolling body-delta command adapter

swift/sidewalk_nav/data/agentic_episode_manifest.py
  EpisodeTraceV2 -> session SFT/DAgger rows

swift/sidewalk_nav/rollout/urbannav_env.py
  strict reset/step/event RouteServer client
```

---

## 15. 当前实现状态

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| typed System2/System1/runtime interfaces | 已实现 | Pydantic + Protocol |
| ToolSpecRegistry 和 capability conformance | 已实现 | live/replay profile |
| RouteSession map operations | 已实现 | 通过 map tool 暴露 |
| single open-loop Stage2 | 已实现 | live API teacher 和 ms-swift backend 路径 |
| LangGraph 多工具/HITL/System1/env 图 | 已实现框架 | 包含事件 gates、两阶段 HITL 和执行反馈 race |
| ExternalUserEvent 主动入口 | 已实现 | new/modify/pause/resume/cancel/response/emergency |
| Task/Dialogue/Control revision state | 已实现 | checkpoint-safe Pydantic contract |
| action freshness 与短时域抢占 | 已实现框架 | task/route/control/observation 四重校验 |
| complete/abort/cancel 终态合同 | 已实现 | 到达验证、结构化原因、幂等 halt 和双读迁移 |
| RouteSession mutation CAS/idempotency | 已实现，单进程 | expected revision + idempotency key；分布式 store 未完成 |
| EpisodeTraceV2 | 已实现 | event-sourced contract |
| v2 assistant-suffix readout | 已实现 | `[SYS1_i]` 位于 finish 后且 LM mask |
| isolated latent queries | 已实现 | 独立 query table |
| coordinate Goal Encoder | 已实现可选项 | 顶层 geometry，不进入 LM target |
| Direct Action 与 CFM 消融 | 已实现 | 相同 condition/target/output |
| AURA-lite anchor CFM | 已实现可选项 | fixed anchor + classifier + flow init |
| System1 checkpoint save/load/head inference | 已实现 | sidecar files |
| checkpoint -> EpisodeRuntime System1Policy adapter | 未统一 | graph Protocol 已有；生产 checkpoint service adapter 待实现 |
| per-decision session SFT/DAgger exporter | 部分实现 | causal rows 和 preempt mask 已实现；active row 尚需补齐 trajectory_policy |
| UrbanNavEnvClient | 已实现 | strict reset/step/event HTTP client |
| frozen episode benchmark | 已实现 | deterministic transitions、events、preemption 和 stale-action metrics |
| 自动多模态 input-mode resolver | 未完成 | 当前依赖显式 prompt/image roles |
| 生产级 Safety Validator | 未完成 | 只有 Protocol、graph node 和测试 fake |
| safety intervention execution | 已实现框架 | replacement 会成为 executable output；真实 validator 未完成 |
| System1 target 坐标/归一化统一 | 已实现 | v2 exporter 输出 metric cumulative FLU，checkpoint 固定 field/format |
| cumulative trajectory -> body-delta runtime 桥接 | 已实现 | adapter 使用 `R(-yaw_prev)` 显式转换 |
| 持久 external event store | 未完成 | 内置实现仅支持单进程；生产需事务队列 |
| live simulator/robot multi-tick rollout validation | 需要部署验证 | hard preempt、tracking 和真实 safety 尚未验证 |
| GRPO 生产训练 | 未完成 | 必须先满足 replay/reward readiness |
| 独立 async RGB memory branch | 未实现 | 当前使用 LVLM multimodal hidden memory |

---

## 16. 不可破坏的框架约束

1. `tool_response` 必须是上下文，不能计算 LM CE。
2. `[TRAJ_i]/[SYS1_i]` 必须 `labels=-100`，且 v2 readout 位于工具结果和 finish 之后。
3. `trajectory_ego_t/action_ego_t` 只能作为顶层 head supervision，不能写入 assistant target。
4. `<finish>` 只表示 assistant turn 结束；只有通过 ArrivalVerifier 的 `complete_task` 才能产生 `succeeded`。
5. `abort_task/cancel/complete_task` 禁止调用 CFM，必须由 execution layer 的幂等 `halt()` 停车。
6. 生产 checkpoint 固定 `trajectory_ego_t/ego_pose_xyt_flu_v1`；禁止自动回退到 `action_ego_t`，也禁止按 `[T,3]` 静默混用。
7. RouteSession、provider cache 和 safety diagnostics 属于外部 runtime state，不是 LVLM 必须预测的 belief。
8. live、replay、training exporter 必须共享同一 tool name/op/response contract。
9. DAgger 必须使用显式 teacher correction，不能把失败动作回写成监督。
10. RL 必须建立在可复现 transition、numeric reward、stable replay key 和 terminal trace 上。
11. Safety Validator 缺少观测证据时必须拒绝或停止，不能通过不透明 fallback 默认接受。
12. 所有闭环指标必须从环境事件计算，不能只用文本格式正确率替代 arrival、collision 和 recovery。
13. 每个可执行 System1 chunk 必须绑定 `task_revision/route_revision/control_epoch/observation_id/action_id`；stale chunk 不得 dispatch。
14. 外部事件必须经过幂等 event store 和纯 reducer；`pause/cancel/emergency_stop` 不得交给 LVLM 自由解释。
15. RouteSession mutation 必须使用 expected revision 和 idempotency context，模型不能直接修改内部 route object。
16. 新 session trace 必须按 `decision_id` 导出；`tick_id` 仅表示环境 transition，不能覆盖同 tick 多次决策。
17. active System1 trainable row 必须携带与 checkpoint 一致的 `trajectory_policy`，不能依赖 shape 或默认值推断。
