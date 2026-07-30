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

总体图右侧的 `terminal_control` 属于 runtime execution plane，不属于 System1。`complete_task` 只有在权威到达证据通过验证后才转换为成功；`abort_task`、外部 `cancel` 和 runtime failure 都不生成 `[SYS1_i]`，而是进入确定性 `halt()`。被拒绝的完成请求重新进入 System2，不能由 `<finish>` 标签直接结束 episode。

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

### 1.4 当前 canonical 基线

修改后的框架以以下版本组合为唯一生产主线；legacy 字段只允许在输入 adapter 中双读，不能由新 exporter 继续生成：

| 边界 | Canonical contract | Owner |
| --- | --- | --- |
| System2 决策 | `tool_call|ask_user|navigate|complete_task|abort_task` | RouteServer runtime + ms-swift action space v4 |
| assistant 回合 | `<finish>.decision_type`；只有 `navigate` 追加 `assistant_suffix_v2` readout | ms-swift prompt/exporter |
| System1 target | `trajectory_ego_t[T,3]`, `ego_pose_xyt_flu_v1` | RouteServer exporter + ms-swift trainer |
| 工具 | `ToolCallV2/ToolResponseV2` + `ToolSpecRegistry` | RouteServer |
| session trace | `urbannav_episode_trace_v2` | RouteServer runtime |
| session row | `urbannav_episode_per_decision_v3` | ms-swift exporter |
| 任务终态 | `complete_task -> ArrivalVerifier`; `abort_task -> failed`; external `cancel -> cancelled` | RouteServer `terminal_control` |

因此主数据流固定为：

```text
observation + task/dialogue/control revisions
-> one System2 decision
-> optional typed tool/HITL re-entry
-> navigate only: [SYS1_i] -> System1 -> safety -> short execution
-> environment feedback / external event
-> EpisodeTraceV2
-> per-decision session SFT/DAgger row
```

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

五类 System2 决策的执行语义固定如下：

| `decision_type` | 必要 payload | 下一节点 | 调用 System1 | 任务终态 |
| --- | --- | --- | --- | --- |
| `tool_call` | `ToolCallV2` | tool executor -> System2 | 否 | 否 |
| `ask_user` | `user_request` | `prepare_hitl -> await_hitl -> System2` | 否 | 否，进入 waiting |
| `navigate` | symbolic finish + `[SYS1_i]` | System1 -> safety -> dispatch | 是 | 否 |
| `complete_task` | `TerminalDecision(outcome=succeeded)` | `terminal_control` | 否 | 仅 ArrivalVerifier 接受后成功 |
| `abort_task` | `TerminalDecision(outcome=failed, reason_code=...)` | `terminal_control` | 否 | failed |

外部 `pause/emergency_stop/cancel` 不属于模型 action space：`pause/emergency_stop` 抢占当前 lease 后保持 waiting，`cancel` 进入 `cancelled` 并调用幂等 `halt()`。`route_action` 只描述路线操作，不能再次承载 ask/complete/abort 语义。

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

### 4.4 Legacy/辅助 FPV 文本轨迹

旧 Phase A 或独立视觉 grounding 数据曾在 assistant `<finish>` 中使用：

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

这是图像平面可视化/兼容监督，不是 metric ego control，也不是生产 V4 Full CoT
字段。新 exporter 不再写出它。

### 4.5 Stage2 v1 与 v2 finish

两种数据视图必须区分：

**v1 / read-only compatibility row**

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

### 5.3 Reasoning V4.6 单轮 Compact FullCoT

Reasoning V4.6 是 Stage-A 单轮 E2E 导航的最小语言推理合同。它与 runtime 的 agent
`<finish>`、tool loop 和任务终态合同分离。V4.6 不把可恢复但未经验证的 object graph、
route progress 或时序关系包装成训练真值。

```text
shared prompt-visible input
  ├─ deterministic Task canonicalizer
  ├─ independent Scene teacher
  ├─ independent LocalTraversability teacher
  ├─ independent NavigationContext teacher
  ├─ independent InteractionConstraint teacher
  └─ Decision teacher (visible input + five validated groundings)
                         |
                         v
           blind DecisionProposalV4
                         |
private strong direction + deterministic composer
                         |
                         v
       candidate target + immutable reconciliation audit
                         |
     derived candidate/review/formal/artifact state
                         |
                         v
            one compact six-section LM row
```

完整数据、teacher、audit 和 release 规则以
`docs/REASONING_V4_DATA_ANNOTATION_DESIGN.md` 为准。canonical owner 是 RouteServer；
ms-swift 只消费生成的 mirror、formal training view 和 loss payload。

#### 5.3.1 唯一 wire 与 section 职责

```text
<think>
<TaskGrounding>{...}</TaskGrounding>
<SceneGrounding>{...}</SceneGrounding>
<LocalTraversability>{...}</LocalTraversability>
<NavigationContextGrounding>{...}</NavigationContextGrounding>
<InteractionConstraintGrounding>{...}</InteractionConstraintGrounding>
<Decision>{...}</Decision>
</think>
```

V4.6 没有 `Finish`、readout、refs、route progress 或第七段。六段共享读取完整输入；
它们不是严格串行的隐变量链。Decision teacher 可读取五段已验证 grounding，但看不到
private strong label。动作仅在 `Decision.selected_action_label` 出现一次并使用 LM loss。
`stop` 只是短时 hold，不表示 arrival 或 task completion。

六段完整字段和职责如下：

| Section | 字段 | 定义与作用 | 来源 | Student 可推断 | Loss |
| --- | --- | --- | --- | --- | --- |
| TaskGrounding | `objective` | 规范化任务 code | versioned registry/cache | 是 | LM |
| TaskGrounding | `requirements[]` | instruction 明示要求 codes | registry/cache | 是 | LM |
| TaskGrounding | `ambiguities[]` | 会改变动作的任务歧义 codes | registry/cache | 是 | LM |
| SceneGrounding | `scene_type` | 粗可见场景 code | LVLM weak | 是 | LM |
| SceneGrounding | `walkable_support` | 可见步行支撑面 code | LVLM weak FPV | 是 | LM |
| SceneGrounding | `key_entities[]` | 影响局部导航的粗类别 | LVLM weak FPV | 是 | LM |
| SceneGrounding | `traffic_context` | 粗交通控制语境 | LVLM weak FPV | 是 | LM |
| LocalTraversability | `direction_affordances[]` | profile-owned 稀疏物理方向判断 | LVLM weak FPV | 是 | LM |
| NavigationContextGrounding | `task_direction` | 可见的短时移动方向或 fail-closed 状态 | LVLM weak instruction/BEV | 条件可见 | LM |
| InteractionConstraintGrounding | `constraints[]` | 当前高层 slow/yield/stop/avoid 限制 | LVLM weak | 是 | LM |
| Decision | `selected_action_label` | strong-owned 唯一短时 action | private deterministic label 经 composer | 是 | LM |
| Decision | `speed` | normal/slow/stop | constraint deterministic projection | 是 | LM |
| Decision | `response` | proceed/slow/yield/detour/stop | constraint deterministic projection | 是 | LM |

`ReasonCode` 是 lower-snake-case 受约束开放 code，不是固定的 instruction/object 类别
选择题。闭集只用于 wire、状态、profile action、affordance、constraint effect 和最终
行为 tuple。

#### 5.3.2 TaskGrounding 字段

Task canonicalizer 对 instruction 执行 NFKC、case-fold、标点/符号转空格、空白折叠和
SHA-256 fingerprint。已知 route-following 模板固定输出：

```json
{
  "objective": "follow_route",
  "requirements": ["route_adherence"],
  "ambiguities": []
}
```

未知 fingerprint 最多标注一次并进入同 run cache；同义模板不得逐样本重复调用 LVLM。
registry v2 将
`Navigate safely on the sidewalk and follow the visible route.` 确定性规范化为上述 tuple。
恢复 append-only cache 时，旧 annotation schema 或已经进入 registry 的旧 LVLM task
结果都视为 stale，不能覆盖当前 deterministic canonicalizer。

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `objective` | `ReasonCode` | 稳定任务 taxonomy，不读 runtime phase、距离或 waypoint state |
| `requirements` | `list[ReasonCode]` | 只保留 instruction 明示要求；唯一，最多 12 |
| `ambiguities` | `list[ReasonCode]` | 只保留会改变动作的歧义；唯一，最多 12 |

不存在 `goal` object、`local_objective`、teacher-only runtime state 或 GT phase。

#### 5.3.3 SceneGrounding 字段

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `scene_type` | `ReasonCode` | 开放粗场景，例如 `urban_sidewalk` |
| `walkable_support` | `ReasonCode` | 开放可见支撑面，例如 `paved_sidewalk` |
| `key_entities` | `list[ReasonCode]` | navigation-relevant 粗类别；唯一，最多 16 |
| `traffic_context` | `ReasonCode` | 归一化为 traffic_control_present/no_active_control_visible/unknown |

这些字段是 LVLM weak visual summary，不是 instance-level GT。crosswalk 是
`walkable_support`，不是 entity；当前过街行为影响由 Interaction 的
`crossing_interaction` 表达。V4.6 不输出 entity/control ID、位置、mask、bbox、metric
depth、tracking 或 confidence。
Scene postprocessor 将 car/sedan/parked vehicle 统一为 `vehicle`，scooter/motorcycle
统一为 `micromobility`，parked bicycle 统一为 `bicycle`，并折叠单复数。原始回答保留用于
审计，typed target 只使用归一化值。

#### 5.3.4 LocalTraversability 字段

`direction_affordances[]` 的每一项包含：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `direction_label` | `ReasonCode` | 必须来自 action profile 对应的 7-way probe；禁止 stop |
| `affordance` | direct/conditional/blocked/forbidden/unknown | 只描述 FPV 物理可执行性 |
| `clearance` | sufficient/narrow/insufficient/unknown | 粗视觉净空，不是 metric width |
| `limiting_factors` | physical factor list | 唯一，最多 12；非 direct 结论必须给原因 |

物理 factor 闭集为 static obstacle、dynamic occupancy、narrow clearance、non-walkable
surface、curb、wall、vegetation、occlusion、unknown。交通规则不能作为本段 factor。
direct 不能同时声明 insufficient clearance。BEV 不得证明本段的 traversability、
clearance 或 collision safety。

当前 profile pairing：

| Action profile | Probe profile | Probe labels |
| --- | --- | --- |
| `fpv_degree10_terminal_centroid_v2` | `fpv_degree10_affordance7_v1` | left_90/30/10, straight, right_10/30/90 |
| `ego_degree10_12way_terminal_centroid_v2` | `ego_12way_affordance7_v1` | left_50plus/30/10, straight, right_10/30/50plus |

#### 5.3.5 NavigationContextGrounding 字段

```text
NavigationContextGrounding
  task_direction
    status
    optional direction
    source
```

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `status` | aligned/conflict/not_visible/insufficient_evidence | 独立表示方向证据状态 |
| `direction` | nullable `ReasonCode` | 只在 aligned 时存在，必须为 profile moving label，禁止 stop |
| `source` | bev_route/instruction/none | aligned 记录真实可见方向来源；其他状态固定 none |

generic route-following instruction 不能单独推出左右方向。只有明示方向 instruction，或
朝向清楚且 route overlay 可见、具有 `visual_conditional` direction authority 的 BEV，
可以支持 aligned direction。BEV 可以支持 route direction 和 upcoming topology；缺少
local safety authority 只阻止它证明 traversability、clearance 和 collision safety。
NavigationContext teacher 单独使用 `image_detail=high`，避免 BEV route overlay 在
provider 侧压缩后消失。aligned 必须携带 `source=bev_route|instruction`；其余状态必须为
`direction=null, source=none`。V4.6 不输出 goal/source alignments、route progress、
remaining distance 或 arrival。

#### 5.3.6 InteractionConstraintGrounding 字段

`constraints[]` 每一项包含：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `kind` | `ReasonCode` | 开放粗约束类别 |
| `status` | applicable/uncertain | 当前适用性 |
| `effect` | slow/yield/stop/avoid | 最小行为影响 |
| `affected_direction_labels` | action label list | 只有 avoid 可使用非空子集 |

`constraints=[]` 是正常负判断。slow/yield/stop 必须为空；avoid 必须明确非空受影响
子集。该子集的合同名称仍为 `affected_direction_labels`，但语义冻结为
directions-requiring-avoidance，不是 forbidden directions：命中时要求 slow/detour，
未命中时不施加 avoid tuple，stop 安全满足。kind postprocessor 折叠为
crossing/social/local-obstacle/traffic-control 等粗类，特别是
`traffic_signal -> traffic_control`。本段没有 object ID、source state、temporal
relation、tracker、ego-motion、TTC 或 normative-policy GT。

#### 5.3.7 Decision、strong ownership 与 tuple

`DecisionProposalV4` 是 blind LVLM proposal，字段与 final Decision 相同，但只作 audit：

```text
selected_action_label
speed
response
```

composer 用 frozen `DirectionSupervisionV4.label_name` 覆盖 proposal 的 action，不允许
teacher 覆盖 strong direction。行为 tuple 按 applicable effects 确定性映射：

| 条件 | `speed` | `response` |
| --- | --- | --- |
| 无 effect | normal | proceed |
| slow | slow | slow |
| yield | slow | yield |
| avoid | slow | detour |
| strong action = stop | stop | stop |

strong moving 与 applicable global stop 冲突时 candidate structure invalid 并进入 review；
不能把 moving label 静默改写为 stop。moving action 禁止 stop speed/response；stop
必须严格 stop/stop。V4.6 不输出 decision_type、motion、social_action、fallback、
uncertainty 或 support/evidence refs。

#### 5.3.8 Source 与 private supervision

`NavObservationV4` 是 teacher/student 共享输入：

| 字段 | 定义与作用 |
| --- | --- |
| `source_group_id` | source/annotation/view join ID |
| `dataset_source` | 数据集族与覆盖统计 |
| `source_record_id` | 原始样本回溯 |
| `split_group_id` | 同源防泄漏 split |
| `images/image_roles` | 有序多模态输入；恰有一个 `fpv.current` |
| `instruction/travel_mode` | 真实任务文本和步行辅助语境 |
| `model_visible_context` | 额外 prompt-visible 非私有上下文 |
| `evidence_catalog` | 输入地址簿，不是输出 ref graph |
| `input_modalities` | 实际模态 availability |
| `direction_context` | public action/probe profile identity |
| `temporal_evidence_eligible` | 仅共享 VQA V2 兼容；FullCoT 不消费 |

BEV observation 中保留的 route-progress presentation metadata 也仅用于共享 VQA source
兼容。Reasoning V4.6 不读取 route progress、terminal 或 local-safety truth。共享 source
envelope 仍使用 `urbannav_reasoning_source_v4_5` 是有意的；本轮输入结构未变，不能据此
复用旧 V4.5 annotation/cache。

`ReasoningSourceV4` 额外保存 private sidecars：

| 字段 | 定义与作用 |
| --- | --- |
| `direction_supervision` | 完整、已有、版本化的 deterministic direction stamp |
| `trajectory_ego_t/trajectory_policy/action_mask` | 可选既有 System1 supervision |
| `warnings` | source 构造诊断 |
| `source_fingerprint` | 除 fingerprint 外整个 source 的 SHA-256 |

方向 stamp 必须同时包含 profile ID/version、index、name、choice、angle、source、
bin mode 和 derivation version。缺字段、identity 不一致、profile 不在 allowlist 或
source 使用 route theta 时 fail closed。V4.6 不从 row trajectory、BEV、GPS、route、
goal 或 LVLM 回答重新派生 direction。

#### 5.3.9 Annotation、audit 与 append-only resolution

`ReasoningAnnotationV4` 的字段：

| 字段 | 定义与作用 |
| --- | --- |
| `schema_version/source_group_id/source_fingerprint` | 绑定 source |
| `record_status` | complete/error，只表示 bounded pipeline 是否完成 |
| `grounding_outputs` | 五个可独立存在的 typed grounding |
| `decision_proposal` | blind LVLM proposal |
| `action_reconciliation` | immutable proposal/strong/grounding audit |
| `composer_candidate/candidate_target` | 未经隐式改写的 strong-owned candidate |
| `candidate_structure_record` | valid/invalid/not_available 及 composer 诊断 |
| `stage_records` | 六段状态与 producer provenance |
| `errors/warnings` | record-level 诊断 |
| `teacher_provider/teacher_model/metadata` | run provenance |

每个 stage record 显式区分 `producer_kind=deterministic|lvlm`，并记录
`producer_profile_id/derivation_version/prompt_policy_digest`。Task registry 结果不伪装成
LVLM label；LVLM stage 必须绑定 prompt digest。

Action Reconciliation 不是第七段。audit 只保存：

```text
audit_id
source_fingerprint
proposal_digest
groundings_digest
comparison
probe_projections
four deterministic checks
review_trigger_codes
```

全部等距离最近 probe 都必须保存；例如 `right_20` 同时对应 `right_10/right_30`，不能
按列表顺序任取一个。聚合结果为
`all_supported/all_unsupported/mixed/unknown/not_applicable`，
`mixed/all_unsupported/unknown` 进入 review。
action profile 的 10° compatibility tolerance 不等于人工阈值：relation 仍可在角差
超过 10° 时记录 `large_delta`，但纯角度 review 只在移动 action 角差严格 `>20°` 时触发。
sign、stop/move、probe、constraint 和 tuple hard trigger 不受该阈值影响。

它不复制 profile、proposal、strong label 或 candidate，不持久化 review priority/status。
expanded review queue 通过 source/annotation join 生成可读对照。

人工裁决写入 append-only `ActionReviewResolutionV4`，包含 audit ID、resolved/rejected、
resolved action、reason code、reviewer、timezone-aware timestamp 和 note。resolution
不能偷偷修正 source GT；若 resolved action 不等于 frozen strong label，release gate 要求
修正上游 direction、生成新 source fingerprint 并重新标注。

#### 5.3.10 独立状态轴与 formal release

状态必须独立：

| 轴 | 值 |
| --- | --- |
| record | complete/error |
| section | accepted/rejected/error/skipped |
| candidate structure | valid/invalid/not_available |
| review | not_required/pending_human/resolved/rejected |
| formal eligibility | true/false |
| artifact status | preview_only/training_ready |
| artifact eligibility | true/false |

正式 training eligible 当且仅当 record complete、六段全部 accepted、candidate structure
valid、
audit 存在且可重建、review 为 not_required/resolved、resolved action 仍等于 frozen
strong，并且 candidate 对 source/profile 再验证通过。

`review_preview` 固定
`artifact_status=preview_only, artifact_training_eligible=false`，但可展示独立的
`formal_training_eligible`；它仍不能训练。`formal_training` 必须同时为
`artifact_status=training_ready`、artifact/formal eligibility true。ms-swift 在 Pydantic
defaults 之前检查这些原始字段以及 schema、canonical digest、audit ID 和 review，防止
preview 或 stale row 混入训练。

#### 5.3.11 Training view 与 LM loss

`ReasoningTrainingViewV4` 完整字段组：

| 字段组 | 字段 | 作用 |
| --- | --- | --- |
| identity | schema/digest/sample/source/split/dataset | contract、join、防泄漏 |
| release | release track/audit/review/artifact status/artifact eligibility/formal eligibility | formal admission |
| visible input | images/roles/instruction/context/catalog/modalities | 构造多模态 prompt |
| public vocabulary | direction context | action/probe allowlist |
| language target | assistant text + typed target | exact six-section LM target |
| binding | renderer ID + prompt profile/digest | 防 renderer/prompt 漂移 |
| private supervision | direction stamp | strong action identity校验 |
| optional System1 | trajectory/policy/action mask；ms-swift 派生 query contract | 既有 trajectory loss 与 16-slot runtime binding |
| metadata | audit identity与非语义 provenance | 审计，不进入 assistant |

assistant message 的 `loss=true`，`reasoning_v4_6_sections` 对六个 tag 全部启用 LM loss。
`selected_action_label` 只在 Decision 中训练；没有独立 direction head，也没有 Finish 的
重复 action target。可选 System1 trajectory 继续走既有 trajectory loss。此类 row 必须
同时携带 16 步 `trajectory_ego_t`、16 位 `action_mask`，并在最后一个 user turn 放置
`[TRAJ_0]...[TRAJ_15]`；assistant 不得包含 query，query LM label 为 `-100`。query count
、target horizon、action-mask length 和有效 mask 和都必须为 16。V4.6 的
`system1_query_contract.query_target_alignment=query_i_to_target_i`，因此第 i 个 query
hidden state 直接作为第 i 个 target step 的局部 condition；expanded encode 找不到完整
query 或 Q/T 不一致时 fail closed。Q 与 T 不同的历史 row 只允许显式 legacy pooled
compatibility。

固定审计指标包括 Sources、Record Complete、All Sections Accepted、Candidate
Structurally Valid、Pending Human、Formal Training Eligible、Artifact Training
Eligible，以及 task canonicalization drift、equal-nearest probe loss、preview leak。
正式批量标注前先运行 64-row diverse smoke，生成 self-contained HTML；full runner
同时要求上述机器 gate 全通过和人工显式批准。
### 5.4 Navigation VQA V2 原子监督链

VQA V2 是与 Reasoning V4 并列的 answer-only LM contract，不是 Full CoT 的 section
projection，也不是在线 Agent node。完整 27-task 定义、来源与 capability 状态见
`docs/VQA_V2_DATA_ANNOTATION_DESIGN.md`。

架构上的核心 partition 是：

```text
VQASourceV2
  observation             student-visible NavObservationV4
  direction_supervision   private existing-trajectory derived label
  interfaces              private future external-GT bindings
  source_fingerprint      complete source identity

VQAAnnotationV2
  question                answer-independent typed question
  answer                  one task-discriminated minimal payload
  provenance              source/producer/prompt/evidence digests

VQATrainingViewV2
  visible observation + question + minimal assistant_text
  no direction_supervision/interfaces/trajectory/Full-CoT/dense target
```

`VQAQuestionV2` 的字段职责为：

| 字段 | 作用 |
| --- | --- |
| `task_id/template_id/question_text` | 冻结原子科研问题及模板版本 |
| `direction_profile_id` | 标识本题使用哪套方向合同 |
| `conditioned_direction_label` | B/D 判断题明确要求检查的方向；禁止复制到 answer |
| `candidate_direction_labels` | selection/comparison 题的唯一合法方向集合 |
| `target_refs/evidence_refs` | 限定 prompt-visible target 与 raw evidence 闭包 |
| `candidate_set_id/visible_candidates/target_candidate_id` | 未来 E1–E3 的 candidate ID、student-visible 表示和指定 target；当前任务关闭 |

student prompt 只渲染 `question.evidence_refs` 命中的 evidence catalog 项，并显式列出
`target_refs`。这与 weak teacher 的可见 evidence 闭包一致；C4 将来启用时，
`bev.*`/`fpv.*` 两个 cross-view ID 也必须先成为真实 prompt-visible evidence。

`VQAAnnotationV2` 的 `label_source` 严格区分 `derived_gt/lvlm_weak/external_gt`。
`source_field_refs` 对 derived GT 必填；weak label 必须携带 teacher model、prompt profile 和
policy digest。只保存 accepted annotation，unavailable/rejected/error 只进入 audit report，
不能生成空 target。

当前 capability profile 打开 18 个 task：

- A1–A3、B1–B6、C1、D1–D6、E4：仅用 student-visible 输入的 LVLM weak label；
- E0：按共享 direction profile 从数据集中已有的 trajectory GT 确定性派生；
- C2 pixel/waypoint、C3–C6、E1–E3：contract/source interface ready，但 producer、
  question generation、training export 和 evaluation 全部关闭。

关闭任务的 interface 只存在于 private source schema；不进入 training view 或 ms-swift
manifest。C3 禁止 GPS、route matching 和 route theta；C4 当前不假定 cross-view entity
ID，未来接口只接受已经 prompt-visible 的 `bev.*`/`fpv.*` ID；C5 不把
current/future direction 混成一个 label；E1–E3 必须等待真正
prompt-visible candidate set 和对应 GT。

ms-swift 最终构造：

```text
system: canonical positive VQA policy
user: actual multimodal input + one atomic question
assistant: <VQAAnswer>{minimal payload}</VQAAnswer>
```

整个 assistant target 使用普通 LM loss。E0 direction 不走独立 head；VQA row 的
System1、trajectory、segmentation loss 均关闭。SegHead、SAM/SoM 与 mIoU 只保留显式
unavailable 状态，不能包装成当前能力。

### 5.5 Legacy V2 导出边界

V2 exporter 只保留 canonical Full CoT/System1 view。旧的 Map/Social section-QA
registry、投影器和弱 teacher gate 已删除；后续 VQA 使用独立 typed
question/rationale/answer 合同，不能从 V2 或 V4 Full CoT 静默拆段。

```text
ReasoningTraceV2 + finish + optional trajectory_ego_t
  -> Full CoT/System1 row
```

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

#### 8.1.1 终态控制的权威边界

需要严格区分以下五个名称相近但职责不同的对象：

| 名称 | 所属层 | 语义 |
| --- | --- | --- |
| assistant `<finish>` | 文本协议 | 结束当前 assistant turn，不结束任务 |
| `decision_type=complete_task` | System2 | 请求完成，仍需验证 |
| `terminal_control` | runtime | 验证终态、调用 `halt()`、确定 episode status |
| graph `finish` node | runtime | 只消费已确定的 status/reason 并写入 TaskState/trace |
| `navigate + stop_hold` | System1 | 临时保持轨迹，不是终态 |

默认 `RuntimeTerminalValidator` 只接受 runtime 中值严格为 `true` 的权威字段：

```text
observation_info.arrival_success|goal_reached|route_completed
environment_result.info.arrival_success|goal_reached|route_completed
successful ToolResponse.result.arrival_success|goal_reached|route_completed
```

模型给出的 `TerminalDecision.evidence_refs` 只进入审计 metadata。`complete_task` 验证失败时写入 `terminal_validation(accepted=false)`，清空待提交终态并重新进入 System2；超过 `max_terminal_rejections` 后以 `terminal_validation_retry_exhausted` 失败。环境自身返回 `done=true` 时可以直接提供 terminal status，但仍必须经过同一个 `terminal_control -> halt -> finish` 路径。

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

`ask_user` 不调用 CFM。当前 runtime 会把 `motion_policy_while_waiting` 持久化到 `PendingInteraction`；真实 robot adapter 必须把 `stop/slow_down/handover_to_safety_controller` 映射为确定性的控制器行为。该硬件等待策略尚不能由 LangGraph checkpoint 本身证明，不能用模型生成一条未验证的零轨迹替代。

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
  episode_started / system_message

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

terminal control:
  terminal_validation / control_halted / episode_finished
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
  -> complete_task -> terminal_validation
       accepted -> control_halted -> episode_finished(succeeded)
       rejected -> next model_output
  -> abort_task -> terminal_validation -> control_halted -> episode_finished(failed)
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

完整 navigate row 才可同时启用 LM 与 System1 loss；`ask_user/complete_task/abort_task`
不含 readout，也不启用 System1 loss。

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
  "image_roles": ["fpv", "bev"],
  "available_modalities": ["instruction", "fpv", "bev", "route"],
  "source_group_id": "episode-001:decision:3",
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
    "action_space_contract": "sidewalk_action_space_v4",
    "system1_readout_placement": "assistant_suffix_v2",
    "system1_readout_lm_label": -100,
    "environment_observation_lm_label": -100,
    "external_event_lm_label": -100,
    "runtime_state_lm_label": -100,
    "terminal_validation_lm_label": -100,
    "tool_response_lm_label": -100,
    "reasoning_schema_version": "urbannav_reasoning_trace_v2"
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

### 10.2 同源多任务视图

V4.6 每个 eligible `source_group_id` 最多生成一个 compact FullCoT row：

| Row view | assistant target | LM | System1 | seg/pixel traj |
| --- | --- | --- | --- | --- |
| V4.6 FullCoT | 精确六段、无 Finish/readout | 六段全开，动作只在 Decision | optional existing 16-step `trajectory_ego_t/action_mask` + 16 prompt queries | 否 |

独立 VQA 生成器也不得依据图片数量猜模态。`images` 保留原始有序列表，
`image_roles` 显式区分 `fpv_history/fpv/bev/goal_image`；否则多帧 FPV 会被误当成
FPV+BEV 并产生错误的 CrossModalAlignment 标签。

### 10.3 联合损失

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

### 10.4 训练课程

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

### 10.5 Multi-tick / per-decision exporter

对新 event-driven trace，ms-swift exporter 为每个 System2 decision 构造历史前缀：

\[
row_j=Export(e_{0:end(decision_j)}).
\]

同一 tick 的 `tool_call -> tool_response -> next model_output` 会产生多个 row，分别学习工具选择和工具结果后的新决策。可通过 `window_ticks` 截断物理 tick 历史，但不能打乱保留窗口内 event 顺序：

\[
row_j=Export(e_{tick\ge t_j-W+1:\,end(decision_j)}).
\]

环境 observation、external event、runtime state、terminal validation 和 tool response 保持 context role，只有 assistant/tool-call action 是 LM target。被拒绝的完成请求作为下一轮 System2 的 `<TerminalValidation>` user context，label 固定为 `-100`，不能训练模型复述 validator。System1 target 从当前 `decision_id` 的 `system1_output` 读取，并且必须存在 `action_dispatched` 且不存在 `action_preempted`。DAgger correction 的主键为：

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
| `status` | `active|paused|cancelled|completed|failed` | 任务生命周期 |
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

assistant `FinishV2` 公共字段为：

```text
schema_version = urbannav_finish_v2
decision_type
route_action optional
tool_used optional
```

按决策分区：

```text
navigate:
  motion / speed / social_action / confidence_level / trajectory_type
ask_user:
  user_request.question / choices / motion_policy_while_waiting
complete_task | abort_task:
  reason_code / reason / evidence_refs / retryable
```

不同分区字段不能混用。`route_action` 只表示路线操作，不承担任务终态；
`decision_type` 才决定 runtime 控制流。旧 V1 的 `intent`、
`trajectory_validity`、`map_context_available` 和 `trajectory_grid1000` 仅由
adapter 读取。

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
image_roles             # 与 images 一一对应，禁止按数量猜 BEV
available_modalities    # evidence capability gate
source_group_id         # Full CoT/System1 同源关联
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
reasoning_schema_version = urbannav_reasoning_trace_v4_5 | urbannav_reasoning_trace_v2 | legacy_v1
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
| ReasoningTraceV4 Full CoT | derived row | 是 | compact 六段 + 唯一 action location；readout mask=false | navigate row | schema/grounding |
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

### 11.11 Reasoning V4.6 离线标注链

当前主线由 `ReasoningSourceV4 -> deterministic Task + four independent LVLM
groundings -> blind DecisionProposal -> immutable Action Reconciliation audit ->
strong-owned composer candidate/CandidateStructureRecord -> ReasoningAnnotationV4 ->
append-only resolution -> preview/formal ReasoningTrainingViewV4` 构成。source、label、
review projection 和 view 分别通过
`build_reasoning_v4_sources.py`、`label_reasoning_v4.py`、
`export_reasoning_v4_reconciliation_review.py` 和 `export_reasoning_v4_views.py`
处理。preview artifact 永远不可训练；只有 artifact/formal eligibility 全部通过的
formal view 才能进入 ms-swift。完整合同和数据集 trust policy 见
`docs/REASONING_V4_DATA_ANNOTATION_DESIGN.md`。

#### 11.11.1 Legacy Evidence-conditioned Reasoning V2

既有 V2 Phase-A/E2E 标注使用三个不可混用的 artifact：

```text
existing FPV/BEV/instruction/route assets
  -> ReasoningSourceV2          # 冻结 teacher/model 可见输入
  -> ReasoningAnnotationV2      # 原始回答、修复历史、验证结果
  -> ReasoningTrainingViewV2    # Full CoT/System1 only
  -> ms-swift manifest row      # messages + navigate-only System1 sidecar
```

`ReasoningSourceV2` 是唯一 teacher 输入合同。核心字段为：

| 字段组 | 字段 | 语义 |
| --- | --- | --- |
| identity | `source_group_id/dataset_source/source_record_id/split_group_id` | 同源视图关联和防泄漏 split |
| vision | `images/image_roles/available_modalities` | 显式 FPV history/current、BEV current、goal image 顺序 |
| task | `instruction/travel_mode/model_visible_context` | 当前指令以及已冻结 goal/route/waypoint/tool evidence |
| grounding | `evidence_catalog` | teacher 可引用的完整地址空间 |
| optional condition | `goal_geometry/route_points_ego_t` | Goal Encoder 或路线条件，不进入 assistant target |
| System1 sidecar | `trajectory_ego_t/trajectory_policy/action_mask/system1_enabled` | 仅训练连续 head，teacher 不可见 |
| audit | `reasoning_qa_label_validity/teacher_private_context/metadata/source_fingerprint` | 冻结旧 source 的兼容审计字段、私有审计和不可变绑定 |

route/waypoint 只有在来源可验证时才进入模型：显式
`independent_route_evidence=true`，或 provenance 为 map provider、route session、tool
response、frozen route、人工标注。仅出现 `route_context`、`route_points_ego_t` 或
`intent_gt` 不足以证明独立性；无来源字段会被删除并记录
`metadata.unverified_route_context_dropped=true`。由 future trajectory、demonstration、
ground truth 构造的 pseudo goal 默认也被隔离。

teacher 只接收 image、instruction、`model_visible_context` 和
`evidence_catalog`。以下字段不进入请求：

```text
trajectory_ego_t / action_mask / segmentation / pixel trajectory
teacher_private_context / future GT / private teacher judgment
```

teacher 返回 JSON 形式的 `ReasoningTargetV2`，随后经过强 schema、引用可解析、
constraint-to-path applicability、finish 合法性校验。当前 E2E 标注阶段固定
`decision_type=navigate`，且没有实际 tool call，因此 `tool_evidence_refs=[]`。
非法回答使用 validator error 原因做有限 repair；没有 rule/template fallback。

`reasoning_qa_label_validity` 只为已冻结 V2 source/fingerprint 保留，不再触发 QA
projection 或训练 view。新 VQA 不复用该字段。

实现入口：

```text
contracts/reasoning_dataset.py       # 三个 Pydantic artifact
data/reasoning_source.py             # legacy assets -> frozen source
teacher/reasoning_v2_teacher.py      # VLM call, repair, resume
data/reasoning_views.py              # accepted target -> Full CoT/System1
tools/model_processing/
  build_reasoning_v2_sources.py
  label_reasoning_v2.py
  export_reasoning_v2_views.py
```

服务器 pilot 命令：

```bash
cd Agent_part/RouteServer

PYTHONPATH=src python tools/model_processing/build_reasoning_v2_sources.py \
  --input-jsonl /data/input.jsonl \
  --output-jsonl /data/reasoning_source_v2.jsonl \
  --report-json /data/reasoning_source_v2.report.json \
  --dataset-source sidewalk \
  --limit 10

# Provider/model/base URL/API key 只通过环境变量注入；凭据不得写入 artifact。
PYTHONPATH=src python tools/model_processing/label_reasoning_v2.py \
  --source-jsonl /data/reasoning_source_v2.jsonl \
  --annotation-jsonl /data/reasoning_annotation_v2.jsonl \
  --report-json /data/reasoning_annotation_v2.report.json \
  --concurrency 2 \
  --max-retries 2

PYTHONPATH=src python tools/model_processing/export_reasoning_v2_views.py \
  --source-jsonl /data/reasoning_source_v2.jsonl \
  --annotation-jsonl /data/reasoning_annotation_v2.jsonl \
  --output-jsonl /data/reasoning_training_view_v2.jsonl \
  --report-json /data/reasoning_training_view_v2.report.json
```

这条链只训练“已有 evidence 条件下如何做 grounded navigate 决策”。tool policy、
ask-user、route update、complete/abort 和 multi-tick recovery 必须在后续 Stage2/session
trace 中监督，不能由本阶段 teacher 伪造。

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
-> terminal requests pass terminal_control and deterministic halt
-> EpisodeTraceV2
```

适合部署和冻结 benchmark，`System2Policy/System1Policy/SafetyValidator/TerminalValidator/EpisodeEnvironment` 通过 Protocol 注入。外部事件通过 `EpisodeRuntime.submit_event()` 独立进入，不能伪装成下一次普通 `Env.step`。任务成功只能来自 `complete_task + accepted terminal_validation` 或环境提供的权威 arrival terminal；graph `finish` 不允许缺省生成成功状态。

### 12.3 ms-swift-owned rollout scheduler

`UrbanNavEnvClient` 通过严格 HTTP 合同连接 RouteServer：

```text
POST /v1/episodes/reset
POST /v1/episodes/{episode_id}/step
POST /v1/episodes/{episode_id}/events
```

`UrbanNavEnvClient.step()` 要求 numeric reward，并在 replay/RL 模式要求 `replay_key`。`submit_event()` 支持主动任务修改、HITL 回答和暂停/恢复，`resume_if_waiting` 只用于唤醒已经持久化 waiting state 的 graph。该路径用于 session rollout、DAgger 数据采集和后续 RL，而不是在 ms-swift 内复制环境。

### 12.4 两层闭环的关系

代码校阅时应先区分两个不同时间尺度的循环：

```text
Agent 大闭环（低频、任务级）
  observation / external event
  -> System2 decision
  -> tool | ask_user | navigate | complete_task | abort_task
  -> task/route/dialogue/control state update
  -> next System2 decision or terminal

E2E 导航小闭环（高频、运动级）
  navigate decision + current multimodal observation
  -> LVLM latent readout
  -> Direct/CFM System1
  -> canonical short trajectory
  -> safety
  -> execute short prefix
  -> controller/environment feedback
```

二者不是两个并列 agent。小闭环是大闭环中 `navigate` 分支的一次短时执行：

\[
\text{AgentLoop}
=\text{System2}\Big(
\text{Tool/HITL/Terminal},
\underbrace{\text{System1}\rightarrow\text{Safety}\rightarrow\text{Execute}}_{\text{仅 navigate 时进入}}
\Big).
\]

因此代码责任固定为：

| 问题 | 权威模块 | 不应由谁隐式处理 |
| --- | --- | --- |
| 当前任务、路线和对话是否需要修改 | RouteServer System2/runtime | CFM head |
| 是否调用工具、询问用户或请求终止 | `System2Policy` | controller |
| 当前 `navigate` 应生成什么短轨迹 | ms-swift LVLM + System1 head | route tool |
| 轨迹是否可执行 | `SafetyValidator` | LM 文本声明 |
| 执行多少步、何时抢占 | execution manager/controller | CFM sampler |
| 是否真正到达 | `TerminalValidator` / environment | `<finish>` 标签 |
| episode 如何回流训练 | `EpisodeTraceV2` exporter | 在线 graph 内临时拼 row |

### 12.5 Agent 大闭环：代码逐步对应

#### 12.5.1 入口和状态所有权

大闭环入口是：

```text
Agent_part/RouteServer/src/mcp_server/runtime/graph.py
  build_episode_graph()
  EpisodeRuntime.run()
  EpisodeRuntime.resume()
  EpisodeRuntime.submit_event()
  EpisodeRuntime.resume_with_event()
```

`EpisodeRuntime.run()` 建立一个 episode 的初始 `TaskState`、`DialogueState`、`ControlState` 和 `EpisodeTraceV2`；`build_episode_graph()` 只编排节点，不在节点外维护第二套隐式 session。外部用户事件由 `ExternalEventStore` 持有，并通过 `runtime/events.py::reduce_external_event()` 以纯状态变换更新任务、对话和控制状态。

每次决策所需的最小权威状态为：

```text
episode_id / tick_id
observation / observation_id / observation_info
task_state.task_revision / task_state.route_revision
dialogue_state / control_state.control_epoch
decision_id / active action_id
last_tool_response / latest_external_event
episode_trace
```

不能额外创建一个与这些 revision 无关的 `route_context` 或 model-private session 作为执行真值。

#### 12.5.2 LangGraph 节点和事件

当前真实节点链可按下表逐项核对：

| 阶段 | 代码入口 | 主要输入 | 主要输出 | 关键 trace 事件 | 后继 |
| --- | --- | --- | --- | --- | --- |
| 初始化观测 | `graph.py::observe()` | `initial_context` 或上一 tick observation | `observation_id`、replay info | `OBSERVATION_RECEIVED` | external event gate |
| 外部事件闸门 | `graph.py::event_gate()` | pending event + 三类 state | revisioned state、event route | `EXTERNAL_EVENT_RECEIVED`、`TASK_STATE_UPDATED`、`REVISION_CONFLICT` 等 | System2 / wait / terminal / 原节点 |
| 慢策略决策 | `graph.py::system2()` | 当前完整 graph state | typed `System2Action`、`decision_id` | `MODEL_OUTPUT` | tool / HITL / System1 / terminal |
| 工具执行 | `graph.py::execute_tool()` | `ToolCallV2` + task/route revision | `ToolResponseV2`、state delta | `TOOL_CALL`、`TOOL_RESPONSE`、`ROUTE_STATE_DELTA` | 回到 System2 |
| 等待用户 | `prepare_interaction()`、`await_interaction()` | `user_request` | persisted `PendingInteraction` | `INTERACTION_REQUESTED`、`USER_INTERRUPT` | interrupt/resume 后回 System2 |
| 轨迹生成 | `graph.py::system1()` | `navigate` decision + observation/state | bound `System1Output` | `SYSTEM1_OUTPUT` | safety |
| 安全验证 | `graph.py::safety()` | canonical trajectory | accept / replacement / reject | `SAFETY_VALIDATION`、可选 `SAFETY_INTERVENTION` | dispatch / System2 / terminal |
| 下发动作 | `graph.py::dispatch_action()` | fresh executable output | `ExecutionLease` | `ACTION_DISPATCHED` | await execution |
| 执行竞争 | `graph.py::await_execution()` | lease + event store | feedback 或 preemption | `CONTROLLER_FEEDBACK`、`ENVIRONMENT_STEP` 或 `ACTION_PREEMPTED` | observe / System2 / terminal |
| 终态控制 | `graph.py::terminal_control()` | terminal request 或环境终态 | accepted terminal status + halt ack | `TERMINAL_VALIDATION`、`CONTROL_HALTED` | System2 或 finish |
| 最终提交 | `graph.py::finish()` | 已验证 status/reason + stopped control | final task/trace | `EPISODE_FINISHED` | END |

图边在同一文件的 `StateGraph` 注册段集中声明。代码校阅时，不应只看某个 node 的实现，还要核对 conditional edge 是否把拒绝、等待和抢占送回正确分支。

#### 12.5.3 一次完整 episode 的伪代码

下面的伪代码与当前 graph 的控制语义对应，不是另起一套实现：

```python
state = initialize_episode()
state.observation = await environment.reset(state.initial_context)

while not state.done:
    # 每个高延迟边界前先消费外部事件；任务修改可以抢占当前动作。
    state = await ingest_external_event_if_any(state)
    if state.paused:
        state = await interrupt_and_resume(state)
        continue
    if state.cancelled:
        break

    decision = await system2_policy.decide(state)

    if decision.action_type == "tool_call":
        state.last_tool_response = await tool_executor.execute(
            bind_task_route_revision(decision.tool_call, state)
        )
        continue

    if decision.action_type == "ask_user":
        persist_pending_interaction(state, decision.user_request)
        await interrupt_and_enqueue_correlated_response(state)
        continue

    if decision.action_type == "navigate":
        chunk = await system1_policy.predict(state)
        chunk = bind_execution_context(chunk, state)
        verdict = await safety_validator.validate(chunk, state)
        if not verdict.accepted and not verdict.intervention:
            continue
        lease = await execution_manager.dispatch(verdict.output, state)
        feedback_or_event = await race_controller_and_external_event(lease)
        if feedback_or_event.is_event:
            await execution_manager.preempt(state.episode_id)
            continue
        state.observation = feedback_or_event.result.observation
        state.tick_id += 1
        continue

    # complete_task / abort_task 都先验证并确定性停车。
    terminal = await terminal_validator.validate(decision, state)
    if not terminal.accepted:
        state.terminal_validation = terminal
        continue
    await execution_manager.halt(state.episode_id, terminal.reason_code)
    state = finalize_episode(state, terminal)
```

#### 12.5.4 大闭环的六条不变量

1. **只有 `navigate` 调用 System1。** `tool_call`、`ask_user`、`complete_task`、`abort_task` 不生成 `[SYS1_i]` chunk。
2. **工具结果必须回到 System2。** 工具不能直接拼接一条 controller action 绕过模型的最终决策。
3. **`tick_id` 只在真实 `ENVIRONMENT_STEP` 完成后递增。** tool turn、HITL turn 和被抢占的 chunk 都不是新环境 tick。
4. **所有慢边界前检查外部事件。** 任务修改、暂停和取消不能等待完整 action chunk 执行结束。
5. **System1 output 必须通过 revision freshness。** `runtime/execution.py::bind_system1_execution_context()` 绑定任务、路线、控制代、观测和 action id；dispatch 前由 execution manager 拒绝 stale output。
6. **`finish()` 不能自行判成功。** `complete_task` 必须经 `TerminalValidator` 接受，所有终态必须经幂等 `halt()` 获得停车确认。

### 12.6 E2E 导航小闭环：代码逐步对应

#### 12.6.1 小闭环的起止边界

小闭环从一个已确定的 `navigate` decision 开始，到短轨迹前缀产生 controller/environment feedback 结束：

```text
System2 navigate
  -> canonical assistant finish + [SYS1_i]
  -> multimodal template encode
  -> LVLM final hidden states
  -> System1 condition
  -> Direct/CFM trajectory head
  -> System1Output trajectory_ego_t
  -> freshness + safety
  -> execute first M steps
  -> next observation / preemption
```

它不负责修改目的地、判断任务最终成功或维护多轮 route session；这些都在大闭环中完成。

#### 12.6.2 Prompt、因果顺序和 readout

训练/推理 prompt 的主要入口是：

```text
ms-swift/swift/sidewalk_nav/prompt.py::SideWalkPromptManager
ms-swift/swift/sidewalk_nav/tokens.py
ms-swift/swift/sidewalk_nav/models/latent_query.py
ms-swift/swift/sidewalk_nav/models/goal_encoder.py
```

其中当前 Full CoT V4 使用：

```text
reasoning_v4_full_cot_sft:
  system = shared accessibility / evidence / safety / uncertainty policy
  user   = ordered images + instruction + model-visible conditions + evidence catalog
  target = six compact sections + exactly one action location + optional masked navigate readout
```

V4 的 source/annotation/training-view schema、prompt profile digest 和 LM span mask
必须版本一致。训练后还必须加载生成出的 adapter 做至少一条真实生成回读；仅完成
forward/backward 不能视为推理链已打通。

以下 Reasoning V2 只保留 legacy Full-CoT 模板入口：

```text
reasoning_v2_cot_system1:
  system = DEFAULT_REASONING_V2_SYSTEM_PROMPT
  user   = multimodal evidence + full reasoning instruction
  target = five sections + finish + navigate-only readout
```

旧 `Scene/FPVScene/BEVContext/CrossView/Plan` prompt 仅用于 legacy 数据读取和消融，
不能与 V4 生产 target 混写。`DEFAULT_REASONING_V2_SYSTEM_PROMPT` 中的
社会导航原则是推理约束，不是每条 row 都必须填充的社会标签；没有 agent/path/control
证据时，SocialNav 字段应为空或 `unknown`。

`navigate` row 的因果顺序固定为：

```text
system + user observation/instruction + images + optional goal
-> optional tool_call
-> tool_response
-> assistant <think>/<finish decision_type=navigate>
-> [SYS1_0] ... [SYS1_Q-1]
```

编码后的监督规则为：

| token/字段 | attention 可见 | LM label | System1 用途 |
| --- | --- | --- | --- |
| system/user/images/goal | 是 | `-100` | LVLM 条件 |
| tool response | 是 | `-100` | 工具返回条件，不做 CE |
| assistant symbolic answer | 因果可见 | 正常 token label | System2 LM CE |
| `[SYS1_i]` / legacy `[TRAJ_i]` | 是 | 强制 `-100` | 提取连续控制 latent |
| top-level `trajectory_ego_t` | 不进入 token 序列 | 无 | System1 target |

`tokens.py::find_traj_token_indices()` 定位 readout；`mask_readout_token_labels()` 保证它们不被 LM CE 当作要生成的普通词。启用 InternNav-style isolated query 后，`latent_query.py::SidewalkLatentQueryTable` 在 embedding lookup 时执行：

\[
e_{q_i}=Q_i\quad\text{而不是}\quad E_{vocab}([SYS1_i]),
\]

其中 token id 只负责稳定定位，独立参数 \(Q_i\) 才是被 System1 loss 优化的 latent query。可选 `[GOAL_i]` 由 `SidewalkGoalEncoder` 把规范化坐标目标映射到 LVLM hidden space；图像目标仍作为图像输入，不应伪装成数值坐标 token。

#### 12.6.3 LVLM 到 System1 condition

设 LVLM 最后一层输出为：

\[
H=F_\theta(I_{fpv},I_{bev},I_{goal},U,R_{tool},Q_{sys1})
\in\mathbb R^{B\times S\times D}.
\]

`system1_condition.py::build_system1_condition()` 提供三个可消融模式：

```text
traj_query_only: z = H[SYS1 positions], memory = None
vlm_memory:      z = masked_mean(H_prompt), memory = H_prompt
hybrid:          z = H[SYS1 positions], memory = H_prompt
```

这里 `H_prompt` 的 mask 由 expanded attention mask 和 LM labels 构造。`labels != -100` 的 teacher-forced assistant target 会被排除，防止 action head 通过 cross-attention 直接读取训练答案；位于它之前的 tool response 保持可见，因为它是当前决策的合法环境条件。默认训练还会 detach prompt memory，避免 CFM cross-attention 无约束地更新整段 prompt token 表征。

#### 12.6.4 CFM 前向、损失和采样

生产 target 为固定 ego FLU 参考帧中的累计相对轨迹：

\[
X_1=\{[x_i^{forward},y_i^{left},\psi_i^{ccw}]\}_{i=1}^{T}
\in\mathbb R^{T\times3}.
\]

代码入口是 `models/flow_action_head.py::FlowActionTrajectoryHead`。基础 CFM 使用：

\[
X_t=(1-t)X_0+tX_1,
\qquad V^*=X_1-X_0,
\]

\[
\hat V_\phi
=D_\phi\big(E_{step}+W_z z+W_xX_t+W_t\gamma(t),\ C\big),
\]

\[
\mathcal L_{CFM}
=\frac{\sum_{i}m_i\|\hat V_{\phi,i}-V_i^*\|_2^2}
{\sum_i m_i}.
\]

其中 \(C\) 是可选 prompt memory，\(m_i\) 是 `action_mask`。启用 AURA-lite anchor 时，\(X_0\) 来自分类器选择的 motion anchor 加噪声，并叠加 anchor CE、endpoint、smoothness 和 kinematic validity loss；不开启时保持基础 zero/noise CFM。

推理时 `sample_actions()` 用 Euler 积分：

\[
X^{k+1}=X^k+\Delta t\,\hat V_\phi(X^k,t_k\mid z,C),
\qquad \Delta t=1/K.
\]

公平消融 `models/action_head_heads.py::DirectQueryActionHead` 复用相同 \((z,C)\)、相同 `[T,3]` target 和 mask，只把 flow objective 换成一次性 masked regression。比较时不能再启用旧 FPV pixel trajectory head 作为所谓 CFM baseline。

#### 12.6.5 训练前向和梯度边界

训练入口是 `trainers/trainer.py::SidewalkSeq2SeqTrainer.compute_loss()`：

```text
1. 从 batch 弹出 trajectory_ego_t/action_mask 等 sidecar 字段
2. 调用 ms-swift 原生 forward 计算 LM CE，并捕获最后层 hidden states
3. 从 [SYS1_i] 位置构造 System1Condition
4. 校验 checkpoint 绑定的 target field/format 和 trajectory_policy
5. 调用 action_head.loss()
6. L = L_LM + lambda_system1 * L_System1 + optional segmentation losses
```

生产配置只允许：

```text
system1_target_field = trajectory_ego_t
system1_target_format = ego_pose_xyt_flu_v1
```

缺少该字段的纯 LM row 将关闭本 row 的 action loss；trainer 不再隐式回退到 `action_ego_t`。`action_ego_t` 只允许在显式 delta-action 消融配置和独立 checkpoint 中使用。

梯度边界由配置控制，但语义必须保持：

```text
LM CE                 -> assistant/tool-call token 参数
System1 loss          -> action head + isolated query；是否回传 LVLM 由 freeze/detach 配置决定
tool response         -> 作为上下文，不产生 LM CE
trajectory_ego_t      -> 只作为 top-level tensor target，不写进 assistant target
```

#### 12.6.6 Head 输出到执行反馈

离线 head 推理入口为 `cli/infer.py::infer_head_dataset()`。它输出：

```yaml
system1_chunk_ego: [[x_forward_m, y_left_m, yaw_rel_ccw_rad], ...]
action_chunk_metadata:
  system1_target_field: trajectory_ego_t
  system1_target_format: ego_pose_xyt_flu_v1
  system1_condition_mode: traj_query_only | vlm_memory | hybrid
  normalized: false
  flow_steps: K
trajectory_policy:
  representation: cumulative_relative_pose
  coordinate_frame: current_ego_flu
  dt_sec: 0.2
```

`adapters/action_runtime_adapter.py::ActionRuntimeAdapter.adapt_inference_row()` 严格读取 metadata，不根据 `[T,3]` shape 猜测累计轨迹还是 delta action。若 controller 消费 rolling body delta，则执行：

\[
\Delta p_i^{body}=R(-\psi_{i-1})(p_i-p_{i-1}),
\qquad
\Delta\psi_i=wrap(\psi_i-\psi_{i-1}).
\]

RouteServer 接收的权威结构是 `runtime/interfaces.py::System1Output`。随后：

```text
bind_system1_execution_context()
-> SafetyValidator.validate()
-> EnvironmentExecutionManager.dispatch()
-> execute first M steps
-> controller feedback or external-event preemption
-> EnvironmentStepResult
```

当前需要明确保留的实现边界是：`ms-swift` 已有 checkpoint/head 推理和 `ActionRuntimeAdapter`，RouteServer 已有 `System1Policy` Protocol、freshness/safety/execution graph；但将一个生产 checkpoint 服务统一封装成 RouteServer `System1Policy.predict()` 的 live adapter 仍需在部署层完成。文档不能把这条尚未统一的桥接写成已经端到端部署完成。

### 12.7 Trace 回流和逐 episode 校阅方法

#### 12.7.1 从闭环事件到训练 row

大闭环的训练真值来自 `EpisodeTraceV2`，不是 graph state 的最终快照。`ms-swift/swift/sidewalk_nav/data/agentic_episode_manifest.py::build_agentic_episode_trainable_rows()` 按 `decision_id` 切分：

```text
MODEL_OUTPUT(tool_call)
-> TOOL_CALL
-> TOOL_RESPONSE
-> MODEL_OUTPUT(navigate)
-> SYSTEM1_OUTPUT
-> SAFETY_VALIDATION
-> ACTION_DISPATCHED
-> CONTROLLER_FEEDBACK / ACTION_PREEMPTED
```

导出规则必须保持：

- tool response 进入后续 messages 上下文，但 labels 为 `-100`。
- 只有 `navigate` decision 可以带 `[SYS1_i]` 和 System1 top-level supervision。
- 只有经过 safety、实际 dispatch 且未被抢占/判 stale 的 chunk 才是有效 action supervision。
- terminal validation rejection 要作为下一次 System2 的可见反馈，但 validator label 不能成为 LM target。
- `validate_episode_for_rl()` 只有在 environment transition、numeric reward 和 replay key 可复现时才允许该 episode 进入 RL。

#### 12.7.2 推荐的代码校阅顺序

针对一个具体 episode，按以下顺序检查可以最快定位层级错误：

1. **先看 trace 因果链。** 用 `episode_id -> tick_id -> decision_id -> action_id` 排序，确认事件没有跨 revision 重排。
2. **再看 System2 action。** 对照 `runtime/interfaces.py::System2Action`，确认五类 action payload 互斥且完整。
3. **工具 case 看 revision。** 对照 `ToolCallV2`、`ToolResponseV2` 和 `ROUTE_STATE_DELTA`，确认 route mutation 后 `route_revision` 增长。
4. **HITL case 看 correlation。** `interaction_id` 必须从 request 延续到 external user response，旧回复不能解除新等待。
5. **navigate case 看 readout。** 确认 `[SYS1_i]` 位于 final symbolic answer 之后、attention 为 1、labels 为 `-100`。
6. **看 System1 metadata。** 必须是 metric `ego_pose_xyt_flu_v1`，且 task/route/control/observation binding 与当前 state 一致。
7. **看执行证据。** 只有 `ACTION_DISPATCHED` 后接 `CONTROLLER_FEEDBACK + ENVIRONMENT_STEP` 才增加 tick；`ACTION_PREEMPTED` 不算成功执行。
8. **看终态证据。** `complete_task` 后必须存在 accepted `TERMINAL_VALIDATION` 和 `CONTROL_HALTED`，然后才允许 `EPISODE_FINISHED:succeeded`。
9. **最后看导出 row。** 检查 messages 因果顺序、loss mask、`system1_supervision_valid` 和 top-level trajectory 是否与 trace 同一 decision 对齐。

#### 12.7.3 按故障现象定位代码

| 现象 | 优先检查 |
| --- | --- |
| 工具调用成功但模型没有利用结果 | trace 的 tool response 顺序、exporter messages、prompt labels |
| 用户修改目的地后仍执行旧轨迹 | event gate、task/route revision、freshness check、execution lease |
| ask-user 无法恢复或串了回复 | `PendingInteraction.interaction_id`、event reducer、checkpointer |
| `<finish>` 被直接记为到达 | action parser、`TerminalValidator`、`terminal_control()` |
| CFM loss 异常低但闭环差 | teacher-answer memory leakage、坐标/归一化 metadata、action mask |
| CFM 轨迹方向左右相反 | exporter 到 canonical FLU 的边界转换、yaw 正方向 |
| safety 拒绝后仍下发动作 | safety conditional edge、`executable_system1_output`、dispatch freshness |
| 被抢占 chunk 仍进入训练 | `ACTION_PREEMPTED` trace、per-decision exporter validity |
| 离线 head 正常但 RouteServer 无法执行 | checkpoint-to-`System1Policy` adapter、runtime metadata、controller adapter |

这套对照关系的校阅原则是：先验证 **事件和状态转移**，再验证 **模型张量与 loss**，最后验证 **controller 物理执行**。只检查最终 answer、最终轨迹或最终 success 任意一个结果，都不足以证明闭环正确。

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

src/mcp_server/contracts/reasoning_v4.py
  ReasoningTraceV4 / exact six shared-input FullCoT sections / no Finish

src/mcp_server/contracts/reasoning_dataset_v4.py
  V4.5 shared source envelope / V4.6 annotation, audit, resolution and training-view contracts

src/mcp_server/teacher/reasoning_v4_batch.py
  deterministic Task, four independent groundings, blind Decision, retry

src/mcp_server/data/reasoning_v4_composer.py
  validated sections + frozen strong direction -> canonical action tuple

src/mcp_server/agent/reasoning_v4_validation.py
  V4.6 provenance, profile, action-tuple and constraint consistency

src/mcp_server/contracts/vqa_v2.py
  27-task answer-independent question and minimal typed answer contract

src/mcp_server/contracts/vqa_dataset_v2.py
  visible/private source partition, provenance annotation, capability and training view

src/mcp_server/data/vqa_v2_*.py
  source conversion, eligibility, questions, producer routing, view export and audit

src/mcp_server/teacher/vqa_v2_*.py
  visible-only task profiles, strict LVLM response parser and resumable batch

src/mcp_server/eval/vqa_v2.py
  stable-id offline parse/exact/field/direction evaluation

src/mcp_server/contracts/reasoning.py
  Legacy ReasoningTraceV2 / FinishV2 / five strongly typed sections

src/mcp_server/agent/reasoning_text.py
  V2 renderer/parser and conservative V1 adapter

src/mcp_server/agent/reasoning_validation.py
  evidence resolution, constraint applicability and finish/readout validation

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

swift/sidewalk_nav/reasoning.py
  training-side V2 contract, parser, renderer and legacy adapter

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

swift/sidewalk_nav/data/vqa_v2_manifest.py
  generated VQA view -> fresh full-target LM-only messages row

swift/sidewalk_nav/data/vqa_v2_generation_smoke.py
  strict model-response parse, exact and per-task audit

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
| terminal validation 训练回流 | 已实现 | rejected completion 作为 masked user context 回到下一次 System2 row |
| RouteSession mutation CAS/idempotency | 已实现，单进程 | expected revision + idempotency key；分布式 store 未完成 |
| EpisodeTraceV2 | 已实现 | event-sourced contract |
| v2 assistant-suffix readout | 已实现 | `[SYS1_i]` 位于 finish 后且 LM mask |
| Reasoning V4.6 Compact FullCoT | 已实现本地合同与 focused tests，待真实 LVLM smoke | 最小六段、BEV provenance/high-detail、strong action ownership、probe tie gate、显式 formal/artifact eligibility、两仓 parity |
| Reasoning V4.6 真实 LVLM/人工裁决/训练/生成 smoke | 待完成 | 先做 64 条 diverse review；只有全部 P0 gate 通过的 formal views 可训练 |
| Navigation VQA V2 contract/pipeline | 已实现本地合同与 focused tests | 27 task、18 active/9 dormant、E0 derived GT、17 visible weak tasks、两仓 artifact/parser parity |
| Navigation VQA V2 真实 LVLM/训练/生成 smoke | 待完成 | 尚未调用真实 teacher；人工 review、实际产量、短训练和 checkpoint 生成回读未验证 |
| Navigation VQA V1 | 已删除 | 被 V2.0 取代；不保留 parser、manifest、test、设计文档或双写 |
| Legacy ReasoningTraceV2 五段 CoT | 已实现 | 仅供既有 V2 数据和旧 agentic 实验 |
| Legacy V2 section QA | 已删除 | schema、投影器、teacher gate、artifact 和训练消费入口均已下线 |
| V1 CoT/finish 读取兼容 | 已实现 | adapter 双读；新 Stage2 exporter 只写 V2 |
| isolated latent queries | 已实现 | 独立 query table |
| coordinate Goal Encoder | 已实现可选项 | 顶层 geometry，不进入 LM target |
| Direct Action 与 CFM 消融 | 已实现 | 相同 condition/target/output |
| AURA-lite anchor CFM | 已实现可选项 | fixed anchor + classifier + flow init |
| System1 checkpoint save/load/head inference | 已实现 | sidecar files |
| checkpoint -> EpisodeRuntime System1Policy adapter | 未统一 | graph Protocol 已有；生产 checkpoint service adapter 待实现 |
| per-decision session SFT/DAgger exporter | 部分实现 | causal rows、terminal context 和 preempt mask 已实现；active row 尚需补齐 trajectory_policy |
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
18. 新 Full CoT 只能写六段 `ReasoningTraceV4`；V2 和更早 modality-oriented CoT 只允许经 versioned adapter 显式读取。
19. V4 Full CoT 禁止裸 `ToolResultUse`；tool-aware/agentic target 必须使用未来独立 `AgentStep` profile。Legacy V2 的 `ToolResultUse` 位置约束只适用于 V2。
20. VQA 不得从 V4 Full CoT 直接拆段冒充独立 QA 标签；必须使用独立 question/rationale/answer schema、validator 和 loss，且不得恢复 legacy V2 section-QA 入口。
21. 多帧输入必须携带 `image_roles`；禁止用 `len(images)>1` 推断 BEV。
