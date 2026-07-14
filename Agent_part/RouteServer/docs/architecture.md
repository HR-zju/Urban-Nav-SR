# RouteServer Architecture

本文档记录 RouteServer 当前主链路的数学抽象和模块边界。README 只保留快速入口，细节放在这里。

## 1. Route Session / Route Planning

当前 route planning 不是一次性函数，而是有状态 session。

\[
\mathcal S_t = (o_t,\ d_t,\ V_t,\ m,\ \Gamma_t,\ h_t)
\]

其中：

- \(o_t\): 当前起点或当前位置
- \(d_t\): 最终目的地
- \(V_t=[v_1,\dots,v_n]\): 途经点序列
- \(m\): 导航模式，`foot/car/bike`
- \(\Gamma_t\): 当前缓存 route plan
- \(h_t\): session metadata，例如创建时间、更新时间

创建 session：

\[
\mathcal S_0 = \mathrm{CreateSession}(o_0,d_0,V_0,m)
\]

自然语言动态增加途经点：

\[
v_{new} = \mathrm{ResolveInstruction}(u_t,x_t,m)
\]

\[
V_{t+1} = \mathrm{Insert}(V_t,v_{new})
\]

\[
\Gamma_{t+1} =
\mathrm{Route}(o_t,\ [V_{t+1}],\ d_t,\ m)
\]

缓存读取：

\[
\mathrm{GetSession}(\mathcal S_t) = (\mathcal S_t,\Gamma_t)
\]

只有显式 `force_replan=true` 时重新请求地图 route API：

\[
\Gamma_t =
\mathrm{Route}(o_t,V_t,d_t,m)
\]

工程上对应：

- `create_route_session`
- `get_route_session`
- `add_via_point_to_session`
- `remove_via_point_from_session`
- `replace_destination_in_session`
- `replan_route_session_from_origin`

## 2. Demand / Itinerary Planning

单目标：

\[
d_t^* = \mathrm{Recommend}(u_t,x_t,m,\rho,k)
\]

多目标需求：

\[
G_t =
\mathrm{Decompose}(u_t)
=
[g_1,g_2,\dots,g_N]
\]

每个 goal 先检索 Top-K：

\[
C_i =
\mathrm{RetrievePOI}(g_i,x_i,m,\rho_i,k)
\]

再由开放词汇 LLM judge 选择：

\[
z_i^* =
\mathrm{Judge}(g_i,\ C_i,\ x_i,\ z_{<i})
\]

最后：

\[
V_t=[z_1^*,\dots,z_{N-1}^*],
\quad
d_t=z_N^*
\]

\[
\Gamma_t =
\mathrm{Route}(o_t,V_t,d_t,m)
\]

当前实现已经支持 goal decomposition 和 per-goal open-vocabulary candidate judge。尚未实现全局 itinerary 组合优化：

\[
(V_t^*,d_t^*)
=
\arg\max_{\mathcal I}
Score(\mathcal I \mid G_t,x_t,m)
\]

后续全局 score 应包含：

- goal satisfaction
- total walking time
- detour penalty
- fatigue penalty
- spatial compactness
- opening hours / availability

## 3. RouteContext

RouteContext 是在线局部状态压缩层：

\[
(P_t,W_t,S_t)=\Gamma_t
\]

\[
r_t =
\mathrm{RouteContext}(x_t,\psi_t,\Gamma_t)
\]

先做 route matching：

\[
\hat q_t=\Pi_{P_t}(x_t)
\]

得到：

\[
(i_t,\alpha_t,\hat q_t)
=
\mathrm{Match}(x_t,P_t)
\]

并计算：

\[
e_t=\|x_t-\hat q_t\|_\perp
\]

\[
\Delta\psi_t=\psi_t-\theta_t
\]

再做 progress：

\[
s_t=\mathrm{arcpos}(\hat q_t)
\]

\[
d^{turn}_t=s(w_{next})-s_t
\]

输出：

- `pose_match`
- `progress`
- `local_semantics`
- `geometry`

## 4. LocalBEVContext

LocalBEVContext 是 route-side geometric prior / debugger state：

\[
b_t =
\mathrm{LocalBEV}(r_t,\Gamma_t)
\]

输出局部 ego-aligned 表示：

\[
b_t =
(p^{ego}_t,\ P^{local}_t,\ L^{local}_t,\ C_t,\ T_t,\ B_t)
\]

字段：

- `current_pose_bev`
- `route_polyline_local`
- `local_segment_bev`
- `connector_bev`
- `next_turn_point_bev`
- `bev_bbox`

它适合作为 teacher/debug representation，不是最终 visual prompt 的强制路径。

## 5. Prompt Proposal / Visual Prompt

BEV prior：

\[
\mathcal P^{bev}_t = f(r_t,b_t)
\]

单个 proposal：

\[
P_i = (\tau_i,\theta_i,\Delta\theta_i,\mathrm{range}_i,\pi_i)
\]

PoV weak projection：

\[
\mathcal P^{pov,v1}_t =
g(\mathcal P^{bev}_t,r_t,H,W)
\]

RGB-aware candidates：

\[
V_t = \Psi(I_t)
\]

\[
\mathcal C_t =
\mathrm{OptionalCandidates}(r_t,b_t,I_t)
\]

当前在线主决策不再强制：

```text
BEV proposal -> PoV candidate -> candidate arbiter
```

而是：

\[
\hat p_t =
A_{\theta}^{zero-shot}
(
I_t,\ r_t,\ m,\ \mathcal C_t^{optional}
)
\]

其中：

- \(I_t\): FPV image
- \(r_t\): compact RouteContext
- \(m\): navigation mode
- \(\mathcal C_t\): optional BEV/PoV candidates
- \(\hat p_t\): structured visual prompt

输出结构：

```json
{
  "type": "arrow_path",
  "path_uv": [[160, 175], [190, 135], [220, 90]],
  "instruction": "Walk along the sidewalk opening.",
  "confidence": 0.72,
  "reason": "The route intent is forward and the visible sidewalk is the safest opening."
}
```

mode-aware feasibility constraint：

\[
\hat p_t \in \mathcal F(I_t,m)
\]

对于 `mode=foot`：

\[
\mathcal F(I_t,\text{foot})
=
\text{sidewalk/pedestrian path/crosswalk/open plaza}
-
\text{vehicle road/obstacles/walls/fences}
\]

最终：

\[
O_t =
\mathrm{Overlay}(I_t,\hat p_t)
\]

## 6. Total Chain

推荐总链：

\[
u_t,x_t,\psi_t,I_t,\mathcal S_{t-1}
\rightarrow
\mathcal S_t
\rightarrow
\Gamma_t
\rightarrow
r_t
\rightarrow
\hat p_t
\rightarrow
Overlay(I_t,\hat p_t)
\]

工程对应：

```text
Route Session / Route Plan
-> RouteContext
-> mode-aware VisualPromptAgent
-> SVG overlay
```

BEV/PoV candidates 保留为：

- explanation/debug output
- teacher prior
- future distillation data

## 7. Episode Runtime V2

闭环主线不再把 `RouteContext` 或额外 belief 对象设为模型必须消费的中间表示。
外部环境、工具状态和连续观测通过 typed tool 与 episode event 暴露：

\[
s_t=(o_t,h_t,r_t,d_t,b_t),\quad
a_t^{(2)}=\pi_{S2}(s_t)
\]

其中 `System2Action` 只允许：

```text
tool_call | ask_user | navigate | finish | stop
```

当 \(a_t^{(2)}=\texttt{navigate}\) 时，System1 根据 LVLM latent 生成 ego-frame
连续轨迹块：

\[
X_{t:t+T}^{ego}=\pi_{S1}(H_t),\qquad
X^{ego}\in\mathbb R^{T\times3}
\]

每一步必须先经过安全验证，再作用于环境：

\[
v_t=V_{safe}(X_{t:t+T}^{ego},o_t),\qquad
(o_{t+1},r_t,d_t,i_t)=Env.step(X_{t:t+T}^{ego})
\]

当前 LangGraph 节点图为：

```text
observe -> system2
  tool_call -> tool -> system2 / hitl
  ask_user -> interrupt -> resume -> system2
  navigate -> system1 -> safety -> environment_step -> observe
  finish/stop -> finish
```

HITL graph 使用 checkpointer，并在 `run/resume` 边界启用同步 durability。所有
conditional router 都是 async，避免异步图把路由工作泄漏到默认线程池。

## 8. 单一工具合同

工具合同按三层拆分，但由同一 registry 生成：

```text
contracts/tool.py   Pydantic ToolCallV2 / ToolResponseV2
tools/specs.py      map/image/user argument models and legacy aliases
tools/registry.py   ToolOperationSpec / ToolSpecRegistry / capability profiles
```

模型可见 tool name 保持兼容：

```text
map_tool
image_tool
user_interaction_tool
```

live 与 replay executor 必须显式报告能力集合
\(C_e=\{(tool,op)\}\)，conformance test 要求：

\[
C_e=C_{registry}(profile_e)
\]

这保证 schema 中公开的 op、executor 可执行 op 和 replay 可复现 op 不再由三套
常量分别维护。provider-specific 字段只能出现在工具内部结果中，不能改变公共
`ToolResponseV2` 外壳。

## 9. EpisodeTraceV2

三个旧 runner 仍可保留各自内部控制流，但输出统一附带：

```text
contracts/episode.py       EpisodeEvent / EpisodeTraceV2
contracts/trace_adapter.py legacy AgentTrace -> EpisodeTraceV2
```

事件只记录真实发生的因果链：

```text
observation_received
model_output
tool_call / tool_response / route_state_delta
user_interrupt / user_resume
system1_output / safety_validation / environment_step
episode_finished
```

每个事件具有稳定 `event_id`、`parent_event_id`、`tick_id` 和 `source`。它是跨
runtime、评测和训练的数据合同，不是额外的模型输入 belief。

## 10. 冻结 Episode Benchmark

离线评测代码位于 `eval/episode/`，按职责拆分：

```text
schema.py       immutable case/transition schema
io.py           JSONL loading and writing
environment.py frozen deterministic transition environment
metrics.py      main metrics and diagnostics
runner.py       injected-policy episode runner
service.py      externally driven replay service
```

主指标限制为 arrival、collision-free、recovery、normalized efficiency 和 total
reward。tool call 数量、安全拒绝次数等只作为 diagnostics。冻结 transition 必须有
`replay_key`；HTTP benchmark 必须使用 `--episode-only`，该模式不会初始化地图
provider。

## 11. RouteServer 与 ms-swift 边界

```text
RouteServer
  tools + environment + LangGraph rollout + EpisodeTraceV2 + frozen benchmark

ms-swift
  EpisodeTraceV2 -> causal multi-tick rows
  session SFT / explicit DAgger corrections
  LM/tool/readout masks + System1 action loss
  UrbanNavEnvClient
```

GRPO 不是默认 fallback。只有每个 `environment_step` 同时具有 numeric reward、
稳定 `replay_key` 和 terminal event 时，episode 才通过 RL readiness gate。当前顺序
固定为：

```text
session SFT -> DAgger aggregation -> frozen transition/reward audit -> GRPO
```
