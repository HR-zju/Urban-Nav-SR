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
