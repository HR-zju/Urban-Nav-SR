# Tool-Interactive End-to-End Navigation Agent

本文档重新定义 RouteServer 与 ms-swift 之间的全局理解规划、局部视觉导航、训练数据回流和在线推理闭环。核心不是在全局规划和局部导航之间再堆一个手工中间状态，而是让端到端模型通过稳定工具接口主动获取外部信息，并学习如何把工具返回、视觉观测和用户目标联合起来输出局部可执行动作。

```text
instruction / dialogue + FPV / BEV + tool registry
-> E2E agent policy
-> optional typed tool calls
-> tool responses
-> final trajectory / action / route update / clarification
-> validator / replay / training feedback
```

也就是说，系统主干是 **tool-interactive E2E policy**。RouteSession 是 map tool 维护的外部状态，RouteContext 只应作为诊断、评测或 teacher artifact，不应作为训练和推理的必经输入。

相关代码边界：

- RouteServer agent/tool 协议：`src/mcp_server/agent/*`
- Route session：`src/mcp_server/core/route_session_ops.py`
- RouteContext / LocalBEV 诊断工具：`src/mcp_server/route_context/*`
- Visual prompt 在线链路：`src/mcp_server/services/visual_prompt_service.py`
- Agentic Stage2 trace / exporter / online runner：`src/mcp_server/teacher/agentic_stage2_trace_builder.py`、`src/mcp_server/eval/exporters.py`、`src/mcp_server/eval/online_agent_runner.py`
- ms-swift SideWalk 训练推理：`/home/ubuntu/WorkSpace/project/SR/Code/ms-swift/swift/sidewalk_nav/*`

## 1. 当前系统真实状态

当前仓库已经不是空白设计，它已经有四条可复用主线。

### 1.1 全局目的地与路线能力

UrbanNav Module 1 已经围绕以下任务建立了离线 benchmark：

```text
instruction + origin
-> destination / ordered stops / route-update target
-> route
```

当前 final 15K split 覆盖：

- `poi_navigation`
- `implicit_need_navigation`
- `multi_stop_route`
- `route_update_add_via`

评估框架使用冻结 replay cache，避免不同模型或不同时间访问 live map API 导致不可比。

### 1.2 Route session 已经有工程基础

`route_session_ops.py` 已支持：

```text
create_route_session
get_route_session_snapshot
list_route_sessions
add_via_point
remove_via_point
replace_destination
replan_route_session_from_origin
```

这意味着 route-update 和 multi-stop 不应该再被建模成一次性 prompt 输出，而应该升级为有状态 route session 上的操作。

### 1.3 RouteContext 不是 E2E 主路径

当前 `route_context` 已经实现了一套基于 pose、heading、route polyline 和 coarse waypoints 的工程化局部路线状态：

```text
current pose + heading + route_polyline + coarse_waypoints
-> PoseMatch
-> RouteProgress
-> LocalSemantics
-> LocalGeometry
-> LocalBEVContext
```

关键字段包括：

```text
off_route
route_confidence
cross_track_error_m
heading_delta_deg
remaining_distance_m
distance_to_next_turn_m
next_turn_type
turn_semantics
corridor_type
instruction_hint
route_polyline_local
local_segment_bev
connector_bev
corridor_sector_deg
turn_sector_deg
```

但这套表示不应作为新的端到端主路径，原因是：

- 它强绑定 current pose / heading / route geometry，训练数据会过度依赖中间工程标签。
- 它把全局路线到局部行为的对齐问题提前手工拆解，削弱模型端到端学习 tool result、视觉观测和动作之间关系的空间。
- 它适合 debug、teacher、评测切片和可视化，但不适合作为每个训练样本必须提供的隐式输入。

因此新的框架中，`route_context` 降级为：

```text
optional diagnostic artifact
optional teacher signal
optional evaluator slice
not mandatory model input
not mandatory runtime contract
```

上层 agent 只需要看到稳定工具 schema、历史消息、图像和工具返回。需要路线、POI、route session、重规划或用户确认时，由模型学习主动调用工具。

### 1.4 Agentic Stage2 已经打通 RouteServer 到 ms-swift

当前 Stage2 协议固定为 single-tool cold start：

```text
user image/request
assistant AgentState + ToolDecision
optional tool_call
optional tool_response
user final navigation readout + [TRAJ_0]...[TRAJ_15]
assistant final CoT + <finish>
```

它有一个关键因果约束：

```text
[TRAJ_i] 必须出现在 tool_response 之后。
```

ms-swift SideWalk trajectory/action head 从 `[TRAJ_i]` 的 hidden states 读出轨迹或 action chunk。自回归模型中当前位置 hidden state 只能看到之前 token，因此工具结果、route state、视觉上下文必须先进入上下文，再放 `[TRAJ_i]`。

ms-swift 训练侧已经支持：

- `messages/tools/images` agent-template row
- `pixel_trajectory`
- `trajectory_fpv_t`
- `trajectory_ego_t`
- `semantic_segmentation_path`
- trajectory head
- optional action head
- optional segmentation head
- `build_agentic_stage2_trainable_row()` 将 RouteServer canonical trace 转成模板兼容训练视图

## 2. 现有设计缺口

当前文档和代码的主要断点不是缺少某个术语，而是缺少统一闭环。

### 2.1 推理缺口

当前线上链路更像：

```text
instruction -> one-shot tool / route / visual prompt -> local trajectory
```

还缺：

- 工具 schema、工具执行、工具返回和模型输出之间缺少统一的端到端训练 contract。
- route session 已经存在，但还没有作为模型可交互的外部工具状态进入在线 E2E runtime。
- multi-stop 的逐站状态机，离线能评测，在线还没有完整状态。
- route-update、replan、ask-user 等动作还没有稳定地作为模型可学习 tool policy。
- 全局工具结果如何影响局部 trajectory/action，还需要由训练样本显式对齐。
- 运行时 trace 回流训练的闭环数据标准。

### 2.2 训练缺口

当前 Stage2 可以训练“是否调用一个工具、如何消费工具结果、如何输出局部轨迹”，但它不是完整 session-level 闭环训练。

还缺：

- 多 tick / 多 turn session trace。
- tool decision、tool arguments、tool result use 的监督。
- route session op 序列监督，包括 active stop / active leg 的更新。
- route-update / replan / ask-user 的失败恢复监督。
- 工具返回到最终局部 trajectory/action 的端到端监督。
- 在线失败样本的自动归档、教师修正和再训练流程。

## 3. 优化后总体架构

推荐将整体框架包装为 **Tool-Interactive End-to-End Navigation Framework**，简称
`TI-E2E-Nav`。论文表述上，它不是手工全局-局部分层规划器，而是一个三模块闭环系统：

```text
Module I   Tool Environment and State
Module II  End-to-End Agent Policy
Module III Closed-Loop Trace Learning
```

三者分别回答三个问题：

```text
外部世界通过哪些稳定工具暴露给模型？
模型如何端到端决定调用工具、理解返回并输出动作？
运行失败如何变成可回放训练样本，继续提升模型？
```

### 3.1 三模块总览

```text
User instruction / dialogue
FPV / BEV / optional history
Tool registry
        |
        v
┌──────────────────────────────────────────────┐
│ Module I: Tool Environment and State          │
│ map_tool / image_tool / user_interaction_tool │
│ RouteSession as external tool state           │
│ deterministic executor and tool_response      │
└──────────────────────────────────────────────┘
        |
        | tool schema / prior tool responses
        v
┌──────────────────────────────────────────────┐
│ Module II: End-to-End Agent Policy            │
│ multimodal LM decides whether to call tools   │
│ consumes tool_response in context             │
│ emits final trajectory / action / update      │
└──────────────────────────────────────────────┘
        |
        | action / tool_call / final answer
        v
┌──────────────────────────────────────────────┐
│ Module III: Closed-Loop Trace Learning        │
│ validator / runtime logger / failure mining   │
│ AgentTrace -> ms-swift row                    │
│ LM loss + trajectory/action/seg loss          │
└──────────────────────────────────────────────┘
        |
        v
Improved checkpoint -> online tool-interactive inference
```

### 3.2 模块一：Tool Environment and State

**目标**：把地图、路线、图像操作和用户交互统一暴露为模型可学习调用的 typed tools。

输入：

```text
tool schema
tool arguments generated by model
replay cache or live provider
optional persistent RouteSession
```

输出：

```text
tool_response
updated RouteSession when a session tool is called
artifact metadata
structured failure when execution fails
```

这一层对应 RouteServer 当前已经具备或正在稳定的能力：

- `map_tool`：POI search、route planning、route session create/get/update/replan。
- `image_tool`：crop、zoom-in 等可审计图像操作。
- `user_interaction_tool`：ask-user、request-clarification、confirm-route-change。
- `RouteSession`：作为 map tool 内部维护的外部状态，而不是默认塞进模型上下文的手工状态。

在论文中，这一层可以表述为：

```text
TES: a typed tool environment that exposes map, route-session, visual-operation,
and user-interaction APIs to an end-to-end navigation policy.
```

它的关键是工具接口稳定，而不是中间表示复杂：

```text
model-generated tool_call -> deterministic executor -> tool_response -> model context
```

### 3.3 模块二：End-to-End Agent Policy

**目标**：用一个端到端多模态 agent policy 学习全局工具交互和局部动作输出。

输入：

```text
instruction / dialogue
FPV / BEV images
tool registry
dialogue history
previous tool_response, when available
```

输出：

```text
ToolDecision
optional tool_call
final navigation CoT
trajectory through [TRAJ_i] hidden states
optional ego action chunk
optional route update or clarification request
```

这一层不要求预先计算 RouteContext，也不要求每个样本绑定 current pose。模型只看到：

```text
视觉输入
用户目标
可用工具 schema
已经发生的工具返回
```

如果模型认为需要全局信息，它应学习调用工具：

```text
search_poi
plan_route
create_route_session
get_route_session
add_via_and_replan
replace_destination
replan_from_current_pose
request_clarification
```

在论文中，这一层可以表述为：

```text
EAP: an end-to-end multimodal agent policy that jointly learns tool use,
tool-result interpretation, and local navigation action generation.
```

训练时保持 Stage2 的关键因果顺序：

```text
user / images / tool schema
-> assistant tool decision
-> optional tool_call
-> optional tool_response
-> [TRAJ_0] ... [TRAJ_15]
-> final trajectory/action
```

只有这样，trajectory/action head 的 hidden states 才能看到工具结果。没有工具时，模型直接根据 instruction + FPV/BEV 输出局部动作。

### 3.4 模块三：Closed-Loop Trace Learning

**目标**：把在线推理、工具交互、局部动作和失败恢复都变成可回放、可训练的数据。

输入：

```text
online/replay execution
model tool_call and final output
tool_response
validator report
optional human/teacher correction
```

输出：

```text
replayable AgentTrace
optional SessionTrace
ms-swift trainable row
failure slice report
updated checkpoint
```

这一层对应当前 RouteServer 与 ms-swift 的连接：

```text
AgentTrace
-> trace_to_ms_swift_stage2_row()
-> build_agentic_stage2_trainable_row()
-> swift.sidewalk_nav.cli.train
-> OnlineAgentRunner / MsSwiftInferenceAdapter
```

在论文中，这一层可以表述为：

```text
CTL: closed-loop trace learning that converts tool-interactive navigation
runs into supervised multimodal language, tool-use, trajectory, action,
and segmentation training targets.
```

### 3.5 RouteContext 的位置

`RouteContext` 不再是 Module II 的主干。它保留三个用途：

```text
debugging: 解释某次 route/session/pose 是否一致
teacher: 生成或审计部分工具交互样本
evaluation: 做 off-route、progress、maneuver 等诊断切片
```

它不应默认进入 E2E 模型输入，也不应成为所有样本必须具备的 supervision。端到端模型应主要从工具返回和视觉输入中学习。

### 3.6 推理闭环

一次在线推理可以写成：

```text
1. Model observes instruction, images, history, tool registry.
2. Model either emits a tool_call or directly emits final navigation output.
3. Runtime executes typed tool_call when present.
4. tool_response is appended to context.
5. Model emits final navigation CoT and [TRAJ_i]-conditioned trajectory/action.
6. Validator checks parseability, safety, route/session consistency when applicable.
7. Runtime logs replayable trace for training and failure mining.
```

### 3.7 训练闭环

```text
RouteServer online/replay execution
-> AgentTrace / SessionTrace
-> validation and failure slicing
-> ms-swift agent-template row
-> LM loss + trajectory/action/seg loss
-> checkpoint
-> OnlineAgentRunner / RouteServer deployment
```

核心思想保持不变：

```text
RouteServer 管工具环境、executor、route session、validator、日志和评测。
ms-swift 管 E2E agent policy、trajectory/action/seg head、agent-template 编码和模型推理。
两者之间通过稳定 trace / row contract 连接。
```

因此，论文级框架可以总结为：

```text
TI-E2E-Nav exposes navigation-relevant external state through typed tools,
trains a multimodal E2E policy to decide tool use and local actions,
and closes the loop by replaying validated tool-interactive traces.
```

## 4. 核心运行时契约

三大模块落地时不需要新的全局-局部中间状态。闭环 runtime 只需要维护四类与端到端工具交互对齐的工程契约：

```text
ToolRegistry
ToolCall
ToolResponse
AgentTrace / SessionTrace
```

其中 `ToolRegistry / ToolCall / ToolResponse` 对应 Module I 和 Module II 的交互边界，`AgentTrace / SessionTrace` 对应 Module III 的训练闭环。

### 4.1 ToolRegistry

`ToolRegistry` 是模型可见的外部能力列表。它应该稳定、少而强，避免暴露大量细粒度内部函数。

```json
{
  "tools": [
    {"name": "map_tool", "ops": ["geocode", "reverse_geocode", "search_poi", "search_category", "plan_route", "get_route", "create_route_session", "get_route_session", "add_via_and_replan", "remove_via_and_replan", "replace_destination", "replan_from_current_pose"]},
    {"name": "image_tool", "ops": ["crop", "zoom_in"]},
    {"name": "user_interaction_tool", "ops": ["ask_user", "request_clarification", "confirm_route_change"]}
  ]
}
```

这些 op 名称是 student-visible canonical API。训练 target、validator、executor 和
ms-swift held-out eval 必须使用同一组名称，不在样本中混入 `create_route`、
`route_session_id` 等旧别名。

### 4.2 ToolCall

`ToolCall` 是模型需要学习生成的结构化动作。训练时它参与 LM loss；推理时由 runtime 解析和执行。

```json
{
  "name": "map_tool",
  "arguments": {
    "op": "create_route_session",
    "origin": {"latitude": 30.0, "longitude": 120.0},
    "destination": {"latitude": 30.01, "longitude": 120.01},
    "mode": "foot"
  }
}
```

注意：current pose 只有在模型选择调用需要位姿的工具时才作为参数出现，例如 `replan_from_current_pose`。它不应成为所有样本都必须绑定的隐式条件。

### 4.3 ToolResponse

`ToolResponse` 是模型后续推理和局部动作输出可以使用的唯一外部事实来源。它不参与 assistant LM target，但必须出现在 `[TRAJ_i]` readout 之前。

```json
{
  "tool": "map_tool",
  "success": true,
  "result": {
    "session": {"session_id": "route_xxx"},
    "route_plan": {"route_summary": {"distance_m": 350.0}}
  },
  "error": null,
  "artifacts": [],
  "call_id": "optional"
}
```

失败也要结构化返回：

```json
{
  "tool": "map_tool",
  "success": false,
  "result": {},
  "error": {"code": "route_not_found", "message": "No pedestrian route is available."},
  "artifacts": []
}
```

`tool_response` 是环境上下文，不参与 assistant LM target。最终 assistant 只能通过
`<ToolResultUse>` 总结工具结果如何影响导航决策。

### 4.4 Canonical Finish

Stage2/Stage3 新生成的 final assistant 使用 canonical `<finish>`。文本中的轨迹
字段固定为 ms-swift prompt 对齐的 `trajectory_grid1000` 和
`trajectory_coord_frame=norm1000`：

```json
{
  "route_action": "use_map_context",
  "tool_used": true,
  "tool": "map_tool",
  "op": "search_poi",
  "route_session_id": "route_xxx",
  "trajectory_grid1000": [[500, 900], [510, 840]],
  "trajectory_coord_frame": "norm1000",
  "intent": "go_straight",
  "confidence_level": "medium",
  "trajectory_type": "local_navigation",
  "trajectory_validity": "valid",
  "teacher_source": "routeserver_teacher"
}
```

top-level `pixel_trajectory`、`trajectory_fpv_t`、`trajectory_ego_t` 等字段继续作为
trajectory/action head 的连续监督。validator 会把 finish 轨迹转回 norm01 后与
top-level `pixel_trajectory` 做等价比较。旧 `finish.trajectory` 只作为兼容输入，
不再作为 RouteServer 新生成样本的 canonical 输出。

### 4.5 AgentTrace / SessionTrace

`AgentTrace` 是单次工具交互和局部导航输出的可回放样本。`SessionTrace` 是多 tick 在线运行的外层容器，可拆成多个 `AgentTrace` 训练样本。

```json
{
  "turns": [
    {"role": "user", "content": "<image><image>..."},
    {"role": "assistant", "content": "<think>...<ToolDecision>...</ToolDecision></think>"},
    {"role": "tool_call", "content": "{\"name\":\"map_tool\",\"arguments\":{\"op\":\"search_poi\"}}"},
    {"role": "tool_response", "content": "{\"tool\":\"map_tool\",\"success\":true,\"result\":{...}}"},
    {"role": "user", "content": "Produce the final sidewalk navigation decision... [TRAJ_0] ... [TRAJ_15]"},
    {"role": "assistant", "content": "<think>...</think><finish>{...}</finish>"}
  ],
  "supervision": {
    "pixel_trajectory": [],
    "trajectory_ego_t": null,
    "semantic_segmentation_path": null
  }
}
```

## 5. Tool Policy 设计

### 5.1 学生可见工具保持三类元工具

当前 `tool_schema.py` 的方向是正确的：

```text
map_tool
image_tool
user_interaction_tool
```

`map_tool` 负责：

```text
geocode
reverse_geocode
search_poi
search_category
plan_route
get_route
create_route_session
get_route_session
add_via_and_replan
remove_via_and_replan
replace_destination
replan_from_current_pose
```

`image_tool` 当前只暴露可审计图像操作：

```text
crop
zoom_in
```

`user_interaction_tool` 负责：

```text
ask_user
request_clarification
confirm_route_change
```

不要把 teacher LVLM、intent parser、visual prompt teacher 伪装成学生工具。它们只能进入 `trace.metadata["teacher_internal"]`。

### 5.2 Stage2 与 Stage3 的工具预算不同

当前 Stage2 保持：

```text
max_tool_calls = 1
```

这是为了稳定训练 tool decision、tool result use 和 `[TRAJ_i]` 因果位置。

后续 session-level runtime 可以升级为 bounded loop：

```text
max_tool_steps_per_tick = 3
max_route_updates_per_user_turn = 1
max_user_clarification_per_turn = 1
allowed_tools_by_stage = {
  "intent_grounding": ["map_tool.search_poi", "map_tool.search_category", "user_interaction_tool.request_clarification"],
  "route_planning": ["map_tool.plan_route", "map_tool.create_route_session"],
  "route_tracking": ["map_tool.get_route_session", "map_tool.replan_from_current_pose"],
  "route_update": ["map_tool.add_via_and_replan", "map_tool.replace_destination", "user_interaction_tool.confirm_route_change"],
  "local_navigation": ["image_tool.crop", "image_tool.zoom_in"]
}
```

注意：Stage3 的多步 loop 是 runtime 层的 session loop，不应直接破坏 Stage2 单条训练样本格式。可以把一次 session 拆成多条 canonical trace 或引入新的 `SessionTrace`。

## 6. 工具返回到局部动作的接口

新的全局到局部接口不是 `RouteContext Compiler`，而是模型上下文中的 `tool_response`。模型通过工具拿到外部信息，再端到端学习这些信息如何影响局部 trajectory/action。

```text
tool registry
-> model-generated tool_call
-> deterministic tool_response
-> model consumes tool_response with FPV/BEV
-> final trajectory/action
```

### 6.1 工具返回是唯一强制外部上下文

对模型而言，外部环境只通过消息历史和工具返回进入上下文：

```json
{
  "tool": "map_tool",
  "success": true,
  "result": {
    "op": "get_route_session",
    "session": {"session_id": "route_xxx", "destination": "120.01,30.01"},
    "route_plan": {
      "route_summary": {"distance_m": 350.0, "duration_s": 420.0},
      "stop_sequence": [],
      "coarse_waypoints": []
    }
  }
}
```

这比预先构造 RouteContext 更符合端到端训练：

```text
model learns when route info is needed
model learns what route fields matter
model learns how route tool responses affect local trajectory
model can ignore tools when visual/local context is sufficient
```

### 6.2 current pose 只作为工具参数

current pose 不应成为框架强绑定输入。只有当模型选择调用需要位姿的工具时，它才作为工具参数出现：

```text
replan_from_current_pose(origin=current_pose)
search_poi(location=current_pose)
plan_route(origin=current_pose, destination=...)
```

这使训练数据更接近真实端到端决策：模型需要学习“什么时候需要当前位置”和“当前位置应该传给哪个工具”，而不是每个样本都被动接收手工 pose-route 中间字段。

### 6.3 RouteContext 的保留边界

`route_context` 可以继续服务三类非主干用途：

- debug：定位 route/session/pose 是否异常。
- teacher：辅助生成少量工具使用或失败恢复样本。
- evaluation：作为诊断指标切片，而不是训练必需输入。

不要默认把 `RouteContext`、`LocalBEVContext` 或它们的派生字段写入模型 prompt；如果未来确实需要，也应包装为显式工具返回，让模型通过 tool interaction 学会使用。

## 7. 在线推理闭环

### 7.1 单 tick 推理流程

每个导航 tick 执行：

```text
Observe
  instruction/dialogue, FPV/BEV, previous messages, tool registry

Generate Pre-tool Decision
  no_tool | map_tool | image_tool | user_interaction_tool

Execute Typed Tool When Needed
  ToolExecutor -> map_ops / route_session_ops / image op / user interaction

Append ToolResponse
  success result or structured failure

Generate Final Navigation
  ms-swift Stage2/Stage3 model sees images + tool_response + [TRAJ_i]
  -> final CoT + trajectory/action

Validate Local Output
  parse finish, trajectory range, segmentation safety, tool-result consistency

Act and Log
  publish trajectory/action, persist canonical trace, update route session if a tool changed it
```

### 7.2 Backtracking 条件

触发二次工具调用、replan 或用户澄清的条件应显式化：

```text
model requests replan_from_current_pose
route/session tool_response failed
route/session tool_response is stale or inconsistent
user changes destination or stops
validator rejects local trajectory
local trajectory rejected for N consecutive ticks
tool result says provider route unavailable
model emits request_clarification
```

### 7.3 已落地的 session-level runtime 边界

```text
src/mcp_server/agent/session_runtime.py
  RouteSessionRuntime
  run_tick()
  run_until_stop_condition()
  RouteSessionRunResult

future optional split:
src/mcp_server/agent/validators.py
  validate_tool_call()
  validate_tool_response()
  validate_local_navigation_output()

src/mcp_server/agent/session_trace.py
  SessionTrace
  SessionTickTrace
  export_tick_to_agent_trace()
```

当前 `AgentToolRuntime` 不需要推倒重写。它继续作为 Stage2 single-tool runtime。`RouteSessionRuntime` 已作为上层薄调度器落地：输入是一组有序 `AgentStage2Case` tick，内部顺序调用 `OnlineAgentRunner` / `AgentRunner`，输出 `RouteSessionRunResult`，其中每个 tick 仍然是可独立 replay/export 的 canonical `AgentTrace`。

固定停止原因：

```text
completed_all_ticks
max_ticks
user_interaction
invalid_trace
```

其中 `user_interaction` 由 `user_interaction_tool`、`ToolResponse.result.await_user=true`
或 `finish.route_action="request_user_input"` 触发。这个层只做 session 编排、validation 聚合和日志封装；它不负责构造强制 RouteContext，也不改变 Stage2 的 tool role / finish schema。

## 8. 训练闭环

### 8.1 数据来源

训练数据不应只来自静态 teacher。应分为四类：

| 数据源 | 作用 |
| --- | --- |
| UrbanNav Module 1 final 15K | destination grounding、multi-stop、route-update、route success |
| SideWalk Phase1 archive | FPV/BEV CoT、pixel trajectory、seg supervision |
| Agentic Stage2 traces | tool decision、tool_call、tool_response use、final navigation |
| Online session logs | off-route recovery、route update、失败工具恢复、clarification、validator rejection |

### 8.2 RouteServer canonical trace

RouteServer 应保存审计友好的 canonical 顺序：

```text
user
assistant tool decision
tool_call
tool_response
user final navigation readout + [TRAJ_i]
assistant final
```

session-level 闭环可以外包一层：

```json
{
  "schema_version": "route_session_trace_v1",
  "session_id": "nav_xxx",
  "ticks": [
    {
      "tick_id": 0,
      "observation": {},
      "tool_registry": [],
      "agent_trace": {},
      "tool_calls": [],
      "tool_responses": [],
      "route_session_snapshot": {},
      "validators": {},
      "runtime_result": {}
    }
  ]
}
```

训练前可按目标切分：

- tool decision row
- final navigation row
- no-tool row
- failed-tool recovery row
- route-update row
- off-route replan row

### 8.3 ms-swift trainable view

RouteServer exporter 不应把所有监督字段都当作每条样本必备字段。它输出的是按阶段启用的
superset：

```text
common:
  sidewalk_agentic_stage2_sft_v1
  messages
  images
  sidewalk_meta

tool rows:
  tools
  tool_call / tool_response roles
  route_session metadata, when available

trajectory rows:
  pixel_trajectory
  trajectory_fpv_t
  trajectory_ego_t

action rows:
  action_ego_t
  action_mask

segmentation rows:
  semantic_segmentation_path
  segmentation_valid_mask
```

SpotAgent-style cold start 的核心训练视图是 agentic trajectory SFT，而不是直接叠加所有
navigation heads。RouteServer canonical trace：

```text
user image/request + tools
-> assistant <AgentState> + <ToolDecision>
-> optional tool_call
-> optional tool_response
-> user final navigation readout + [TRAJ_i]
-> assistant final navigation
```

ms-swift 训练前只做模板兼容转换：

```text
tool_response -> user final readout + [TRAJ_i] -> assistant final
```

转换为：

```text
tool_response + final-readout prompt + [TRAJ_i] -> assistant final
```

这样既满足 ms-swift agent-template 的 tool/assistant 配对，又保持：

```text
tool_response context 在 [TRAJ_i] 之前
```

其中 `[TRAJ_i]` 是 prompt-side readout query token：`attention_mask=1`，但
`labels=-100`。它本身不参与 LM CE，只给 trajectory/action head 提供因果位置。
没有轨迹监督的 cold-start row 可以保留 `[TRAJ_i]` 以稳定格式，但必须关闭
`L_traj/L_action` 或不提供对应 target。

### 8.4 Loss 归属与阶段化目标

不要把训练目标写成所有阶段共享的固定总公式。loss 应由当前阶段的数据字段、head 开关和
loss mask 决定。

本地 `Spotagent/` 代码给出的阶段边界是：

- `Spotagent/train/sft/qwen2.5_vl_geo_sft.yaml`：`stage: sft` 的基础 SFT，可全量训练，
  `freeze_vision_tower: false`。
- `Spotagent/train/cold_start/cold_start_mp16_reason_latlon.sh`：仍然调用
  `llamafactory-cli train --stage sft`，只是在基础 checkpoint 上用 LoRA、长
  `cutoff_len=16384`、`freeze_vision_tower=true` 注入 agentic/tool 轨迹。
- `Spotagent/train/rl/run_mp16_1_5_rw_linear_hard.sh`：RL 阶段才切到 `verl.trainer.main_ppo`
  且 `algorithm.adv_estimator=grpo`。
- `Spotagent/train/custom/mp16_geo_linear_hard.py`：reward 只解析 `<answer>` 中的经纬度，
  按 haversine 距离做分段线性奖励；没有单独的 tool-policy head loss。

因此工具冷启动的核心 loss 不是新 head loss，而是 agentic trajectory 的 token-level SFT。

#### 8.4.1 Token-level LM loss mask

ms-swift 的 LM loss 是 labels 非 `-100` token 上的自回归 CE：

```text
assistant generated text -> LM CE
tool_call generated text -> LM CE
system/user/images/tools -> labels=-100
tool_response -> labels=-100
[TRAJ_i] query tokens -> labels=-100
pixel/action/seg labels -> not serialized into assistant target
```

SpotAgent 的 agentic cold start 对应的是：

```text
L_cold_start =
  CE(pre-tool assistant reasoning + tool decision)
+ CE(tool_call/action tokens, if any)
+ CE(final assistant reasoning / answer)
```

`tool_response` 是 observation/context，只作为后续 token 的条件，不要求模型复现。
这也是 Stage2-A 的正确口径：先训练会不会发起工具、发起什么工具、如何在 final answer
里使用工具返回；不在这一阶段强行打开 trajectory/action/seg loss。

#### 8.4.2 SideWalk head losses

SideWalk 扩展的 head loss 只在目标字段存在且阶段启用时计算：

```text
L_traj:
  source = pixel_trajectory or pixel_coords
  readout = [TRAJ_i] hidden states
  default = curve / point / WTA candidate loss by head type

L_action:
  source = action_ego_t + action_mask
  readout = same [TRAJ_i] hidden states
  default = flow / CFM action loss

L_seg:
  source = semantic_segmentation_path or semantic_segmentation
  readout = FPV visual feature map
  default = CE + Dice, optional navigation-group auxiliary loss
```

字段缺失时，不应把缺失字段当成负样本监督。当前 trainer 为了让 head 参数进入
autograd graph 可能添加 zero-module loss，但这不是有效监督。

#### 8.4.3 Per-phase objective

训练课程中的每个阶段应使用不同目标：

| 阶段 | 目标函数 | 说明 |
| --- | --- | --- |
| Phase A mixed local E2E | `L_A = L_lm(VQA + nav CoT/finish) + 1_traj * lambda_traj * L_traj + 1_action * lambda_action * L_action + 1_seg * lambda_seg * L_seg + optional L_dense` | 一个训练里混合 FPV-only、FPV-BEV、VQA、trajectory CoT 和 trajectory/action head。按 row 字段决定哪些 loss 生效；不启用 tools。 |
| Phase B tool cold start | `L_B = L_cold_start` | 对齐本地 SpotAgent cold-start 脚本：仍是 SFT token CE，只监督 assistant/tool_call/final answer token；tool_response、images、tools 不算 CE。 |
| Phase B' tool + trajectory bridge | `L_B' = L_cold_start + 1_traj * lambda_traj * L_traj + 1_action * lambda_action * L_action` | 可选桥接阶段，只在 cold-start 已稳定且 row 有有效轨迹/action target 时启用，用来训练工具返回影响 `[TRAJ_i]` 后的动作。 |
| Phase C joint route-session SFT | `L_C = L_lm + 1_traj * lambda_traj * L_traj + 1_action * lambda_action * L_action + 1_seg * lambda_seg * L_seg` | 混合 Phase A/B/B' 和 session rows；tool policy 仍主要通过 LM CE 学，除非未来新增独立 tool-policy head。 |
| Phase D online / RL | `L_D = L_C` for curated SFT; optional `L_pref` or GRPO-style RL objective | DAgger 数据回流继续用 joint SFT；偏好/RL 只在 joint SFT 稳定后启用。 |

不要把以下内容写入 assistant target：

- `pixel_trajectory`
- `trajectory_fpv_t`
- `trajectory_ego_t`
- action label
- segmentation label
- teacher LVLM 私有判断
- ground-truth route labels

这些只能作为顶层监督字段、validator report 或 metadata。

### 8.5 训练课程

推荐分阶段训练，但阶段顺序应先建立端到端局部导航能力，再增量学习工具交互。
不要一开始就把 route grounding / tool policy 当作主任务，否则模型会先学会输出
map/tool 文本，却还没有稳定的 FPV/BEV 到 trajectory/action 的基础映射。
本地 SpotAgent 代码也是这个顺序：先 SFT basic alignment，再 cold-start SFT 注入
tool trajectories，最后才用 GRPO/RL 做结果级优化。

#### Phase A：Mixed FPV/BEV E2E navigation foundation

目标：

```text
instruction / VQA question / local navigation command
+ FPV-only or FPV+BEV
-> VQA answer or final navigation CoT
-> optional <finish>.trajectory_grid1000
-> optional trajectory/action head
-> optional segmentation / dense grounding
```

数据：

- SideWalk Phase1 FPV-only rows
- SideWalk Phase1 FPV-BEV rows
- VQA / scene QA / `qa_bev_context_semantic` BEV topology QA rows, if available
- `structured_cot_trajectory` rows
- `trajectory_fpv_only_cot` rows
- teacher / human final navigation CoT and VQA answers
- `pixel_trajectory` / `trajectory_fpv_t` / `trajectory_ego_t`, when available
- optional action chunk
- optional segmentation / dense grounding sidecars

输出按 row 类型变化：

- VQA row：assistant 输出 answer / structured scene reasoning。
- navigation CoT row：assistant 输出 `<think>` + `<finish>`。
- trajectory text row：`<finish>.trajectory_grid1000` 作为 LM target 的一部分。
- trajectory/action row：`[TRAJ_i]` hidden states 接 trajectory/action head。
- segmentation/dense row：只通过顶层 sidecar 监督，不写入 assistant target。

Loss：

```text
L_A = L_lm(VQA answer + nav CoT/finish)
    + 1_traj * lambda_traj * L_traj
    + 1_action * lambda_action * L_action
    + 1_seg * lambda_seg * L_seg
    + optional L_dense
```

训练方式：

- 这是一个 mixed SFT / multi-task E2E 训练，不需要拆成 FPV-only 和 FPV-BEV 两个 phase。
- 通过 batch mix 或 dataset ratio 控制 FPV-only、FPV-BEV、VQA、trajectory CoT、trajectory head、
  action head、seg/dense rows 的比例。
- VQA row 只开 `L_lm`；没有 trajectory target 时不计算 `L_traj/L_action`。
- trajectory row 同时训练文本 `<finish>.trajectory_grid1000` 和可选 head loss。
- FPV-only row 使用 FPV-only prompt，并把 `<BEVContext>` 明确设为 unavailable。
- FPV-BEV row 使用 BEV 作为拓扑/意图上下文，但局部安全仍以 FPV 为准。
- 这一阶段不暴露 tools、不训练 route session，也不要求模型输出 map op。

#### Phase B：SpotAgent-style tool cold start

```text
FPV/BEV + tool registry + optional route/session text
-> AgentState / ToolDecision
-> at most one tool_call
-> tool_response
-> final navigation CoT
-> optional [TRAJ_i] trajectory/action bridge
```

数据：

- Agentic Stage2 traces
- UrbanNav final 15K 转换出的 map/tool decision rows
- route-update / multi-stop replay 转换出的 single-step tool rows
- no-tool / failed-tool / stale-tool_response cases
- 少量 Phase A no-tool navigation replay，避免冷启动后工具过调用

输出：

- `ToolDecision`
- `tool_call`
- tool result use explanation
- final navigation CoT
- optional trajectory/action, only for Stage2-B bridge rows

Loss：

```text
Stage2-A / pure cold start:
  L_B = L_cold_start

Stage2-B / tool-trajectory bridge:
  L_B' = L_cold_start
       + lambda_traj * L_traj
       + optional lambda_action * L_action
```

这一阶段参考本地 SpotAgent，但只采用最小可控的 single-tool cold start：每条样本最多一个工具，
失败写成 `tool_response.success=false`，并继续生成 final navigation。关键因果顺序保持不变：

```text
tool_response context -> [TRAJ_i] -> trajectory/action head
```

工具冷启动应从 Phase A 的基础导航 checkpoint 继续训练，而不是从纯 route/tool 文本模型开始。
UrbanNav 在这里作为 map_tool / route session op 的监督来源，而不是整个课程的第一阶段。
本地 SpotAgent 的 cold-start 脚本也是 SFT：`llamafactory-cli train --stage sft`，通过
assistant `<think>`、`<tool_call>`、tool observation 后的 final answer 来学习工具轨迹。

#### Phase C：Joint tool-E2E route-session SFT

目标：

```text
FPV/BEV + tool registry + session history + optional tool_response
-> tool/no-tool decision
-> route session op / replan / ask-user / continue
-> final navigation CoT
-> trajectory/action
```

数据：

- Phase A mixed local navigation rows
- Phase B single-tool rows
- RouteSession multi-turn traces
- route-update / active-stop / off-route replan replay
- validator-corrected failure traces

Loss：

```text
L_C = L_lm
    + 1_traj * lambda_traj * L_traj
    + 1_action * lambda_action * L_action
    + 1_seg * lambda_seg * L_seg
```

训练方式：

- 统一训练 `L_lm + L_traj + L_action + L_seg`，但按 batch mix 控制比例。
- session trace 仍可切成多条单步 agent rows，保持 ms-swift `[TRAJ_i]` 位置约束。
- 对 tool-required 样本要求 final trajectory/action 使用工具结果。
- 对 no-tool 样本显式惩罚不必要工具调用，保护纯局部导航能力。

这一阶段才开始真正训练闭环路线状态更新，包括 active stop、current origin、route-update、
off-route replan、到达后停止或 ask-user。它不是单独的 route policy SFT，而是工具文本、
工具返回使用和局部动作头的联合训练。

#### Phase D：Online failure mining + optional RL / preference optimization

流程：

```text
deploy joint checkpoint
-> run online/replay sessions
-> collect failures
-> teacher or rule validator corrects
-> append curated traces
-> continue joint SFT or DAgger-style aggregation
-> optional preference / reward optimization
```

Loss / objective：

```text
curated replay / DAgger:
  continue optimizing L_C

preference:
  L_C + lambda_pref * L_pref

RL:
  maximize audited reward with KL to the joint SFT checkpoint
```

重点采集：

- off-route 后未 replan
- route update 没用 current origin
- multi-stop active stop 错误
- tool parse / schema failure
- tool result 未被 final navigation 使用
- trajectory 被 segmentation validator 拒绝
- 到达状态仍继续前进
- 需要工具的样本中 trajectory 没随 tool_response 改变
- no-tool 场景中产生不必要 tool_call

RL 或偏好优化只应在 Phase C 的 joint SFT 稳定后再考虑。奖励应来自可审计指标：

```text
destination success
route success
route-update success
tool_op_accuracy
tool_efficiency
trajectory ADE/FDE
safety validator pass rate
off-route recovery success
clarification correctness
arrival stop correctness
unnecessary tool-call penalty
```

## 9. 评测闭环

必须分层评测，避免一个总分掩盖错误来源。

| 层级 | 指标 |
| --- | --- |
| Destination grounding | DS、Task Hit@5、Route Success、Task SPL |
| Multi-stop | stop recall、sequence hit、sequence success、per-leg route success |
| Route update | current origin usage、updated destination success、route-update success |
| Tool policy | tool name accuracy、op accuracy、argument validity、tool efficiency、parse exception rate |
| Tool-result grounding | tool_response usage accuracy、failure recovery accuracy、route-result consistency、unnecessary tool-call rate |
| Route-session tool use | active stop update、current origin usage、session update correctness、replan trigger precision/recall |
| E2E local navigation | ADE/FDE、curve loss、validator pass rate、safe-support ratio、collision/social risk |
| Closed-loop recovery | recovery success、arrival success、steps to recovery、unnecessary replan rate |

当前已有：

- `EVALUATION.md` 的 Module 1 表。
- `SIDEWALK_AGENT_EVAL_PROTOCOL.md` 的 Stage2 protocol metrics。
- ms-swift `eval_agentic_stage2.py` / `eval_action.py` / trajectory metrics。

后续应增加：

```text
route_session_closed_loop_eval.py
session_trace_validator.py
tool_result_grounding_eval.py
```

## 10. 上下层作用与实验验证

TI-E2E-Nav 的价值不应只用单一 ADE 或单一 destination success 解释。它的收益来自“上层工具交互”和“下层局部动作”之间的耦合：

```text
工具调用让模型能主动改变路线状态、获取全局信息和请求澄清；
局部轨迹/action head 让工具结果真正影响下一步可执行动作；
validator 和 replay trace 把失败闭环成下一轮训练数据。
```

因此实验必须分层、分 slice 验证。

### 10.1 关键假设

建议围绕以下假设设计实验：

| 假设 | 需要证明什么 | 主要指标 |
| --- | --- | --- |
| H1 工具交互提升路线更新能力 | add-via、replace-destination、replan case 上，训练后的 agent 比 no-tool 或 tool-untrained baseline 更会调用正确工具并使用结果 | route-update success、current origin usage、session update correctness、tool op accuracy |
| H2 工具结果能影响局部动作 | 给定 route/session tool_response 后，final trajectory/action 更符合更新后的路线或用户目标 | tool-response grounding、route-result consistency、conditional ADE/FDE、validator pass |
| H3 训练闭环提升恢复能力 | off-route、工具失败、局部 validator reject 后，模型更会 replan、fallback 或 ask-user | recovery success、steps to recovery、failed-tool recovery accuracy |
| H4 工具交互不会伤害纯局部导航 | 不需要工具的常规局部样本上，tool-trained 模型不应明显退化 | no-tool ADE/FDE、no-tool validator pass、unnecessary tool-call rate |
| H5 端到端联合训练优于分离训练 | 只训练 tool text 或只训练 trajectory 都不如 LM+tool+trajectory/action 联合训练 | composite score、tool-needed ADE、route-update success、local safety |

### 10.2 ADE 是否应该提升

ADE/FDE 不一定在所有样本上整体提升。更准确的判断是：

```text
纯局部、无路线变化、无交互需求的样本：
  ADE/FDE 应保持不退化，提升可能有限。

需要工具的样本，例如 route_update、add_via、replace_destination、off-route replan：
  如果 GT trajectory/action 是基于更新后路线或新目标生成的，
  tool-interactive E2E 模型应该在 conditional ADE/FDE 上提升。

工具返回错误、过期或模型误调用工具：
  ADE/FDE 可能变差，因此必须同时报告 unnecessary tool-call rate 和 failed-tool recovery。
```

因此不要只报 overall ADE。需要至少报告：

```text
ADE_all
ADE_no_tool_required
ADE_tool_required
ADE_route_update
ADE_replan_recovery
ADE_after_failed_tool
FDE_* 同样分桶
```

如果 `ADE_all` 没明显提升，但 `ADE_tool_required`、`ADE_route_update` 和 recovery slice 明显提升，同时 no-tool slice 不退化，这仍然说明框架有效。

### 10.3 实验组设计

建议使用以下 ablation：

| 组别 | 描述 | 目的 |
| --- | --- | --- |
| `local-only` | FPV/BEV -> trajectory/action，不给 tools，不训练 tool roles | 局部端到端 baseline |
| `tool-schema-only` | prompt 中有 tools，但未用 tool trace SFT | 测试工具 schema 暴露本身是否足够 |
| `tool-sft-lm-only` | 训练 tool decision/tool_call/final text，但不开 trajectory/action head loss | 分离 tool 学习和局部动作学习 |
| `traj-only-with-route-text` | 给固定路线文本或人工 route hint，但不让模型调用工具 | 对比手工中间条件 |
| `full-tool-e2e` | LM + tool_call + tool_response use + trajectory/action 联合训练 | 主方法 |
| `full-tool-e2e-online-mined` | 主方法 + online failure mining / curated replay | 验证闭环数据收益 |

### 10.4 实验模块一：路线工具交互评测

目标：验证上层工具能力是否提升。

数据：

```text
UrbanNav final 15K 中的 multi_stop_route / route_update_add_via
新增 replace_destination / replan_from_current_pose synthetic replay cases
工具失败和空结果 cases
```

指标：

```text
tool_name_accuracy
tool_op_accuracy
argument_validity
session_update_correctness
current_origin_usage
route_update_success
multi_stop_sequence_success
failed_tool_recovery_accuracy
unnecessary_tool_call_rate
```

建议实现：

```text
tools/evaluation/run_tool_interactive_route_benchmark.py
src/mcp_server/eval/tool_interactive_metrics.py
```

### 10.5 实验模块二：工具结果 grounding 评测

目标：验证模型不是“会调用工具但不会用工具结果”。

设计三种输入：

```text
correct tool_response
stale / shuffled tool_response
failed tool_response
```

比较 final navigation 是否随工具结果合理变化。

指标：

```text
tool_response_usage_accuracy
route_result_consistency
destination_after_update_match
trajectory_direction_consistency
failure_recovery_action_accuracy
```

核心 sanity check：

```text
正确 tool_response 下性能高；
shuffled tool_response 下性能下降；
failed tool_response 下模型能 fallback 或 ask-user；
这才能说明模型真实使用了工具结果。
```

建议实现：

```text
tools/evaluation/run_tool_result_grounding_eval.py
```

### 10.6 实验模块三：局部 E2E 导航评测

目标：验证工具交互是否改善局部 trajectory/action，而不是只改善文本工具指标。

数据分桶：

```text
no_tool_required
tool_required_search_or_route
route_update_add_via
replace_destination
replan_from_current_pose
failed_tool_recovery
```

指标：

```text
ADE_all / FDE_all
ADE_no_tool_required / FDE_no_tool_required
ADE_tool_required / FDE_tool_required
ADE_route_update / FDE_route_update
trajectory_curve_loss
validator_pass_rate
safe_support_ratio
collision/social_risk
```

解释标准：

- `full-tool-e2e` 在 `tool_required` 和 `route_update` slice 上应优于 `local-only`。
- `no_tool_required` slice 不应明显退化。
- 如果 overall ADE 提升不大，但 tool-required slice 提升明显，应按 slice 报告而不是否定框架。

建议实现：

```text
swift.sidewalk_nav.cli.eval_agentic_stage2
swift.sidewalk_nav.cli.eval_action
tools/evaluation/compare_tool_e2e_navigation_slices.py
```

### 10.7 实验模块四：闭环 session replay

目标：验证多 tick 交互、route update、局部失败恢复是否有效。

Replay 场景：

```text
用户中途增加途经点
用户更换目的地
局部 validator 连续拒绝 trajectory
map_tool 返回失败
模型需要 request_clarification
```

指标：

```text
closed_loop_success
arrival_success
route_update_success
recovery_success
steps_to_recovery
tool_calls_per_success
unnecessary_replan_rate
trace_exception_rate
```

建议实现：

```text
tools/evaluation/run_route_session_closed_loop_benchmark.py
tools/evaluation/mine_tool_interactive_failures.py
```

### 10.8 论文主表建议

主表不要混成一个分数，建议拆成三张表：

1. 全局工具与路线更新表：

```text
DS / Route Success / route-update success / multi-stop sequence success / tool op accuracy
```

2. 局部端到端导航表：

```text
ADE/FDE by slice / validator pass / safe-support ratio / action error
```

3. 闭环恢复表：

```text
closed-loop success / recovery success / steps to recovery / unnecessary tool-call rate
```

这样可以清楚回答：

```text
上层工具训练是否让 agent 更会改路线？
工具返回是否真实影响了局部动作？
局部 ADE 是否在需要工具的样本上提升？
闭环 replay 是否让失败恢复变好？
```

## 11. 推荐实施路线

### Step 1：先稳定 ToolRegistry / ToolExecutor / ToolResponse

目标：不改模型，先保证所有学生可见工具都有稳定 schema、fake-provider 测试、结构化失败返回和 replay cache。

新增：

```text
tests/test_agent_tool_executor_real_ops.py
tests/test_agent_tool_schema_contract.py
```

验收：

- `map_tool` / `image_tool` / `user_interaction_tool` 的 schema 与 executor 一致。
- tool failure 不抛异常，写成 `tool_response.success=false`。
- route session create/get/update/replan 可在 fake provider 下回放。

### Step 2：扩展 OnlineAgentRunner 为 tool-interactive E2E runtime

新增：

```text
agent/session_runtime.py
agent/session_trace.py
```

目标：

```text
model pre-tool generation
-> runtime executes typed tool
-> append tool_response
-> model final navigation generation
-> validator
-> AgentTrace / SessionTrace
```

验收：

- tool/no-tool/failed-tool 都能在线跑通。
- current pose 只在模型调用相关工具时进入 tool arguments。
- 每个 tick 都能导出 ms-swift-compatible AgentTrace。

### Step 3：把 route_update / multi-stop 接入 session state

目标：

```text
multi_stop_route 不再只是离线 sequence，而是 active_stop_index + per-leg route session。
route_update_add_via 不再只是输出 updated route，而是 session op。
```

验收：

- add-via 后 route_version 增加。
- duplicate via 不触发无意义重算。
- replace destination / replan_from_current_pose 都记录 reason。

### Step 4：在线日志回流 ms-swift

流程：

```text
SessionTrace
-> curated AgentTrace
-> trace_to_ms_swift_stage2_row()
-> build_agentic_stage2_trainable_row()
-> swift.sidewalk_nav.cli.train
```

验收：

- tool/no-tool/failed-tool 三类 row 都能训练前 smoke。
- `[TRAJ_i]` 位置通过 validator。
- `pixel_trajectory` 等字段未进入 assistant target。

### Step 5：闭环评测与 failure mining

新增：

```text
tools/evaluation/run_route_session_closed_loop_benchmark.py
tools/evaluation/mine_tool_interactive_failures.py
```

输出：

```text
session_summary.json
session_traces.jsonl
failure_slices.json
curated_stage2_rows.jsonl
```

## 12. 最终框架定位

优化后的框架不是一个“大 prompt agent”，而是一个状态机和训练闭环：

```text
ToolRegistry 暴露地图、路线、图像和用户交互能力。
ToolExecutor 把模型 tool_call 转成确定性 tool_response。
RouteSession 只是 map_tool 维护的外部状态。
ms-swift SideWalk E2E 模型学习 tool decision、tool-result use、final navigation、trajectory/action。
Validator 和 AgentTrace/SessionTrace 把在线失败变成下一轮训练数据。
```

这样可以同时满足：

- 全局理解规划：通过 map tools 学习目的地、隐式需求、多站点、route update。
- 局部导航：FPV/BEV、segmentation、SoM、trajectory/action。
- 在线闭环：off-route、replan、ask-user、failure recovery。
- 离线可复现：replay cache、canonical trace、validator、benchmark。
- 训练闭环：RouteServer trace 到 ms-swift SFT/head loss，再回到 RouteServer online runner。

短期最关键的工程判断：

```text
不要把 Stage2 single-tool runtime 直接扩成无边界 ReAct。
先稳定 tool runtime、tool_response contract 和 replay trace；
把每个 tick 导出为稳定、可训练、可回放的 agent trace。
```
