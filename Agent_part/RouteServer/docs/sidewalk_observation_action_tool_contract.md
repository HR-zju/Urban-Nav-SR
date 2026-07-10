# SideWalk Observation / Action / Tool Contract

本文档定义 Urban-Nav-SR RouteServer 与 ms-swift SideWalk 之间的统一协议。目标是把
现有 Stage2 agent tools、导航语义决策、trajectory head、CFM action head 和在线
runtime 拆成稳定契约，避免把所有字段混进 `<finish>` 或自由 `observation` dict。

## 1. 总体边界

当前框架是 contract-driven tool-augmented E2E navigator：

```text
NavObservation
  -> System2 LM emits TextAction
  -> RouteServer executes optional tool
  -> ToolResponse / ContextAsset enters next context
  -> System2 LM emits NavigationDecision
  -> System1 heads read [TRAJ_i] hidden states
  -> TrajectoryOutput / CFMControlOutput
```

两条输出层必须分开：

```text
System2 / LM text:
  ExternalToolAction
  UserInteractionAction
  NavigationDecisionAction

System1 / heads:
  TrajectoryOutput
  CFMControlOutput
```

`ControlActionChunk` 不写进 `<finish>`，它是 ms-swift head 输出，不是 LM 文本动作。

## 2. NavObservation

代码契约：`mcp_server.agent.contracts.NavObservation`

```json
{
  "contract_version": "sidewalk_nav_observation_v1",
  "observation_id": "case_001_t000",
  "request": {
    "active_instruction": "go to the nearest convenience store",
    "latest_user_utterance": "go to the nearest convenience store",
    "dialogue_history": [],
    "instruction_updates": [],
    "resolved_pending_interaction": null
  },
  "sensors": {
    "fpv_images": [],
    "bev_images": [],
    "history_fpv_images": [],
    "history_bev_images": [],
    "camera": {},
    "timestamp": null
  },
  "route_state": {
    "mode": "foot",
    "current_pose": null,
    "route_session_id": null,
    "active_goal": null,
    "active_route_id": null
  },
  "dialogue_state": {
    "pending_user_interaction": null
  },
  "tool_context": {
    "available_tools": [],
    "tool_responses": [],
    "context_assets": []
  },
  "runtime_meta": {
    "case_id": "case_001",
    "split": "train"
  }
}
```

边界规则：

- Observation 只包含模型和 runtime 真实可见的信息。
- `pixel_trajectory`、`trajectory_ego_t`、`action_ego_t`、`semantic_segmentation_path`
  是 supervision，不进入 Observation。
- `RouteContext` 是 debug / teacher / evaluator artifact，不是默认模型输入。
- 旧 `AgentStage2Case.observation` 仍兼容；`AgentStage2Case.to_nav_observation()`
  返回规范视图。

## 3. TextAction

代码契约：`mcp_server.agent.contracts.ToolAction` 与
`mcp_server.agent.contracts.NavigationDecision`。

### 3.1 ExternalToolAction

```json
{
  "contract_version": "sidewalk_nav_action_v1",
  "action_type": "external_tool_call",
  "tool": "map_tool",
  "arguments": {
    "op": "search_poi",
    "query": "nearby convenience store"
  },
  "decision_meta": {
    "reason_code": "need_map_context"
  }
}
```

该动作仍导出为 `tool_call` role，参与 LM loss。

### 3.2 UserInteractionAction

```json
{
  "contract_version": "sidewalk_nav_action_v1",
  "action_type": "user_interaction",
  "tool": "user_interaction_tool",
  "arguments": {
    "op": "present_options",
    "message": "附近有两家便利店，你想去哪个？",
    "options": [
      {"option_id": "opt_1", "label": "7-Eleven", "candidate_id": "poi_1"}
    ],
    "recommended_option_id": "opt_1",
    "requires_user_response": true,
    "interaction_mode": "blocking",
    "motion_policy_while_waiting": "stop"
  }
}
```

v1 支持的 `user_interaction_tool` op：

```text
ask_user
request_clarification
present_options
confirm_route_change
notify_issue
answer_query
```

`requires_user_response=true` 时，RouteServer runtime 应进入 `WAITING_FOR_USER`，
并写入 `dialogue_state.pending_user_interaction`。当前基础实现不接真实 UI，只
记录 pending object，给后续 App/UI runtime 使用。

### 3.3 NavigationDecisionAction

```json
{
  "contract_version": "sidewalk_nav_action_v1",
  "action_type": "navigation_decision",
  "route_action": "follow_route",
  "intent": "turn_right",
  "instruction_text": "前方约 20 米右转。",
  "tool_used": true,
  "user_message": null
}
```

该动作由 `<finish>` 解析得到。`<finish>` 保留 LM 可解释导航决策，不承载 CFM
连续控制 action chunk。

## 4. System1Output

代码契约：`mcp_server.agent.contracts.System1ControlOutput`。

```json
{
  "output_type": "system1_control",
  "trajectory_norm": [[0.5, 0.9], [0.52, 0.84]],
  "trajectory_grid1000": [[500, 900], [520, 840]],
  "action_space": "ego_delta_xyt",
  "action_chunk_ego": [[0.0, 0.25, 0.01]],
  "action_mask": [true],
  "runtime_commands": []
}
```

训练时：

```text
Stage2-B: L = L_LM + lambda_traj * L_traj
Stage2-C: L = L_LM + lambda_traj * L_traj + lambda_action * L_CFM
```

`System1Output` 的监督来自 top-level row 字段，不从 assistant text 解析。

## 5. ToolContract

代码契约：`mcp_server.agent.tool_schema`。

### 5.1 map_tool

当前保留三类能力：

```text
Location:
  geocode
  reverse_geocode

Destination / POI:
  search_destination
  search_poi
  search_category

Route / session:
  plan_route
  get_route
  get_route_context
  create_route_session
  get_route_session
  add_via
  add_via_and_replan
  remove_via_and_replan
  replace_destination
  replan_from_current_pose
```

`get_route_context` 是新增统一上下文 op。不要为 `get_next_waypoint`、
`get_route_instruction`、`get_bev_view` 拆出多个碎 op；这些应作为
`context_assets` 返回。

### 5.2 image_tool

```text
crop
zoom_in
```

不暴露 `inspect`，除非未来有真实学生可调用视觉工具。

### 5.3 user_interaction_tool

```text
ask_user
request_clarification
present_options
confirm_route_change
notify_issue
answer_query
```

用户输入本身不是工具调用；新用户话语应进入
`NavObservation.request.latest_user_utterance`。

## 6. ContextAsset

代码契约：`mcp_server.agent.contracts.ContextAsset`。

`ContextAsset` 是工具产生的模型可见输入，不是 supervision。

```json
{
  "asset_id": "asset_001",
  "asset_type": "next_waypoint",
  "source_tool": "map_tool",
  "source_tool_response_id": "tr_001",
  "modality": "structured",
  "visibility": "model_visible",
  "frame": "ego",
  "payload": {
    "coordinate_convention": "x_right_y_front_theta_yaw",
    "waypoint_ego": [20.0, -2.0, -0.3],
    "maneuver": "turn_right"
  }
}
```

v1 `get_route_context` 支持：

```text
route_summary
route_instruction_text
next_maneuver
next_waypoint
```

图像类 asset 只留接口：

```text
route_bev_view
waypoint_image
entrance_image
```

如果没有真实 `image_ref`，executor 必须写 `render_status=not_implemented`，不能伪造图像。

## 7. AgentStepResult 与 SessionTrace

代码契约：

```text
mcp_server.agent.contracts.AgentStepResult
mcp_server.agent.session_runtime.SessionTrace
```

`AgentStepResult` 是单 tick 统一输出：

```json
{
  "schema_version": "sidewalk_agent_step_result_v1",
  "step_id": "case_001",
  "runtime_state": "READY_FOR_CONTROL",
  "observation": {},
  "system2": {
    "tool_action": null,
    "navigation_decision": {}
  },
  "system1": {},
  "tool_trace": [],
  "validation": {},
  "artifacts": [],
  "replay": {}
}
```

`SessionTrace` 是多 tick 外层容器，当前由 `RouteSessionRuntime` 生成。它不替代
Stage2 单 tick trace，每个 tick 仍可独立导出 ms-swift row。

## 8. ms-swift row 兼容

RouteServer exporter 保留旧 flat 字段：

```text
messages
tools
images
pixel_trajectory
trajectory_fpv_t
trajectory_ego_t
action_ego_t
action_mask
action_stats
semantic_segmentation_path
sidewalk_meta
```

同时新增显式分组：

```text
observation_meta
system2_supervision
system1_supervision
contract_meta
```

ms-swift trainer 继续使用旧 flat 字段；新增分组只用于 review、后续 parser/evaluator
迁移和 SessionTrace replay。

## 9. 实施顺序

当前基础实现只完成接口和最小运行能力：

1. 定义 Observation / Action / Tool / Step contracts。
2. 让 `AgentStage2Case` 适配 `NavObservation`。
3. exporter 双轨输出 flat fields 与 contract fields。
4. `user_interaction_tool` v1 返回 pending dialogue state。
5. `map_tool.get_route_context` v1 返回 text/structured ContextAsset。
6. `RouteSessionRuntime` 输出 `AgentStepResult` 视图。

后续再做：

- 真实 UI 等待用户输入。
- Map BEV image / waypoint image 作为 tool image。
- 多轮 bounded Contracted ReAct runtime。
- Session-level failure mining 与自动回流训练。
