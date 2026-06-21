# SideWalk Agentic Stage2 Eval Protocol

本文档定义 RouteServer 最终 Agentic Cold Start 协议。它直接对齐 ms-swift
`agent_template`：工具调用使用独立 `tool_call` role，工具返回使用独立
`tool_response` role，`tools` 在导出时写成 JSON 字符串。旧的 assistant content 内嵌
`<tool_call>...</tool_call>` 格式不再作为主流程保留。

## 1. 数据流

```text
sidewalk_agent_stage2_case_v1
  -> AgentRunner(rule / RouteServer teacher / external LVLM / E2E checkpoint)
  -> sidewalk_agent_trace_v2
  -> metrics
  -> sidewalk_agentic_stage2_sft_v1 ms-swift JSONL
```

核心模块：

```text
src/mcp_server/agent/
  schemas.py
  stage2.py
  tool_schema.py
  tool_parser.py
  tool_executor.py

src/mcp_server/eval/
  cases.py
  runners.py
  metrics.py
  exporters.py
```

## 2. 固定因果顺序

Stage2 的训练样本必须让工具上下文出现在导航 readout token 之前：

```text
user image/request
assistant <think><AgentState>...<ToolDecision>...</think>
tool_call
tool_response
user "Produce the final sidewalk navigation decision..." + [TRAJ_0]...[TRAJ_15]
assistant final CoT + <finish>
```

原因是 ms-swift SideWalk trajectory head 从 `[TRAJ_*]` token 的 hidden states 读出轨迹。
只有把这些 token 放在 `tool_response` 之后，trajectory head 才能因果看到工具结果。

## 3. AgentEvalCase JSONL

每行一个 JSON object：

```json
{
  "schema_version": "sidewalk_agent_stage2_case_v1",
  "case_id": "case-0001",
  "task_type": "sidewalk_navigation_agent",
  "user_request": "帮我找一个可以休息的地方",
  "observation": {
    "images": {
      "fpv": "/data/fpv.jpg",
      "bev": "/data/bev.jpg"
    },
    "current_location": "120.0,30.0",
    "origin": "120.0,30.0",
    "destination": "120.01,30.01"
  },
  "ground_truth": {
    "expected_tool_ops": ["search_poi"],
    "pixel_trajectory": [[0.5, 0.9], [0.55, 0.75]],
    "trajectory_fpv_t": null,
    "trajectory_ego_t": null,
    "semantic_segmentation_path": null
  },
  "split": "train",
  "locale": "zh-CN"
}
```

`observation.images` 支持 dict 或 list。dict 中推荐使用 `fpv`、`bev` 键，导出
ms-swift JSONL 时会按 FPV 在前、BEV 在后的顺序写入 `images`。

Phase1/E2E 已有监督字段继续放在 `ground_truth` 中，包括：

```text
pixel_trajectory
trajectory_fpv_t
trajectory_fpv_tv
trajectory_ego_t
trajectory_policy
semantic_segmentation_path
semantic_mask_path
```

这些字段不进入 LM messages；exporter 会透传给 SideWalk trainer 的 traj/seg loss。

## 4. Student-Visible Tools

只暴露三个元工具：

```text
map_tool
image_tool
user_interaction_tool
```

旧 RouteServer 内部的 intent parser、goal decomposer、POI recommendation、visual prompt
teacher 只能进入 `trace.metadata["teacher_internal"]`，不能作为学生可见工具暴露。

### 4.1 map_tool

用于地图查询、路线规划、route session 修改。支持的 `op`：

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

### 4.2 image_tool

用于对 FPV/BEV 图像做可审计操作，例如：

```text
crop
zoom_in
```

### 4.3 user_interaction_tool

用于多轮对话中的用户澄清和确认：

```text
ask_user
request_clarification
confirm_route_change
```

离线 evaluator 不真正等待用户输入，只记录结构化动作和 synthetic tool response。

## 5. AgentTrace

runner 输出统一 trace：

```json
{
  "schema_version": "sidewalk_agent_trace_v2",
  "case_id": "case-0001",
  "task_type": "sidewalk_navigation_agent",
  "request": {"...": "original AgentEvalCase"},
  "turns": [
    {"role": "user", "content": "<image><image>帮我找一个可以休息的地方"},
    {"role": "assistant", "content": "<think>...<AgentState>...<ToolDecision>...</think>"},
    {"role": "tool_call", "tool": "map_tool", "content": {"name": "map_tool", "arguments": {"op": "search_poi"}}},
    {"role": "tool_response", "tool": "map_tool", "content": {"tool": "map_tool", "success": true, "result": {}}},
    {"role": "user", "content": "Produce the final sidewalk navigation decision...\n\n[TRAJ_0] ... [TRAJ_15]"},
    {"role": "assistant", "content": "<think>...<Scene>...<ToolResultUse>...</think>\n<finish>{...}</finish>"}
  ],
  "supervision": {"expected_tool_ops": ["search_poi"]},
  "metrics": {}
}
```

最终 assistant 必须保留 Phase1 结构：

```text
<Scene>
<FPVScene>
<BEVContext>
<CrossView>
<Plan>
<ToolResultUse>
<finish>
```

如果输入 case 已有 Phase1 CoT response，Stage2 builder 会把 `<ToolResultUse>` 插入
`</think>` 之前，并补齐 `<finish>` 中的工具字段。没有 Phase1 response 时，会生成
一个 deterministic fallback CoT，便于 smoke/review。

## 6. ms-swift JSONL Export

`eval/exporters.py` 将高质量 `AgentTrace` 导出为：

```json
{
  "schema_version": "sidewalk_agentic_stage2_sft_v1",
  "stage": "agentic_cold_start",
  "task_type": "sidewalk_navigation_agent",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image><image>..."},
    {"role": "assistant", "content": "<think>...<ToolDecision>...</think>"},
    {"role": "tool_call", "content": "{\"name\":\"map_tool\",\"arguments\":{\"op\":\"search_poi\"}}"},
    {"role": "tool_response", "content": "{\"tool\":\"map_tool\",\"success\":true,\"result\":{}}"},
    {"role": "user", "content": "Produce the final sidewalk navigation decision...\n\n[TRAJ_0] ... [TRAJ_15]"},
    {"role": "assistant", "content": "<think>...<ToolResultUse>...</think>\n<finish>{...}</finish>"}
  ],
  "tools": "[{\"type\":\"function\",\"function\":{\"name\":\"map_tool\"}}]",
  "images": ["/data/fpv.jpg", "/data/bev.jpg"],
  "pixel_trajectory": [[0.5, 0.9], [0.55, 0.75]],
  "sidewalk_meta": {
    "schema_version": "sidewalk_agentic_stage2_sft_v1",
    "agent_trace_schema": "sidewalk_agent_trace_v2",
    "case_id": "case-0001"
  }
}
```

Loss 归属：

```text
assistant content: LM loss，包括 <think>、Phase1 CoT tags、<finish>
tool_call role: ms-swift agent_template 转成模型需要生成的 tool-call 文本，算 loss
tool_response role: 环境上下文，不算 loss
user/system/images/tools: 不算 loss
pixel_trajectory / trajectory_fpv_t / semantic_segmentation_path: SideWalk trainer 额外监督
```

## 7. Metrics

当前实现以下协议级指标：

```text
parse_success
tool_name_accuracy
tool_op_accuracy
argument_validity
finish_validity
stage2_role_order_valid
trajectory_readout_order_valid
final_cot_validity
finish_trajectory_consistency
route_update_success
```

重点指标：

- `stage2_role_order_valid`: 是否满足 `assistant -> tool_call -> tool_response -> readout user -> final assistant`。
- `trajectory_readout_order_valid`: `[TRAJ_0]...[TRAJ_15]` 是否出现在最后一个 `tool_response` 之后。
- `final_cot_validity`: 最终 assistant 是否包含 Phase1 CoT tags、`<ToolResultUse>` 和 `<finish>`。

这些指标只评价协议和可执行性，不评价自然语言解释质量。

## 8. RouteServer Teacher Wrapper

teacher wrapper 位于：

```text
src/mcp_server/teacher/trace_builder.py
```

它把旧 RouteServer 能力封装为最终 Stage2 trace：

```text
IntentParser
GoalDecomposer
DestinationRecommendationEngine
run_zero_shot_visual_prompt_pipeline
```

这些模块的输出只进入：

```text
trace.metadata["teacher_internal"]
```

学生可见部分只有三类元工具和最终导航 CoT。
