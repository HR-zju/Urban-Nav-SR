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
  phase1_archive_adapter.py
  stage2_validator.py

src/mcp_server/teacher/
  agentic_stage2_trace_builder.py
  trace_builder.py
```

## 2. 固定因果顺序

Stage2 的训练样本必须让工具上下文出现在导航 readout token 之前。tool 路径：

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
显式 no-tool replay 使用同一 schema 的短序列：

```text
user image/request
assistant <think><AgentState>...<ToolDecision>{"should_call_tool":false,...}</ToolDecision></think>
user "Produce the final sidewalk navigation decision..." + [TRAJ_0]...[TRAJ_15]
assistant final CoT + <finish>
```

no-tool 样本必须通过 `observation.force_no_tool=true`、`ground_truth.tool_policy="no_tool"`
或 `metadata.tool_policy="no_tool"` 显式标注，构建脚本负责控制 replay 比例。

正式 Python API：

```text
AgentStage2Case
AgenticStage2TraceBuilder.build_trace()
trace_to_ms_swift_stage2_row()
write_ms_swift_stage2_jsonl()
validate_stage2_case()
validate_stage2_trace()
validate_ms_swift_stage2_row()
```

旧的 `AgentEvalCase`、`trace_to_ms_swift_row()`、`write_ms_swift_jsonl()` 只作为兼容入口保留。

## 2.1 Phase1 Archive 接入

真实 Phase1 archive 通过以下 adapter 接入 Stage2：

```text
src/mcp_server/eval/phase1_archive_adapter.py
phase1_archive_row_to_stage2_case()
load_phase1_archive_stage2_cases()
```

支持两类输入：

- `sidewalk_phase1_archive/manifests/*.jsonl`：`messages/images/pixel_trajectory/sidewalk_meta`。
- 归档前 `phase1_cot_*.jsonl`：`prompt/response/fpv_image_path/bev_image_path`。

adapter 只做字段迁移：

- Phase1 assistant CoT 保存为 `ground_truth.phase1_response`。
- `pixel_trajectory`、`trajectory_fpv_t`、`trajectory_fpv_tv`、`trajectory_ego_t`、`trajectory_policy` 原样进入 `ground_truth`。
- `semantic_segmentation_path`、`semantic_mask_path` 等 seg 监督路径原样进入 `ground_truth`。
- 图片路径默认按 Phase1 manifest 所在目录解析相对路径。

注意：adapter 不重写轨迹、不重新采样标签。Stage2 的数学约束是把工具结果
`C_tool` 插入在 readout token 之前，让 `[TRAJ_i]` hidden states 能看到工具上下文；
监督目标 `Y` 仍然是 Phase1 archive 已验证过的轨迹/seg 标签。

## 3. AgentStage2Case JSONL

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
    "tool_policy": "tool",
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

no-tool replay 推荐写法：

```json
{
  "observation": {"force_no_tool": true},
  "ground_truth": {"tool_policy": "no_tool"}
}
```

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
  "request": {"...": "original AgentStage2Case"},
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

tool 路径最终 assistant 必须保留 Phase1 结构并追加工具使用结论：

```text
<Scene>
<FPVScene>
<BEVContext>
<CrossView>
<Plan>
<ToolResultUse>
<finish>
```

其中 `<ToolResultUse>` 是 teacher 对工具返回的可学习摘要，不是原始 `tool_response`
的复制。字段包括：

```json
{
  "tool_used": true,
  "tool": "map_tool",
  "op": "search_poi",
  "success": false,
  "used_for": "final_navigation_decision",
  "result_summary": "map_tool.search_poi failed: offline map index unavailable",
  "route_update": "none",
  "navigation_effect": "tool_failed_use_fpv_bev_and_existing_route_context",
  "failure_recovery": "fallback_to_existing_route_state_and_visible_fpv_bev_context"
}
```

`failure_recovery` 只在 `success=false` 的 failed-tool trace 中出现。no-tool 路径不写
`<ToolResultUse>`，最终 assistant 只保留 Phase1 tags 和 `<finish>`，并在
`<finish>` 中写入 `"tool_used": false`。

如果输入 case 已有 Phase1 CoT response，Stage2 builder 会把 `<ToolResultUse>` 插入
`</think>` 之前；no-tool 路径不会插入。两条路径都会补齐 `<finish>` 中的 Stage2
字段。没有 Phase1 response 时，会生成一个 deterministic fallback CoT，便于
smoke/review。

## 6. ms-swift JSONL Export

`eval/exporters.py::trace_to_ms_swift_stage2_row()` 将高质量 `AgentTrace` 导出为：

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
tool_result_use_validity
tool_decision_consistency
unnecessary_tool_call_rate
finish_trajectory_consistency
route_update_success
```

重点指标：

- `stage2_role_order_valid`: 是否满足 tool 序列或显式 no-tool 序列。
- `trajectory_readout_order_valid`: tool 路径中 `[TRAJ_0]...[TRAJ_15]` 是否出现在最后一个 `tool_response` 之后；no-tool 路径中是否出现在 `<ToolDecision>` 之后。
- `final_cot_validity`: 最终 assistant 是否包含 Phase1 CoT tags 和 `<finish>`；tool 路径还必须包含 `<ToolResultUse>`。
- `tool_result_use_validity`: tool 路径中 `<ToolResultUse>.tool_used=true`；no-tool 路径中不出现 `<ToolResultUse>` 或显式 `tool_used=false`。
- `tool_decision_consistency`: `<ToolDecision>.should_call_tool` 是否与后续是否存在 `tool_call` role 一致。
- `unnecessary_tool_call_rate`: 对显式 no-tool 样本，是否仍然调用了工具。

这些指标只评价协议和可执行性，不评价自然语言解释质量。

## 8. RouteServer Teacher Wrapper

Stage2 trace 拼装入口位于：

```text
src/mcp_server/teacher/agentic_stage2_trace_builder.py
```

它只接收 case、tool call、tool response，然后输出固定顺序 trace。RouteServer teacher
wrapper 位于：

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

## 9. Validator 与 Review Build

独立 validator 位于：

```text
src/mcp_server/eval/stage2_validator.py
```

验证范围：

```text
case schema/version/basic fields
trace tool/no-tool role order and readout token order
tool_call/tool_response JSON shape
final assistant Phase1 tags + optional <ToolResultUse> + <finish>
ms-swift row tools/tool_call/tool_response JSON string type
```

生成 review 样例：

```bash
PYTHONPATH=src python tools/evaluation/build_sidewalk_agent_stage2_review_samples.py
```

默认输入输出：

```text
examples/agent_eval/sidewalk_agent_stage2_cases_review.jsonl
examples/agent_eval/sidewalk_agent_stage2_traces_review.jsonl
examples/agent_eval/sidewalk_agentic_stage2_ms_swift_review.jsonl
examples/agent_eval/sidewalk_agent_stage2_report_review.json
```

接真实 Phase1 archive 并使用 ToolExecutor smoke：

```bash
PYTHONPATH=src python tools/evaluation/build_sidewalk_agent_stage2_review_samples.py \
  --phase1-archive-jsonl /path/to/sidewalk_phase1_archive/manifests/train.jsonl \
  --limit 16 \
  --execute-tools \
  --tool-context fake \
  --stage2-cases-out examples/agent_eval/sidewalk_phase1_stage2_cases_smoke.jsonl \
  --traces-out examples/agent_eval/sidewalk_phase1_stage2_traces_smoke.jsonl \
  --ms-swift-out examples/agent_eval/sidewalk_phase1_agentic_stage2_ms_swift_smoke.jsonl \
  --report-out examples/agent_eval/sidewalk_phase1_stage2_report_smoke.json \
  --split-output-dir examples/agent_eval/sidewalk_phase1_stage2_splits_smoke \
  --review-samples-out examples/agent_eval/sidewalk_phase1_stage2_review_samples_smoke.jsonl \
  --review-index-out examples/agent_eval/sidewalk_phase1_stage2_review_index_smoke.csv
```

`--tool-context fake` 使用本地假地图客户端，但仍走真实 `ToolExecutor -> map_ops /
route_session_ops` 调用路径，适合离线 smoke。`--tool-context provider` 会创建真实
RouteServer map provider，可能依赖网络和 provider API key。默认不传
`--execute-tools` 时仍使用 synthetic tool response，便于协议 review 和单测。

第 6 步完整流水线会额外输出：

- `--stage2-cases-out`: Phase1 archive 适配后的 `sidewalk_agent_stage2_case_v1`。
- `--split-output-dir`: `train/val/heldout` 三组 cases/traces/ms-swift rows。
- `--review-samples-out`: 按 `(split, tool_policy, op)` 抽样的人工 review JSONL。
- `--review-index-out`: 便于人工筛选的 CSV index。
- report 中的 `slices` 会按 split、tool/no-tool/failed-tool、tool name、op、
  validation、Phase1 intent 聚合计数和协议 metrics。

注意：真实 Phase1 CoT 中的 `<finish>.trajectory` 可能经过文本四舍五入。Stage2
最终 assistant 会以 `ground_truth.pixel_trajectory` 覆盖 `<finish>.trajectory`，
保证 LM 监督文本、ms-swift row 的 `pixel_trajectory`、trajectory head 监督三者
严格一致。
