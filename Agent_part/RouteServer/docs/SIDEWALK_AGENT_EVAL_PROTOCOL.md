# SideWalk Agent Eval Protocol

本文档定义 RouteServer 第一版 Agent/Eval 协议。目标是先得到可记录、可回放、
可评估、可导出训练数据的最小闭环；不在协议层调用外部 LVLM，也不把旧 teacher
中的 LLM/LVLM 能力暴露成学生模型可调用工具。

## 1. 数据流

```text
sidewalk_agent_eval_case_v1
  -> AgentRunner(rule / teacher / external LVLM / E2E checkpoint)
  -> sidewalk_agent_trace_v1
  -> metrics
  -> ms-swift JSONL exporter
```

核心模块：

```text
src/mcp_server/agent/
  schemas.py
  tool_schema.py
  tool_parser.py
  tool_executor.py

src/mcp_server/eval/
  cases.py
  runners.py
  metrics.py
  exporters.py
```

## 2. AgentEvalCase JSONL

每行一个 JSON object：

```json
{
  "schema_version": "sidewalk_agent_eval_case_v1",
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
    "pixel_trajectory": [[0.5, 0.9], [0.55, 0.75]]
  },
  "split": "eval",
  "locale": "zh-CN"
}
```

`observation.images` 支持 dict 或 list。dict 中推荐使用 `fpv`、`bev` 键，导出
ms-swift JSONL 时会按 FPV 在前、BEV 在后的顺序写入 `images`。

## 3. Student-Visible Tools

第一版只暴露三个元工具：

```text
map_tool
image_tool
user_interaction_tool
```

### 3.1 map_tool

用于地图查询、路线规划、route session 修改。

支持的 `op`：

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

`map_tool` 内部由 `tool_executor.py` 调用 `core.map_ops` 和
`core.route_session_ops`。这里不包含 POI 排序、意图解析、目标推荐等 teacher
LVLM 能力；这些能力应该由 E2E LVLM 学习，或在 teacher runner 中生成监督 trace。

### 3.2 image_tool

用于对 FPV/BEV 图像做可审计操作，例如：

```text
crop
zoom_in
```

第一版 executor 支持 crop/zoom-in 形态的 bbox 裁剪；如果没有执行环境，也可以只把
tool call 记录进 trace，用于后续标注和离线回放。

### 3.3 user_interaction_tool

用于多轮对话中的用户澄清和确认：

```text
ask_user
request_clarification
confirm_route_change
```

离线 evaluator 不真正等待用户输入，只记录结构化动作：

```json
{
  "await_user": true,
  "question": "...",
  "choices": [],
  "proposal": {}
}
```

## 4. Assistant Output Blocks

推荐 assistant 输出同级 block，不把工具调用放进 `<think>`：

```text
<reasoning_summary>{"decision":"tool_call"}</reasoning_summary>
<tool_call>{"tool":"map_tool","op":"search_poi","query":"rest stop","location":"120.0,30.0"}</tool_call>
```

或最终输出：

```text
<finish>{"route_action":"use_tool_result","safe_direction":"front_left"}</finish>
```

`<tool_response>` 是环境/tool 返回给模型的上下文，不是 assistant 需要预测的目标。
exporter 会把 tool response 写成 `role="tool"` 的消息：

```text
<tool_response>{"tool":"map_tool","success":true,"result":{...}}</tool_response>
```

`tool_parser.py` 会显式拒绝把 `<tool_call>` 或 `<finish>` 放在 `<think>` /
`<reasoning_summary>` 内部的输出，避免训练时把可执行动作藏进自由推理文本。

## 5. AgentTrace

runner 输出统一 trace：

```json
{
  "schema_version": "sidewalk_agent_trace_v1",
  "case_id": "case-0001",
  "task_type": "sidewalk_navigation_agent",
  "request": {"...": "original AgentEvalCase"},
  "turns": [
    {"role": "user", "content": "<image><image>帮我找一个可以休息的地方"},
    {"role": "assistant", "content": "<tool_call>{...}</tool_call>"},
    {"role": "tool_response", "tool": "map_tool", "content": {"success": true}},
    {"role": "assistant", "content": "<finish>{...}</finish>"}
  ],
  "supervision": {"expected_tool_ops": ["search_poi"]},
  "metrics": {}
}
```

trace 是后续人工 review、teacher distillation、SFT 数据导出的统一中间产物。

## 6. Metrics

第一版实现以下指标：

```text
parse_success
tool_name_accuracy
tool_op_accuracy
argument_validity
finish_validity
route_update_success
```

含义：

- `parse_success`: assistant 是否输出可解析的 `<tool_call>` 或 `<finish>`。
- `tool_name_accuracy`: 工具名是否正确；有 GT 时比较序列，否则检查是否属于允许工具。
- `tool_op_accuracy`: 有 `expected_tool_ops` 时比较 op 序列。
- `argument_validity`: tool call 参数是否通过轻量 schema 校验。
- `finish_validity`: 是否输出合法 JSON `<finish>`。
- `route_update_success`: route-session 类 tool response 是否成功返回 route/session 结果。

这些指标只评价协议和可执行性，不评价自然语言解释质量。

## 7. ms-swift JSONL Export

`eval/exporters.py` 将高质量 `AgentTrace` 导出为：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image><image>..."},
    {"role": "assistant", "content": "<tool_call>{...}</tool_call>"},
    {"role": "tool", "content": "<tool_response>{...}</tool_response>"},
    {"role": "assistant", "content": "<finish>{...}</finish>"}
  ],
  "tools": [{"type": "function", "function": {"name": "map_tool"}}],
  "images": ["/data/fpv.jpg", "/data/bev.jpg"],
  "pixel_trajectory": [[0.5, 0.9], [0.55, 0.75]],
  "sidewalk_meta": {
    "agent_trace_schema": "sidewalk_agent_trace_v1",
    "case_id": "case-0001",
    "task_type": "sidewalk_navigation_agent"
  }
}
```

这样后续可以直接比较：

```text
RouteServer teacher
rule baseline
external LVLM
本项目 E2E LVLM checkpoint
```

也可以把 teacher/human trace 转成统一 SFT 数据。

## 8. RouteServer Teacher Wrapper

第一版 teacher wrapper 位于：

```text
src/mcp_server/teacher/trace_builder.py
```

它把旧 RouteServer 能力封装为 `AgentTrace`：

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

不会进入学生可见 `tools` schema。学生模型仍然只学习：

```text
map_tool
image_tool
user_interaction_tool
```

人工 review 小样例位于：

```text
examples/agent_eval/sidewalk_agent_eval_cases_review.jsonl
```

可用以下命令生成 teacher trace 和 ms-swift JSONL：

```bash
PYTHONPATH=src python tools/evaluation/build_sidewalk_agent_review_samples.py
```

默认使用 `RouteServerTeacherRunner(use_llm=False)`，不会调用外部 LLM/LVLM 或地图服务。
