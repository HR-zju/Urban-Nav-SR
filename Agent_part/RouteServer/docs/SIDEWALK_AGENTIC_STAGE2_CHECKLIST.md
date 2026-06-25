# SideWalk Agentic Stage2 Implementation Checklist

本文档是 Urban-Nav-SR / RouteServer 与 ms-swift 之间的 Stage2 agentic cold-start
执行清单。它用于约束后续修改顺序，避免 schema、executor、runtime、训练和评测各自漂移。

## 0. 固定原则

- [x] Stage2 主训练格式保持 single-tool interaction，不混入多轮 SpotAgent loop。
- [x] `tools` 保持顶层 JSON string，由 ms-swift agent-template 注入，不写入 `<Scene>` 等 CoT tag。
- [x] `tool_call` 使用独立 role，不塞进 assistant content 作为最终训练格式。
- [x] `tool_response` 使用独立 role，原始工具返回不进入 assistant LM target。
- [x] `[TRAJ_0]...[TRAJ_15]` 必须位于最后一个 `tool_response` 之后。
- [x] 最终 assistant 只写工具结果使用结论：`<ToolResultUse>{...}</ToolResultUse>`。
- [x] no-tool replay 不写 `tool_call`、`tool_response`、`<ToolResultUse>`。
- [x] `AgentState.available_context` 只列运行时可观测上下文；`pixel_trajectory` / `trajectory_fpv_t` / `trajectory_ego_t` / seg label 只作为顶层监督字段，不进入 assistant target。

验收标准：

```bash
cd Agent_part/RouteServer
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_agent_eval_framework.py
```

## 1. RouteServer Schema / Builder / Exporter

- [x] `AgentStage2Case` 作为 Stage2 case 主类。
- [x] `AgentTrace` 支持 `tool_call` / `tool_response` 独立 role。
- [x] `AgenticStage2TraceBuilder` 支持 tool / no-tool / failed-tool 三类 trace。
- [x] `trace_to_ms_swift_stage2_row()` 导出 ms-swift agent-template row。
- [x] exporter 透传 `pixel_trajectory` / `trajectory_fpv_t` / `semantic_segmentation_path` 等额外监督字段。
- [x] `stage2_validator` 校验 tool/no-tool 两种 role sequence。

验收标准：

- exported row 的 `tools` 是 JSON 字符串。
- exported row 的 `tool_call.content` / `tool_response.content` 是 JSON 字符串。
- `finish.trajectory == pixel_trajectory`。
- tool 路径中 `[TRAJ_i]` 在 `tool_response` 后。
- no-tool 路径中 `[TRAJ_i]` 在 first assistant think 后。

## 2. ToolExecutor 逐工具单测

目标：把“协议支持工具”推进到“executor 真实分发可测”。所有测试默认使用 fake provider /
fake store，不访问 live map API。

| Tool op | 状态 | 必测 case | 验收标准 |
| --- | --- | --- | --- |
| `map_tool.search_poi` | Pending | success / provider empty / provider error / missing app_context | 返回 `provider/query/results/count`；失败时 `success=false` |
| `map_tool.create_route_session` | Pending | success / invalid origin-destination / provider route error | 返回 stable `session_id`，不污染真实 session |
| `map_tool.get_route_session` | Pending | existing / missing session / force_replan | snapshot 可序列化，失败结构化 |
| `map_tool.add_via_and_replan` | Pending | create then add_via / duplicate via / missing session | route session 更新后可 snapshot |
| `map_tool.replace_destination` | Pending | create then replace / provider failure | `route_update=replace_destination` 可被 `<ToolResultUse>` 摘要 |
| `map_tool.geocode` | Pending | success / empty / provider error | 坐标字段归一化 |
| `map_tool.reverse_geocode` | Pending | success / provider error | address metadata 可序列化 |
| `map_tool.search_category` | Pending | bbox success / empty / malformed bbox | bbox 参数校验稳定 |
| `image_tool.crop` | Pending | normalized bbox / pixel bbox / missing image / artifact_dir | 返回 crop artifact metadata 或文件 |
| `image_tool.zoom_in` | Pending | 与 crop alias 行为一致或明确区分 | 行为与 schema 文档一致 |
| `user_interaction_tool.request_clarification` | Pending | question / choices / proposal | 返回 `await_user=true` 离线响应 |
| `user_interaction_tool.ask_user` | Pending | question / choices | 返回 `await_user=true` 离线响应 |
| `user_interaction_tool.confirm_route_change` | Pending | proposal accept/reject payload | 返回可审计 proposal |

建议测试文件：

```text
Agent_part/RouteServer/tests/test_agent_tool_executor_real_ops.py
```

建议命令：

```bash
cd Agent_part/RouteServer
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_agent_tool_executor_real_ops.py
```

## 3. Stage2 Single-tool Runtime

- [x] 新增 `AgentToolRuntime`。
- [x] 支持解析 first assistant 的 `<tool_call>`。
- [x] 将 `<tool_call>` 规范化成独立 `tool_call` role。
- [x] 调用 `ToolExecutor.execute()`。
- [x] tool 失败时写入 `tool_response.success=false`，不中断 trace。
- [x] 追加 final user readout prompt 和 `[TRAJ_i]`。
- [x] 支持 tool / no-tool / failed-tool 三类 smoke。
- [x] 增加 runtime budget 配置：`max_tool_calls=1`、per-tool op allowlist。
- [x] 增加 runtime log：conversation、tool_call、tool_response、errors、artifacts。
- [x] 将 runtime trace logger 与 review artifact 目录打通。

当前测试：

```bash
cd Agent_part/RouteServer
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_agent_eval_framework.py
```

下一步建议测试文件：

```text
Agent_part/RouteServer/tests/test_agent_online_runtime.py
```

## 4. Online Model Runner

目标：把 fake first/final assistant output 替换成真实模型调用，但输出仍统一为
`AgentTrace`。

- [x] 定义 `ModelAdapter` 协议：
  - [x] `generate_pre_tool(messages, tools, images) -> str`
  - [x] `generate_final(messages, tools, images) -> str`
- [x] 新增 `OnlineAgentRunner`：
  - [x] 构造 system/user/images/tools。
  - [x] 第一次调用模型生成 `AgentState + ToolDecision + optional tool_call`。
  - [x] 调用 `AgentToolRuntime` 执行工具。
  - [x] 追加 `tool_response` 和 `[TRAJ_i]`。
  - [x] 第二次调用模型生成 final CoT + `<finish>`。
  - [x] 输出 `AgentTrace`、ms-swift row、validation report。
- [x] 支持 fake model adapter 测试。
- [x] 支持 ms-swift inference adapter。

建议文件：

```text
Agent_part/RouteServer/src/mcp_server/eval/online_agent_runner.py
Agent_part/RouteServer/tests/test_agent_online_runtime.py
```

验收标准：

- fake model 可以跑 tool/no-tool/failed-tool 三类 online trace。
- 所有输出通过 `validate_stage2_trace()`。
- 可导出 `trace_to_ms_swift_stage2_row()`。

## 5. `image_tool.inspect` 决策

- [x] 当前不暴露 `image_tool.inspect`，除非语义明确。
- [x] 方案 A：不实现 inspect，继续只暴露 `crop` / `zoom_in`。
- [x] 方案 B：暂缓；只有定义为真实外部视觉工具并返回结构化视觉结论后才可启用。
- [x] 明确禁止把 teacher LVLM 直接伪装成学生可调用工具。
- [x] 已补 schema / executor / exported tools 防回归单测，确保 `image_tool.inspect` 不进入 Stage2。

决策记录：

Stage2 选择方案 A。`image_tool` 当前只提供可审计、确定性的图像操作：
`crop` 与 `zoom_in`。`inspect` 含义不明确，容易被错误实现为 teacher LVLM
能力暴露给学生模型，导致训练目标把教师视觉推理能力建模成环境工具。若后续要启用
`image_tool.inspect`，必须先定义独立外部视觉 runtime、输入输出 schema、failure
语义、artifact 记录和 fake-provider 单测，再加入工具 schema。

验收标准：

- tool schema 与 executor op 集合一致。
- exported ms-swift `tools` 不包含 `inspect`。
- `ToolExecutor` 对 `image_tool.inspect` 返回 `invalid_tool_call`。
- ms-swift prompt 不把 `image_tool.inspect` 说明写入 CoT target。
- 如果未来启用 inspect，返回必须作为 `tool_response`，原始结果不进入 assistant target。

## 6. Phase1 Archive 到 Stage2 标注流水线

- [x] Phase1 archive adapter 透传原始 CoT / trajectory / seg supervision。
- [x] build script 支持 synthetic / fake executor 生成 review 样例。
- [x] 用真实 Phase1 archive 生成 Stage2 cases。
- [x] 使用 fake/real executor 生成 Stage2 traces。
- [x] 输出 ms-swift rows。
- [x] 输出 report，按 tool / no-tool / failed-tool / op 做 slice。
- [x] 人工 review 小样本。
- [x] 固定 train/val/held-out split。

当前实现：

- `eval/phase1_archive_adapter.py` 将真实 Phase1 archive/pre-archive JSONL 适配为
  `AgentStage2Case`，原始 CoT 保存到 `ground_truth.phase1_response`，轨迹和 seg
  监督字段原样透传。
- `eval/stage2_pipeline.py` 负责 deterministic hash split、slice report、review
  sample、review CSV index 和 split 文件写出。
- `tools/evaluation/build_sidewalk_agent_stage2_review_samples.py` 支持从真实
  Phase1 archive 一次性输出 cases、traces、ms-swift rows、report、review samples
  和 train/val/heldout split 文件。
- `--tool-context fake` 不访问外网，但会经过真实 `ToolExecutor -> map_ops /
  route_session_ops` 路径；`--tool-context provider` 是显式真实 provider 入口。
- final assistant 的 `<finish>.trajectory` 会以 `ground_truth.pixel_trajectory`
  覆盖 Phase1 CoT 中可能四舍五入的 trajectory，保证 LM target 与 trajectory
  head 监督严格对齐。

验收输出：

```text
sidewalk_agent_stage2_cases_*.jsonl
sidewalk_agent_stage2_traces_*.jsonl
sidewalk_agentic_stage2_ms_swift_*.jsonl
sidewalk_agent_stage2_report_*.json
```

当前 smoke：

```bash
cd Agent_part/RouteServer
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_stage2_pipeline.py tests/test_agent_eval_framework.py

PYTHONPATH=src python tools/evaluation/build_sidewalk_agent_stage2_review_samples.py \
  --phase1-archive-jsonl /path/to/phase1_archive/manifests/train.jsonl \
  --limit 3 \
  --execute-tools \
  --tool-context fake \
  --stage2-cases-out /tmp/sidewalk_stage2_step6_smoke/sidewalk_agent_stage2_cases_smoke.jsonl \
  --traces-out /tmp/sidewalk_stage2_step6_smoke/sidewalk_agent_stage2_traces_smoke.jsonl \
  --ms-swift-out /tmp/sidewalk_stage2_step6_smoke/sidewalk_agentic_stage2_ms_swift_smoke.jsonl \
  --report-out /tmp/sidewalk_stage2_step6_smoke/sidewalk_agent_stage2_report_smoke.json \
  --split-output-dir /tmp/sidewalk_stage2_step6_smoke/splits \
  --review-samples-out /tmp/sidewalk_stage2_step6_smoke/sidewalk_agent_stage2_review_samples_smoke.jsonl \
  --review-index-out /tmp/sidewalk_stage2_step6_smoke/sidewalk_agent_stage2_review_index_smoke.csv
```

## 7. ms-swift 训练接入

- [x] Stage2 prompt template：`agentic_stage2_cot_trajectory`。
- [x] parser 支持 `AgentState` / `ToolDecision` / `ToolResultUse`。
- [x] tokenize / forward smoke。
- [x] 使用 RouteServer 真实 Stage2 rows 跑 dataset smoke。
- [x] 使用 UrbanNav mini Stage2 rows 跑 frozen-forward framework smoke，不加载 base LLM、不建 optimizer、不更新参数。
- [x] Stage2-A 小数据训练入口与 dry-run command smoke：`--sidewalk_phase phase0`。
- [ ] Stage2-A 真实小数据训练：待 frozen-forward / held-out smoke 完成后，再固定 `BASE_MODEL`、train/val split、输出目录和 GPU 环境。
- [x] Stage2-A held-out 对比入口与 evaluator smoke。
- [ ] Stage2-A 真实 held-out 对比：待 Stage2-A checkpoint。
- [ ] Stage2-B 打开 trajectory loss：
  - [x] 入口与 dry-run command smoke：`--sidewalk_phase phase1`
  - [x] 入口与 dry-run command smoke：`--sidewalk_phase1_supervision lm_traj`
  - [x] frozen-forward smoke 验证 `[TRAJ_i]` readout 与 trajectory head forward wiring。
  - [ ] 真实训练：待固定 `BASE_MODEL` / Stage2-A checkpoint、train/val split、输出目录和 GPU 环境。
- [x] Stage2-B held-out 对比入口与 evaluator smoke。
- [ ] Stage2-B 真实 held-out 对比：待 Stage2-B checkpoint。

当前 ms-swift 入口：

```text
swift/sidewalk_nav/cli/smoke_agentic_stage2.py
examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh dataset_smoke
examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh frozen_forward_smoke
examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh train_stage2a
examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh train_stage2b
examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh benchmark_stage2a
examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh benchmark_stage2b
```

当前 smoke：

```bash
cd /home/ubuntu/WorkSpace/project/SR/Code/ms-swift
source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate internnav

PYTHONPYCACHEPREFIX=/tmp/pycache_ms_swift_stage2_step7 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
pytest -q -p no:cacheprovider \
  tests/test_sidewalk_agentic_stage2_dataset_smoke.py \
  tests/test_sidewalk_agentic_stage2_prompt.py

STAGE2_MANIFEST=/tmp/sidewalk_stage2_step6_smoke/sidewalk_agentic_stage2_ms_swift_smoke.jsonl \
OUTPUT_ROOT=/tmp/ms_swift_stage2_step7_smoke \
SMOKE_LIMIT=3 \
bash examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh dataset_smoke

STAGE2_MANIFEST=/tmp/sidewalk_stage2_urbannav_mini/sidewalk_agentic_stage2_urbannav_mini.jsonl \
OUTPUT_ROOT=/tmp/sidewalk_stage2_urbannav_mini/ms_swift \
SMOKE_LIMIT=6 \
bash examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh frozen_forward_smoke
```

RouteServer 真实 Stage2 row smoke 结果：

- `valid_rows=3`
- `decision_agentic_summary.metrics.protocol_parse_success=1.0`
- `decision_agentic_summary.metrics.should_call_tool_match=1.0`
- `final_agentic_summary.metrics.final_cot_tag_validity=1.0`
- `final_agentic_summary.metrics.tool_result_use_valid=1.0`
- `final_text_traj_summary.ade=0.0`
- `final_text_traj_summary.fde=0.0`

UrbanNav mini frozen-forward smoke 结果：

- `num_rows=6`
- `tool_policy_counts={"tool":4,"no_tool":1,"failed_tool":1}`
- `validation.valid_rows=6`
- `frozen_forward.valid_rows=6`
- `parameter_update_attempted=false`
- `optimizer_step_attempted=false`
- `base_llm_loaded=false`
- 所有 tool / failed-tool 样本满足 `first_traj_token_index > tool_response_token_index`
- 每条样本 frozen trajectory head 输出 `trajectory_output_shape=[1,3,16,2]`

验收指标：

- parse success
- tool decision consistency
- tool op accuracy
- final CoT validity
- finish trajectory consistency
- ADE / FDE
- tool/no-tool slice
- failed-tool recovery slice

## 8. Urban-Nav-SR Benchmark / 推理闭环

- [ ] 固定 base checkpoint。
- [ ] 固定 held-out benchmark manifest。
- [ ] 跑 base checkpoint zero-shot benchmark。
- [ ] 接 Stage2-A checkpoint 推理。
- [ ] 接 Stage2-B checkpoint 推理。
- [ ] 同一 held-out 集合前后对比。
- [ ] 输出 per-case trace、summary report、error report、artifact index。

验收标准：

- 同一 case_id 可对齐 base / Stage2-A / Stage2-B 输出。
- route action、tool decision、trajectory 指标可聚合。
- failed case 可以回放完整 messages / tool_response / artifacts。

## 9. 最终推理接口

- [ ] RouteServer online runtime 输入：
  - [ ] images
  - [ ] user request
  - [ ] route state
  - [ ] available tools
- [ ] RouteServer online runtime 输出：
  - [ ] final assistant text
  - [ ] parsed `<finish>`
  - [ ] trajectory
  - [ ] tool_call / tool_response trace
  - [ ] artifacts
  - [ ] validation report
- [ ] ms-swift `infer.py` 输出可以接入 RouteServer runtime。
- [ ] RouteServer runtime 可以保存 ms-swift-compatible replay row。

验收标准：

- 单 case CLI 可以端到端运行。
- batch benchmark 可以端到端运行。
- 输出 trace 可以复用现有 validator/exporter/metrics。

## 10. 当前下一步

优先做第 2 节：`ToolExecutor` 每个 op 的 fake-provider 单测。

完成第 2 节后再做：

1. 第 3 节 runtime budget / logger。
2. 第 4 节 `OnlineAgentRunner + ModelAdapter`。
3. 第 6 节真实 Phase1 archive Stage2 数据生成。
4. 第 7-8 节训练与前后对比。
