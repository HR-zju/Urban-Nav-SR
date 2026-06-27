# SideWalk Agentic Stage2 Implementation Checklist

本文档是 Urban-Nav-SR / RouteServer 与 ms-swift 之间的 Stage2 agentic cold-start
执行清单。它用于约束后续修改顺序，避免 schema、executor、runtime、训练和评测各自漂移。

> 实现注释：本文档按完整闭环排序：先固定协议，再做数据标注，再接 ms-swift
> 训练，最后做 benchmark 和在线推理。每一节都对应一类可验证产物，避免直接进入训练后
> 才发现 role sequence、loss mask 或 trajectory readout 位置错位。

## 0. 固定原则

> 实现注释：这一节是 Stage2 的“不可变协议”。核心约束来自自回归模型因果性：
> `[TRAJ_i]` 的 hidden state 只能看到它之前的 token，所以工具交互必须先发生，
> `tool_response` 必须先进入上下文，最后才放导航 readout token 和 final assistant target。

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

> 实现注释：RouteServer 是 teacher trace 和 review trace 的来源，内部保存 canonical
> 审计顺序；exporter 再把 trace 转为 ms-swift agent-template row。这里不要把
> ms-swift 的模板兼容细节反向污染 RouteServer schema。

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

> 实现注释：Stage2 框架已经支持工具 role，但“工具可训练”不等于“真实工具可靠”。
> 本节要求每个暴露给模型的 op 都有 fake-provider 单测，保证 executor 分发、失败返回、
> artifact 和 session 状态可审计；默认不访问 live API，避免测试不稳定。

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

> 实现注释：`AgentToolRuntime` 是 SpotAgent-style runtime policy 的最小可控版本：
> 只执行第一轮里最多一个工具，并把失败写成 `tool_response.success=false`。这样既能训练
> “何时调用工具/如何使用工具结果”，又不会让多轮工具循环破坏 Stage2 trajectory head 的
> 因果位置。

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

> 实现注释：`OnlineAgentRunner` 把 teacher replay 推进到真实在线评测：第一次模型生成
> 工具决策，runtime 执行工具，第二次模型在 tool result 和 `[TRAJ_i]` 可见后生成
> final navigation。输出仍统一为 `AgentTrace`，所以可以复用 validator/exporter/metrics。

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

> 实现注释：这里的重点是防止数据泄漏。`inspect` 如果只是 teacher LVLM 的别名，
> 模型会学到“调用一个不存在的视觉专家工具”，而不是学习 FPV/BEV 自身推理。因此当前只暴露
> 可审计的图像操作，真正的 inspect 必须等外部视觉 runtime 和失败语义都明确后再启用。

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

> 实现注释：Stage2 不重做 Phase1 的 BEV/FPV 轨迹监督，而是在已有 E2E CoT、trajectory、
> seg supervision 基础上补 agentic teacher trace。最终 row 同时服务两类 loss：
> assistant/tool_call 文本 loss，以及顶层 `pixel_trajectory` / segmentation 的额外监督 loss。

- [x] Phase1 archive adapter 透传原始 CoT / trajectory / seg supervision。
- [x] build script 支持 synthetic / fake executor 生成 review 样例。
- [x] 用真实 Phase1 archive 生成 Stage2 cases。
- [x] 使用 fake/real executor 生成 Stage2 traces。
- [x] 输出 ms-swift rows。
- [x] 输出 report，按 tool / no-tool / failed-tool / op 做 slice。
- [x] 人工 review 小样本。
- [x] 固定 train/val/held-out split。

当前实现：

> 实现注释：`--tool-context fake` 是“真实 executor 路径 + fake provider 数据”，适合
> 批量标注前 smoke；`--tool-context provider` 才是真实地图 provider 入口，运行前需要固定
> key、配额、城市范围和失败重试策略。

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

> 实现注释：ms-swift 侧只负责“如何训练/推理这条 row”，不重新定义 RouteServer
> canonical schema。对 tool trace，训练前会生成 template-compatible view：把 final
> readout 合并到 preceding `tool_response`，保持 `tool_response -> [TRAJ_i] -> final assistant`
> 的因果顺序，同时满足 ms-swift agent-template 的 tool/assistant 配对要求。

- [x] Stage2 prompt template：`agentic_stage2_cot_trajectory`。
- [x] parser 支持 `AgentState` / `ToolDecision` / `ToolResultUse`。
- [x] tokenize / forward smoke。
- [x] 使用 RouteServer 真实 Stage2 rows 跑 dataset smoke。
- [x] 使用 UrbanNav mini Stage2 rows 跑 frozen-forward framework smoke，不加载 base LLM、不建 optimizer、不更新参数。
- [x] Stage2-A 小数据训练入口与 dry-run command smoke：`--sidewalk_phase phase0`。
- [x] Stage2-A 安全真实训练 smoke 入口：`train_stage2a_smoke`，默认 `max_steps=1`、`max_length=4096`、`lora_rank=2`、`target_modules=q_proj v_proj`、小样本采样，避免 OOM。
- [x] Stage2-A CUDA 容器真实 smoke：UrbanNav mini train/val rows + Qwen3.5-4B + LoRA 1-step 已通过。
- [x] Stage2-A 真实小数据训练：UrbanNav mini train/val rows + Qwen3.5-4B + LoRA 4-step 已通过，不再使用 smoke sample cap。
- [x] Stage2-A held-out 对比入口与 evaluator smoke。
- [x] Stage2-A 真实 held-out 对比：同一 UrbanNav mini held-out split 已完成 base / Stage2-A 对比。
- [ ] Stage2-B 打开 trajectory loss：
  - [x] 入口与 dry-run command smoke：`--sidewalk_phase phase1`
  - [x] 入口与 dry-run command smoke：`--sidewalk_phase1_supervision lm_traj`
  - [x] frozen-forward smoke 验证 `[TRAJ_i]` readout 与 trajectory head forward wiring。
  - [x] 安全真实训练 smoke 入口：`train_stage2b_smoke`，使用 smoke-only `residual_points` 小 head 跑 1-step 验证。
  - [x] Stage2-B CUDA 容器真实 smoke：UrbanNav mini train/val rows + LM/traj loss 1-step 已通过。
  - [x] 真实训练：基于 Stage2-A checkpoint，使用生产 `causal_multi_query_curve` head、LM + trajectory loss 4-step 已通过。
- [x] Stage2-B held-out 对比入口与 evaluator smoke。
- [x] Stage2-B 真实 held-out 对比：同一 UrbanNav mini held-out split 已完成，trajectory head 输出可评估。

当前 Step7 状态：

> 实现注释：当前 mini 结果只证明工程闭环，不代表模型能力结论。特别是 held-out 只有
> 1 条 failed-tool case，文本指标为 0 时应进入 error review，而不是直接否定训练入口。

- 工程闭环已完成：dataset smoke、frozen-forward、Stage2-A train、Stage2-A held-out、
  Stage2-B train、Stage2-B held-out、compare 均已跑通。
- 当前 split 是 UrbanNav mini：train=4、val=1、held-out=1，held-out 单条为
  failed-tool case。该结果只证明训练/推理/评测框架可运行，不作为研究级模型质量指标。
- 为避免 RTX 3090 OOM，正式 mini 训练保持 `target_modules=q_proj v_proj`、
  `lora_rank=2`、`max_length=4096`、`per_device_train_batch_size=1`。
- Stage2-B 使用生产 `causal_multi_query_curve` trajectory head，但关闭
  `sidewalk_train_traj_token_embedding`，避免把整张 embedding 表加入可训练参数。
- 训练日志中 `grad_norm=nan` 仍需后续排查；当前 `loss`、`eval_loss`、
  `traj_loss` 均为有限值，checkpoint 可保存并可用于推理。

当前 ms-swift 入口：

```text
swift/sidewalk_nav/cli/smoke_agentic_stage2.py
examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh dataset_smoke
examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh frozen_forward_smoke
examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh train_stage2a_smoke
examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh train_stage2a
examples/sidewalk_nav/agentic_stage2/run_agentic_stage2_pipeline.sh train_stage2b_smoke
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

当前 CUDA 训练 smoke 使用容器而不是宿主 `internnav`：

```text
container: sidewalk-ms-swift
container torch: 2.10.0+cu128
container cuda_available: true
host internnav torch: 2.11.0+cu130
host internnav cuda_available: false on current 570/CUDA 12.8 driver
```

已通过的真实 smoke：

```text
Stage2-A:
  action: train_stage2a_smoke
  model: /workspace/ms-swift/.cache/ms-swift/modelscope/models/Qwen/Qwen3___5-4B
  data: /tmp/sidewalk_stage2_urbannav_mini/train.jsonl#2, val.jsonl#1
  trainable_parameters: 0.2294M
  memory: 13.93GiB
  checkpoint: /tmp/ms_swift_stage2a_real_smoke_container_4096/stage2a_agentic_text_smoke/.../checkpoint-1

Stage2-B:
  action: train_stage2b_smoke
  smoke head: residual_points
  train_traj_token_embedding: false
  trainable_parameters: 6.8675M
  memory: 18.00GiB
  traj_loss: train=0.01379, eval=0.007365
  checkpoint: /tmp/ms_swift_stage2b_real_smoke_container_small4/stage2b_agentic_traj_smoke/.../checkpoint-1
```

已通过的完整 mini 闭环：

```text
Stage2-A mini:
  action: train_stage2a
  model: /workspace/ms-swift/.cache/ms-swift/modelscope/models/Qwen/Qwen3___5-4B
  data: /tmp/sidewalk_stage2_urbannav_mini/train.jsonl#4, val.jsonl#1
  trainable_parameters: 0.2294M
  memory: 14.01GiB
  train_loss: 1.647
  eval_loss: 1.712
  checkpoint: /tmp/ms_swift_stage2_full_mini/stage2a_agentic_text/v0-20260626-154725/checkpoint-4

Stage2-B mini:
  action: train_stage2b
  init_adapter: /tmp/ms_swift_stage2_full_mini/stage2a_agentic_text/v0-20260626-154725/checkpoint-4
  head: causal_multi_query_curve
  train_traj_token_embedding: false
  memory: 18.51GiB
  train_loss: 1.666
  eval_loss: 1.726
  traj_loss: train≈0.2087-0.2123, eval=0.2123
  checkpoint: /tmp/ms_swift_stage2_full_mini/stage2b_agentic_traj/v0-20260626-155228/checkpoint-4

Held-out mini benchmark:
  split: /tmp/sidewalk_stage2_urbannav_mini/heldout.jsonl#1
  base/stage2a/stage2b text parse metrics: 0.0 on the single failed-tool held-out case
  stage2b trajectory head parse_success: 1.0
  stage2b trajectory head strict_success: 1.0
  stage2b trajectory head ADE/FDE: 0.0852 / 0.1290
  comparison_dir: /tmp/ms_swift_stage2_full_mini/comparison
```

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

> 实现注释：第 8 节解决“同一 held-out 上怎么公平比较 base / Stage2-A / Stage2-B”。
> RouteServer 不在这里做模型推理细节，而是收集 ms-swift 输出，按 case_id 对齐成
> per-case report，并把 parse/tool/trajectory 失败整理成 error report 供人工复盘。

- [x] 固定 base checkpoint。
- [x] 固定 held-out benchmark manifest。
- [x] 跑 base checkpoint zero-shot benchmark。
- [x] 接 Stage2-A checkpoint 推理。
- [x] 接 Stage2-B checkpoint 推理。
- [x] 同一 held-out 集合前后对比。
- [x] 输出 per-case trace、summary report、error report、artifact index。

当前实现：

> 实现注释：collector 的输入是 ms-swift 已经落盘的 refs/predictions/metrics。
> 这样 benchmark packaging 可以离线复跑，不依赖 GPU、地图 API 或 LLM key。

- `eval/agentic_stage2_benchmark.py` 是 RouteServer 侧 benchmark packaging
  模块，负责读取 ms-swift benchmark 输出并生成 Urban review 包。
- `tools/evaluation/run_sidewalk_agentic_stage2_benchmark.py` 是薄 CLI，支持：
  - `--action collect`：只收集已有 ms-swift 输出，不加载模型。
  - `--action run`：调用 ms-swift 现有 Stage2 pipeline。
  - `--action run-and-collect`：先跑 pipeline，再生成 Urban review 包。
- collector 输出：
  - `config.json`
  - `per_case_report.jsonl`
  - `error_report.jsonl`
  - `artifact_index.json`
  - `summary_report.json`
  - `summary_report.md`
- 单测使用 fake ms-swift 输出，离线验证 case 对齐、error report、
  artifact index 和 CLI collect，不访问地图 API、LLM 或 GPU。

当前 mini 闭环：

```text
run_id: stage2-full-mini-20260626
base_model: /workspace/ms-swift/.cache/ms-swift/modelscope/models/Qwen/Qwen3___5-4B
stage2a_checkpoint: /tmp/ms_swift_stage2_full_mini/stage2a_agentic_text/v0-20260626-154725/checkpoint-4
stage2b_checkpoint: /tmp/ms_swift_stage2_full_mini/stage2b_agentic_traj/v0-20260626-155228/checkpoint-4
heldout_manifest: /tmp/sidewalk_stage2_urbannav_mini/heldout.jsonl
ms_swift_output_root: /tmp/ms_swift_stage2_full_mini_host
urban_review: /tmp/ms_swift_stage2_full_mini_host/urban_review
num_cases: 1
tool_policy_counts: {"failed_tool": 1}
stage2b_head_parse_success: 1.0
stage2b_head_ADE/FDE: 0.0852 / 0.1290
```

边界说明：

- 当前是 4/1/1 UrbanNav mini split 的 held-out collect，只证明
  base / Stage2-A / Stage2-B 推理、评测、对齐和 review packaging 闭环可运行。
- 正式研究结论仍需要扩大第 6 节真实 Phase1 archive split 后复跑。
- 文本 agentic 指标在单条 failed-tool held-out 上仍为 0；这被记录进
  `error_report.jsonl`，用于人工 review 和后续扩大数据后的回归对比。

验收标准：

- [x] 同一 case_id 可对齐 base / Stage2-A / Stage2-B 输出。
- [x] route action、tool decision、trajectory 指标可聚合。
- [x] failed case 可以回放完整 messages / tool_response / artifacts。

## 9. 最终推理接口

> 实现注释：第 9 节是应用层接口，不再是训练数据构造。输入来自真实 RouteServer
> runtime，输出既要能被业务消费（final text / parsed finish / trajectory），也要能被
> 研发回放（完整 trace / tool_response / validation / ms-swift replay row）。

- [x] RouteServer online runtime 输入：
  - [x] images
  - [x] user request
  - [x] route state
  - [x] available tools
- [x] RouteServer online runtime 输出：
  - [x] final assistant text
  - [x] parsed `<finish>`
  - [x] trajectory
  - [x] tool_call / tool_response trace
  - [x] artifacts
  - [x] validation report
- [x] ms-swift `infer.py` 输出可以接入 RouteServer runtime。
- [x] RouteServer runtime 可以保存 ms-swift-compatible replay row。

当前实现：

> 实现注释：当前 ms-swift backend 通过 subprocess 调 `infer.py`，适合 smoke 和
> 单机验证；生产服务应把模型加载变成常驻进程，否则每个 case 重复加载 4B 模型成本过高。

- `agent/final_inference.py` 提供应用层稳定 API：
  - `RouteServerAgentInferenceRequest`
  - `RouteServerAgentInferenceConfig`
  - `RouteServerAgentInferenceResult`
  - `run_routeserver_final_inference()`
  - `write_routeserver_final_inference_artifacts()`
- `tools/evaluation/run_sidewalk_agentic_stage2_inference.py` 提供单 case CLI：
  - `--model-backend fake` 用于离线 smoke 和 schema 回归。
  - `--model-backend ms-swift` 使用 `MsSwiftInferenceAdapter` 调
    `swift.sidewalk_nav.cli.infer`。
- 输出文件：
  - `result.json`
  - `agent_trace.json`
  - `ms_swift_replay_row.json`
  - `validation_report.json`
  - `artifact_index.json`
- `ms_swift_replay_row.json` 保持 agent-template 兼容字段，可回放或进入后续
  review/SFT 数据闭环。

当前边界：

- CLI 的 ms-swift backend 是 subprocess 形式，适合本机/容器内执行；真正服务化
  时应在第 10 步封装成长期模型服务或容器 RPC，避免每个 case 重复加载模型。
- `--execute-tools` 当前只创建基础 `ToolExecutor`；真实地图 provider/session
  context 需要由上层 RouteServer 服务注入。

验收标准：

- [x] 单 case CLI 可以端到端运行。
- [x] batch benchmark 可以端到端运行。
- [x] 输出 trace 可以复用现有 validator/exporter/metrics。

## 10. 当前下一步

> 实现注释：此处列的是工程进入正式数据规模前的风险收敛项。优先级高于继续加新功能：
> 工具 executor 单测、真实 split 扩大、服务化推理、完整 benchmark 复跑、以及数值稳定性排查。

Stage2 工程链路已经覆盖标注、训练、benchmark packaging 和单 case 最终推理接口。
下一步不再是继续补入口，而是扩大数据、补齐真实 tool executor 测试，并把最终推理
接口服务化：

1. 第 2 节：补 `ToolExecutor` 每个 op 的 fake-provider 单测，保证真实 executor
   能稳定覆盖所有暴露工具。
2. 扩大第 6 节 Phase1 archive split，生成非 mini 的 train/val/held-out，并保持
   tool / no-tool / failed-tool slice 可控。
3. 将第 9 节 CLI 后端升级为长期 ms-swift 模型服务/RPC，避免单 case subprocess
   重复加载 4B 模型。
4. 复跑第 7-8 节完整训练与对比，使用固定完整 split，而不是 4/1/1 mini split。
5. 排查训练日志中的 `grad_norm=nan`，确认是日志/AMP 缩放问题还是梯度数值问题。
