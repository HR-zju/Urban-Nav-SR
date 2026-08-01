# RouteServer Docs

本文档是 `docs/` 目录的维护入口。新增内容应优先合并到现有正式文档，避免为一次性实验单独增加长期维护文件。

## 正式设计

| 文档 | 用途 |
| --- | --- |
| [architecture.md](architecture.md) | RouteServer 模块边界、核心链路和服务结构。 |
| [FRAMEWORK_MATH_MODEL.md](FRAMEWORK_MATH_MODEL.md) | Agent 框架数学模型、数据接口和地图接口边界。 |
| [URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md](URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md) | UrbanNav benchmark 的任务定义、真值构造、采样策略、指标和 Agent baseline。 |
| [EVALUATION.md](EVALUATION.md) | UrbanNav Module 1 一体化评测入口、prediction contract、核心指标、内部相对分数比较和扩展任务指标。 |
| [ROUTE_BELIEF_GUIDED_AGENT_DESIGN.md](ROUTE_BELIEF_GUIDED_AGENT_DESIGN.md) | Route-Belief Guided Interactive Navigation Agent 的状态、工具、VoP 和全局-局部导航统一设计。 |
| [YELP_INSTRUCTION_DATA_DESIGN.md](YELP_INSTRUCTION_DATA_DESIGN.md) | Yelp 需求挖掘、taxonomy 扩展和 instruction set 构造流程。 |
| [SIDEWALK_AGENT_EVAL_PROTOCOL.md](SIDEWALK_AGENT_EVAL_PROTOCOL.md) | SideWalk Agentic Stage2 评估协议。 |
| [SIDEWALK_PHASE1_DATA_PIPELINE.md](SIDEWALK_PHASE1_DATA_PIPELINE.md) | SideWalk Phase1 数据生成与标注流水线。 |
| [REASONING_V4_DATA_ANNOTATION_DESIGN.md](REASONING_V4_DATA_ANNOTATION_DESIGN.md) | Reasoning V4 共享输入六段合同、独立 section teacher、强 sidecar 边界与 LM/System1 loss 路由。 |
| [VQA_V2_DATA_ANNOTATION_DESIGN.md](VQA_V2_DATA_ANNOTATION_DESIGN.md) | Navigation VQA V2 的 27-task typed contract、18/9 capability gate、标签来源、可见性、LM loss、审阅与评测边界。 |
| [STAGE_II_CFM_DATA_CONTRACT.md](STAGE_II_CFM_DATA_CONTRACT.md) | Stage II 的 2.0 秒/16 步 metric FLU trajectory、严格 mask/QC、predicted context cache 与 exchange-view 边界。 |

## 工程运行

| 文档 | 用途 |
| --- | --- |
| [commands.md](commands.md) | 常用运行命令。 |
| [debugging.md](debugging.md) | 调试、排错和运行状态检查。 |
| [URBANNAV_25K_BUILD.md](URBANNAV_25K_BUILD.md) | UrbanNav final-25K 数据构建、质量约束、版本演进与验证记录。 |

## 历史参考

这些文档保留为实现背景或实验记录，不应作为当前 benchmark 的唯一依据：

| 文档 | 状态 |
| --- | --- |
| [SIDEWALK_AGENTIC_STAGE2_CHECKLIST.md](SIDEWALK_AGENTIC_STAGE2_CHECKLIST.md) | Stage2 实现 checklist；长期协议以 `SIDEWALK_AGENT_EVAL_PROTOCOL.md` 为准。 |
| [change_log.md](change_log.md) | 历史变更记录。 |

## 已合并或删除

- `DATASET_TASK_DESIGN.md`：早期 CitySeeker 迁移和 GT 构造草稿，核心内容已合并到 `URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md`。
- `INSTRUCTION_REVIEW_EXAMPLES.md`：早期指令 taxonomy review 表，当前任务类型和样例已由 benchmark 技术设计与评测协议覆盖。
- `TASK_INSTRUCTION_ROUTE_EVAL_DESIGN.md`：早期分层 benchmark 草稿，正式口径已合并到 `URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md` 和 `EVALUATION.md`。
- `URBANNAV_FAKE_EVAL_REVIEW.md`：fake evaluator smoke 记录，核心指标语义已合并到 `EVALUATION.md`。
- `URBANNAV_COMPLEX_EVAL_EXAMPLES.md`：复杂样例记录，multi-stop 与 route-update 语义已合并到正式评测文档。
- `URBANNAV_REAL_API_REVIEW.md`：真实 API review 命令草稿，当前正式采集流程以 `EVALUATION.md` 和工具脚本为准。
- `FRODOBOT_SYSTEM2_PLAN.md`：早期计划错误地把近似 FPV trajectory 当作可训练监督，已由当前 FrodoBots trust policy 取代。
- `BEV_GENERATOR_FINAL_SUMMARY.md`：一次性 BEV 生成阶段报告，现行转换链路由 `SIDEWALK_PHASE1_DATA_PIPELINE.md` 和代码维护。
- `UrbanNav Reasoning V4 Canonical Contract.pdf`：`REASONING_V4_DATA_ANNOTATION_DESIGN.md` 的重复导出，已删除以避免双源漂移。
- `coco script.pdf`：与 RouteServer 当前合同无直接关系的二进制附件，已删除。
