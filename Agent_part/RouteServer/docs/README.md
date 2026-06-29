# RouteServer Docs

本文档是 `docs/` 目录的维护入口。新增内容应优先合并到现有正式文档，避免为一次性实验单独增加长期维护文件。

## 正式设计

| 文档 | 用途 |
| --- | --- |
| [architecture.md](architecture.md) | RouteServer 模块边界、核心链路和服务结构。 |
| [FRAMEWORK_MATH_MODEL.md](FRAMEWORK_MATH_MODEL.md) | Agent 框架数学模型、数据接口和地图接口边界。 |
| [URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md](URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md) | UrbanNav benchmark 的任务定义、真值构造、采样策略、指标和 Agent baseline。 |
| [EVALUATION.md](EVALUATION.md) | UrbanNav Module 1 一体化评测入口、prediction contract、核心指标、内部相对分数比较和扩展任务指标。 |
| [YELP_INSTRUCTION_DATA_DESIGN.md](YELP_INSTRUCTION_DATA_DESIGN.md) | Yelp 需求挖掘、taxonomy 扩展和 instruction set 构造流程。 |
| [SIDEWALK_AGENT_EVAL_PROTOCOL.md](SIDEWALK_AGENT_EVAL_PROTOCOL.md) | SideWalk Agentic Stage2 评估协议。 |
| [SIDEWALK_PHASE1_DATA_PIPELINE.md](SIDEWALK_PHASE1_DATA_PIPELINE.md) | SideWalk Phase1 数据生成与标注流水线。 |

## 工程运行

| 文档 | 用途 |
| --- | --- |
| [commands.md](commands.md) | 常用运行命令。 |
| [debugging.md](debugging.md) | 调试、排错和运行状态检查。 |

## 历史参考

这些文档保留为实现背景或实验记录，不应作为当前 benchmark 的唯一依据：

| 文档 | 状态 |
| --- | --- |
| [FRODOBOT_SYSTEM2_PLAN.md](FRODOBOT_SYSTEM2_PLAN.md) | FrodoBot System2 计划。 |
| [BEV_GENERATOR_FINAL_SUMMARY.md](BEV_GENERATOR_FINAL_SUMMARY.md) | BEV 生成器阶段总结。 |
| [SIDEWALK_AGENTIC_STAGE2_CHECKLIST.md](SIDEWALK_AGENTIC_STAGE2_CHECKLIST.md) | Stage2 实现 checklist；长期协议以 `SIDEWALK_AGENT_EVAL_PROTOCOL.md` 为准。 |
| [change_log.md](change_log.md) | 历史变更记录。 |

## 已合并或删除

- `DATASET_TASK_DESIGN.md`：早期 CitySeeker 迁移和 GT 构造草稿，核心内容已合并到 `URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md`。
- `INSTRUCTION_REVIEW_EXAMPLES.md`：早期指令 taxonomy review 表，当前任务类型和样例已由 benchmark 技术设计与评测协议覆盖。
- `TASK_INSTRUCTION_ROUTE_EVAL_DESIGN.md`：早期分层 benchmark 草稿，正式口径已合并到 `URBANNAV_BENCHMARK_TECHNICAL_DESIGN.md` 和 `EVALUATION.md`。
- `URBANNAV_FAKE_EVAL_REVIEW.md`：fake evaluator smoke 记录，核心指标语义已合并到 `EVALUATION.md`。
- `URBANNAV_COMPLEX_EVAL_EXAMPLES.md`：复杂样例记录，multi-stop 与 route-update 语义已合并到正式评测文档。
- `URBANNAV_REAL_API_REVIEW.md`：真实 API review 命令草稿，当前正式采集流程以 `EVALUATION.md` 和工具脚本为准。
